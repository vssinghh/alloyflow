"""Stage 1 vision-only encoder pretraining on rendered multi-camera 4-object layouts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import h5py
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from envs.base import CAMERA_NAMES, OBJECT_NAMES, TASK_SPECS, norm_gripper_to_raw
from envs.sim_env import SimEnv
from training.model import TaskConditionedVisionFlowPolicy


def compute_layout_target_12d(
    spawns_xy: dict[str, tuple[float, float]] | dict[str, np.ndarray],
    task_id: int,
) -> np.ndarray:
    """Build 12D tabletop XY target: [src_xy(2), tgt_xy(2), pen_xy(2), cup_xy(2), bowl_xy(2), cube_xy(2)]."""
    spec = TASK_SPECS[int(task_id)]
    src_xy = np.asarray(spawns_xy[spec.source_object][:2], dtype=np.float32)
    tgt_xy = np.asarray(spawns_xy[spec.target_object][:2], dtype=np.float32)
    all_xy = np.concatenate(
        [np.asarray(spawns_xy[name][:2], dtype=np.float32) for name in OBJECT_NAMES],
        axis=0,
    )
    return np.concatenate([src_xy, tgt_xy, all_xy], axis=0).astype(np.float32)


def render_localization_dataset(
    out_path: str | Path = "data/loc_layouts_3000.npz",
    demo_h5_path: str | Path = "data/sim_demos.h5",
    num_layouts: int = 3000,
    base_seed: int = 50000,
    varied_arm_prob: float = 0.50,
) -> Path:
    """Render 3-camera labeled layouts (clean + DR, varied arm poses, all 4 object positions)."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load realistic arm poses from the 300 training demos for varied-arm-pose rendering
    arm_pose_pool: list[np.ndarray] = []
    h5_file = Path(demo_h5_path)
    if h5_file.exists():
        with h5py.File(h5_file, "r") as f:
            root = f.get("data", f)
            for k in sorted(k for k in root if k.startswith("demo_")):
                arm_pose_pool.append(root[k]["obs/proprio"][:].astype(np.float32))
    poses_flat = (
        np.concatenate(arm_pose_pool, axis=0)
        if arm_pose_pool
        else np.zeros((1, 6), dtype=np.float32)
    )

    envs = {
        dr: SimEnv(include_rgb=False, rgb_cameras=CAMERA_NAMES, domain_rand=dr)
        for dr in (False, True)
    }
    rng = np.random.default_rng(12345)

    imgs_by_cam: dict[str, np.ndarray] = {
        cam: np.empty((num_layouts, 128, 128, 3), dtype=np.uint8) for cam in CAMERA_NAMES
    }
    proprios = np.empty((num_layouts, 6), dtype=np.float32)
    task_ids = np.empty((num_layouts,), dtype=np.int64)
    domains_dr = np.empty((num_layouts,), dtype=np.bool_)
    targets_12d = np.empty((num_layouts, 12), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(num_layouts):
        tid = i % 3
        seed = base_seed + i
        is_dr = bool(i % 2 == 1)
        env = envs[is_dr]
        env.reset(task_id=tid, seed=seed)

        # Record exact settled object XY coordinates before varying the arm pose
        obj_pos = {
            name: env.data.body(env._obj_body_ids[name]).xpos[:2].copy().astype(np.float32)
            for name in OBJECT_NAMES
        }
        targets_12d[i] = compute_layout_target_12d(obj_pos, tid)

        # Vary arm pose on varied_arm_prob of layouts without moving the 4 tabletop objects
        if len(poses_flat) > 1 and rng.random() < varied_arm_prob:
            pose_6d = poses_flat[int(rng.integers(0, len(poses_flat)))].copy()
            for j_idx in range(5):
                q_adr = env._qpos_adrs[j_idx]
                env.data.qpos[q_adr] = float(pose_6d[j_idx])
            env.data.qpos[env._qpos_adrs[5]] = norm_gripper_to_raw(float(pose_6d[5]))
            mujoco.mj_forward(env.model, env.data)
            proprios[i] = pose_6d
        else:
            obs_now = env.get_obs()
            proprios[i] = obs_now["proprio"]

        for cam in CAMERA_NAMES:
            imgs_by_cam[cam][i] = env.render_camera(camera_name=cam, width=128, height=128)

        task_ids[i] = tid
        domains_dr[i] = is_dr

        if (i + 1) % 600 == 0 or (i + 1) == num_layouts:
            elapsed = time.perf_counter() - t0
            print(
                f"[RenderLoc] {i + 1}/{num_layouts} layouts rendered ({elapsed:.1f}s)",
                flush=True,
            )

    for env in envs.values():
        env.close()

    np.savez_compressed(
        out_file,
        rgb_third_person_cam=imgs_by_cam["third_person_cam"],
        rgb_overhead_cam=imgs_by_cam["overhead_cam"],
        rgb_wrist_cam=imgs_by_cam["wrist_cam"],
        proprio=proprios,
        task_id=task_ids,
        domain_dr=domains_dr,
        targets_12d=targets_12d,
    )
    print(f"[RenderLoc] Saved {num_layouts} 3-camera layouts to {out_file}", flush=True)
    return out_file


def load_demo_frame0_validation(
    demo_h5_path: str | Path = "data/sim_demos.h5",
) -> dict[str, torch.Tensor]:
    """Load frame-0 images and exact 12D object XY targets from the 300 demos in sim_demos.h5."""
    dummy_env = SimEnv(include_rgb=False)
    imgs_by_cam: dict[str, list[np.ndarray]] = {cam: [] for cam in CAMERA_NAMES}
    tasks: list[int] = []
    targets: list[np.ndarray] = []

    with h5py.File(demo_h5_path, "r") as f:
        root = f.get("data", f)
        for k in sorted(k for k in root if k.startswith("demo_")):
            grp = root[k]
            tid = int(grp.attrs["task_id"])
            seed = int(grp.attrs["seed"])
            spawns_xy = dummy_env._sample_object_spawns(np.random.default_rng(seed))
            targets.append(compute_layout_target_12d(spawns_xy, tid))
            tasks.append(tid)
            for cam in CAMERA_NAMES:
                imgs_by_cam[cam].append(grp[f"obs/rgb_{cam}"][0])

    dummy_env.close()
    out: dict[str, torch.Tensor] = {
        "task_id": torch.as_tensor(np.array(tasks), dtype=torch.long),
        "targets_12d": torch.as_tensor(np.stack(targets), dtype=torch.float32),
    }
    for cam in CAMERA_NAMES:
        out[f"rgb_{cam}"] = (
            torch.as_tensor(np.stack(imgs_by_cam[cam]), dtype=torch.uint8)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
    return out


def evaluate_policy_localization(
    policy: TaskConditionedVisionFlowPolicy,
    val_data: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate per-camera localization error (cm) and R^2 on the 300 demo layouts."""
    policy.eval()
    n = len(val_data["task_id"])
    y_true = val_data["targets_12d"]  # (N, 12)
    preds_by_head: dict[str, list[torch.Tensor]] = {cam: [] for cam in CAMERA_NAMES}

    with torch.no_grad():
        for i in range(0, n, 100):
            tid_b = val_data["task_id"][i : i + 100].to(device)
            obs_b: dict[str, torch.Tensor] = {}
            for cam in CAMERA_NAMES:
                obs_b[f"rgb_{cam}"] = (
                    val_data[f"rgb_{cam}"][i : i + 100].to(device).float().div_(255.0)
                )
            cam_tokens, z_task = policy.encode_vision_tokens(obs_b, tid_b)
            pos_preds = policy.predict_aux_positions(cam_tokens, z_task)
            for k, v in pos_preds.items():
                preds_by_head[k].append(policy.unnormalize_obj_xy(v).cpu())

    results: dict[str, Any] = {}
    for head_name, chunks in preds_by_head.items():
        pred = torch.cat(chunks, dim=0)  # (N, 12)
        src_err_cm = float((pred[:, 0:2] - y_true[:, 0:2]).norm(dim=1).mean().item() * 100.0)
        tgt_err_cm = float((pred[:, 2:4] - y_true[:, 2:4]).norm(dim=1).mean().item() * 100.0)
        all4_err_cm = float(
            (pred[:, 4:12].reshape(n, 4, 2) - y_true[:, 4:12].reshape(n, 4, 2))
            .norm(dim=2)
            .mean()
            .item()
            * 100.0
        )
        ss_res = ((pred[:, 0:4] - y_true[:, 0:4]) ** 2).sum(dim=0)
        ss_tot = ((y_true[:, 0:4] - y_true[:, 0:4].mean(dim=0)) ** 2).sum(dim=0)
        r2 = (1.0 - ss_res / ss_tot).numpy()
        results[head_name] = {
            "src_err_cm": src_err_cm,
            "tgt_err_cm": tgt_err_cm,
            "all4_err_cm": all4_err_cm,
            "r2_src_xy": [float(r2[0]), float(r2[1])],
            "r2_tgt_xy": [float(r2[2]), float(r2[3])],
        }
    return results


def pretrain_vision_encoders(
    policy: TaskConditionedVisionFlowPolicy,
    loc_npz_path: str | Path = "data/loc_layouts_3000.npz",
    demo_h5_path: str | Path = "data/sim_demos.h5",
    steps: int = 3000,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | str = "mps",
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Pretrain camera_encoders, task_embedding, and per-camera position heads on 3,000 layouts."""
    dev = torch.device(device)
    npz_file = Path(loc_npz_path)
    if not npz_file.exists():
        render_localization_dataset(out_path=npz_file, demo_h5_path=demo_h5_path)

    raw = np.load(npz_file)
    imgs_cpu: dict[str, torch.Tensor] = {
        cam: torch.as_tensor(raw[f"rgb_{cam}"], dtype=torch.uint8).permute(0, 3, 1, 2).contiguous()
        for cam in CAMERA_NAMES
    }
    task_cpu = torch.as_tensor(raw["task_id"], dtype=torch.long)
    targets_cpu = torch.as_tensor(raw["targets_12d"], dtype=torch.float32)

    obj_mean = targets_cpu.mean(dim=0)
    obj_std = targets_cpu.std(dim=0).clamp(min=1e-2)
    policy.set_obj_xy_stats(obj_mean, obj_std)
    policy.to(dev)

    # Vision-only Stage 1 optimizer: task_embedding + camera_encoders + aux_cam_pos_heads only
    vis_params = (
        list(policy.task_embedding.parameters())
        + list(policy.camera_encoders.parameters())
        + list(policy.aux_cam_pos_heads.parameters())
    )
    opt = torch.optim.AdamW(vis_params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-5)

    n = len(task_cpu)
    val_data = load_demo_frame0_validation(demo_h5_path)
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        policy.train()
        idx = torch.randint(0, n, (batch_size,))
        tid_b = task_cpu[idx].to(dev)
        y_norm = policy.normalize_obj_xy(targets_cpu[idx].to(dev))
        obs_b: dict[str, torch.Tensor] = {}
        for cam in CAMERA_NAMES:
            obs_b[f"rgb_{cam}"] = imgs_cpu[cam][idx].to(dev).float().div_(255.0)

        cam_tokens, z_task = policy.encode_vision_tokens(obs_b, tid_b)
        preds = policy.predict_aux_positions(cam_tokens, z_task)

        loss = (
            F.mse_loss(preds["overhead_cam"], y_norm)
            + F.mse_loss(preds["third_person_cam"], y_norm)
            + 0.5 * F.mse_loss(preds["wrist_cam"], y_norm)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(vis_params, 1.0)
        opt.step()
        sched.step()

        if step % 600 == 0 or step == steps:
            val_metrics = evaluate_policy_localization(policy, val_data, dev)
            ov = val_metrics["overhead_cam"]
            tp = val_metrics["third_person_cam"]
            wr = val_metrics["wrist_cam"]
            print(
                f"[PretrainVis {step:04d}/{steps:04d}] loss={loss.item():.4f} | "
                f"ov_src={ov['src_err_cm']:.2f}cm (tgt={ov['tgt_err_cm']:.2f}cm, all4={ov['all4_err_cm']:.2f}cm, R2={ov['r2_src_xy'][0]:+.2f},{ov['r2_src_xy'][1]:+.2f}) | "
                f"tp_src={tp['src_err_cm']:.2f}cm | wr_src={wr['src_err_cm']:.2f}cm | {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    final_val = evaluate_policy_localization(policy, val_data, dev)
    if save_path is not None:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "val_localization": final_val,
            },
            sp,
        )
        print(f"[PretrainVis] Saved pretrained encoder weights to {sp}", flush=True)

    return final_val
