"""Pre-extract z_fused in .eval() mode and run 5-epoch obs_dropout=0.0 cooldown on Exp08.

Source checkpoint:
  checkpoints/exp08_dart_full_900/best_policy.pt

Settings:
  - Stage A (task_embedding, proprio_mlp, camera_encoders) frozen in .eval() mode
  - Stage B (predict_velocity: obs_proj, act_proj, time_encoder, res_blocks, out_head) trained with dropout=0.0
  - Mode "cfm_only" (promoted to baseline): 5 epochs, batch_size=256, lr=5e-5 -> 1e-6 (cosine), weight_decay=1e-4, k_flow=4
  - Mode "distill_mc": 5 epochs, batch_size=256, lr=1e-4 -> 1e-6 (cosine), mc_teacher_k=8

Output paths:
  checkpoints/exp08_cooldown_cfm_only/best_policy.pt (promoted to checkpoints/exp08b_cooldown_k0/best_policy.pt)
  checkpoints/exp08_cooldown_distill_mc/best_policy.pt

120-episode benchmark result (k=0, policy.eval()):
  cfm_only (exp08b_cooldown_k0): 112 / 120 (93.3% total: 58/60 train, 54/60 test; BLOCKED 21/120)
  distill_mc:                    110 / 120 (91.7% total: 58/60 train, 52/60 test; BLOCKED 28/120)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from envs.base import CAMERA_NAMES
from training.dataset import build_action_chunks
from training.flow_matching import ConditionalFlowMatcher
from training.trainer import load_policy_checkpoint, save_policy_checkpoint


CACHE_PATH = Path("checkpoints/exp08_z_fused_cache.pt")
CKPT_IN = Path("checkpoints/exp08_dart_full_900/best_policy.pt")


def extract_or_load_cache(device: torch.device) -> dict[str, torch.Tensor]:
    if CACHE_PATH.exists():
        print(f"Loading cached z_fused from {CACHE_PATH} ...")
        return torch.load(CACHE_PATH, map_location="cpu", weights_only=False)

    t0 = time.time()
    policy, cfg, _ = load_policy_checkpoint(CKPT_IN, device=device)
    policy.eval()

    z_list: list[torch.Tensor] = []
    x1_list: list[torch.Tensor] = []
    task_list: list[torch.Tensor] = []
    val_mask_list: list[torch.Tensor] = []

    h5_path = Path("data/sim_demos_v2_dart_full_900.h5")
    with h5py.File(h5_path, "r") as f:
        demo_names = sorted(k for k in f.keys() if k.startswith("demo_"))
        num_demos = len(demo_names)
        print(f"Extracting z_fused for {num_demos} demos on {device} ...")
        for idx, dname in enumerate(demo_names):
            grp = f[dname]
            task_id = int(grp.attrs["task_id"])
            seed = int(grp.attrs["seed"])
            # Hold out 10% of demos per task as validation
            is_val = (idx % 10) == 9

            proprio_np = grp["obs"]["proprio"][:].astype(np.float32)
            actions_np = grp["actions"][:].astype(np.float32)
            chunks_np = build_action_chunks(actions_np, chunk_size=cfg.chunk_size)
            t_len = proprio_np.shape[0]

            obs_batch: dict[str, torch.Tensor] = {
                "proprio": torch.as_tensor(proprio_np, dtype=torch.float32, device=device)
            }
            for cam in CAMERA_NAMES:
                obs_batch[f"rgb_{cam}"] = torch.as_tensor(
                    grp["obs"][f"rgb_{cam}"][:], dtype=torch.uint8, device=device
                )
            tid_t = torch.full((t_len,), task_id, dtype=torch.long, device=device)

            with torch.no_grad():
                z_fused = policy.extract_obs_features(obs_batch, tid_t, return_aux=False)
                chunks_t = torch.as_tensor(chunks_np, dtype=torch.float32, device=device)
                x1_norm = policy.normalize_actions(chunks_t)

            z_list.append(z_fused.cpu().to(torch.float16))
            x1_list.append(x1_norm.cpu().to(torch.float16))
            task_list.append(tid_t.cpu().to(torch.uint8))
            val_mask_list.append(torch.full((t_len,), is_val, dtype=torch.bool))

            if (idx + 1) % 100 == 0 or (idx + 1) == num_demos:
                elapsed = time.time() - t0
                print(f"  [{idx + 1}/{num_demos}] demos extracted ({elapsed:.1f}s)")

    cache = {
        "z_fused": torch.cat(z_list, dim=0),
        "x1_norm": torch.cat(x1_list, dim=0),
        "task_id": torch.cat(task_list, dim=0),
        "is_val": torch.cat(val_mask_list, dim=0),
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE_PATH)
    print(f"Saved cache ({cache['z_fused'].shape[0]} samples) to {CACHE_PATH}")
    return cache


@torch.no_grad()
def evaluate_k0_vs_teacher(
    student_policy: torch.nn.Module,
    teacher_policy: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    z_val: torch.Tensor,
    x1_val_norm: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    student_policy.eval()
    teacher_policy.eval()

    b = z_val.shape[0]
    steps = 5
    dt = 1.0 / float(steps)
    mc_k = 16

    gen = torch.Generator(device=device)
    gen.manual_seed(12345)
    x0 = torch.randn(
        (b, matcher.config.chunk_size, matcher.config.action_dim),
        device=device,
        dtype=torch.float32,
        generator=gen,
    )

    # 1. Student k=0 rollout in .eval() mode
    x_stu = x0.clone()
    for s in range(steps):
        tau = torch.full((b,), float(s) * dt, device=device, dtype=torch.float32)
        v = student_policy.predict_velocity(z_val, x_stu, tau)
        x_stu = x_stu + dt * v

    # 2. Teacher k=16 MC rollout
    teacher_policy.obs_dropout.train()
    z_mc = z_val.unsqueeze(1).expand(b, mc_k, -1).reshape(b * mc_k, -1)
    x_tea = x0.clone()
    for s in range(steps):
        tau_mc = torch.full((b * mc_k,), float(s) * dt, device=device, dtype=torch.float32)
        x_mc = (
            x_tea.unsqueeze(1)
            .expand(b, mc_k, matcher.config.chunk_size, matcher.config.action_dim)
            .reshape(b * mc_k, matcher.config.chunk_size, matcher.config.action_dim)
        )
        v = (
            teacher_policy.predict_velocity(z_mc, x_mc, tau_mc)
            .reshape(b, mc_k, matcher.config.chunk_size, matcher.config.action_dim)
            .mean(dim=1)
        )
        x_tea = x_tea + dt * v
    teacher_policy.eval()

    diff_vs_teacher = x_stu - x_tea
    pan_bias_norm = float(diff_vs_teacher[:, 0, 0].mean().item())
    rmse_vs_teacher_norm = float(diff_vs_teacher.pow(2).mean().sqrt().item())

    stu_unnorm = student_policy.unnormalize_actions(x_stu)
    gt_unnorm = student_policy.unnormalize_actions(x1_val_norm)
    val_ode_mse = float((stu_unnorm - gt_unnorm).pow(2).mean().item())
    tea_unnorm = teacher_policy.unnormalize_actions(x_tea)
    tea_ode_mse = float((tea_unnorm - gt_unnorm).pow(2).mean().item())

    return {
        "pan_bias_vs_mc16_norm": pan_bias_norm,
        "rmse_vs_mc16_norm": rmse_vs_teacher_norm,
        "student_k0_ode_mse": val_ode_mse,
        "teacher_mc16_ode_mse": tea_ode_mse,
    }


def run_cooldown(
    mode: str,
    cache: dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-4,
) -> Path:
    teacher_policy, cfg, raw_meta = load_policy_checkpoint(CKPT_IN, device=device)
    teacher_policy.eval()

    student_cfg = dataclasses.replace(cfg, dropout=0.0)
    student_policy, _, _ = load_policy_checkpoint(CKPT_IN, device=device)
    student_policy.config = student_cfg
    # Disable all dropout in student velocity head
    student_policy.obs_dropout.p = 0.0
    for blk in student_policy.res_blocks:
        for mod in blk.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
    student_policy.eval()

    matcher = ConditionalFlowMatcher(student_cfg)
    dim_w = matcher._dim_weights.to(device=device).view(1, 1, 1, -1)

    train_mask = ~cache["is_val"]
    val_mask = cache["is_val"]

    z_train = cache["z_fused"][train_mask].to(device=device, dtype=torch.float32)
    x1_train = cache["x1_norm"][train_mask].to(device=device, dtype=torch.float32)

    z_val = cache["z_fused"][val_mask][:1024].to(device=device, dtype=torch.float32)
    x1_val = cache["x1_norm"][val_mask][:1024].to(device=device, dtype=torch.float32)

    # Freeze vision/proprio/task encoders; optimize only the velocity head
    vel_params = (
        list(student_policy.obs_proj.parameters())
        + list(student_policy.act_proj.parameters())
        + list(student_policy.time_encoder.parameters())
        + list(student_policy.res_blocks.parameters())
        + list(student_policy.out_head.parameters())
    )
    optimizer = torch.optim.AdamW(vel_params, lr=lr, weight_decay=1e-4)
    n_train = z_train.shape[0]
    steps_per_epoch = n_train // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-6
    )

    init_stats = evaluate_k0_vs_teacher(
        student_policy, teacher_policy, matcher, z_val, x1_val, device
    )
    print(f"\n=== Cooldown Mode: {mode} (epochs={epochs}, lr={lr}) ===")
    print(f"  [Epoch 0 (Raw Exp08 k=0)] {init_stats}")

    k_flow = 4
    mc_teacher_k = 8

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        perm = torch.randperm(n_train, device=device)
        ep_loss = 0.0

        # Keep student in .eval() so all dropout is 100% disabled
        student_policy.eval()
        teacher_policy.eval()
        teacher_policy.obs_dropout.train()

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            z_b = z_train[idx]  # (B, 192)
            x1_b = x1_train[idx]  # (B, 16, 6)
            b = z_b.shape[0]

            tau = matcher.sample_stratified_timesteps(b, k_flow, device)  # (B, K)
            x0 = torch.randn(
                (b, k_flow, cfg.chunk_size, cfg.action_dim), device=device, dtype=torch.float32
            )
            x1_exp = x1_b.unsqueeze(1).expand(b, k_flow, cfg.chunk_size, cfg.action_dim)
            tau_exp = tau.unsqueeze(-1).unsqueeze(-1)
            x_tau = (1.0 - tau_exp) * x0 + tau_exp * x1_exp
            u_target = x1_exp - x0

            v_stu = student_policy.predict_velocity(z_b, x_tau, tau)

            if mode == "cfm_only":
                sq_err = (v_stu - u_target).pow(2)
                loss = (torch.sum(sq_err * dim_w, dim=-1) / torch.sum(dim_w)).mean()
            elif mode == "distill_mc":
                with torch.no_grad():
                    # Average teacher over mc_teacher_k dropout masks
                    z_rep = (
                        z_b.unsqueeze(1)
                        .expand(b, mc_teacher_k, -1)
                        .reshape(b * mc_teacher_k, -1)
                    )
                    x_rep = (
                        x_tau.unsqueeze(1)
                        .expand(b, mc_teacher_k, k_flow, cfg.chunk_size, cfg.action_dim)
                        .reshape(b * mc_teacher_k, k_flow, cfg.chunk_size, cfg.action_dim)
                    )
                    tau_rep = (
                        tau.unsqueeze(1)
                        .expand(b, mc_teacher_k, k_flow)
                        .reshape(b * mc_teacher_k, k_flow)
                    )
                    v_tea = (
                        teacher_policy.predict_velocity(z_rep, x_rep, tau_rep)
                        .reshape(b, mc_teacher_k, k_flow, cfg.chunk_size, cfg.action_dim)
                        .mean(dim=1)
                    )
                sq_distill = (v_stu - v_tea).pow(2)
                loss_distill = (torch.sum(sq_distill * dim_w, dim=-1) / torch.sum(dim_w)).mean()
                sq_cfm = (v_stu - u_target).pow(2)
                loss_cfm = (torch.sum(sq_cfm * dim_w, dim=-1) / torch.sum(dim_w)).mean()
                loss = loss_distill + 0.05 * loss_cfm
            else:
                raise ValueError(f"Unknown mode: {mode}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vel_params, 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += float(loss.item())

        teacher_policy.eval()
        stats = evaluate_k0_vs_teacher(
            student_policy, teacher_policy, matcher, z_val, x1_val, device
        )
        print(
            f"  [Epoch {ep}/{epochs}] loss={ep_loss / steps_per_epoch:.6f} "
            f"({time.time() - t_ep:.1f}s) | {stats}"
        )

    out_dir = Path(f"checkpoints/exp08_cooldown_{mode}")
    out_path = save_policy_checkpoint(
        out_dir / "best_policy.pt",
        policy=student_policy,
        config=student_cfg,
        epoch=40 + epochs,
        metrics=stats,
        include_rng=False,
    )
    print(f"Saved cooldown checkpoint to {out_path}")
    return out_path


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cache = extract_or_load_cache(device)
    run_cooldown("cfm_only", cache, device, epochs=5, batch_size=256, lr=5e-5)
    run_cooldown("distill_mc", cache, device, epochs=5, batch_size=256, lr=1e-4)


if __name__ == "__main__":
    main()
