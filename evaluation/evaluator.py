"""Closed-loop multi-task simulation policy evaluator with automatic GIF diagnostics."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
import multiprocessing as mp
from pathlib import Path
from typing import Any

import h5py
import mujoco
import numpy as np
import torch

from envs.base import CAMERA_NAMES, TASK_SPECS
from envs.sim_env import SimEnv
from evaluation.visualizer import RolloutVisualizer
from training.dataset import build_action_chunks, build_proprio_history_step
from training.flow_matching import ConditionalFlowMatcher, TemporalEnsembler
from training.trainer import load_policy_checkpoint


@dataclass
class EpisodeEvalResult:
    """Structured result and diagnostic metrics for a single closed-loop rollout."""

    episode_label: str
    task_id: int
    task_name: str
    seed: int
    domain: str
    success: bool
    all_constraints_passed: bool
    status_text: str
    min_pinch_src_xy_cm: float
    max_src_z_cm: float
    max_src_lift_cm: float
    final_src_tgt_xy_cm: float
    max_bystander_disp_cm: float
    bystander_tipped: bool
    max_target_disp_cm: float
    target_tipped: bool
    mean_latency_ms: float
    grasp_close_step: int | None = None
    grasp_close_xy_cm: float | None = None
    grasp_close_dz_cm: float | None = None
    grasp_class: str | None = None
    grasp_blocked: bool = False
    blocked_pad: str | None = None
    gif_path: str | None = None
    strip_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_grasp_steps(steps_75: list[dict[str, Any]]) -> tuple[str, str]:
    """Classify approach/descent into BLOCKED, EARLY_CLOSE, or OTHER and detect first contact pad."""
    if not steps_75:
        return "OTHER", "neighbor"
    tc = next((s["step"] for s in steps_75 if s["grip_cmd"] < 0.48), min(48, len(steps_75) - 1))
    sc = steps_75[tc]
    close_dz = float(sc["pinch_src_z_cm"])

    w5 = steps_75[max(0, tc - 5) : tc + 1]
    zgap_tc = float(sc.get("z_gap_cmd_cm", 0.0))
    max_zgap_w5 = float(max(s.get("z_gap_cmd_cm", 0.0) for s in w5))
    dz_vel_3 = float(
        (steps_75[tc]["pinch_src_z_cm"] - steps_75[max(0, tc - 3)]["pinch_src_z_cm"]) / 3.0
    )
    disp_pre = float(max(s["src_disp_xy_cm"] for s in steps_75[: tc + 1]))
    z_comp_mm = float(
        max(
            0.0,
            (steps_75[0]["src_z_cm"] - min(s["src_z_cm"] for s in steps_75[: tc + 1])) * 10.0,
        )
    )

    is_stalled_or_collapsed = (dz_vel_3 >= -0.10) or (z_comp_mm >= 10.0)
    is_high_or_shoved = (close_dz >= 1.15) or (disp_pre >= 1.40 and max_zgap_w5 >= 1.00)
    has_large_zgap = (max_zgap_w5 >= 0.80) or (zgap_tc >= 0.65)

    if is_high_or_shoved and has_large_zgap and is_stalled_or_collapsed:
        grasp_cls = "BLOCKED"
    elif close_dz >= 1.15 and not has_large_zgap and dz_vel_3 < -0.10:
        grasp_cls = "EARLY_CLOSE"
    else:
        grasp_cls = "OTHER"

    first_pad = "neighbor"
    for s in steps_75[: tc + 1]:
        cons = s.get("gripper_src_contacts", [])
        if cons:
            has_f = any("fixed_jaw_pad" in c for c in cons)
            has_m = any("moving_jaw_pad" in c for c in cons)
            if has_f and has_m:
                first_pad = "both"
            elif has_f:
                first_pad = "fixed_jaw_pad"
            elif has_m:
                first_pad = "moving_jaw_pad"
            else:
                first_pad = "other_geom"
            break
    return grasp_cls, first_pad


def _benchmark_worker(args: tuple[str, str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Worker process for parallel benchmark evaluation."""
    ckpt_path, device_str, jobs = args
    if device_str == "cpu":
        torch.set_num_threads(1)
    evaluator = SimPolicyEvaluator(
        checkpoint_path=ckpt_path,
        device=device_str,
    )
    out: list[dict[str, Any]] = []
    try:
        for job in jobs:
            res = evaluator.run_episode(**job)
            out.append(res.to_dict())
    finally:
        evaluator.close()
    return out


def diagnose_rollout_status(
    *,
    constraints: dict[str, Any],
    min_pinch_src_xy_cm: float,
    max_src_lift_cm: float,
    final_src_tgt_xy_cm: float,
    grasp_close_xy_cm: float | None = None,
    grasp_close_dz_cm: float | None = None,
) -> str:
    """Return a concise human-readable PASS or FAIL diagnosis string."""
    if constraints["all_constraints_passed"]:
        return "PASS: Goal & constraints satisfied"
    if constraints["bystander_tipped"] or constraints["max_bystander_displacement_m"] > 0.03:
        return (
            f"FAIL: Bystander disturbed "
            f"({constraints['max_bystander_displacement_m'] * 100:.1f}cm)"
        )
    if constraints["target_tipped"] or constraints["max_target_displacement_m"] > 0.03:
        return (
            f"FAIL: Target shoved/tipped "
            f"({constraints['max_target_displacement_m'] * 100:.1f}cm)"
        )
    if min_pinch_src_xy_cm > 3.0:
        return f"FAIL: Missed source reach (min_xy={min_pinch_src_xy_cm:.1f}cm)"
    if max_src_lift_cm < 0.8:
        if grasp_close_xy_cm is not None and grasp_close_xy_cm >= 1.5:
            dz_str = f", dz={grasp_close_dz_cm:+.1f}cm" if grasp_close_dz_cm is not None else ""
            return f"FAIL: Closed off-center (close_xy={grasp_close_xy_cm:.1f}cm{dz_str})"
        return f"FAIL: Reached src ({min_pinch_src_xy_cm:.1f}cm), failed grasp/lift"
    return f"FAIL: Lifted (+{max_src_lift_cm:.1f}cm), missed target ({final_src_tgt_xy_cm:.1f}cm)"


class SimPolicyEvaluator:
    """Closed-loop evaluator for AlloyFlow policies with per-episode GIF generation."""

    def __init__(
        self,
        checkpoint_path: str | Path = "checkpoints/exp09_e2e_nodropout/best_policy.pt",
        device: str = "auto",
        cam_render_size: int = 256,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.policy, self.config, self.ckpt_meta = load_policy_checkpoint(
            self.checkpoint_path, device=device
        )
        self.policy.eval()
        self.device = next(self.policy.parameters()).device
        self.matcher = ConditionalFlowMatcher(self.config)
        self.env = SimEnv(domain_rand=False, include_rgb=True)
        self._fk_data = mujoco.MjData(self.env.model)
        self._geom_id_to_name = {
            i: mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
            for i in range(self.env.model.ngeom)
        }
        self.visualizer = RolloutVisualizer(self.env, cam_size=cam_render_size)

    def close(self) -> None:
        """Release MuJoCo renderers."""
        self.visualizer.close()
        self.env.close()

    def _prepare_obs_tensors(
        self,
        obs: dict[str, Any],
        proprio_history: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert single-step SimEnv obs dict into batched (1, ...) device tensors."""
        out: dict[str, torch.Tensor] = {
            "proprio": torch.as_tensor(obs["proprio"][None, :], dtype=torch.float32, device=self.device)
        }
        if proprio_history is not None:
            out["proprio_history"] = torch.as_tensor(
                proprio_history[None, ...], dtype=torch.float32, device=self.device
            )
        for cam in CAMERA_NAMES:
            key = f"rgb_{cam}"
            out[key] = torch.as_tensor(obs[key][None, ...], dtype=torch.uint8, device=self.device)
        return out

    def run_episode(
        self,
        *,
        task_id: int = 0,
        seed: int = 2000,
        domain_rand: bool = False,
        episode_label: str | None = None,
        gt_proprio: np.ndarray | None = None,
        gt_actions: np.ndarray | None = None,
        max_steps: int = 280,
        ode_steps: int | None = None,
        seed_offset: int = 0,
        exec_horizon: int = 1,
        use_temporal_ensemble: bool = True,
        save_gif: bool = False,
        gif_dir: str | Path = "checkpoints/eval_gifs",
        gif_filename: str | None = None,
        save_strip: bool = True,
        fps: int = 15,
        frame_stride: int = 2,
    ) -> EpisodeEvalResult:
        """Execute one closed-loop episode and optionally save a diagnostic GIF + strip."""
        spec = TASK_SPECS[int(task_id)]
        dom_str = "sim_dr" if domain_rand else "sim_clean"
        label = episode_label or f"task{task_id}_seed{seed}"
        steps_ode = int(ode_steps if ode_steps is not None else 5)

        self.env.domain_rand = bool(domain_rand)
        obs = self.env.reset(task_id=int(task_id), seed=int(seed))
        ensembler = TemporalEnsembler(
            chunk_size=self.config.chunk_size,
            action_dim=self.config.action_dim,
            decay=self.config.temporal_ensemble_decay,
        )
        if save_gif:
            self.visualizer.reset()

        gt_chunks = (
            build_action_chunks(gt_actions, chunk_size=self.config.chunk_size)
            if gt_actions is not None
            else None
        )

        src_bid = self.env._obj_body_ids[spec.source_object]
        tgt_bid = self.env._obj_body_ids[spec.target_object]
        init_src_pos = self.env.data.body(src_bid).xpos.copy()
        init_src_z_cm = float(init_src_pos[2]) * 100.0

        min_pinch_src_xy_cm = float("inf")
        max_src_z_cm = init_src_z_cm
        grasp_close_step: int | None = None
        grasp_close_xy_cm: float | None = None
        grasp_close_dz_cm: float | None = None
        latencies_ms: list[float] = []
        steps_75: list[dict[str, Any]] = []

        cached_chunk: np.ndarray | None = None
        cached_kp: dict[str, np.ndarray] = {}
        cached_weights: np.ndarray = np.full((len(CAMERA_NAMES),), 1.0 / len(CAMERA_NAMES), dtype=np.float32)
        prop_history: list[np.ndarray] = []
        lags = self.config.proprio_history_lags
        passed_constraints: dict[str, Any] | None = None

        for step in range(max_steps):
            prop_cur = obs["proprio"].astype(np.float32).copy()
            prop_history.append(prop_cur)

            need_query = use_temporal_ensemble or cached_chunk is None or (step % max(1, exec_horizon) == 0)
            if need_query:
                prop_hist_np = build_proprio_history_step(
                    prop_history, len(prop_history) - 1, lags=lags
                )
                obs_t = self._prepare_obs_tensors(obs, proprio_history=prop_hist_np)
                torch.manual_seed(int(seed) * 1000 + step + int(seed_offset))
                t0 = time.perf_counter()
                if save_gif:
                    chunk_t, aux = self.matcher.sample_action_chunk(
                        self.policy,
                        obs_t,
                        task_id=int(task_id),
                        ode_steps=steps_ode,
                        clip_to_limits=True,
                        return_aux=True,
                    )
                    cached_kp = {
                        cam: kp_t[0].detach().cpu().numpy()
                        for cam, kp_t in aux["keypoints"].items()
                    }
                    if "camera_weights" in aux:
                        cached_weights = aux["camera_weights"][0].detach().cpu().numpy()
                else:
                    chunk_t = self.matcher.sample_action_chunk(
                        self.policy,
                        obs_t,
                        task_id=int(task_id),
                        ode_steps=steps_ode,
                        clip_to_limits=True,
                        return_aux=False,
                    )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(latency_ms)
                assert isinstance(chunk_t, torch.Tensor)
                cached_chunk = chunk_t[0].detach().cpu().numpy()
            else:
                latency_ms = latencies_ms[-1] if latencies_ms else 0.0

            assert cached_chunk is not None
            if use_temporal_ensemble:
                action_6d = ensembler.update(cached_chunk)
                viz_pred_chunk = cached_chunk
            else:
                offset = step % max(1, exec_horizon)
                action_6d = cached_chunk[offset]
                viz_pred_chunk = cached_chunk[offset:]

            pinch_pos = self.env.data.site(self.env._pinch_site_id).xpos
            src_pos = self.env.data.body(src_bid).xpos
            tgt_pos = self.env.data.body(tgt_bid).xpos

            cur_pinch_src_xy_cm = float(np.linalg.norm(pinch_pos[:2] - src_pos[:2])) * 100.0
            cur_pinch_src_dz_cm = float(pinch_pos[2] - src_pos[2]) * 100.0
            cur_src_z_cm = float(src_pos[2]) * 100.0
            cur_src_tgt_xy_cm = float(np.linalg.norm(src_pos[:2] - tgt_pos[:2])) * 100.0
            min_pinch_src_xy_cm = min(min_pinch_src_xy_cm, cur_pinch_src_xy_cm)
            max_src_z_cm = max(max_src_z_cm, cur_src_z_cm)

            if step < 75:
                self._fk_data.qpos[:5] = action_6d[:5]
                mujoco.mj_kinematics(self.env.model, self._fk_data)
                cmd_pinch_z_cm = float(self._fk_data.site(self.env._pinch_site_id).xpos[2]) * 100.0
                z_gap_cmd_cm = float(pinch_pos[2] * 100.0 - cmd_pinch_z_cm)
                src_disp_xy_cm = float(np.linalg.norm(src_pos[:2] - init_src_pos[:2])) * 100.0
                gripper_src_contacts: list[str] = []
                for c_i in range(self.env.data.ncon):
                    con = self.env.data.contact[c_i]
                    g1 = self._geom_id_to_name[int(con.geom1)]
                    g2 = self._geom_id_to_name[int(con.geom2)]
                    pair = f"{g1}<->{g2}"
                    is_grp = "jaw" in pair or "palm" in pair or "gripper" in pair
                    if is_grp and (
                        spec.source_object in pair or "cup" in pair or "pen_holder" in pair
                    ):
                        gripper_src_contacts.append(pair)
                steps_75.append(
                    {
                        "step": int(step),
                        "grip_cmd": float(action_6d[5]),
                        "pinch_src_xy_cm": cur_pinch_src_xy_cm,
                        "pinch_src_z_cm": cur_pinch_src_dz_cm,
                        "src_z_cm": cur_src_z_cm,
                        "src_disp_xy_cm": src_disp_xy_cm,
                        "z_gap_cmd_cm": z_gap_cmd_cm,
                        "gripper_src_contacts": gripper_src_contacts,
                    }
                )

            if grasp_close_step is None and float(action_6d[5]) < 0.48:
                grasp_close_step = int(step)
                grasp_close_xy_cm = cur_pinch_src_xy_cm
                grasp_close_dz_cm = cur_pinch_src_dz_cm

            cur_constraints = self.env.check_constraints(int(task_id))
            cur_succ = bool(cur_constraints["all_constraints_passed"])

            if save_gif:
                gt_idx = min(step, len(gt_proprio) - 1) if gt_proprio is not None else 0
                cur_gt_prop = gt_proprio[gt_idx] if gt_proprio is not None else None
                cur_gt_chunk = gt_chunks[gt_idx] if gt_chunks is not None else None
                live_status = (
                    "PASS: Goal & constraints satisfied"
                    if cur_succ
                    else f"IN PROGRESS (min_src_xy={min_pinch_src_xy_cm:.1f}cm)"
                )
                self.visualizer.record_step(
                    step_idx=step,
                    max_steps=max_steps,
                    task_id=int(task_id),
                    episode_label=label,
                    domain=dom_str,
                    pred_chunk=viz_pred_chunk,
                    gt_chunk=cur_gt_chunk,
                    keypoints=cached_kp,
                    camera_weights=cached_weights,
                    proprio=obs["proprio"],
                    gt_proprio=cur_gt_prop,
                    pinch_src_xy_cm=cur_pinch_src_xy_cm,
                    min_pinch_src_xy_cm=min_pinch_src_xy_cm,
                    src_z_cm=cur_src_z_cm,
                    max_src_z_cm=max_src_z_cm,
                    src_tgt_xy_cm=cur_src_tgt_xy_cm,
                    bystander_disp_cm=float(self.env.max_bystander_displacement) * 100.0,
                    task_success=cur_succ,
                    status_text=live_status,
                    latency_ms=latency_ms,
                )

            if cur_succ:
                passed_constraints = cur_constraints
                break

            obs = self.env.step(action_6d)
            post_constraints = self.env.check_constraints(int(task_id))
            if bool(post_constraints["all_constraints_passed"]):
                passed_constraints = post_constraints
                break

        constraints = (
            passed_constraints
            if passed_constraints is not None
            else self.env.check_constraints(int(task_id))
        )
        src_pos_end = self.env.data.body(src_bid).xpos
        tgt_pos_end = self.env.data.body(tgt_bid).xpos
        final_src_tgt_xy_cm = float(np.linalg.norm(src_pos_end[:2] - tgt_pos_end[:2])) * 100.0
        max_src_lift_cm = max(0.0, max_src_z_cm - init_src_z_cm)
        grasp_cls, first_pad = classify_grasp_steps(steps_75)

        status_text = diagnose_rollout_status(
            constraints=constraints,
            min_pinch_src_xy_cm=min_pinch_src_xy_cm,
            max_src_lift_cm=max_src_lift_cm,
            final_src_tgt_xy_cm=final_src_tgt_xy_cm,
            grasp_close_xy_cm=grasp_close_xy_cm,
            grasp_close_dz_cm=grasp_close_dz_cm,
        )

        saved_gif_path: str | None = None
        saved_strip_path: str | None = None
        if save_gif and self.visualizer.records:
            # Stamp the final diagnosis onto the last 12 frames so the GIF end-card shows the exact reason
            for tail_idx in range(max(0, len(self.visualizer.records) - 12), len(self.visualizer.records)):
                rec = self.visualizer.records[tail_idx]
                rec.task_success = bool(constraints["all_constraints_passed"])
                rec.status_text = status_text
                self.visualizer.frames[tail_idx] = self.visualizer._render_dashboard_frame(rec)

            out_dir = Path(gif_dir)
            fname = gif_filename or f"{label}_task{task_id}.gif"
            gif_out = self.visualizer.save_gif(out_dir / fname, fps=fps, frame_stride=frame_stride)
            saved_gif_path = str(gif_out.resolve())
            if save_strip:
                strip_out = self.visualizer.save_summary_strip(
                    gif_out.with_name(f"{gif_out.stem}_strip.png")
                )
                saved_strip_path = str(strip_out.resolve())

        return EpisodeEvalResult(
            episode_label=label,
            task_id=int(task_id),
            task_name=spec.name,
            seed=int(seed),
            domain=dom_str,
            success=bool(constraints["task_success"]),
            all_constraints_passed=bool(constraints["all_constraints_passed"]),
            status_text=status_text,
            min_pinch_src_xy_cm=min_pinch_src_xy_cm,
            max_src_z_cm=max_src_z_cm,
            max_src_lift_cm=max_src_lift_cm,
            final_src_tgt_xy_cm=final_src_tgt_xy_cm,
            max_bystander_disp_cm=float(constraints["max_bystander_displacement_m"]) * 100.0,
            bystander_tipped=bool(constraints["bystander_tipped"]),
            max_target_disp_cm=float(constraints["max_target_displacement_m"]) * 100.0,
            target_tipped=bool(constraints["target_tipped"]),
            mean_latency_ms=float(np.mean(latencies_ms)) if latencies_ms else 0.0,
            grasp_close_step=grasp_close_step,
            grasp_close_xy_cm=grasp_close_xy_cm,
            grasp_close_dz_cm=grasp_close_dz_cm,
            grasp_class=grasp_cls,
            grasp_blocked=(grasp_cls == "BLOCKED"),
            blocked_pad=first_pad if grasp_cls == "BLOCKED" else None,
            gif_path=saved_gif_path,
            strip_path=saved_strip_path,
        )

    def run_demo_case(
        self,
        demo_key: str = "demo_0000",
        h5_path: str | Path = "data/sim_demos_v2_dart_full_900.h5",
        *,
        max_steps: int = 280,
        ode_steps: int | None = None,
        seed_offset: int = 0,
        exec_horizon: int = 1,
        use_temporal_ensemble: bool = True,
        save_gif: bool = True,
        gif_dir: str | Path = "checkpoints/eval_gifs",
        save_strip: bool = True,
        fps: int = 15,
        frame_stride: int = 2,
    ) -> EpisodeEvalResult:
        """Evaluate the policy on a specific HDF5 demo seed alongside ground-truth telemetry."""
        with h5py.File(h5_path, "r") as f:
            root = f.get("data", f)
            if demo_key not in root:
                raise KeyError(f"Demo key '{demo_key}' not found in {h5_path}.")
            grp = root[demo_key]
            task_id = int(grp.attrs["task_id"])
            seed = int(grp.attrs["seed"])
            domain = str(grp.attrs.get("domain", "sim_clean"))
            gt_proprio = grp["obs/proprio"][:].astype(np.float32)
            gt_actions = grp["actions"][:].astype(np.float32)

        return self.run_episode(
            task_id=task_id,
            seed=seed,
            domain_rand=(domain == "sim_dr"),
            episode_label=demo_key,
            gt_proprio=gt_proprio,
            gt_actions=gt_actions,
            max_steps=max_steps,
            ode_steps=ode_steps,
            seed_offset=seed_offset,
            exec_horizon=exec_horizon,
            use_temporal_ensemble=use_temporal_ensemble,
            save_gif=save_gif,
            gif_dir=gif_dir,
            gif_filename=f"{demo_key}_task{task_id}.gif",
            save_strip=save_strip,
            fps=fps,
            frame_stride=frame_stride,
        )

    def evaluate_benchmark(
        self,
        *,
        episodes_per_task: int = 20,
        train_h5_path: str | Path = "data/sim_demos_v2_dart_full_900.h5",
        test_base_seed: int = 9000,
        domain_rand: bool = False,
        max_steps: int = 280,
        ode_steps: int | None = None,
        seed_offset: int = 0,
        num_workers: int = 1,
        exec_horizon: int = 1,
        use_temporal_ensemble: bool = True,
        save_report_path: str | Path | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run strict evaluation on both training seeds and held-out test seeds (>=20 per task)."""
        n_per_task = max(1, int(episodes_per_task))
        bench_ode_steps = int(ode_steps if ode_steps is not None else 5)
        train_seeds_by_task: dict[int, list[int]] = {0: [], 1: [], 2: []}
        target_dom = "sim_dr" if domain_rand else "sim_clean"

        h5_file = Path(train_h5_path)
        if h5_file.exists():
            with h5py.File(h5_file, "r") as f:
                root = f.get("data", f)
                for k in sorted(k for k in root if k.startswith("demo_")):
                    grp = root[k]
                    tid = int(grp.attrs.get("task_id", 0))
                    dom = str(grp.attrs.get("domain", "sim_clean"))
                    seed_val = int(grp.attrs.get("seed", 1000))
                    if dom == target_dom and len(train_seeds_by_task[tid]) < n_per_task:
                        train_seeds_by_task[tid].append(seed_val)

        for tid in range(3):
            while len(train_seeds_by_task[tid]) < n_per_task:
                idx = len(train_seeds_by_task[tid])
                base = (1500 if domain_rand else 1000) + tid * 10000 + idx
                train_seeds_by_task[tid].append(base)

        def _run_split(split_name: str, seeds_map: dict[int, list[int]]) -> dict[str, Any]:
            all_eps: list[EpisodeEvalResult] = []
            per_task: dict[int, dict[str, Any]] = {}
            if verbose:
                print(
                    f"\n=== [{split_name.upper()} SPLIT] ({n_per_task} seeds/task, "
                    f"domain={target_dom}, seed_offset={seed_offset}, "
                    f"strict all_constraints_passed) ==="
                )

            jobs: list[dict[str, Any]] = []
            for tid in range(3):
                for s in seeds_map[tid]:
                    jobs.append(
                        {
                            "task_id": tid,
                            "seed": s,
                            "domain_rand": domain_rand,
                            "episode_label": f"{split_name}_t{tid}_s{s}",
                            "max_steps": max_steps,
                            "ode_steps": bench_ode_steps,
                            "seed_offset": seed_offset,
                            "exec_horizon": exec_horizon,
                            "use_temporal_ensemble": use_temporal_ensemble,
                            "save_gif": False,
                            "save_strip": False,
                        }
                    )

            if num_workers > 1:
                nw = min(int(num_workers), len(jobs))
                worker_args = [
                    (str(self.checkpoint_path), str(self.device), jobs[i::nw])
                    for i in range(nw)
                ]
                with mp.get_context("spawn").Pool(nw) as pool:
                    worker_lists = pool.map(_benchmark_worker, worker_args)
                by_label = {d["episode_label"]: EpisodeEvalResult(**d) for sub in worker_lists for d in sub}
                all_eps = [by_label[j["episode_label"]] for j in jobs]
            else:
                all_eps = [self.run_episode(**j) for j in jobs]

            for tid in range(3):
                t_eps = [r for r in all_eps if r.task_id == tid]
                passed = sum(1 for r in t_eps if r.all_constraints_passed)
                total = len(t_eps)
                rate = passed / float(max(total, 1))
                blocked_eps = [r for r in t_eps if r.grasp_blocked]
                pads = Counter(r.blocked_pad for r in blocked_eps)
                close_xys = [r.grasp_close_xy_cm for r in t_eps if r.grasp_close_xy_cm is not None]
                close_dzs = [r.grasp_close_dz_cm for r in t_eps if r.grasp_close_dz_cm is not None]
                centered_closes = sum(
                    1
                    for r in t_eps
                    if r.grasp_close_xy_cm is not None and r.grasp_close_xy_cm < 1.5
                )
                per_task[tid] = {
                    "passed": passed,
                    "total": total,
                    "strict_success_rate": rate,
                    "blocked_count": len(blocked_eps),
                    "blocked_by_pad": {
                        "fixed_jaw_pad": pads.get("fixed_jaw_pad", 0),
                        "moving_jaw_pad": pads.get("moving_jaw_pad", 0),
                        "neighbor": pads.get("neighbor", 0),
                    },
                    "centered_grasp_close_count": centered_closes,
                    "centered_grasp_close_rate": centered_closes / float(max(total, 1)),
                    "mean_grasp_close_xy_cm": float(np.mean(close_xys)) if close_xys else float("nan"),
                    "mean_grasp_close_dz_cm": float(np.mean(close_dzs)) if close_dzs else float("nan"),
                    "mean_min_src_xy_cm": float(np.mean([r.min_pinch_src_xy_cm for r in t_eps])),
                    "mean_max_lift_cm": float(np.mean([r.max_src_lift_cm for r in t_eps])),
                    "mean_final_tgt_xy_cm": float(np.mean([r.final_src_tgt_xy_cm for r in t_eps])),
                    "mean_latency_ms": float(np.mean([r.mean_latency_ms for r in t_eps])),
                }
                if verbose:
                    print(
                        f"  Task {tid} ({TASK_SPECS[tid].name}): "
                        f"{passed}/{total} ({rate * 100.0:.1f}%) | "
                        f"blocked={len(blocked_eps)}/{total} | "
                        f"centered_close={centered_closes}/{total} "
                        f"(close_xy={per_task[tid]['mean_grasp_close_xy_cm']:.2f}cm, "
                        f"close_dz={per_task[tid]['mean_grasp_close_dz_cm']:+.2f}cm) | "
                        f"mean_src_xy={per_task[tid]['mean_min_src_xy_cm']:.2f}cm | "
                        f"mean_lift=+{per_task[tid]['mean_max_lift_cm']:.2f}cm | "
                        f"mean_tgt_xy={per_task[tid]['mean_final_tgt_xy_cm']:.2f}cm"
                    )
            tot_passed = sum(1 for r in all_eps if r.all_constraints_passed)
            tot_count = len(all_eps)
            tot_rate = tot_passed / float(max(tot_count, 1))
            tot_blocked = [r for r in all_eps if r.grasp_blocked]
            tot_pads = Counter(r.blocked_pad for r in tot_blocked)
            tot_centered = sum(
                1
                for r in all_eps
                if r.grasp_close_xy_cm is not None and r.grasp_close_xy_cm < 1.5
            )
            all_close_xys = [r.grasp_close_xy_cm for r in all_eps if r.grasp_close_xy_cm is not None]
            all_close_dzs = [r.grasp_close_dz_cm for r in all_eps if r.grasp_close_dz_cm is not None]
            if verbose:
                print(
                    f"--> {split_name.upper()} STRICT PASS TOTAL: "
                    f"{tot_passed}/{tot_count} ({tot_rate * 100.0:.1f}%) | "
                    f"BLOCKED: {len(tot_blocked)}/{tot_count} | "
                    f"CENTERED CLOSE (<1.5cm): {tot_centered}/{tot_count} "
                    f"({tot_centered / float(max(tot_count, 1)) * 100.0:.1f}%)"
                )
            return {
                "passed": tot_passed,
                "total": tot_count,
                "strict_success_rate": tot_rate,
                "blocked_count": len(tot_blocked),
                "blocked_by_pad": {
                    "fixed_jaw_pad": tot_pads.get("fixed_jaw_pad", 0),
                    "moving_jaw_pad": tot_pads.get("moving_jaw_pad", 0),
                    "neighbor": tot_pads.get("neighbor", 0),
                },
                "centered_grasp_close_count": tot_centered,
                "centered_grasp_close_rate": tot_centered / float(max(tot_count, 1)),
                "mean_grasp_close_xy_cm": float(np.mean(all_close_xys)) if all_close_xys else float("nan"),
                "mean_grasp_close_dz_cm": float(np.mean(all_close_dzs)) if all_close_dzs else float("nan"),
                "mean_latency_ms": float(np.mean([r.mean_latency_ms for r in all_eps])) if all_eps else 0.0,
                "per_task": per_task,
                "episodes": [r.to_dict() for r in all_eps],
            }

        test_seeds_by_task = {
            tid: [int(test_base_seed) + tid * 10000 + i for i in range(n_per_task)]
            for tid in range(3)
        }

        train_summary = _run_split("train", train_seeds_by_task)
        test_summary = _run_split("test", test_seeds_by_task)

        overall_passed = int(train_summary["passed"]) + int(test_summary["passed"])
        overall_total = int(train_summary["total"]) + int(test_summary["total"])
        overall_blocked = int(train_summary["blocked_count"]) + int(test_summary["blocked_count"])

        report = {
            "checkpoint": str(self.checkpoint_path),
            "domain": target_dom,
            "episodes_per_task": n_per_task,
            "seed_offset": int(seed_offset),
            "overall_passed": overall_passed,
            "overall_total": overall_total,
            "overall_strict_success_rate": overall_passed / float(max(overall_total, 1)),
            "overall_blocked_count": overall_blocked,
            "overall_blocked_by_pad": {
                k: int(train_summary["blocked_by_pad"].get(k, 0))
                + int(test_summary["blocked_by_pad"].get(k, 0))
                for k in ("fixed_jaw_pad", "moving_jaw_pad", "neighbor")
            },
            "overall_mean_latency_ms": float(
                0.5 * (train_summary["mean_latency_ms"] + test_summary["mean_latency_ms"])
            ),
            "train_split": train_summary,
            "test_split": test_summary,
        }
        out_report = (
            Path(save_report_path)
            if save_report_path is not None
            else self.checkpoint_path.parent / f"eval_benchmark_{target_dom}.json"
        )
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, indent=2) + "\n")
        return report
