"""Closed-loop multi-task simulation policy evaluator with automatic GIF diagnostics."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from envs.base import CAMERA_NAMES, TASK_SPECS
from envs.sim_env import SimEnv
from evaluation.visualizer import RolloutVisualizer
from training.dataset import build_action_chunks
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
    gif_path: str | None = None
    strip_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_rollout_status(
    *,
    constraints: dict[str, Any],
    min_pinch_src_xy_cm: float,
    max_src_lift_cm: float,
    final_src_tgt_xy_cm: float,
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
        return f"FAIL: Reached src ({min_pinch_src_xy_cm:.1f}cm), failed grasp/lift"
    return f"FAIL: Lifted (+{max_src_lift_cm:.1f}cm), missed target ({final_src_tgt_xy_cm:.1f}cm)"


class SimPolicyEvaluator:
    """Closed-loop evaluator for AlloyFlow policies with per-episode GIF generation."""

    def __init__(
        self,
        checkpoint_path: str | Path = "checkpoints/sim_only/best_policy.pt",
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
        max_steps: int = 154,
        ode_steps: int | None = None,
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
        steps_ode = int(ode_steps if ode_steps is not None else self.config.ode_steps)

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
        init_src_z_cm = float(self.env.data.body(src_bid).xpos[2]) * 100.0

        min_pinch_src_xy_cm = float("inf")
        max_src_z_cm = init_src_z_cm
        latencies_ms: list[float] = []

        cached_chunk: np.ndarray | None = None
        cached_kp: dict[str, np.ndarray] = {}
        cached_weights: np.ndarray = np.full((len(CAMERA_NAMES),), 1.0 / len(CAMERA_NAMES), dtype=np.float32)
        prop_history: list[np.ndarray] = []
        has_clamped = False
        lags = self.config.proprio_history_lags

        for step in range(max_steps):
            prop_cur = obs["proprio"].astype(np.float32).copy()
            if prop_cur[5] < 0.56:
                has_clamped = True
            if not has_clamped and prop_cur[5] >= 0.58:
                prop_cur[5] = 0.68
            prop_history.append(prop_cur)

            need_query = use_temporal_ensemble or cached_chunk is None or (step % max(1, exec_horizon) == 0)
            if need_query:
                cur_t = len(prop_history) - 1
                prop_hist_np = np.stack(
                    [prop_history[max(0, cur_t - int(lag))] for lag in lags],
                    axis=0,
                )
                obs_t = self._prepare_obs_tensors(obs, proprio_history=prop_hist_np)
                torch.manual_seed(int(seed) * 1000 + step)
                t0 = time.perf_counter()
                chunk_t, aux = self.matcher.sample_action_chunk(
                    self.policy,
                    obs_t,
                    task_id=int(task_id),
                    ode_steps=steps_ode,
                    clip_to_limits=True,
                    return_aux=True,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(latency_ms)
                cached_chunk = chunk_t[0].detach().cpu().numpy()
                cached_kp = {
                    cam: kp_t[0].detach().cpu().numpy()
                    for cam, kp_t in aux["keypoints"].items()
                }
                cached_weights = aux["camera_weights"][0].detach().cpu().numpy()
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
            cur_src_z_cm = float(src_pos[2]) * 100.0
            cur_src_tgt_xy_cm = float(np.linalg.norm(src_pos[:2] - tgt_pos[:2])) * 100.0
            min_pinch_src_xy_cm = min(min_pinch_src_xy_cm, cur_pinch_src_xy_cm)
            max_src_z_cm = max(max_src_z_cm, cur_src_z_cm)

            if save_gif:
                gt_idx = min(step, len(gt_proprio) - 1) if gt_proprio is not None else 0
                cur_gt_prop = gt_proprio[gt_idx] if gt_proprio is not None else None
                cur_gt_chunk = gt_chunks[gt_idx] if gt_chunks is not None else None
                cur_succ = self.env.check_task_success(int(task_id))
                live_status = (
                    "PASS: Goal satisfied"
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

            obs = self.env.step(action_6d)

        constraints = self.env.check_constraints(int(task_id))
        src_pos_end = self.env.data.body(src_bid).xpos
        tgt_pos_end = self.env.data.body(tgt_bid).xpos
        final_src_tgt_xy_cm = float(np.linalg.norm(src_pos_end[:2] - tgt_pos_end[:2])) * 100.0
        max_src_lift_cm = max(0.0, max_src_z_cm - init_src_z_cm)

        status_text = diagnose_rollout_status(
            constraints=constraints,
            min_pinch_src_xy_cm=min_pinch_src_xy_cm,
            max_src_lift_cm=max_src_lift_cm,
            final_src_tgt_xy_cm=final_src_tgt_xy_cm,
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
            gif_path=saved_gif_path,
            strip_path=saved_strip_path,
        )

    def run_demo_case(
        self,
        demo_key: str = "demo_0000",
        h5_path: str | Path = "data/sim_demos.h5",
        *,
        max_steps: int = 154,
        ode_steps: int | None = None,
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
            exec_horizon=exec_horizon,
            use_temporal_ensemble=use_temporal_ensemble,
            save_gif=save_gif,
            gif_dir=gif_dir,
            gif_filename=f"{demo_key}_task{task_id}.gif",
            save_strip=save_strip,
            fps=fps,
            frame_stride=frame_stride,
        )
