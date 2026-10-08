"""Check train-mode vs eval-mode action prediction shift on validation demo frames.

Loads a policy checkpoint, samples ~240 evenly spaced frames from the held-out
validation demos (idx % 10 == 9 in the 900-demo HDF5 dataset), runs
extract_obs_features + predict_velocity (Euler ODE integration) across M=16
train-mode passes (different noise/dropout each pass, same initial ODE noise x0)
and compares the 16-pass train-mode average against the deterministic eval-mode
prediction. Also reports the single-pass |train - eval| mean absolute difference
as a per-frame noise reference.

Reports two comparisons:
  1. Full Model (extract_obs_features + predict_velocity): 16-pass .train()
     average vs .eval(), plus single-pass |diff| noise reference.
  2. Velocity Head Only (predict_velocity .train() vs .eval() with clean .eval()
     z_fused): 16-pass .train() average vs .eval(), isolating velocity-head
     dropout shift.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.base import ALL_JOINT_NAMES, CAMERA_NAMES
from training.model import TaskConditionedVisionFlowPolicy
from training.trainer import load_policy_checkpoint


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_validation_frames(
    h5_path: Path,
    num_frames: int = 240,
    device: torch.device = torch.device("cpu"),
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Load ~num_frames evenly spaced frames from validation demos (idx % 10 == 9)."""
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset not found: {h5_path}")

    val_refs: list[tuple[str, int, int]] = []
    with h5py.File(h5_path, "r") as f:
        root = f["data"] if "data" in f and isinstance(f["data"], h5py.Group) else f
        demo_names = sorted(k for k in root.keys() if k.startswith("demo_"))
        for idx, dname in enumerate(demo_names):
            if (idx % 10) != 9:
                continue
            grp = root[dname]
            task_id = int(grp.attrs["task_id"])
            t_len = int(grp["obs/proprio"].shape[0])
            for t in range(t_len):
                val_refs.append((dname, t, task_id))

        total_val = len(val_refs)
        pick_indices = np.linspace(0, total_val - 1, num=min(num_frames, total_val), dtype=int)
        picked = [val_refs[i] for i in pick_indices]

        proprio_list: list[np.ndarray] = []
        task_list: list[int] = []
        cam_lists: dict[str, list[np.ndarray]] = {cam: [] for cam in CAMERA_NAMES}

        for dname, t, task_id in picked:
            grp = root[dname]
            proprio_list.append(grp["obs/proprio"][t].astype(np.float32))
            task_list.append(task_id)
            for cam in CAMERA_NAMES:
                cam_lists[cam].append(grp[f"obs/rgb_{cam}"][t])

    obs: dict[str, torch.Tensor] = {
        "proprio": torch.as_tensor(np.stack(proprio_list, axis=0), dtype=torch.float32, device=device),
    }
    for cam in CAMERA_NAMES:
        obs[f"rgb_{cam}"] = torch.as_tensor(
            np.stack(cam_lists[cam], axis=0), dtype=torch.uint8, device=device
        )
    task_ids = torch.as_tensor(task_list, dtype=torch.long, device=device)
    return obs, task_ids


@torch.no_grad()
def integrate_ode(
    policy: TaskConditionedVisionFlowPolicy,
    z_fused: torch.Tensor,
    x0: torch.Tensor,
    ode_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate predict_velocity from tau=0 to 1 starting at x0; return (x_norm, actions_raw)."""
    b = z_fused.shape[0]
    dt = 1.0 / float(max(ode_steps, 1))
    x_tau = x0.clone()
    for s in range(ode_steps):
        tau = torch.full((b,), float(s) * dt, device=z_fused.device, dtype=torch.float32)
        v = policy.predict_velocity(z_fused, x_tau, tau)
        x_tau = x_tau + dt * v
    actions_raw = policy.unnormalize_actions(x_tau)
    return x_tau, actions_raw


def print_shift_table(
    title: str,
    act_train_avg: torch.Tensor,
    act_train_pass0: torch.Tensor,
    act_eval: torch.Tensor,
    norm_train_avg: torch.Tensor,
    norm_eval: torch.Tensor,
    mc_passes: int,
) -> None:
    """Print per-joint mean signed shift (avg_train - eval) and single-pass |diff| reference."""
    diff_avg_raw = act_train_avg - act_eval  # (B, H, 6)
    diff_pass0_raw = act_train_pass0 - act_eval  # (B, H, 6)
    diff_avg_norm = norm_train_avg - norm_eval  # (B, H, 6)

    col_s0 = f"Step-0 Mean (avg{mc_passes}-eval)"
    col_ch = f"Chunk Mean (avg{mc_passes}-eval)"
    print(f"\n=== {title} ===")
    print(
        f"{'Joint':<16} | {col_s0:>24} | {'Step-0 1p |diff|':>16} | "
        f"{col_ch:>24} | {'Chunk 1p |diff|':>15} | {'Step-0 Norm Mean':>16}"
    )
    print("-" * 125)
    for j, jname in enumerate(ALL_JOINT_NAMES):
        unit = "norm" if j == 5 else "rad "
        s0_signed = float(diff_avg_raw[:, 0, j].mean().item())
        s0_abs_1p = float(diff_pass0_raw[:, 0, j].abs().mean().item())
        ch_signed = float(diff_avg_raw[:, :, j].mean().item())
        ch_abs_1p = float(diff_pass0_raw[:, :, j].abs().mean().item())
        s0_norm_signed = float(diff_avg_norm[:, 0, j].mean().item())
        if abs(s0_signed) < 1e-12:
            s0_signed = 0.0
        if abs(ch_signed) < 1e-12:
            ch_signed = 0.0
        if abs(s0_norm_signed) < 1e-12:
            s0_norm_signed = 0.0
        print(
            f"{jname:<16} | {s0_signed:+19.6f} {unit} | {s0_abs_1p:11.6f} {unit} | "
            f"{ch_signed:+19.6f} {unit} | {ch_abs_1p:10.6f} {unit} | {s0_norm_signed:+16.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check train vs eval action prediction shift on validation frames."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/exp09_e2e_nodropout/best_policy.pt",
        help="Path to policy checkpoint.",
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/sim_demos_v2_dart_full_900.h5",
        help="Path to simulation demonstrations HDF5 file.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=240,
        help="Number of validation frames to evaluate.",
    )
    parser.add_argument(
        "--mc-passes",
        type=int,
        default=16,
        help="Number of independent train-mode passes to average per frame.",
    )
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=None,
        help="Number of Euler ODE steps (defaults to checkpoint config.ode_steps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/eval comparison.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, mps, cuda, or cpu.",
    )
    args = parser.parse_args()

    policy, cfg, _ = load_policy_checkpoint(args.checkpoint, device=args.device)
    device = next(policy.parameters()).device
    ode_steps = int(args.ode_steps if args.ode_steps is not None else cfg.ode_steps)
    mc_passes = max(1, int(args.mc_passes))

    obs, task_ids = load_validation_frames(
        Path(args.h5_path), num_frames=args.num_frames, device=device
    )
    b = task_ids.shape[0]

    print(
        f"Checkpoint : {args.checkpoint}\n"
        f"Config     : dropout={cfg.dropout}, keypoint_noise={cfg.keypoint_noise}, "
        f"proprio_noise_std={cfg.proprio_noise_std}, proprio_drop_prob={cfg.proprio_drop_prob}, "
        f"wrist_cam_drop_prob={cfg.wrist_cam_drop_prob}, shift_pad={cfg.shift_pad}\n"
        f"Frames     : {b} validation frames | mc_passes={mc_passes} | ode_steps={ode_steps} | seed={args.seed} | device={device}"
    )

    # Fixed initial ODE noise x0 shared across all passes
    _seed_all(args.seed)
    x0 = torch.randn(
        (b, cfg.chunk_size, cfg.action_dim), device=device, dtype=torch.float32
    )

    # 1. Eval pass: extract_obs_features + predict_velocity in .eval() mode
    with torch.no_grad():
        policy.eval()
        _seed_all(args.seed)
        z_eval = policy.extract_obs_features(obs, task_ids, return_aux=False)
        norm_eval, act_eval = integrate_ode(policy, z_eval, x0, ode_steps)

    # 2. Full train passes (M=16): extract_obs_features + predict_velocity in .train() mode
    with torch.no_grad():
        policy.train()
        _seed_all(args.seed)
        full_norm_list: list[torch.Tensor] = []
        full_act_list: list[torch.Tensor] = []
        for _ in range(mc_passes):
            z_tr = policy.extract_obs_features(obs, task_ids, return_aux=False)
            n_tr, a_tr = integrate_ode(policy, z_tr, x0, ode_steps)
            full_norm_list.append(n_tr)
            full_act_list.append(a_tr)
        norm_full_train_avg = torch.stack(full_norm_list, dim=0).mean(dim=0)
        act_full_train_avg = torch.stack(full_act_list, dim=0).mean(dim=0)
        act_full_train_pass0 = full_act_list[0]

    # 3. Velocity-head-only train passes (M=16): extract_obs_features in .eval(), predict_velocity in .train()
    with torch.no_grad():
        policy.train()
        _seed_all(args.seed)
        vel_norm_list: list[torch.Tensor] = []
        vel_act_list: list[torch.Tensor] = []
        for _ in range(mc_passes):
            n_vtr, a_vtr = integrate_ode(policy, z_eval, x0, ode_steps)
            vel_norm_list.append(n_vtr)
            vel_act_list.append(a_vtr)
        norm_vel_train_avg = torch.stack(vel_norm_list, dim=0).mean(dim=0)
        act_vel_train_avg = torch.stack(vel_act_list, dim=0).mean(dim=0)
        act_vel_train_pass0 = vel_act_list[0]
        policy.eval()

    print_shift_table(
        f"1. Full Model (extract_obs_features + predict_velocity): {mc_passes}-Pass .train() Avg vs .eval()",
        act_full_train_avg,
        act_full_train_pass0,
        act_eval,
        norm_full_train_avg,
        norm_eval,
        mc_passes=mc_passes,
    )
    print_shift_table(
        f"2. Velocity Head Only (predict_velocity {mc_passes}-Pass .train() Avg vs .eval(), clean z_fused)",
        act_vel_train_avg,
        act_vel_train_pass0,
        act_eval,
        norm_vel_train_avg,
        norm_eval,
        mc_passes=mc_passes,
    )


if __name__ == "__main__":
    main()
