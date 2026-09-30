"""Unit tests for the AlloyFlow demonstration collection package."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from collection.collector import (
    collect_real_demos,
    collect_sim_demos,
    inspect_hdf5_dataset,
    verify_hdf5_action_replay,
)
from collection.sim_expert import SimExpertPlanner, trim_stationary_frames
from envs.base import CAMERA_NAMES
from envs.sim_env import SimEnv


def test_trim_stationary_frames_removes_zero_delta_steps() -> None:
    """Verify leading and trailing stationary steps with ||dq|| < 1e-3 are removed."""
    q0 = np.zeros(6, dtype=np.float32)
    q1 = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    steps = [
        {"proprio": q0, "action": q0.copy()},  # Leading stationary frame -> removed
        {"proprio": q0, "action": q1.copy()},  # Active transition -> kept
        {"proprio": q1, "action": q1.copy()},  # Trailing stationary frame -> removed
    ]
    trimmed = trim_stationary_frames(steps, min_delta=1e-3)
    assert len(trimmed) == 1
    assert np.allclose(trimmed[0]["action"], q1)


def test_sim_expert_solves_all_three_tasks_with_constraints() -> None:
    """Verify SimExpertPlanner solves Tasks 0, 1, and 2 while passing all 4 constraint checks."""
    env = SimEnv(domain_rand=False)
    planner = SimExpertPlanner(env)

    for task_id in (0, 1, 2):
        ep = planner.generate_episode(task_id=task_id, seed=1 + task_id, trim_dwell=True)
        constraints = ep["constraints"]
        assert constraints["spawn_clearance_ok"], f"Task {task_id} failed spawn clearance"
        assert constraints["bystander_ok"], (
            f"Task {task_id} disturbed bystander: {constraints['max_bystander_displacement_m']:.4f}m"
        )
        assert constraints["task_success"], f"Task {task_id} failed physical goal check"
        assert constraints["no_stationary_dwell"], f"Task {task_id} had stationary frames"
        assert constraints["all_constraints_passed"]

        T = ep["num_steps"]
        assert T == 144
        assert ep["actions"].shape == (T, 6)
        assert ep["proprio"].shape == (T, 6)
        for cam in CAMERA_NAMES:
            assert ep[f"rgb_{cam}"].shape == (T, 128, 128, 3)
            assert ep[f"rgb_{cam}"].dtype == np.uint8

        # Check no stationary dwell frames exist anywhere in the trajectory
        cmd_deltas = np.linalg.norm(ep["actions"] - ep["proprio"], axis=-1)
        motion_deltas = np.linalg.norm(np.diff(ep["proprio"], axis=0), axis=-1)
        assert np.all(cmd_deltas >= 1e-3)
        assert np.all(motion_deltas >= 1e-3)

    env.close()


def test_collect_sim_and_real_hdf5_schema_and_exact_replay(tmp_path: Path) -> None:
    """Verify Sim and Real HDF5 schema parity AND 0-error open-loop replay of saved Sim demos."""
    sim_h5 = tmp_path / "sim_demos.h5"
    real_h5 = tmp_path / "real_demos.h5"

    sim_summary = collect_sim_demos(
        output_path=sim_h5,
        task_ids=[0, 1, 2],
        episodes_per_task=2,
        split_clean_and_dr=True,
        base_seed=1,
        verbose=False,
    )
    assert sim_summary["total_saved"] == 6

    sim_stats = inspect_hdf5_dataset(sim_h5)
    assert sim_stats["num_episodes"] == 6
    assert sim_stats["task_counts"] == {0: 2, 1: 2, 2: 2}
    assert sim_stats["domain_counts"] == {"sim_clean": 3, "sim_dr": 3}

    # Replay all 6 saved episodes (3 clean + 3 DR) open-loop and verify 0 float/pixel drift
    replay_audit = verify_hdf5_action_replay(sim_h5, check_rgb_pixels=True)
    assert replay_audit["verified_episodes"] == 6
    assert replay_audit["max_proprio_err"] == 0.0
    assert replay_audit["max_rgb_err"] == 0
    assert replay_audit["all_replays_exact"]

    real_summary = collect_real_demos(
        output_path=real_h5,
        task_ids=[0, 1, 2],
        episodes_per_task=1,
        max_steps=16,
        mock_hardware=True,
        verbose=False,
    )
    assert real_summary["total_saved"] == 3

    real_stats = inspect_hdf5_dataset(real_h5)
    assert real_stats["num_episodes"] == 3
    assert real_stats["task_counts"] == {0: 1, 1: 1, 2: 1}
    assert real_stats["domain_counts"] == {"real": 3}

    with h5py.File(sim_h5, "r") as fs, h5py.File(real_h5, "r") as fr:
        assert list(fs["demo_0000/obs"].keys()) == list(fr["demo_0000/obs"].keys())
