"""Unit tests for the AlloyFlow evaluation and GIF visualization workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from envs.sim_env import SimEnv
from evaluation.__main__ import parse_args
from evaluation.evaluator import SimPolicyEvaluator, diagnose_rollout_status
from evaluation.visualizer import project_3d_to_camera_pixels
from training.config import AlloyTrainConfig
from training.model import TaskConditionedVisionFlowPolicy


def test_camera_projection_and_failure_diagnosis() -> None:
    """Verify 3D-to-2D camera projection and human-readable rollout diagnosis."""
    env = SimEnv(domain_rand=False, include_rgb=False)
    try:
        env.reset(task_id=0, seed=1000)
        pts_3d = np.array([[0.20, 0.0, 0.05], [0.22, -0.05, 0.08]], dtype=np.float32)
        px_2d, valid = project_3d_to_camera_pixels(
            env.model, env.data, "third_person_cam", pts_3d, width=256, height=256
        )
        assert px_2d.shape == (2, 2)
        assert valid.shape == (2,)
        assert bool(np.all(valid)) is True
    finally:
        env.close()

    base_constraints = {
        "all_constraints_passed": False,
        "bystander_tipped": False,
        "max_bystander_displacement_m": 0.0,
        "target_tipped": False,
        "max_target_displacement_m": 0.0,
    }
    msg_reach = diagnose_rollout_status(
        constraints=base_constraints,
        min_pinch_src_xy_cm=8.5,
        max_src_lift_cm=0.0,
        final_src_tgt_xy_cm=12.0,
    )
    assert "Missed source reach" in msg_reach

    msg_grasp = diagnose_rollout_status(
        constraints=base_constraints,
        min_pinch_src_xy_cm=1.2,
        max_src_lift_cm=0.2,
        final_src_tgt_xy_cm=12.0,
    )
    assert "failed grasp/lift" in msg_grasp

    msg_off_center = diagnose_rollout_status(
        constraints=base_constraints,
        min_pinch_src_xy_cm=1.1,
        max_src_lift_cm=0.0,
        final_src_tgt_xy_cm=12.0,
        grasp_close_xy_cm=3.4,
        grasp_close_dz_cm=1.8,
    )
    assert "Closed off-center" in msg_off_center

    msg_target = diagnose_rollout_status(
        constraints=base_constraints,
        min_pinch_src_xy_cm=0.8,
        max_src_lift_cm=4.5,
        final_src_tgt_xy_cm=9.4,
    )
    assert "missed target" in msg_target


def test_evaluator_saves_gif_and_strip(tmp_path: Path) -> None:
    """Verify SimPolicyEvaluator executes a rollout and saves both GIF and keyframe strip."""
    cfg = AlloyTrainConfig(device="cpu", ode_steps=2)
    policy = TaskConditionedVisionFlowPolicy(cfg)
    ckpt_path = tmp_path / "dummy_policy.pt"
    torch.save(
        {
            "config": cfg.to_dict(),
            "model_state_dict": policy.state_dict(),
            "norm_stats": policy.get_norm_stats(),
            "epoch": 1,
            "best_loss": 0.5,
        },
        ckpt_path,
    )

    evaluator = SimPolicyEvaluator(checkpoint_path=ckpt_path, device="cpu", cam_render_size=128)
    try:
        res = evaluator.run_episode(
            task_id=1,
            seed=2001,
            max_steps=6,
            ode_steps=2,
            save_gif=True,
            gif_dir=tmp_path / "gifs",
            frame_stride=2,
        )
    finally:
        evaluator.close()

    assert res.task_id == 1
    assert res.gif_path is not None and Path(res.gif_path).exists()
    assert res.strip_path is not None and Path(res.strip_path).exists()

    parsed = parse_args(
        ["--demos", "demo_0000,demo_0200", "--max-steps", "40", "--seed-offset", "500"]
    )
    assert parsed.demos == "demo_0000,demo_0200"
    assert parsed.max_steps == 40
    assert parsed.seed_offset == 500
