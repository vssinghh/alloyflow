"""Unit tests for AlloyFlow SO-ARM101 assets and environments (envs/)."""

from __future__ import annotations

import numpy as np
import pytest

from envs import (
    CAMERA_NAMES,
    HOME_PROPRIO_6D,
    NUM_TASKS,
    OBJECT_NAMES,
    TASK_SPECS,
    RealEnv,
    SimEnv,
    norm_gripper_to_raw,
    radians_to_ticks,
    raw_gripper_to_norm,
    ticks_to_radians,
)


def test_task_specs_and_bystanders() -> None:
    """Verify the 3 locked tasks and their source, target, and bystander objects."""
    assert NUM_TASKS == 3
    assert set(TASK_SPECS.keys()) == {0, 1, 2}

    assert TASK_SPECS[0].source_object == "pen_holder"
    assert TASK_SPECS[0].target_object == "bowl"
    assert set(TASK_SPECS[0].bystander_objects) == {"cup", "rubiks_cube"}

    assert TASK_SPECS[1].source_object == "cup"
    assert TASK_SPECS[1].target_object == "bowl"
    assert set(TASK_SPECS[1].bystander_objects) == {"pen_holder", "rubiks_cube"}

    assert TASK_SPECS[2].source_object == "cup"
    assert TASK_SPECS[2].target_object == "rubiks_cube"
    assert set(TASK_SPECS[2].bystander_objects) == {"pen_holder", "bowl"}


def test_unit_conversions_roundtrip() -> None:
    """Verify gripper [0.0, 1.0] and Feetech STS3215 tick conversions."""
    for val in (0.0, 0.25, 0.5, 0.75, 1.0):
        raw = norm_gripper_to_raw(val)
        recovered = raw_gripper_to_norm(raw)
        assert np.isclose(val, recovered, atol=1e-5)

    rads = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 0.2], dtype=np.float32)
    ticks = radians_to_ticks(rads)
    recovered_rads = ticks_to_radians(ticks)
    assert np.allclose(rads, recovered_rads, atol=2e-3)


def test_sim_env_loads_and_renders_3_cameras() -> None:
    """Verify SimEnv loads scene.xml and renders 3 non-blank 128x128 RGB streams."""
    env = SimEnv(include_rgb=True)
    try:
        obs = env.reset(task_id=0, seed=0)
        assert obs["proprio"].shape == (6,)
        assert obs["proprio"].dtype == np.float32
        assert 0.0 <= float(obs["proprio"][5]) <= 1.0

        for cam in CAMERA_NAMES:
            key = f"rgb_{cam}"
            assert key in obs
            img = obs[key]
            assert img.shape == (128, 128, 3)
            assert img.dtype == np.uint8
            assert float(img.std()) > 5.0, f"Camera {cam} rendered a blank frame."
    finally:
        env.close()


def test_spawn_clearance_across_50_seeds() -> None:
    """Verify all 4 objects spawn >= 8 cm apart with zero initial collisions across 50 seeds."""
    env = SimEnv(include_rgb=False)
    try:
        for seed in range(50):
            obs = env.reset(task_id=seed % 3, seed=seed)
            positions = obs["object_positions"]
            assert not env.has_initial_collision(), f"Seed {seed} had an initial collision."

            for i, name_i in enumerate(OBJECT_NAMES):
                pos_i = positions[name_i][:2]
                for name_j in OBJECT_NAMES[i + 1 :]:
                    pos_j = positions[name_j][:2]
                    dist = float(np.linalg.norm(pos_i - pos_j))
                    assert dist >= SimEnv.MIN_SPAWN_CLEARANCE_M - 1e-3, (
                        f"Seed {seed}: {name_i} and {name_j} spawned too close ({dist:.3f}m < 0.08m)"
                    )
    finally:
        env.close()


def test_task_success_and_bystander_constraints_all_3_tasks() -> None:
    """Verify check_task_success and check_constraints for Task 0, Task 1, and Task 2."""
    env = SimEnv(include_rgb=False)
    try:
        fixed_spawns = {
            "pen_holder": (0.20, 0.12),
            "cup": (0.20, -0.02),
            "bowl": (0.27, -0.13),
            "rubiks_cube": (0.27, 0.05),
        }

        # Task 0: Pen holder inside bowl
        env.reset(task_id=0, seed=0, object_xy_overrides=fixed_spawns)
        assert not env.check_task_success(0)
        bowl_pos = env.data.body("bowl").xpos.copy()
        q_pen = env._obj_qpos_adrs["pen_holder"]
        env.data.qpos[q_pen : q_pen + 3] = [bowl_pos[0], bowl_pos[1], bowl_pos[2] + 0.035]
        env.step(HOME_PROPRIO_6D)
        report_t0 = env.check_constraints(0)
        assert report_t0["task_success"] is True
        assert report_t0["bystander_ok"] is True
        assert report_t0["all_constraints_passed"] is True
        # Should NOT count as success for Task 1 (which requires the cup)
        assert env.check_task_success(1) is False

        # Task 1: Cup inside bowl, then test bystander violation
        env.reset(task_id=1, seed=1, object_xy_overrides=fixed_spawns)
        bowl_pos = env.data.body("bowl").xpos.copy()
        q_cup = env._obj_qpos_adrs["cup"]
        env.data.qpos[q_cup : q_cup + 3] = [bowl_pos[0], bowl_pos[1], bowl_pos[2] + 0.020]
        env.step(HOME_PROPRIO_6D)
        assert env.check_constraints(1)["all_constraints_passed"] is True

        # Now bump bystander pen_holder by 3.0 cm (> 1.5 cm threshold)
        q_pen = env._obj_qpos_adrs["pen_holder"]
        env.data.qpos[q_pen] += 0.030
        env.step(HOME_PROPRIO_6D)
        report_bumped = env.check_constraints(1)
        assert report_bumped["task_success"] is True
        assert report_bumped["bystander_ok"] is False
        assert report_bumped["all_constraints_passed"] is False

        # Task 2: Stack cup on top of Rubik's cube
        env.reset(task_id=2, seed=2, object_xy_overrides=fixed_spawns)
        cube_pos = env.data.body("rubiks_cube").xpos.copy()
        q_cup = env._obj_qpos_adrs["cup"]
        env.data.qpos[q_cup : q_cup + 3] = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.048]
        env.data.qvel[:] = 0.0
        env.step(HOME_PROPRIO_6D)
        report_t2 = env.check_constraints(2)
        assert report_t2["task_success"] is True
        assert report_t2["bystander_ok"] is True
        assert report_t2["all_constraints_passed"] is True
    finally:
        env.close()


def test_domain_randomization_and_restore() -> None:
    """Verify domain randomization perturbs model parameters and restores cleanly."""
    env = SimEnv(include_rgb=False, domain_rand=True)
    try:
        init_mass = env._init_body_mass.copy()
        env.reset(task_id=0, seed=42)
        assert env.last_domain_params is not None
        assert not np.allclose(env.model.body_mass, init_mass)

        env.restore_nominal_domain()
        assert env.last_domain_params is None
        assert np.allclose(env.model.body_mass, init_mass)
    finally:
        env.close()


def test_real_env_mock_contract_parity() -> None:
    """Verify RealEnv(mock_hardware=True) matches SimEnv's observation and action API."""
    real_env = RealEnv(mock_hardware=True)
    try:
        obs = real_env.reset(task_id=2)
        assert obs["task_id"] == 2
        assert obs["proprio"].shape == (6,)
        assert obs["proprio"].dtype == np.float32
        for cam in CAMERA_NAMES:
            assert obs[f"rgb_{cam}"].shape == (128, 128, 3)
            assert obs[f"rgb_{cam}"].dtype == np.uint8

        target = np.array([0.1, -0.4, 0.5, 0.6, -1.57, 0.3], dtype=np.float32)
        obs2 = real_env.step(target)
        assert np.allclose(obs2["proprio"], target, atol=1e-5)

        with pytest.raises(ValueError):
            real_env.step(np.zeros(5, dtype=np.float32))
    finally:
        real_env.close()
