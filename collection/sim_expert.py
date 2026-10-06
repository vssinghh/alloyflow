"""Scripted 6-DoF Damped Least-Squares IK expert planner for the simulated SO-ARM101."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from envs.base import (
    CAMERA_NAMES,
    JOINT_LIMITS_HIGH,
    JOINT_LIMITS_LOW,
    TASK_SPECS,
    norm_gripper_to_raw,
)
from envs.sim_env import SimEnv


def trim_stationary_frames(
    episode_steps: list[dict[str, Any]],
    min_delta: float = 1e-3,
    filter_interior: bool = False,
) -> list[dict[str, Any]]:
    """Remove stationary frames where joint motion falls below min_delta.

    Args:
        episode_steps: Recorded step dictionaries containing 'proprio' and 'action'.
        min_delta: Minimum L2 joint change (radians) required to count as active motion.
        filter_interior: If True (used for human teleop), strips mid-episode pauses as well
            using consecutive state/action deltas so steady-state gravity lag does not fool
            the filter. If False (used for scripted sim), strips only leading/trailing idle
            frames to preserve exact 20 Hz open-loop replay continuity.
    """
    n = len(episode_steps)
    if n <= 1:
        return episode_steps

    def _is_active(idx: int) -> bool:
        q_cur = episode_steps[idx]["proprio"]
        act_cur = episode_steps[idx]["action"]
        if idx + 1 < n:
            q_next = episode_steps[idx + 1]["proprio"]
            act_next = episode_steps[idx + 1]["action"]
            motion_delta = float(np.linalg.norm(q_next - q_cur))
            act_step_delta = float(np.linalg.norm(act_next - act_cur))
        elif idx > 0:
            q_prev = episode_steps[idx - 1]["proprio"]
            act_prev = episode_steps[idx - 1]["action"]
            motion_delta = float(np.linalg.norm(q_cur - q_prev))
            act_step_delta = float(np.linalg.norm(act_cur - act_prev))
        else:
            motion_delta = float(np.linalg.norm(act_cur - q_cur))
            act_step_delta = motion_delta

        if filter_interior:
            return max(motion_delta, act_step_delta) >= min_delta

        cmd_delta = float(np.linalg.norm(act_cur - q_cur))
        return motion_delta >= min_delta and cmd_delta >= min_delta

    if filter_interior:
        return [step for idx, step in enumerate(episode_steps) if _is_active(idx)]

    start_idx = 0
    while start_idx < n and not _is_active(start_idx):
        start_idx += 1

    end_idx = n - 1
    while end_idx >= start_idx and not _is_active(end_idx):
        end_idx -= 1

    if start_idx > end_idx:
        return []
    return episode_steps[start_idx : end_idx + 1]


class SimExpertPlanner:
    """5-phase blended cosine-linear IK trajectory generator for all 3 SO-ARM101 tasks."""

    SETTLE_STEPS: int = 10

    def __init__(self, env: SimEnv, trajectory_version: str = "v1") -> None:
        if trajectory_version not in ("v1", "v2", "v2_dart"):
            raise ValueError(
                f"Unsupported trajectory_version '{trajectory_version}', expected 'v1', 'v2', or 'v2_dart'."
            )
        self.env = env
        self.trajectory_version = trajectory_version
        self._ik_data = mujoco.MjData(env.model)
        self._fixed_pad_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad"
        )
        self._moving_pad_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad"
        )
        self._gripper_body_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_pitch: float = 1.25,
        grip_for_center: float = 0.55,
        q_init: np.ndarray | None = None,
        max_iters: int = 80,
    ) -> np.ndarray:
        """Solve 5-DoF arm joint angles that place the jaw pad midpoint at target_pos."""
        model = self.env.model
        data = self._ik_data
        mujoco.mj_resetData(model, data)

        if q_init is not None:
            for idx in range(5):
                data.qpos[self.env._qpos_adrs[idx]] = float(q_init[idx])
        else:
            pan_guess = float(-np.arctan2(target_pos[1], target_pos[0] - 0.0388))
            init_guess = [pan_guess, -0.2, 0.5, 0.95, -np.pi / 2.0]
            for idx in range(5):
                data.qpos[self.env._qpos_adrs[idx]] = float(init_guess[idx])

        data.qpos[self.env._qpos_adrs[4]] = -np.pi / 2.0
        data.qpos[self.env._qpos_adrs[5]] = norm_gripper_to_raw(grip_for_center)

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        dof_idx = self.env._dof_adrs[:4]
        q_idx = self.env._qpos_adrs[:4]
        low = JOINT_LIMITS_LOW[:4].astype(np.float64)
        high = JOINT_LIMITS_HIGH[:4].astype(np.float64)
        j_pitch = np.array([[0.0, 0.25, 0.25, 0.25]], dtype=np.float64)

        for _ in range(max_iters):
            mujoco.mj_forward(model, data)
            cur_pos = 0.5 * (
                data.geom(self._fixed_pad_id).xpos + data.geom(self._moving_pad_id).xpos
            )
            pos_err = np.asarray(target_pos, dtype=np.float64) - cur_pos
            q_cur = np.array([data.qpos[i] for i in q_idx], dtype=np.float64)
            pitch_err = float(target_pitch - (q_cur[1] + q_cur[2] + q_cur[3]))

            if np.linalg.norm(pos_err) < 4e-4 and abs(pitch_err) < 0.03:
                break

            mujoco.mj_jac(model, data, jacp, None, cur_pos, self._gripper_body_id)
            j_pos = jacp[:, dof_idx]
            j_full = np.vstack([j_pos, j_pitch])
            err = np.concatenate([pos_err, [0.25 * pitch_err]])
            dq = j_full.T @ np.linalg.solve(j_full @ j_full.T + 1e-3 * np.eye(4), err)
            q_new = np.clip(q_cur + dq, low, high)
            for i, qi in enumerate(q_idx):
                data.qpos[qi] = float(q_new[i])

        res = np.array([data.qpos[i] for i in self.env._qpos_adrs[:5]], dtype=np.float32)
        res[4] = -np.pi / 2.0
        return res

    def _build_segments(
        self, task_id: int, q_home: np.ndarray, g_home: float
    ) -> list[tuple[np.ndarray, float, np.ndarray, float, int]]:
        """Construct the v1 5-phase trajectory segments (q0, g0, q1, g1, n_steps)."""
        spec = TASK_SPECS[task_id]
        src_pos = self.env.data.body(self.env._obj_body_ids[spec.source_object]).xpos.copy()
        tgt_pos = self.env.data.body(self.env._obj_body_ids[spec.target_object]).xpos.copy()

        grasp_z = float(src_pos[2]) + (0.004 if spec.source_object == "pen_holder" else 0.002)
        place_z = (
            float(tgt_pos[2]) + 0.052
            if spec.goal_type == "place_inside"
            else float(tgt_pos[2]) + 0.055
        )

        # Phase 1: Hover Source (moderate jaw opening g=0.65 avoids bystander collisions)
        q_hover_src = self.solve_ik(
            np.array([src_pos[0], src_pos[1], 0.145]),
            target_pitch=1.20,
            grip_for_center=0.58,
        )
        # Phase 2: Descend to Grasp
        q_grasp = self.solve_ik(
            np.array([src_pos[0], src_pos[1], grasp_z]),
            target_pitch=1.30,
            grip_for_center=0.55,
            q_init=q_hover_src,
        )
        # Phase 3: Clamp & Lift (shift toward fixed jaw and begin 1.2 cm lift so jaws never stall)
        q_clamp = self.solve_ik(
            np.array([src_pos[0], src_pos[1], grasp_z + 0.012]),
            target_pitch=1.30,
            grip_for_center=0.30,
            q_init=q_grasp,
        )
        q_lift = self.solve_ik(
            np.array([src_pos[0], src_pos[1], 0.155]),
            target_pitch=1.20,
            grip_for_center=0.30,
            q_init=q_clamp,
        )
        # Phase 4: Move to Goal & Lower
        q_hover_tgt = self.solve_ik(
            np.array([tgt_pos[0], tgt_pos[1], 0.155]),
            target_pitch=1.20,
            grip_for_center=0.30,
            q_init=q_lift,
        )
        q_place = self.solve_ik(
            np.array([tgt_pos[0], tgt_pos[1], place_z]),
            target_pitch=1.25,
            grip_for_center=0.30,
            q_init=q_hover_tgt,
        )
        # Phase 5: Place & Retract (gentle jaw opening g=0.72 avoids bowl wall wedging)
        q_release = self.solve_ik(
            np.array([tgt_pos[0], tgt_pos[1], place_z + 0.018]),
            target_pitch=1.25,
            grip_for_center=0.45,
            q_init=q_place,
        )
        q_retract = self.solve_ik(
            np.array([tgt_pos[0], tgt_pos[1], 0.155]),
            target_pitch=1.20,
            grip_for_center=0.58,
            q_init=q_release,
        )

        return [
            (q_home, g_home, q_hover_src, 0.65, 22),
            (q_hover_src, 0.65, q_grasp, 0.60, 18),
            (q_grasp, 0.60, q_clamp, 0.05, 10),
            (q_clamp, 0.05, q_lift, 0.05, 20),
            (q_lift, 0.05, q_hover_tgt, 0.05, 24),
            (q_hover_tgt, 0.05, q_place, 0.05, 18),
            (q_place, 0.05, q_release, 0.72, 12),
            (q_release, 0.72, q_retract, 0.75, 20),
        ]

    def _build_segments_v2(
        self, task_id: int, seed: int, q_home: np.ndarray, g_home: float
    ) -> list[tuple[np.ndarray, float, np.ndarray, float, int]]:
        """Construct v2 flat-bottom grasp, open-in-place release, and seeded timing-jitter segments."""
        spec = TASK_SPECS[task_id]
        src_pos = self.env.data.body(self.env._obj_body_ids[spec.source_object]).xpos.copy()
        tgt_pos = self.env.data.body(self.env._obj_body_ids[spec.target_object]).xpos.copy()

        grasp_z = float(src_pos[2]) + (0.004 if spec.source_object == "pen_holder" else 0.002)
        place_z = (
            float(tgt_pos[2]) + 0.056
            if spec.goal_type == "place_inside"
            else float(tgt_pos[2]) + 0.055
        )

        rng = np.random.default_rng(int(seed) * 31337 + int(task_id) * 997 + 7)

        def jitter_steps(nom: int, min_s: int = 4) -> int:
            scale = float(rng.uniform(0.8, 1.2))
            return max(min_s, int(np.rint(nom * scale)))

        dwell_grasp = int(rng.integers(2, 6))  # 2 to 5 steps open at bottom
        hold_closed = int(rng.integers(2, 4))  # 2 to 3 steps closed at bottom
        dwell_place = int(rng.integers(2, 5))  # 2 to 4 steps at place_z before opening
        hold_open = 2                          # 2 steps open at place_z before retracting

        q_hover_src = self.solve_ik(
            np.array([src_pos[0], src_pos[1], 0.145]),
            target_pitch=1.20,
            grip_for_center=0.58,
        )
        q_grasp = self.solve_ik(
            np.array([src_pos[0], src_pos[1], grasp_z]),
            target_pitch=1.30,
            grip_for_center=0.55,
            q_init=q_hover_src,
        )
        # Close in place at grasp_z (compensating XY via grip_for_center=0.30, zero vertical rise)
        q_clamp_in_place = self.solve_ik(
            np.array([src_pos[0], src_pos[1], grasp_z]),
            target_pitch=1.30,
            grip_for_center=0.30,
            q_init=q_grasp,
        )
        q_lift = self.solve_ik(
            np.array([src_pos[0], src_pos[1], 0.155]),
            target_pitch=1.20,
            grip_for_center=0.30,
            q_init=q_clamp_in_place,
        )
        q_hover_tgt = self.solve_ik(
            np.array([tgt_pos[0], tgt_pos[1], 0.155]),
            target_pitch=1.20,
            grip_for_center=0.30,
            q_init=q_lift,
        )
        q_place = self.solve_ik(
            np.array([tgt_pos[0], tgt_pos[1], place_z]),
            target_pitch=1.25,
            grip_for_center=0.30,
            q_init=q_hover_tgt,
        )
        # Open in place at place_z (compensating XY via grip_for_center=0.45, zero vertical rise)
        q_release_in_place = self.solve_ik(
            np.array([tgt_pos[0], tgt_pos[1], place_z]),
            target_pitch=1.25,
            grip_for_center=0.45,
            q_init=q_place,
        )
        q_retract = self.solve_ik(
            np.array([tgt_pos[0], tgt_pos[1], 0.155]),
            target_pitch=1.20,
            grip_for_center=0.58,
            q_init=q_release_in_place,
        )

        n_approach = jitter_steps(22, min_s=16)
        n_descend = jitter_steps(18, min_s=14)
        n_close = jitter_steps(8, min_s=6)
        n_lift = jitter_steps(20, min_s=15)
        n_carry = jitter_steps(24, min_s=18)
        n_lower = jitter_steps(18, min_s=14)
        n_open = int(rng.integers(4, 6))  # 4 to 5 steps
        n_retract = jitter_steps(18, min_s=14)

        return [
            (q_home, g_home, q_hover_src, 0.65, n_approach),
            (q_hover_src, 0.65, q_grasp, 0.62, n_descend),
            (q_grasp, 0.62, q_grasp, 0.60, dwell_grasp),
            (q_grasp, 0.60, q_clamp_in_place, 0.05, n_close),
            (q_clamp_in_place, 0.05, q_clamp_in_place, 0.05, hold_closed),
            (q_clamp_in_place, 0.05, q_lift, 0.05, n_lift),
            (q_lift, 0.05, q_hover_tgt, 0.05, n_carry),
            (q_hover_tgt, 0.05, q_place, 0.05, n_lower),
            (q_place, 0.05, q_place, 0.05, dwell_place),
            (q_place, 0.38, q_release_in_place, 0.72, n_open),
            (q_release_in_place, 0.72, q_release_in_place, 0.72, hold_open),
            (q_release_in_place, 0.72, q_retract, 0.75, n_retract),
        ]

    def generate_episode(
        self,
        task_id: int,
        seed: int,
        trim_dwell: bool = True,
        render_only_on_success: bool = True,
        trajectory_version: str | None = None,
    ) -> dict[str, Any]:
        """Run one scripted 20 Hz trajectory and return observations, actions, and constraint report."""
        ver = trajectory_version or self.trajectory_version
        if ver not in ("v1", "v2", "v2_dart"):
            raise ValueError(
                f"Unsupported trajectory_version '{ver}', expected 'v1', 'v2', or 'v2_dart'."
            )

        prev_include_rgb = self.env.include_rgb
        if render_only_on_success:
            self.env.include_rgb = False

        obs = self.env.reset(task_id=task_id, seed=seed)
        spec = TASK_SPECS[task_id]
        src_bid = self.env._obj_body_ids[spec.source_object]
        init_src_z = float(self.env.data.body(src_bid).xpos[2])

        q_home = obs["proprio"][:5].copy()
        g_home = float(obs["proprio"][5])
        is_dart = (ver == "v2_dart")
        if ver in ("v2", "v2_dart"):
            segments = self._build_segments_v2(
                task_id=task_id, seed=seed, q_home=q_home, g_home=g_home
            )
        else:
            segments = self._build_segments(task_id=task_id, q_home=q_home, g_home=g_home)

        if is_dart:
            rng_dart = np.random.default_rng(int(seed) * 31337 + int(task_id) * 997 + 99)
            angle = float(rng_dart.uniform(0.0, 2.0 * np.pi))
            mag = float(rng_dart.uniform(0.0, 0.025))  # 0 to 2.5 cm
            delta = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float64)
            q_hover_clean = segments[0][2]
            src_xy = self.env.data.body(src_bid).xpos[:2].copy()
            q_hover_pert = self.solve_ik(
                np.array([src_xy[0] + delta[0], src_xy[1] + delta[1], 0.145]),
                target_pitch=1.20,
                grip_for_center=0.58,
            )
            dq_hover = q_hover_pert - q_hover_clean
        else:
            delta = np.zeros(2, dtype=np.float64)
            mag = 0.0
            dq_hover = np.zeros(5, dtype=np.float32)

        raw_steps: list[dict[str, Any]] = []
        lift_start_step: int | None = None
        for seg_idx, (q0, g0, q1, g1, n_steps) in enumerate(segments):
            for s in range(1, n_steps + 1):
                u = s / float(n_steps)
                # Blended cosine-linear profile maintains non-zero velocity across waypoints
                alpha = 0.35 * u + 0.65 * 0.5 * (1.0 - np.cos(np.pi * u))
                q_cmd = (1.0 - alpha) * q0 + alpha * q1
                g_cmd = (1.0 - alpha) * g0 + alpha * g1
                action = np.concatenate([q_cmd, [g_cmd]]).astype(np.float32)

                if is_dart:
                    if seg_idx == 0:
                        # Ramp offset from 0 to full dq_hover at hover point
                        q_exec = q_cmd + alpha * dq_hover
                    elif seg_idx == 1:
                        # Upper descent hold, decay linearly to 0 by pinch dz = 5.5 cm
                        for j in range(5):
                            self._ik_data.qpos[self.env._qpos_adrs[j]] = float(q_cmd[j])
                        mujoco.mj_kinematics(self.env.model, self._ik_data)
                        clean_pinch_z = float(self._ik_data.site(self.env._pinch_site_id).xpos[2])
                        dz = clean_pinch_z - float(self.env.data.body(src_bid).xpos[2])
                        if dz >= 0.085:
                            w = 1.0
                        elif dz >= 0.055:
                            w = (dz - 0.055) / (0.085 - 0.055)
                        else:
                            w = 0.0
                        q_exec = q_cmd + w * dq_hover
                    else:
                        q_exec = q_cmd
                    exec_action = np.concatenate([q_exec, [g_cmd]]).astype(np.float32)
                else:
                    exec_action = action

                cur_src_z = float(self.env.data.body(src_bid).xpos[2])
                if lift_start_step is None and (cur_src_z - init_src_z) >= 0.002:
                    lift_start_step = len(raw_steps)

                step_record: dict[str, Any] = {
                    "proprio": obs["proprio"].copy(),
                    "action": action,
                    "task_id": int(task_id),
                    "qpos": self.env.data.qpos.copy(),
                    "qvel": self.env.data.qvel.copy(),
                }
                if not render_only_on_success and prev_include_rgb:
                    for cam_name in CAMERA_NAMES:
                        step_record[f"rgb_{cam_name}"] = obs[f"rgb_{cam_name}"].copy()

                obs = self.env.step(exec_action)
                raw_steps.append(step_record)

        # Step 10 extra settling frames (not saved in trajectory) to verify physical stability
        hold_action = raw_steps[-1]["action"].copy()
        for _ in range(self.SETTLE_STEPS):
            self.env.step(hold_action)

        constraints = self.env.check_constraints(task_id=task_id)
        if ver in ("v2", "v2_dart"):
            steps = raw_steps
            no_stationary_dwell = bool(len(steps) >= 2 and len(steps) == len(raw_steps))
        else:
            steps = (
                trim_stationary_frames(raw_steps, min_delta=1e-3, filter_interior=False)
                if trim_dwell
                else raw_steps
            )
            # Check that every consecutive step inside the kept trajectory has active motion
            if len(steps) >= 2:
                proprio_arr = np.stack([s["proprio"] for s in steps], axis=0)
                action_arr = np.stack([s["action"] for s in steps], axis=0)
                min_dq = float(np.min(np.linalg.norm(np.diff(proprio_arr, axis=0), axis=-1)))
                min_da = float(np.min(np.linalg.norm(action_arr - proprio_arr, axis=-1)))
                no_stationary_dwell = bool(
                    len(steps) == len(raw_steps) and min_dq >= 1e-3 and min_da >= 1e-3
                )
            else:
                no_stationary_dwell = False

        constraints["no_stationary_dwell"] = no_stationary_dwell
        constraints["all_constraints_passed"] = bool(
            constraints["all_constraints_passed"] and no_stationary_dwell
        )

        # Render the 3 cameras for kept steps if needed
        if render_only_on_success and prev_include_rgb and constraints["all_constraints_passed"]:
            w, h = self.env.rgb_resolution
            saved_qpos = self.env.data.qpos.copy()
            saved_qvel = self.env.data.qvel.copy()
            for step_record in steps:
                self.env.data.qpos[:] = step_record["qpos"]
                self.env.data.qvel[:] = step_record["qvel"]
                mujoco.mj_forward(self.env.model, self.env.data)
                for cam_name in CAMERA_NAMES:
                    step_record[f"rgb_{cam_name}"] = self.env.render_camera(
                        camera_name=cam_name, width=w, height=h
                    )
            self.env.data.qpos[:] = saved_qpos
            self.env.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.env.model, self.env.data)

        self.env.include_rgb = prev_include_rgb

        if steps:
            actions_out = np.stack([s["action"] for s in steps], axis=0).astype(np.float32)
            proprio_out = np.stack([s["proprio"] for s in steps], axis=0).astype(np.float32)
        else:
            actions_out = np.zeros((0, 6), dtype=np.float32)
            proprio_out = np.zeros((0, 6), dtype=np.float32)

        episode_data: dict[str, Any] = {
            "task_id": int(task_id),
            "seed": int(seed),
            "num_steps": len(steps),
            "trajectory_version": ver,
            "perturbation": "dart" if is_dart else "none",
            "delta_mag_cm": float(mag * 100.0) if is_dart else 0.0,
            "delta_xy": [float(delta[0]), float(delta[1])] if is_dart else [0.0, 0.0],
            "lift_start_step": int(lift_start_step if lift_start_step is not None else 45),
            "constraints": constraints,
            "actions": actions_out,
            "proprio": proprio_out,
        }
        has_rendered_rgb = (
            prev_include_rgb
            and bool(steps)
            and ((not render_only_on_success) or constraints["all_constraints_passed"])
        )
        if has_rendered_rgb:
            for cam_name in CAMERA_NAMES:
                key = f"rgb_{cam_name}"
                episode_data[key] = np.stack([s[key] for s in steps], axis=0).astype(np.uint8)

        return episode_data
