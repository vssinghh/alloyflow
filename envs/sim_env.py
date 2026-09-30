"""MuJoCo Digital Twin Environment for the 6-DoF SO-ARM101 Manipulator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import mujoco
import numpy as np

from envs.base import (
    ALL_JOINT_NAMES,
    CAMERA_NAMES,
    HOME_PROPRIO_6D,
    OBJECT_NAMES,
    TASK_SPECS,
    SO101Env,
    norm_gripper_to_raw,
    raw_gripper_to_norm,
)


class SimEnv(SO101Env):
    """MuJoCo tabletop simulation environment for the 6-DoF SO-ARM101 across 3 tasks."""

    # Reachable tabletop workspace bounds for the 4 objects
    DEFAULT_WORKSPACE_BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "x": (0.185, 0.295),
        "y": (-0.165, 0.165),
    }

    # Resting Z heights on tabletop (Z = 0.0) for each object's body origin
    OBJECT_REST_Z: ClassVar[dict[str, float]] = {
        "pen_holder": 0.035,
        "cup": 0.020,
        "bowl": 0.004,
        "rubiks_cube": 0.028,
    }

    # Constraint thresholds locked in DESIGN.md
    MIN_SPAWN_CLEARANCE_M: float = 0.080
    MAX_BYSTANDER_DISTURBANCE_M: float = 0.015
    BOWL_INNER_RADIUS_M: float = 0.032
    CUBE_STACK_XY_TOL_M: float = 0.020

    def __init__(
        self,
        xml_path: str | None = None,
        control_hz: int = 20,
        physics_dt: float = 0.002,
        include_rgb: bool = True,
        rgb_cameras: tuple[str, ...] = CAMERA_NAMES,
        rgb_resolution: tuple[int, int] = (128, 128),
        domain_rand: bool = False,
        workspace_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize the SO-ARM101 MuJoCo simulation environment."""
        if xml_path is None:
            repo_root = Path(__file__).resolve().parent.parent
            xml_path = str(repo_root / "assets" / "so101" / "scene.xml")

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"SO-ARM101 scene XML not found at: {xml_path}")

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.control_hz = control_hz
        self.physics_dt = physics_dt
        self.model.opt.timestep = physics_dt
        self.substeps = round((1.0 / control_hz) / physics_dt)

        self.include_rgb = include_rgb
        self.rgb_cameras = tuple(rgb_cameras)
        self.rgb_resolution = rgb_resolution
        self.domain_rand = domain_rand
        self.workspace_bounds = (
            workspace_bounds
            if workspace_bounds is not None
            else dict(self.DEFAULT_WORKSPACE_BOUNDS)
        )

        # Joint and actuator indices for the 6-DoF SO-ARM101
        self._joint_ids = [self.model.joint(name).id for name in ALL_JOINT_NAMES]
        self._qpos_adrs = [int(self.model.jnt_qposadr[jid]) for jid in self._joint_ids]
        self._dof_adrs = [int(self.model.jnt_dofadr[jid]) for jid in self._joint_ids]
        self._actuator_ids = [self.model.actuator(name).id for name in ALL_JOINT_NAMES]

        # Object body, joint, and site IDs
        self._obj_body_ids: dict[str, int] = {
            name: self.model.body(name).id for name in OBJECT_NAMES
        }
        self._obj_qpos_adrs: dict[str, int] = {
            name: int(self.model.jnt_qposadr[self.model.joint(f"{name}_joint").id])
            for name in OBJECT_NAMES
        }
        self._obj_dof_adrs: dict[str, int] = {
            name: int(self.model.jnt_dofadr[self.model.joint(f"{name}_joint").id])
            for name in OBJECT_NAMES
        }
        self._obj_geom_ids: dict[str, list[int]] = {
            name: [
                gid
                for gid in range(self.model.ngeom)
                if int(self.model.geom_bodyid[gid]) == self._obj_body_ids[name]
            ]
            for name in OBJECT_NAMES
        }

        self._table_geom_id = self.model.geom("desk_top").id
        self._floor_geom_id = self.model.geom("floor").id
        self._pinch_site_id = self.model.site("pinch_site").id
        self._gripper_site_id = self.model.site("gripperframe").id

        # Material and camera IDs for domain randomization
        self._mat_ids = {
            name: self.model.material(name).id
            for name in (
                "wood_table",
                "pen_holder_mat",
                "cup_mat",
                "bowl_mat",
                "bowl_inner_mat",
            )
        }
        self._pen_rim_geom_id = self.model.geom("pen_holder_rim").id
        self._cam_ids = {
            cam: self.model.camera(cam).id
            for cam in CAMERA_NAMES
            if self.model.camera(cam) is not None
        }

        # Snapshot nominal visual and physical model parameters
        self._init_light_pos = self.model.light_pos.copy()
        self._init_light_dir = self.model.light_dir.copy()
        self._init_light_diffuse = self.model.light_diffuse.copy()
        self._init_cam_pos = self.model.cam_pos.copy()
        self._init_cam_quat = self.model.cam_quat.copy()
        self._init_mat_rgba = self.model.mat_rgba.copy()
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_friction = self.model.geom_friction.copy()
        self._init_body_mass = self.model.body_mass.copy()
        self._init_body_inertia = self.model.body_inertia.copy()
        self._init_dof_damping = self.model.dof_damping.copy()

        self.renderer: mujoco.Renderer | None = None
        self.current_task_id: int = 0
        self.current_step: int = 0
        self.initial_object_pos: dict[str, np.ndarray] = {}
        self.initial_collision: bool = False
        self.max_bystander_displacement: float = 0.0
        self.bystander_tipped: bool = False
        self.max_target_displacement: float = 0.0
        self.target_tipped: bool = False
        self.last_domain_params: dict[str, Any] | None = None

    def restore_nominal_domain(self) -> None:
        """Restore all visual and physical parameters to nominal studio settings."""
        self.model.light_pos[:] = self._init_light_pos
        self.model.light_dir[:] = self._init_light_dir
        self.model.light_diffuse[:] = self._init_light_diffuse
        self.model.cam_pos[:] = self._init_cam_pos
        self.model.cam_quat[:] = self._init_cam_quat
        self.model.mat_rgba[:] = self._init_mat_rgba
        self.model.geom_rgba[:] = self._init_geom_rgba
        self.model.geom_friction[:] = self._init_geom_friction
        self.model.body_mass[:] = self._init_body_mass
        self.model.body_inertia[:] = self._init_body_inertia
        self.model.dof_damping[:] = self._init_dof_damping
        self.last_domain_params = None

    def sample_domain_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample Sim-to-Real visual and physical domain randomization parameters."""
        light_pos_delta = rng.uniform(-0.25, 0.25, size=self._init_light_pos.shape)
        light_dir_delta = rng.uniform(-0.18, 0.18, size=self._init_light_dir.shape)
        light_intensity_scale = float(rng.uniform(0.65, 1.35))
        light_rgb_tint = rng.uniform(0.88, 1.12, size=(3,))

        # Hue-preserving luminance scale + bounded channel tint so objects never camouflage
        mat_rgb: dict[str, list[float]] = {}
        for name, mid in self._mat_ids.items():
            base_rgb = self._init_mat_rgba[mid, :3]
            lum = float(rng.uniform(0.72, 1.28))
            tint = rng.uniform(-0.05, 0.05, size=(3,))
            rgb = np.clip(base_rgb * lum + tint, 0.08, 0.96)
            mat_rgb[name] = [float(v) for v in rgb]

        cam_pos_deltas: dict[str, list[float]] = {}
        cam_euler_deltas: dict[str, list[float]] = {}
        for cam_name in self._cam_ids:
            pos_lim = 0.003 if cam_name == "wrist_cam" else 0.008
            rot_lim = 0.008 if cam_name == "wrist_cam" else 0.012
            cam_pos_deltas[cam_name] = [float(v) for v in rng.uniform(-pos_lim, pos_lim, size=(3,))]
            cam_euler_deltas[cam_name] = [float(v) for v in rng.uniform(-rot_lim, rot_lim, size=(3,))]

        obj_mass_scales = {
            name: float(rng.uniform(0.75, 1.40)) for name in OBJECT_NAMES
        }
        arm_damping_scales = [float(v) for v in rng.uniform(0.85, 1.20, size=(6,))]
        friction_scale = float(rng.uniform(0.85, 1.25))

        return {
            "light_pos_delta": [[float(v) for v in row] for row in light_pos_delta],
            "light_dir_delta": [[float(v) for v in row] for row in light_dir_delta],
            "light_intensity_scale": light_intensity_scale,
            "light_rgb_tint": [float(v) for v in light_rgb_tint],
            "mat_rgb": mat_rgb,
            "cam_pos_deltas": cam_pos_deltas,
            "cam_euler_deltas": cam_euler_deltas,
            "obj_mass_scales": obj_mass_scales,
            "arm_damping_scales": arm_damping_scales,
            "friction_scale": friction_scale,
        }

    def apply_domain_params(self, params: dict[str, Any]) -> None:
        """Apply sampled domain randomization parameters to the MuJoCo model."""
        self.restore_nominal_domain()

        self.model.light_pos[:] = self._init_light_pos + np.array(
            params["light_pos_delta"], dtype=np.float64
        )
        if "light_dir_delta" in params:
            raw_dir = self._init_light_dir + np.array(
                params["light_dir_delta"], dtype=np.float64
            )
            norms = np.linalg.norm(raw_dir, axis=-1, keepdims=True) + 1e-8
            self.model.light_dir[:] = raw_dir / norms

        scale = float(params["light_intensity_scale"])
        tint = np.array(params["light_rgb_tint"], dtype=np.float32)[None, :]
        self.model.light_diffuse[:] = np.clip(
            self._init_light_diffuse * scale * tint, 0.05, 1.50
        )

        mat_rgb = params.get("mat_rgb", {})
        for name, mid in self._mat_ids.items():
            if name in mat_rgb:
                self.model.mat_rgba[mid, :3] = np.clip(
                    np.array(mat_rgb[name], dtype=np.float32), 0.08, 0.96
                )
        if "pen_holder_mat" in mat_rgb:
            self.model.geom_rgba[self._pen_rim_geom_id, :3] = np.clip(
                np.array(mat_rgb["pen_holder_mat"], dtype=np.float32) * 0.80,
                0.05,
                0.90,
            )

        cam_pos_deltas = params.get("cam_pos_deltas", {})
        cam_euler_deltas = params.get("cam_euler_deltas", {})
        for cam_name, cid in self._cam_ids.items():
            if cam_name in cam_pos_deltas:
                self.model.cam_pos[cid] = self._init_cam_pos[cid] + np.array(
                    cam_pos_deltas[cam_name], dtype=np.float64
                )
            if cam_name in cam_euler_deltas:
                euler = np.array(cam_euler_deltas[cam_name], dtype=np.float64)
                angle = float(np.linalg.norm(euler))
                if angle > 1e-8:
                    axis = euler / angle
                    dq = np.zeros(4, dtype=np.float64)
                    mujoco.mju_axisAngle2Quat(dq, axis, angle)
                    q_new = np.zeros(4, dtype=np.float64)
                    mujoco.mju_mulQuat(q_new, self._init_cam_quat[cid], dq)
                    self.model.cam_quat[cid] = q_new

        for name, m_scale in params.get("obj_mass_scales", {}).items():
            bid = self._obj_body_ids[name]
            self.model.body_mass[bid] = self._init_body_mass[bid] * float(m_scale)
            self.model.body_inertia[bid] = self._init_body_inertia[bid] * float(m_scale)

        for dof_adr, d_scale in zip(
            self._dof_adrs, params.get("arm_damping_scales", [1.0] * 6)
        ):
            self.model.dof_damping[dof_adr] = self._init_dof_damping[dof_adr] * float(d_scale)

        if "friction_scale" in params:
            f_scale = float(params["friction_scale"])
            self.model.geom_friction[:, 0] = np.clip(
                self._init_geom_friction[:, 0] * f_scale, 0.20, 2.50
            )

        self.last_domain_params = params

    @classmethod
    def is_valid_spawn(
        cls,
        positions_xy: dict[str, tuple[float, float]],
        min_clearance: float = MIN_SPAWN_CLEARANCE_M,
    ) -> bool:
        """Check that all 4 objects are at least min_clearance meters apart in (X, Y)."""
        names = list(positions_xy.keys())
        for i in range(len(names)):
            p_i = np.asarray(positions_xy[names[i]], dtype=np.float64)
            for j in range(i + 1, len(names)):
                p_j = np.asarray(positions_xy[names[j]], dtype=np.float64)
                # Require extra clearance around the wider 10.4cm bowl
                req = min_clearance + (
                    0.015 if "bowl" in (names[i], names[j]) else 0.0
                )
                if float(np.linalg.norm(p_i - p_j)) < req:
                    return False
        return True

    def has_initial_collision(self) -> bool:
        """Return True if any contact exists other than objects resting on desk_top."""
        allowed_support = {self._table_geom_id, self._floor_geom_id}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist > 1e-4:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 not in allowed_support and g2 not in allowed_support:
                return True
        return False

    def _sample_object_spawns(self, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
        """Sample collision-free (X, Y) tabletop coordinates for all 4 objects."""
        x_low, x_high = self.workspace_bounds["x"]
        y_low, y_high = self.workspace_bounds["y"]

        for _ in range(500):
            candidate: dict[str, tuple[float, float]] = {}
            for name in OBJECT_NAMES:
                x = float(rng.uniform(x_low, x_high))
                y = float(rng.uniform(y_low, y_high))
                candidate[name] = (x, y)
            if self.is_valid_spawn(candidate, min_clearance=self.MIN_SPAWN_CLEARANCE_M):
                return candidate

        raise RuntimeError(
            "Failed to sample 4 object spawns satisfying >= 8 cm clearance within 500 tries."
        )

    def reset(
        self,
        task_id: int = 0,
        seed: int | None = None,
        object_xy_overrides: dict[str, tuple[float, float]] | None = None,
        domain_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset the simulation to the home pose and spawn the 4 objects."""
        if task_id not in TASK_SPECS:
            raise ValueError(f"Invalid task_id {task_id}. Must be one of {list(TASK_SPECS)}.")

        self.current_task_id = int(task_id)
        self.current_step = 0
        spawn_rng = np.random.default_rng(seed)
        dr_rng = np.random.default_rng(
            None if seed is None else int(seed) + 1_000_000
        )

        if domain_params is not None:
            self.apply_domain_params(domain_params)
        elif self.domain_rand:
            self.apply_domain_params(self.sample_domain_params(dr_rng))
        else:
            self.restore_nominal_domain()

        mujoco.mj_resetData(self.model, self.data)

        # Set SO-ARM101 to home pose
        raw_home = HOME_PROPRIO_6D.copy()
        raw_home[5] = norm_gripper_to_raw(float(HOME_PROPRIO_6D[5]))
        for idx, (q_adr, act_id) in enumerate(zip(self._qpos_adrs, self._actuator_ids)):
            self.data.qpos[q_adr] = float(raw_home[idx])
            self.data.ctrl[act_id] = float(raw_home[idx])

        # Place the 4 tabletop objects using spawn_rng (independent of domain_rand)
        spawns_xy = (
            object_xy_overrides
            if object_xy_overrides is not None
            else self._sample_object_spawns(spawn_rng)
        )
        for name in OBJECT_NAMES:
            q_adr = self._obj_qpos_adrs[name]
            x, y = spawns_xy[name]
            z = self.OBJECT_REST_Z[name]
            self.data.qpos[q_adr : q_adr + 3] = [x, y, z]
            self.data.qpos[q_adr + 3 : q_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)

        # Settle objects onto the tabletop for 5 control steps
        for _ in range(5 * self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Snapshot initial object 3D positions and reset bystander & target tracking
        self.initial_object_pos = {
            name: self.data.body(self._obj_body_ids[name]).xpos.copy()
            for name in OBJECT_NAMES
        }
        self.initial_collision = self.has_initial_collision()
        self.max_bystander_displacement = 0.0
        self.bystander_tipped = False
        self.max_target_displacement = 0.0
        self.target_tipped = False

        return self.get_obs()

    def _update_bystander_metrics(self) -> None:
        """Track peak XY movement and uprightness of bystander and target objects."""
        spec = TASK_SPECS[self.current_task_id]
        for obj_name in spec.bystander_objects:
            bid = self._obj_body_ids[obj_name]
            cur_pos = self.data.body(bid).xpos
            init_pos = self.initial_object_pos[obj_name]
            xy_disp = float(np.linalg.norm(cur_pos[:2] - init_pos[:2]))
            self.max_bystander_displacement = max(self.max_bystander_displacement, xy_disp)
            up_z = float(self.data.body(bid).xmat.reshape(3, 3)[2, 2])
            if up_z < 0.80:
                self.bystander_tipped = True

        # Also monitor the target receptacle (bowl or rubiks_cube) against shoving or tipping
        tgt_name = spec.target_object
        tgt_bid = self._obj_body_ids[tgt_name]
        tgt_cur = self.data.body(tgt_bid).xpos
        tgt_init = self.initial_object_pos[tgt_name]
        tgt_xy_disp = float(np.linalg.norm(tgt_cur[:2] - tgt_init[:2]))
        self.max_target_displacement = max(self.max_target_displacement, tgt_xy_disp)
        tgt_up_z = float(self.data.body(tgt_bid).xmat.reshape(3, 3)[2, 2])
        if tgt_up_z < 0.90:
            self.target_tipped = True

    def step(self, action_6d: np.ndarray) -> dict[str, Any]:
        """Step the simulation by 50 ms (20 Hz) with a 6D action vector."""
        act = np.asarray(action_6d, dtype=np.float32)
        if act.shape != (6,):
            raise ValueError(f"Expected 6D action vector, got shape {act.shape}")

        # Convert normalized gripper [0.0, 1.0] to raw hinge radians
        for idx in range(5):
            self.data.ctrl[self._actuator_ids[idx]] = float(act[idx])
        self.data.ctrl[self._actuator_ids[5]] = norm_gripper_to_raw(float(act[5]))

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.current_step += 1
        self._update_bystander_metrics()
        return self.get_obs()

    def _is_object_upright(self, obj_name: str, min_cos: float = 0.85) -> bool:
        """Return True if the object's local +Z axis points upward."""
        xmat = self.data.body(self._obj_body_ids[obj_name]).xmat.reshape(3, 3)
        return float(xmat[2, 2]) >= min_cos

    def check_task_success(self, task_id: int | None = None) -> bool:
        """Verify whether the target task goal is physically satisfied and settled."""
        tid = self.current_task_id if task_id is None else int(task_id)
        spec = TASK_SPECS[tid]

        src_pos = self.data.body(self._obj_body_ids[spec.source_object]).xpos
        tgt_pos = self.data.body(self._obj_body_ids[spec.target_object]).xpos
        xy_dist = float(np.linalg.norm(src_pos[:2] - tgt_pos[:2]))

        # Check gripper is open and retracted clear of the placed object
        raw_grip = float(self.data.qpos[self._qpos_adrs[5]])
        grip_norm = raw_gripper_to_norm(raw_grip)
        pinch_pos = self.data.site(self._pinch_site_id).xpos
        gripper_clear = grip_norm >= 0.45 and float(np.linalg.norm(pinch_pos - src_pos)) >= 0.035

        # Check placed object is physically settled and target receptacle was not shoved or tipped
        src_dof = self._obj_dof_adrs[spec.source_object]
        src_speed = float(np.linalg.norm(self.data.qvel[src_dof : src_dof + 6]))
        tgt_init = self.initial_object_pos.get(spec.target_object, tgt_pos)
        tgt_disp = float(np.linalg.norm(tgt_pos[:2] - tgt_init[:2]))
        tgt_stable = (
            self._is_object_upright(spec.target_object, min_cos=0.92)
            and tgt_disp <= self.MAX_BYSTANDER_DISTURBANCE_M
            and not self.target_tipped
        )

        if not gripper_clear or not tgt_stable or src_speed > 0.25:
            return False

        if spec.goal_type == "place_inside":
            # Pen holder or cup seated upright inside the bowl cavity
            in_bowl_xy = xy_dist <= self.BOWL_INNER_RADIUS_M
            in_bowl_z = (tgt_pos[2] + 0.008) <= src_pos[2] <= (tgt_pos[2] + 0.055)
            src_upright = self._is_object_upright(spec.source_object, min_cos=0.85)
            return bool(in_bowl_xy and in_bowl_z and src_upright)

        if spec.goal_type == "stack_on_top":
            # Cup stacked stably on top of the 5.6cm Rubik's cube
            on_cube_xy = xy_dist <= self.CUBE_STACK_XY_TOL_M
            expected_z = float(tgt_pos[2]) + 0.028 + 0.020
            on_cube_z = abs(float(src_pos[2]) - expected_z) <= 0.012
            src_upright = self._is_object_upright(spec.source_object, min_cos=0.88)
            return bool(on_cube_xy and on_cube_z and src_upright)

        return False

    def check_constraints(self, task_id: int | None = None) -> dict[str, Any]:
        """Return detailed breakdown of all simulation quality & constraint checks."""
        tid = self.current_task_id if task_id is None else int(task_id)
        init_xy = {
            name: (float(pos[0]), float(pos[1]))
            for name, pos in self.initial_object_pos.items()
        }
        spawn_clearance_ok = (
            self.is_valid_spawn(init_xy, min_clearance=self.MIN_SPAWN_CLEARANCE_M)
            and not self.initial_collision
        )
        bystander_ok = (
            self.max_bystander_displacement <= self.MAX_BYSTANDER_DISTURBANCE_M
            and not self.bystander_tipped
            and self.max_target_displacement <= self.MAX_BYSTANDER_DISTURBANCE_M
            and not self.target_tipped
        )
        task_success = self.check_task_success(tid)

        return {
            "spawn_clearance_ok": bool(spawn_clearance_ok),
            "bystander_ok": bool(bystander_ok),
            "max_bystander_displacement_m": float(self.max_bystander_displacement),
            "bystander_tipped": bool(self.bystander_tipped),
            "max_target_displacement_m": float(self.max_target_displacement),
            "target_tipped": bool(self.target_tipped),
            "task_success": bool(task_success),
            "all_constraints_passed": bool(
                spawn_clearance_ok and bystander_ok and task_success
            ),
        }

    def render_camera(
        self,
        camera_name: str = "third_person_cam",
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        """Render an RGB image (H, W, 3) uint8 from the specified camera."""
        w = width if width is not None else self.rgb_resolution[0]
        h = height if height is not None else self.rgb_resolution[1]
        if (
            self.renderer is None
            or self.renderer.width != w
            or self.renderer.height != h
        ):
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height=h, width=w)

        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render().copy()

    def get_obs(self) -> dict[str, Any]:
        """Return observation dict with aligned 6D proprio, 3x RGB cameras, and sim metadata."""
        arm_qpos = np.array(
            [self.data.qpos[adr] for adr in self._qpos_adrs[:5]], dtype=np.float32
        )
        raw_grip = float(self.data.qpos[self._qpos_adrs[5]])
        grip_norm = np.float32(raw_gripper_to_norm(raw_grip))
        proprio = np.concatenate([arm_qpos, [grip_norm]], axis=0).astype(np.float32)

        obs: dict[str, Any] = {
            "proprio": proprio,
            "task_id": self.current_task_id,
            "pinch_pos": self.data.site(self._pinch_site_id).xpos.copy().astype(np.float32),
            "pinch_mat": self.data.site(self._pinch_site_id).xmat.copy().reshape(3, 3).astype(np.float32),
            "object_positions": {
                name: self.data.body(self._obj_body_ids[name]).xpos.copy().astype(np.float32)
                for name in OBJECT_NAMES
            },
            "max_bystander_displacement": float(self.max_bystander_displacement),
        }

        if self.include_rgb:
            w, h = self.rgb_resolution
            for cam in self.rgb_cameras:
                obs[f"rgb_{cam}"] = self.render_camera(camera_name=cam, width=w, height=h)

        return obs

    def close(self) -> None:
        """Release the MuJoCo OpenGL renderer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
