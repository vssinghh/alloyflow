"""Leader-follower 20 Hz teleoperation recorder for the physical 6-DoF SO-ARM101."""

from __future__ import annotations

from typing import Any

import numpy as np

from collection.sim_expert import trim_stationary_frames
from envs.base import (
    CAMERA_NAMES,
    HOME_PROPRIO_6D,
    JOINT_LIMITS_HIGH,
    JOINT_LIMITS_LOW,
    raw_gripper_to_norm,
    ticks_to_radians,
)
from envs.real_env import RealEnv


class RealTeleopRecorder:
    """Streams 6D leader arm joint positions to the follower RealEnv at 20 Hz."""

    ADDR_TORQUE_ENABLE: int = 40
    REAL_TRIM_MIN_DELTA: float = 3.5e-3

    def __init__(
        self,
        follower_env: RealEnv,
        leader_port: str = "/dev/tty.usbmodem58760431552",
        baudrate: int = 1_000_000,
        leader_zero_offsets: np.ndarray | None = None,
        mock_hardware: bool = False,
    ) -> None:
        self.follower_env = follower_env
        self.leader_port = leader_port
        self.baudrate = baudrate
        self.leader_zero_offsets = (
            np.asarray(leader_zero_offsets, dtype=np.float32)
            if leader_zero_offsets is not None
            else follower_env.zero_offsets.copy()
        )
        self.mock_hardware = mock_hardware
        self._leader_bus: Any = None

        if not self.mock_hardware:
            self._connect_leader()

    def _connect_leader(self) -> None:
        """Open the USB serial connection to the passive SO-ARM101 leader arm and disable torque."""
        try:
            import scservo_sdk as scs  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Feetech SDK (scservo_sdk) is required for physical leader-follower teleop."
            ) from exc

        port_handler = scs.PortHandler(self.leader_port)
        packet_handler = scs.sms_sts(port_handler)
        if not port_handler.openPort():
            raise RuntimeError(f"Failed to open leader serial port {self.leader_port}")
        if not port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate {self.baudrate} on {self.leader_port}")

        # Disable motor torque on all 6 leader servos so the operator can backdrive the arm freely
        for servo_id in range(1, 7):
            packet_handler.write1ByteTxRx(servo_id, self.ADDR_TORQUE_ENABLE, 0)

        self._leader_bus = (scs, port_handler, packet_handler)

    def read_leader_action(self, step_idx: int = 0) -> np.ndarray:
        """Read the 6-DoF leader arm joint state and format as a follower 6D action."""
        if self.mock_hardware or self._leader_bus is None:
            # Smooth synthetic motion for dry-run testing without USB hardware
            phase = 0.08 * float(step_idx)
            delta = np.array(
                [
                    0.15 * np.sin(phase),
                    0.10 * np.sin(phase),
                    -0.10 * np.cos(phase),
                    0.08 * np.sin(phase),
                    0.0,
                    -0.25 * (1.0 - np.cos(phase)),
                ],
                dtype=np.float32,
            )
            action = HOME_PROPRIO_6D.copy() + delta
            action[:5] = np.clip(action[:5], JOINT_LIMITS_LOW[:5], JOINT_LIMITS_HIGH[:5])
            action[5] = float(np.clip(action[5], 0.0, 1.0))
            return action

        _, _port_handler, packet_handler = self._leader_bus
        ticks = np.zeros(6, dtype=np.int32)
        for servo_id in range(1, 7):
            pos, _, _ = packet_handler.ReadPosSpeed(servo_id)
            ticks[servo_id - 1] = int(pos)

        raw_rads = ticks_to_radians(ticks, self.leader_zero_offsets)
        action = np.zeros(6, dtype=np.float32)
        action[:5] = np.clip(raw_rads[:5], JOINT_LIMITS_LOW[:5], JOINT_LIMITS_HIGH[:5])
        action[5] = raw_gripper_to_norm(float(raw_rads[5]))
        return action

    def record_episode(
        self,
        task_id: int,
        episode_seed: int = 0,
        max_steps: int = 200,
        trim_dwell: bool = True,
    ) -> dict[str, Any]:
        """Record a single 20 Hz leader-follower demonstration episode."""
        obs = self.follower_env.reset(task_id=task_id, seed=episode_seed)

        raw_steps: list[dict[str, Any]] = []
        for step_idx in range(max_steps):
            action = self.read_leader_action(step_idx=step_idx)

            step_record: dict[str, Any] = {
                "proprio": obs["proprio"].copy(),
                "action": action.copy(),
                "task_id": int(task_id),
            }
            for cam_name in CAMERA_NAMES:
                step_record[f"rgb_{cam_name}"] = obs[f"rgb_{cam_name}"].copy()

            # RealEnv.step() handles the 50 ms (20 Hz) pacing sleep before reading s_{t+1}
            obs = self.follower_env.step(action)
            raw_steps.append(step_record)

        steps = (
            trim_stationary_frames(
                raw_steps,
                min_delta=self.REAL_TRIM_MIN_DELTA,
                filter_interior=True,
            )
            if trim_dwell
            else raw_steps
        )
        w, h = self.follower_env.rgb_resolution
        if steps:
            actions_out = np.stack([s["action"] for s in steps], axis=0).astype(np.float32)
            proprio_out = np.stack([s["proprio"] for s in steps], axis=0).astype(np.float32)
        else:
            actions_out = np.zeros((0, 6), dtype=np.float32)
            proprio_out = np.zeros((0, 6), dtype=np.float32)

        episode_data: dict[str, Any] = {
            "task_id": int(task_id),
            "seed": int(episode_seed),
            "num_steps": len(steps),
            "actions": actions_out,
            "proprio": proprio_out,
        }
        for cam_name in CAMERA_NAMES:
            key = f"rgb_{cam_name}"
            if steps:
                episode_data[key] = np.stack([s[key] for s in steps], axis=0).astype(np.uint8)
            else:
                episode_data[key] = np.zeros((0, h, w, 3), dtype=np.uint8)

        return episode_data

    def close(self) -> None:
        """Close the leader serial port if open."""
        if self._leader_bus is not None:
            _, port_handler, _ = self._leader_bus
            port_handler.closePort()
            self._leader_bus = None
