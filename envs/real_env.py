"""Physical 6-DoF SO-ARM101 Environment over USB Serial and OpenCV Webcams."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from envs.base import (
    CAMERA_NAMES,
    HOME_PROPRIO_6D,
    STS3215_CENTER_TICK,
    TASK_SPECS,
    SO101Env,
    norm_gripper_to_raw,
    radians_to_ticks,
    raw_gripper_to_norm,
    ticks_to_radians,
)


class RealEnv(SO101Env):
    """Physical SO-ARM101 environment matching SimEnv's exact 20 Hz observation/action API."""

    def __init__(
        self,
        serial_port: str = "/dev/tty.usbmodem58760431551",
        baudrate: int = 1_000_000,
        camera_indices: dict[str, int] | None = None,
        control_hz: int = 20,
        rgb_resolution: tuple[int, int] = (128, 128),
        zero_offsets: np.ndarray | None = None,
        mock_hardware: bool = False,
    ) -> None:
        """Initialize the physical SO-ARM101 follower arm and 3 USB camera streams."""
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.control_hz = control_hz
        self.step_dt = 1.0 / float(control_hz)
        self.rgb_resolution = rgb_resolution
        self.mock_hardware = mock_hardware
        self.zero_offsets = (
            np.asarray(zero_offsets, dtype=np.float32)
            if zero_offsets is not None
            else np.full(6, float(STS3215_CENTER_TICK), dtype=np.float32)
        )
        self.camera_indices = camera_indices or {
            "third_person_cam": 0,
            "overhead_cam": 1,
            "wrist_cam": 2,
        }

        self.current_task_id: int = 0
        self.current_step: int = 0
        self._mock_proprio = HOME_PROPRIO_6D.copy()
        self._port_handler: Any = None
        self._packet_handler: Any = None
        self._captures: dict[str, Any] = {}

        if not self.mock_hardware:
            self._connect_hardware()

    def _connect_hardware(self) -> None:
        """Open the Feetech STS3215 USB serial bus and 3 OpenCV USB camera streams."""
        try:
            import scservo_sdk as scs  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "feetech-servo-sdk (scservo_sdk) is required for physical SO-ARM101 "
                "deployment. Install with `uv pip install feetech-servo-sdk opencv-python` "
                "or pass `mock_hardware=True` for offline testing."
            ) from exc

        self._port_handler = scs.PortHandler(self.serial_port)
        self._packet_handler = scs.sms_sts(self._port_handler)
        if not self._port_handler.openPort() or not self._port_handler.setBaudRate(
            self.baudrate
        ):
            raise RuntimeError(
                f"Failed to open SO-ARM101 serial port {self.serial_port} at {self.baudrate} baud."
            )

        import cv2  # type: ignore[import-not-found]

        for cam_name, dev_idx in self.camera_indices.items():
            cap = cv2.VideoCapture(dev_idx)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Failed to open USB camera '{cam_name}' at index {dev_idx}."
                )
            self._captures[cam_name] = cap

    def _read_proprio_6d(self) -> np.ndarray:
        """Read 6 Feetech STS3215 encoders and convert to [5x radians, 1x normalized gripper]."""
        if self.mock_hardware:
            return self._mock_proprio.copy()

        ticks = np.zeros(6, dtype=np.int32)
        for servo_id in range(1, 7):
            pos, _, _ = self._packet_handler.ReadPosSpeed(servo_id)
            ticks[servo_id - 1] = int(pos)

        rads = ticks_to_radians(ticks, self.zero_offsets)
        proprio = np.zeros(6, dtype=np.float32)
        proprio[:5] = rads[:5]
        proprio[5] = raw_gripper_to_norm(float(rads[5]))
        return proprio

    def _write_action_6d(self, action_6d: np.ndarray) -> None:
        """Convert 6D action [5x radians, 1x normalized gripper] to ticks and command servos."""
        if self.mock_hardware:
            self._mock_proprio = np.asarray(action_6d, dtype=np.float32).copy()
            self._mock_proprio[5] = float(np.clip(self._mock_proprio[5], 0.0, 1.0))
            return

        raw_rads = np.asarray(action_6d, dtype=np.float32).copy()
        raw_rads[5] = norm_gripper_to_raw(float(action_6d[5]))
        target_ticks = radians_to_ticks(raw_rads, self.zero_offsets)
        for servo_id in range(1, 7):
            self._packet_handler.WritePosEx(
                servo_id, int(target_ticks[servo_id - 1]), 2400, 50
            )

    def _read_cameras(self) -> dict[str, np.ndarray]:
        """Grab and center-crop 128x128 RGB frames from all 3 USB cameras."""
        w, h = self.rgb_resolution
        frames: dict[str, np.ndarray] = {}
        if self.mock_hardware:
            for cam in CAMERA_NAMES:
                frames[f"rgb_{cam}"] = np.zeros((h, w, 3), dtype=np.uint8)
            return frames

        import cv2  # type: ignore[import-not-found]

        for cam, cap in self._captures.items():
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Failed to read frame from camera '{cam}'.")
            fh, fw = bgr.shape[:2]
            side = min(fh, fw)
            y0 = (fh - side) // 2
            x0 = (fw - side) // 2
            cropped = bgr[y0 : y0 + side, x0 : x0 + side]
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
            frames[f"rgb_{cam}"] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return frames

    def reset(self, task_id: int = 0, seed: int | None = None) -> dict[str, Any]:
        """Move the physical SO-ARM101 smoothly to HOME_PROPRIO_6D and return initial obs."""
        if task_id not in TASK_SPECS:
            raise ValueError(f"Invalid task_id {task_id}. Must be one of {list(TASK_SPECS)}.")
        self.current_task_id = int(task_id)
        self.current_step = 0
        self._write_action_6d(HOME_PROPRIO_6D)
        if not self.mock_hardware:
            time.sleep(1.0)
        return self.get_obs()

    def step(self, action_6d: np.ndarray) -> dict[str, Any]:
        """Send 6D joint target at 20 Hz (50 ms pacing) and return new observation."""
        t0 = time.perf_counter()
        act = np.asarray(action_6d, dtype=np.float32)
        if act.shape != (6,):
            raise ValueError(f"Expected 6D action vector, got shape {act.shape}")

        self._write_action_6d(act)
        self.current_step += 1
        obs = self.get_obs()

        if not self.mock_hardware:
            elapsed = time.perf_counter() - t0
            if elapsed < self.step_dt:
                time.sleep(self.step_dt - elapsed)
        return obs

    def get_obs(self) -> dict[str, Any]:
        """Return observation dict with aligned 'proprio' and 'rgb_<cam>' keys."""
        obs: dict[str, Any] = {
            "proprio": self._read_proprio_6d(),
            "task_id": self.current_task_id,
        }
        obs.update(self._read_cameras())
        return obs

    def close(self) -> None:
        """Close OpenCV camera captures and Feetech serial port."""
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None
