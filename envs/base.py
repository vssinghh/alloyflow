"""Shared base interface, task specifications, and unit conversions for SO-ARM101."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ALL_JOINT_NAMES: tuple[str, ...] = (*ARM_JOINT_NAMES, "gripper")

CAMERA_NAMES: tuple[str, ...] = (
    "third_person_cam",
    "overhead_cam",
    "wrist_cam",
)

OBJECT_NAMES: tuple[str, ...] = (
    "pen_holder",
    "cup",
    "bowl",
    "rubiks_cube",
)

# Neutral home pose (5 arm joints in radians + 1 normalized gripper in [0.0, 1.0])
# Keeps the arm retracted at X ~ 0.16m so all 4 objects are unobstructed at t=0,
# with wrist_roll = -pi/2 so the two jaws open horizontally along Y and wrist_cam sits on top.
HOME_PROPRIO_6D = np.array(
    [0.0, -0.95, 0.85, 0.75, -np.pi / 2, 1.0],
    dtype=np.float32,
)

# Physical SO-ARM101 gripper hinge range Mapped to normalized [0.0 (closed), 1.0 (open)]
GRIPPER_RAW_CLOSED: float = -0.15
GRIPPER_RAW_OPEN: float = 0.85

# Feetech STS3215 12-bit magnetic encoder resolution (4096 ticks per 2*pi radians)
STS3215_TICKS_PER_REV: int = 4096
STS3215_CENTER_TICK: int = 2048


@dataclass(frozen=True)
class TaskSpec:
    """Specification for one of the 3 AlloyFlow tabletop manipulation tasks."""

    task_id: int
    name: str
    description: str
    source_object: str
    target_object: str
    goal_type: str  # "place_inside" or "stack_on_top"

    @property
    def bystander_objects(self) -> tuple[str, ...]:
        """Return objects that should not be disturbed during this task."""
        return tuple(
            obj
            for obj in OBJECT_NAMES
            if obj not in (self.source_object, self.target_object)
        )


TASK_SPECS: dict[int, TaskSpec] = {
    0: TaskSpec(
        task_id=0,
        name="pick_pen_holder_to_bowl",
        description="Pick up the pen holder and place it inside the bowl",
        source_object="pen_holder",
        target_object="bowl",
        goal_type="place_inside",
    ),
    1: TaskSpec(
        task_id=1,
        name="pick_cup_to_bowl",
        description="Ignore the pen holder, pick up the small cup, and place it inside the bowl",
        source_object="cup",
        target_object="bowl",
        goal_type="place_inside",
    ),
    2: TaskSpec(
        task_id=2,
        name="stack_cup_on_cube",
        description="Pick up the small cup and stack it on top of the Rubik's cube",
        source_object="cup",
        target_object="rubiks_cube",
        goal_type="stack_on_top",
    ),
}
NUM_TASKS: int = len(TASK_SPECS)


def raw_gripper_to_norm(raw_rad: float) -> float:
    """Convert raw MuJoCo/servo gripper angle (radians) to normalized [0.0, 1.0]."""
    norm = (float(raw_rad) - GRIPPER_RAW_CLOSED) / (GRIPPER_RAW_OPEN - GRIPPER_RAW_CLOSED)
    return float(np.clip(norm, 0.0, 1.0))


def norm_gripper_to_raw(norm_val: float) -> float:
    """Convert normalized gripper command in [0.0, 1.0] to raw angle (radians)."""
    clipped = float(np.clip(norm_val, 0.0, 1.0))
    return float(GRIPPER_RAW_CLOSED + clipped * (GRIPPER_RAW_OPEN - GRIPPER_RAW_CLOSED))


def ticks_to_radians(ticks: np.ndarray, zero_offsets: np.ndarray | None = None) -> np.ndarray:
    """Convert Feetech STS3215 encoder ticks (0..4095) to radians relative to home offsets."""
    arr = np.asarray(ticks, dtype=np.float32)
    offsets = (
        np.asarray(zero_offsets, dtype=np.float32)
        if zero_offsets is not None
        else np.full_like(arr, float(STS3215_CENTER_TICK))
    )
    return (arr - offsets) * (2.0 * np.pi / float(STS3215_TICKS_PER_REV))


def radians_to_ticks(radians: np.ndarray, zero_offsets: np.ndarray | None = None) -> np.ndarray:
    """Convert radians relative to home offsets into Feetech STS3215 encoder ticks (0..4095)."""
    arr = np.asarray(radians, dtype=np.float32)
    offsets = (
        np.asarray(zero_offsets, dtype=np.float32)
        if zero_offsets is not None
        else np.full_like(arr, float(STS3215_CENTER_TICK))
    )
    ticks = np.round(offsets + arr * (float(STS3215_TICKS_PER_REV) / (2.0 * np.pi)))
    return np.clip(ticks, 0, STS3215_TICKS_PER_REV - 1).astype(np.int32)


class SO101Env(ABC):
    """Shared environment contract implemented by both SimEnv and RealEnv."""

    control_hz: int = 20

    @abstractmethod
    def reset(self, task_id: int = 0, seed: int | None = None) -> dict[str, Any]:
        """Reset the environment for a given task ID (0, 1, or 2) and return initial obs."""

    @abstractmethod
    def step(self, action_6d: np.ndarray) -> dict[str, Any]:
        """Execute one 20 Hz control step with a 6D target action and return new obs."""

    @abstractmethod
    def get_obs(self) -> dict[str, Any]:
        """Return current observation dictionary with 'proprio' and 'rgb_<cam>' keys."""

    @abstractmethod
    def close(self) -> None:
        """Release simulation renderer or hardware serial/camera handles."""
