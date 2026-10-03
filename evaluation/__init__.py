"""Closed-loop multi-task evaluation and GIF visualization package for AlloyFlow."""

from evaluation.evaluator import (
    EpisodeEvalResult,
    SimPolicyEvaluator,
    diagnose_rollout_status,
)
from evaluation.visualizer import (
    RolloutStepRecord,
    RolloutVisualizer,
    project_3d_to_camera_pixels,
)

__all__ = [
    "EpisodeEvalResult",
    "RolloutStepRecord",
    "RolloutVisualizer",
    "SimPolicyEvaluator",
    "diagnose_rollout_status",
    "project_3d_to_camera_pixels",
]
