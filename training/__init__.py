"""AlloyFlow policy training package (4-mode multi-task Vision Flow Matching)."""

from training.config import DEFAULT_TRAIN_CONFIG, VALID_TRAIN_MODES, AlloyTrainConfig
from training.dataset import (
    HDF5DemoDataset,
    MultiModeBatchLoader,
    build_action_chunks,
    compute_combined_norm_stats,
)
from training.flow_matching import ConditionalFlowMatcher, TemporalEnsembler
from training.model import (
    ResMLPBlock,
    SinusoidalTimeEmbedding,
    SpatialSoftmax2d,
    SpatialSoftmaxConvNet,
    TaskConditionedVisionFlowPolicy,
)
from training.trainer import (
    PolicyTrainer,
    load_policy_checkpoint,
    save_policy_checkpoint,
)

__all__ = [
    "DEFAULT_TRAIN_CONFIG",
    "VALID_TRAIN_MODES",
    "AlloyTrainConfig",
    "ConditionalFlowMatcher",
    "HDF5DemoDataset",
    "MultiModeBatchLoader",
    "PolicyTrainer",
    "ResMLPBlock",
    "SinusoidalTimeEmbedding",
    "SpatialSoftmax2d",
    "SpatialSoftmaxConvNet",
    "TaskConditionedVisionFlowPolicy",
    "TemporalEnsembler",
    "build_action_chunks",
    "compute_combined_norm_stats",
    "load_policy_checkpoint",
    "save_policy_checkpoint",
]
