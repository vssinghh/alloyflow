"""Frozen training configuration for AlloyFlow 4-mode policy training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch

from envs.base import CAMERA_NAMES, NUM_TASKS

VALID_TRAIN_MODES: tuple[str, ...] = (
    "sim_only",
    "real_only",
    "finetune",
    "cotrain",
)


@dataclass(frozen=True)
class AlloyTrainConfig:
    """Single-source-of-truth hyperparameter configuration for AlloyFlow training."""

    # Training mode & dataset paths
    train_mode: str = "sim_only"
    sim_data_path: str = "data/sim_demos_v2_dart_full_900.h5"
    real_data_path: str = "data/real_demos.h5"
    pretrained_checkpoint: str | None = None
    resume_checkpoint: str | None = None
    default_finetune_checkpoint: str = "checkpoints/exp09_e2e_nodropout/best_policy.pt"
    save_dir: str = ""
    real_ratio: float = 0.5

    # Sensor & action dimensions (SO-ARM101 6-DoF at 20 Hz)
    camera_names: tuple[str, ...] = CAMERA_NAMES
    img_size: int = 128
    proprio_dim: int = 6
    proprio_history_lags: tuple[int, ...] = (0,)
    action_dim: int = 6
    chunk_size: int = 16
    num_tasks: int = NUM_TASKS

    # Modality encoder & attention dimensions
    task_emb_dim: int = 32
    proprio_emb_dim: int = 64
    num_keypoints: int = 32
    vision_feat_dim: int = 32
    num_attn_heads: int = 4

    # Flow Matching ResMLP backbone & regularization dimensions
    hidden_dim: int = 256
    num_res_blocks: int = 4
    time_emb_dim: int = 64
    num_flow_samples: int = 4
    gripper_weight: float = 2.5
    shift_pad: int = 4
    dropout: float = 0.0
    keypoint_noise: float = 0.01
    proprio_noise_std: float = 0.02
    proprio_drop_prob: float = 0.10
    wrist_cam_drop_prob: float = 0.05
    aux_pos_loss_weight: float = 0.5
    pretrain_loc_steps: int = 3000
    loc_data_path: str = "data/loc_layouts_3000.npz"
    val_samples_per_epoch: int = 512

    # Optimization schedule
    batch_size: int = 128
    epochs: int = 40
    lr: float = 5e-4
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Inference & memory settings
    ode_steps: int = 10
    temporal_ensemble_decay: float = 0.05
    rolling_window_size: int = 8192
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.train_mode not in VALID_TRAIN_MODES:
            raise ValueError(
                f"Invalid train_mode '{self.train_mode}'. Must be one of {VALID_TRAIN_MODES}."
            )
        if not self.save_dir:
            object.__setattr__(self, "save_dir", f"checkpoints/{self.train_mode}")
        if not (0.0 < self.real_ratio < 1.0) and self.train_mode == "cotrain":
            raise ValueError(
                f"real_ratio must be in (0.0, 1.0) for cotrain mode, got {self.real_ratio}."
            )
        if self.vision_feat_dim % self.num_attn_heads != 0:
            raise ValueError(
                f"vision_feat_dim ({self.vision_feat_dim}) must be divisible by "
                f"num_attn_heads ({self.num_attn_heads})."
            )
        if self.time_emb_dim % 2 != 0:
            raise ValueError(f"time_emb_dim ({self.time_emb_dim}) must be even.")
        if self.num_flow_samples < 1:
            raise ValueError(f"num_flow_samples must be >= 1, got {self.num_flow_samples}.")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}.")

    @property
    def num_proprio_frames(self) -> int:
        """Number of causal proprioception history frames (e.g., lags 0, 4, 8 -> 3)."""
        return len(self.proprio_history_lags)

    @property
    def proprio_input_dim(self) -> int:
        """Total flattened causal proprioception dimension (3 * 6 = 18)."""
        return self.num_proprio_frames * self.proprio_dim

    @property
    def attended_cam_dim(self) -> int:
        """Total dimension of side-by-side attended camera slots (3 * 32 = 96)."""
        return len(self.camera_names) * self.vision_feat_dim

    @property
    def fused_dim(self) -> int:
        """Total dimension of fused conditioning vector (96 + 64 + 32 = 192)."""
        return self.attended_cam_dim + self.proprio_emb_dim + self.task_emb_dim

    @property
    def effective_lr(self) -> float:
        """Return learning rate for the active training mode."""
        if self.train_mode == "finetune" and self.lr == 5e-4:
            return self.finetune_lr
        return self.lr

    def resolve_device(self) -> torch.device:
        """Resolve 'auto' to 'mps', 'cuda', or 'cpu'."""
        if self.device != "auto":
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        data = asdict(self)
        data["camera_names"] = list(self.camera_names)
        data["proprio_history_lags"] = list(self.proprio_history_lags)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = True) -> AlloyTrainConfig:
        """Reconstruct AlloyTrainConfig from a dictionary."""
        valid_keys = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid_keys
        if strict and unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        if "camera_names" in filtered and isinstance(filtered["camera_names"], list):
            filtered["camera_names"] = tuple(filtered["camera_names"])
        if "proprio_history_lags" in filtered and isinstance(
            filtered["proprio_history_lags"], list
        ):
            filtered["proprio_history_lags"] = tuple(
                int(x) for x in filtered["proprio_history_lags"]
            )
        return cls(**filtered)


DEFAULT_TRAIN_CONFIG = AlloyTrainConfig()
