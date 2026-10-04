"""Task-conditioned Vision Flow Matching policy network for SO-ARM101."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from training.config import DEFAULT_TRAIN_CONFIG, AlloyTrainConfig


class SpatialSoftmax2d(nn.Module):
    """Differentiable 2D Spatial Softmax layer that extracts expected (x, y) keypoints."""

    def __init__(self, height: int = 8, width: int = 8, temperature: float = 1.0) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.temperature = nn.Parameter(torch.tensor(float(temperature), dtype=torch.float32))

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.height),
            torch.linspace(-1.0, 1.0, self.width),
            indexing="ij",
        )
        # Note: pos_y corresponds to horizontal column (X) and pos_x to vertical row (Y)
        self.register_buffer("grid_x", pos_y.reshape(1, 1, self.height * self.width))
        self.register_buffer("grid_y", pos_x.reshape(1, 1, self.height * self.width))

    def forward(
        self,
        features: torch.Tensor,
        return_keypoints: bool = False,
        viz_temperature: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute expected (x, y) coordinates in [-1, 1] for each channel.

        Args:
            features: (B, C, H, W) activation heatmaps.
            return_keypoints: If True, also return (B, C, 2) keypoint tensor.
            viz_temperature: Optional sharp temperature for diagnostic visualization
                to remove post-GroupNorm background dilution without altering flat_coords.

        Returns:
            flat_coords: (B, 2 * C) concatenated [x_0, y_0, x_1, y_1, ...] coordinates.
        """
        b, c, h, w = features.shape
        if h != self.height or w != self.width:
            raise ValueError(
                f"Expected feature map spatial size ({self.height}, {self.width}), got ({h}, {w})."
            )
        temp = self.temperature.abs().clamp(min=1e-4)
        flat = features.reshape(b, c, h * w)
        probs = F.softmax(flat / temp, dim=-1)

        exp_x = torch.sum(probs * self.grid_x, dim=-1)
        exp_y = torch.sum(probs * self.grid_y, dim=-1)
        keypoints = torch.stack([exp_x, exp_y], dim=-1)  # (B, C, 2)
        flat_coords = keypoints.reshape(b, 2 * c)  # (B, 2 * C)

        if return_keypoints:
            if viz_temperature is not None:
                viz_probs = F.softmax(flat / max(float(viz_temperature), 1e-4), dim=-1)
                viz_x = torch.sum(viz_probs * self.grid_x, dim=-1)
                viz_y = torch.sum(viz_probs * self.grid_y, dim=-1)
                viz_kp = torch.stack([viz_x, viz_y], dim=-1)
                return flat_coords, viz_kp
            return flat_coords, keypoints
        return flat_coords


class SpatialSoftmaxConvNet(nn.Module):
    """4-layer ConvNet with GroupNorm, Task FiLM modulation, and 8x8 2D Spatial Softmax."""

    def __init__(
        self,
        num_keypoints: int = DEFAULT_TRAIN_CONFIG.num_keypoints,
        vision_feat_dim: int = DEFAULT_TRAIN_CONFIG.vision_feat_dim,
        task_emb_dim: int = DEFAULT_TRAIN_CONFIG.task_emb_dim,
        img_size: int = DEFAULT_TRAIN_CONFIG.img_size,
        keypoint_noise: float = DEFAULT_TRAIN_CONFIG.keypoint_noise,
    ) -> None:
        super().__init__()
        if img_size % 16 != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by 16.")
        feat_h = img_size // 16
        self.keypoint_noise = float(keypoint_noise)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.conv4 = nn.Conv2d(64, num_keypoints, kernel_size=3, stride=2, padding=1)
        self.gn4 = nn.GroupNorm(4, num_keypoints)

        # Task-conditioned FiLM modulation: (1 + gamma) * f + beta before Spatial Softmax
        self.task_film = nn.Linear(task_emb_dim, 2 * num_keypoints)
        nn.init.normal_(self.task_film.weight, std=0.02)
        nn.init.zeros_(self.task_film.bias)

        self.act4 = nn.SiLU()
        self.spatial_softmax = SpatialSoftmax2d(height=feat_h, width=feat_h, temperature=1.0)
        self.proj = nn.Sequential(
            nn.Linear(2 * num_keypoints, vision_feat_dim),
            nn.LayerNorm(vision_feat_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        img: torch.Tensor,
        z_task: torch.Tensor,
        return_keypoints: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode (B, 3, 128, 128) float32 image in [0, 1] into a (B, 32) camera token."""
        x = self.block1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gn4(self.conv4(x))

        gamma, beta = self.task_film(z_task).chunk(2, dim=-1)
        x = (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) * x + beta.unsqueeze(-1).unsqueeze(-1)
        x = self.act4(x)

        if return_keypoints:
            viz_temp = 0.08 if not self.training else None
            flat_coords, kp = self.spatial_softmax(
                x, return_keypoints=True, viz_temperature=viz_temp
            )
            if self.training and self.keypoint_noise > 0.0:
                flat_coords = flat_coords + torch.randn_like(flat_coords) * self.keypoint_noise
            return self.proj(flat_coords), kp
        flat_coords = self.spatial_softmax(x, return_keypoints=False)
        if self.training and self.keypoint_noise > 0.0:
            flat_coords = flat_coords + torch.randn_like(flat_coords) * self.keypoint_noise
        return self.proj(flat_coords)


class MultiCameraAttention(nn.Module):
    """4-head Query-Key camera attention that keeps weighted camera slots side by side (96D)."""

    def __init__(
        self,
        num_cameras: int = len(DEFAULT_TRAIN_CONFIG.camera_names),
        vision_feat_dim: int = DEFAULT_TRAIN_CONFIG.vision_feat_dim,
        proprio_emb_dim: int = DEFAULT_TRAIN_CONFIG.proprio_emb_dim,
        task_emb_dim: int = DEFAULT_TRAIN_CONFIG.task_emb_dim,
        num_heads: int = DEFAULT_TRAIN_CONFIG.num_attn_heads,
    ) -> None:
        super().__init__()
        self.num_cameras = int(num_cameras)
        self.vision_feat_dim = int(vision_feat_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.vision_feat_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        query_in_dim = proprio_emb_dim + task_emb_dim
        self.q_proj = nn.Linear(query_in_dim, vision_feat_dim)
        self.k_proj = nn.Linear(vision_feat_dim, vision_feat_dim)
        self.v_proj = nn.Sequential(
            nn.Linear(vision_feat_dim, vision_feat_dim),
            nn.LayerNorm(vision_feat_dim),
        )
        self.slot_norm = nn.LayerNorm(vision_feat_dim)

    def forward(
        self,
        cam_tokens: torch.Tensor,
        z_prop: torch.Tensor,
        z_task: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Weight each camera token via Query-Key dot product and keep slots side by side.

        Args:
            cam_tokens: (B, num_cams, vision_feat_dim) stacked camera embeddings.
            z_prop: (B, proprio_emb_dim) proprioception embedding.
            z_task: (B, task_emb_dim) task embedding.
            return_weights: If True, also return mean (B, num_cams) attention weights.

        Returns:
            attended_slots: (B, num_cams * vision_feat_dim) side-by-side weighted slots (96D).
        """
        b, n_cams, d = cam_tokens.shape
        q_in = torch.cat([z_prop, z_task], dim=-1)  # (B, 96)

        # Reshape Q to (B, H, 1, D_h) and K, V to (B, H, N_cams, D_h)
        q = self.q_proj(q_in).reshape(b, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(cam_tokens).reshape(b, n_cams, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(cam_tokens).reshape(b, n_cams, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention weights across the 3 cameras per head: (B, H, 1, N_cams)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)  # sums to 1.0 across the 3 cameras

        # Multiply each camera's Value by (n_cams * attn_weight) with residual + LayerNorm per slot
        weighted_v = (n_cams * attn_weights.transpose(-2, -1)) * v  # (B, H, N_cams, D_h)
        weighted_slots = weighted_v.transpose(1, 2).reshape(b, n_cams, d)  # (B, N_cams, D)
        normed_slots = self.slot_norm(cam_tokens + weighted_slots)
        attended_slots = normed_slots.reshape(b, n_cams * d)  # (B, 96)

        if return_weights:
            mean_weights = attn_weights.squeeze(2).mean(dim=1)  # (B, N_cams)
            return attended_slots, mean_weights
        return attended_slots


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous flow time tau in [0, 1]."""

    def __init__(self, dim: int = DEFAULT_TRAIN_CONFIG.time_emb_dim) -> None:
        super().__init__()
        self.dim = int(dim)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
        self.register_buffer("freqs", freqs)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Map scalar or batched tau (...) to (..., dim) sinusoidal features."""
        angles = tau.unsqueeze(-1).float() * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ResMLPBlock(nn.Module):
    """Pre-norm Residual MLP block with LayerNorm, SiLU, and Dropout."""

    def __init__(
        self,
        hidden_dim: int = DEFAULT_TRAIN_CONFIG.hidden_dim,
        dropout: float = DEFAULT_TRAIN_CONFIG.dropout,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TaskConditionedVisionFlowPolicy(nn.Module):
    """End-to-end 3-Task Vision Flow Matching Policy for the 6-DoF SO-ARM101."""

    def __init__(self, config: AlloyTrainConfig = DEFAULT_TRAIN_CONFIG) -> None:
        super().__init__()
        self.config = config
        cfg = self.config

        # 1. Task Embedding Lookup Table (3 tasks -> 32D)
        self.task_embedding = nn.Embedding(cfg.num_tasks, cfg.task_emb_dim)

        # 2. Causal Proprioception History MLP (18D -> 64D -> 64D)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(cfg.proprio_input_dim, cfg.proprio_emb_dim),
            nn.LayerNorm(cfg.proprio_emb_dim),
            nn.SiLU(),
            nn.Linear(cfg.proprio_emb_dim, cfg.proprio_emb_dim),
            nn.LayerNorm(cfg.proprio_emb_dim),
            nn.SiLU(),
        )

        # 3. Three Independent Spatial Softmax Camera CNNs with Task FiLM
        self.camera_encoders = nn.ModuleDict(
            {
                cam: SpatialSoftmaxConvNet(
                    num_keypoints=cfg.num_keypoints,
                    vision_feat_dim=cfg.vision_feat_dim,
                    task_emb_dim=cfg.task_emb_dim,
                    img_size=cfg.img_size,
                    keypoint_noise=cfg.keypoint_noise,
                )
                for cam in cfg.camera_names
            }
        )

        # 4. Multi-Camera Attention keeping 3 weighted camera slots side by side (96D)
        self.camera_attention = MultiCameraAttention(
            num_cameras=len(cfg.camera_names),
            vision_feat_dim=cfg.vision_feat_dim,
            proprio_emb_dim=cfg.proprio_emb_dim,
            task_emb_dim=cfg.task_emb_dim,
            num_heads=cfg.num_attn_heads,
        )

        # 5. Optimal-Transport Conditional Flow Matching ResMLP Head
        chunk_flat_dim = cfg.chunk_size * cfg.action_dim
        self.obs_proj = nn.Sequential(
            nn.Linear(cfg.fused_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.obs_dropout = nn.Dropout(cfg.dropout)
        self.act_proj = nn.Linear(chunk_flat_dim, cfg.hidden_dim)
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(cfg.time_emb_dim),
            nn.Linear(cfg.time_emb_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.res_blocks = nn.ModuleList(
            [ResMLPBlock(cfg.hidden_dim, dropout=cfg.dropout) for _ in range(cfg.num_res_blocks)]
        )
        self.out_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, chunk_flat_dim),
        )

        # Persistent z-score normalization buffers saved inside model checkpoints
        self.register_buffer("proprio_mean", torch.zeros(cfg.proprio_dim))
        self.register_buffer("proprio_std", torch.ones(cfg.proprio_dim))
        self.register_buffer("action_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("action_std", torch.ones(cfg.action_dim))
        self.register_buffer("stats_initialized", torch.tensor(False, dtype=torch.bool))

    def set_norm_stats(self, stats: dict[str, torch.Tensor | np.ndarray]) -> None:
        """Load dataset z-score normalization statistics into persistent model buffers."""
        device = self.proprio_mean.device
        self.proprio_mean.copy_(torch.as_tensor(stats["proprio_mean"], dtype=torch.float32, device=device))
        self.proprio_std.copy_(
            torch.as_tensor(stats["proprio_std"], dtype=torch.float32, device=device).clamp(min=1e-2)
        )
        self.action_mean.copy_(torch.as_tensor(stats["action_mean"], dtype=torch.float32, device=device))
        self.action_std.copy_(
            torch.as_tensor(stats["action_std"], dtype=torch.float32, device=device).clamp(min=1e-2)
        )
        self.stats_initialized.fill_(True)

    def get_norm_stats(self) -> dict[str, torch.Tensor]:
        """Return current normalization statistics as CPU tensors."""
        return {
            "proprio_mean": self.proprio_mean.detach().cpu().clone(),
            "proprio_std": self.proprio_std.detach().cpu().clone(),
            "action_mean": self.action_mean.detach().cpu().clone(),
            "action_std": self.action_std.detach().cpu().clone(),
        }

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        """Z-score normalize raw 6D proprioception."""
        return (proprio - self.proprio_mean) / self.proprio_std

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Z-score normalize raw 6D actions of shape (..., 6)."""
        return (actions - self.action_mean) / self.action_std

    def unnormalize_actions(self, actions_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized action predictions back to physical SO-ARM101 units."""
        return actions_norm * self.action_std + self.action_mean

    def _augment_shift(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random +-shift_pad pixel translation via bilinear border-padded grid sampling."""
        pad = self.config.shift_pad
        if not self.training or pad <= 0:
            return img
        b, _, h, w = img.shape
        shifts_x = (torch.rand(b, 1, 1, device=img.device) * 2.0 - 1.0) * (pad / float(w / 2.0))
        shifts_y = (torch.rand(b, 1, 1, device=img.device) * 2.0 - 1.0) * (pad / float(h / 2.0))
        base_y, base_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=img.device),
            torch.linspace(-1.0, 1.0, w, device=img.device),
            indexing="ij",
        )
        grid = torch.stack(
            [base_x.unsqueeze(0) + shifts_x, base_y.unsqueeze(0) + shifts_y],
            dim=-1,
        )
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def extract_obs_features(
        self,
        obs: dict[str, torch.Tensor],
        task_ids: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Encode 3 camera views, causal proprioception history, and task ID into a (B, 192) vector."""
        cfg = self.config
        z_task = self.task_embedding(task_ids.long())  # (B, 32)
        if "proprio_history" in obs:
            prop_hist_raw = obs["proprio_history"].float()
        else:
            prop_single = obs["proprio"].float()
            if prop_single.ndim == 2:
                prop_hist_raw = (
                    prop_single.unsqueeze(1)
                    .expand(-1, cfg.num_proprio_frames, -1)
                    .contiguous()
                )
            else:
                prop_hist_raw = prop_single

        q_hist_norm = self.normalize_proprio(prop_hist_raw)  # (B, K_hist, 6)
        q_curr = q_hist_norm[:, 0, :].clone()
        dq_deltas: list[torch.Tensor] = [
            (q_hist_norm[:, 0, :] - q_hist_norm[:, lag_i, :]).clone()
            for lag_i in range(1, cfg.num_proprio_frames)
        ]

        if self.training and cfg.proprio_noise_std > 0.0:
            q_curr = q_curr + torch.randn_like(q_curr) * cfg.proprio_noise_std
            for i in range(len(dq_deltas)):
                dq_deltas[i] = (
                    dq_deltas[i] + torch.randn_like(dq_deltas[i]) * cfg.proprio_noise_std
                )

        prop_feat_in = torch.cat([q_curr, *dq_deltas], dim=-1)
        z_prop = self.proprio_mlp(prop_feat_in)  # (B, 64)
        if self.training and cfg.proprio_drop_prob > 0.0:
            keep_prop = (
                torch.rand((q_curr.shape[0], 1), device=q_curr.device) >= cfg.proprio_drop_prob
            ).float()
            z_prop = z_prop * keep_prop

        cam_tokens_list: list[torch.Tensor] = []
        keypoints_dict: dict[str, torch.Tensor] = {}

        for cam in cfg.camera_names:
            key = f"rgb_{cam}" if f"rgb_{cam}" in obs else cam
            img = obs[key]
            if img.dtype == torch.uint8:
                if img.ndim == 4 and img.shape[-1] == 3:
                    img = img.permute(0, 3, 1, 2).contiguous()
                img = img.float().div_(255.0)
            elif img.ndim == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2).contiguous()

            img = self._augment_shift(img)
            if return_aux:
                token, kp = self.camera_encoders[cam](img, z_task, return_keypoints=True)
                keypoints_dict[cam] = kp
            else:
                token = self.camera_encoders[cam](img, z_task, return_keypoints=False)
            if self.training and cfg.wrist_cam_drop_prob > 0.0 and cam == "wrist_cam":
                keep_cam = (
                    torch.rand((img.shape[0], 1), device=img.device) >= cfg.wrist_cam_drop_prob
                ).float()
                token = token * keep_cam
            cam_tokens_list.append(token)

        cam_tokens = torch.stack(cam_tokens_list, dim=1)  # (B, 3, 32)
        if return_aux:
            attended_cams, attn_weights = self.camera_attention(
                cam_tokens, z_prop, z_task, return_weights=True
            )
        else:
            attended_cams = self.camera_attention(
                cam_tokens, z_prop, z_task, return_weights=False
            )

        z_fused = torch.cat([attended_cams, z_prop, z_task], dim=-1)  # (B, 192)
        if return_aux:
            return z_fused, {
                "keypoints": keypoints_dict,
                "camera_weights": attn_weights,
            }
        return z_fused

    def predict_velocity(
        self,
        z_fused: torch.Tensor,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Predict flow velocity field v_theta(x_tau, tau | z_fused).

        Supports both:
          - Single sample per obs: x_tau (B, H, 6), tau (B,) -> returns (B, H, 6)
          - Stratified K samples:  x_tau (B, K, H, 6), tau (B, K) -> returns (B, K, H, 6)
        """
        cfg = self.config
        if x_tau.ndim == 3:
            b = x_tau.shape[0]
            flat_x = x_tau.reshape(b, cfg.chunk_size * cfg.action_dim)
            obs_emb = self.obs_dropout(self.obs_proj(z_fused))
            h = obs_emb + self.act_proj(flat_x) + self.time_encoder(tau)
            for block in self.res_blocks:
                h = block(h)
            out = self.out_head(h)
            return out.reshape(b, cfg.chunk_size, cfg.action_dim)

        if x_tau.ndim == 4:
            b, k, _, _ = x_tau.shape
            flat_x = x_tau.reshape(b, k, cfg.chunk_size * cfg.action_dim)
            obs_emb = self.obs_dropout(self.obs_proj(z_fused)).unsqueeze(1)  # (B, 1, 256) broadcast across K
            act_emb = self.act_proj(flat_x)  # (B, K, 256)
            time_emb = self.time_encoder(tau)  # (B, K, 256)
            h = obs_emb + act_emb + time_emb
            for block in self.res_blocks:
                h = block(h)
            out = self.out_head(h)
            return out.reshape(b, k, cfg.chunk_size, cfg.action_dim)

        raise ValueError(f"Expected x_tau with 3 or 4 dims, got shape {tuple(x_tau.shape)}.")

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        task_ids: torch.Tensor,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end forward pass from raw observations to velocity chunk."""
        z_fused = self.extract_obs_features(obs, task_ids, return_aux=False)
        return self.predict_velocity(z_fused, x_tau, tau)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
