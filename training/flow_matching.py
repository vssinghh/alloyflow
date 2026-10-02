"""Optimal-Transport Conditional Flow Matching loss, Euler ODE sampler, and Temporal Ensembler."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from envs.base import JOINT_LIMITS_HIGH, JOINT_LIMITS_LOW
from training.config import AlloyTrainConfig
from training.model import TaskConditionedVisionFlowPolicy


class ConditionalFlowMatcher:
    """Optimal-Transport Conditional Flow Matching with K-sample stratified amortization."""

    def __init__(self, config: AlloyTrainConfig | None = None) -> None:
        self.config = config or AlloyTrainConfig()
        weights = torch.ones(self.config.action_dim, dtype=torch.float32)
        weights[-1] = float(self.config.gripper_weight)
        self._dim_weights = weights

    @staticmethod
    def sample_stratified_timesteps(
        batch_size: int,
        k: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample K stratified flow timesteps per sample in [0, 1): shape (B, K)."""
        u = torch.rand(batch_size, k, device=device, dtype=torch.float32)
        bins = torch.arange(k, device=device, dtype=torch.float32).unsqueeze(0)
        return (bins + u) / float(k)

    def compute_loss(
        self,
        policy: TaskConditionedVisionFlowPolicy,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Compute K-stratified Optimal-Transport Conditional Flow Matching loss.

        Runs the 3-camera CNN + Attention encoder once per minibatch to obtain z_fused (B, 192),
        then evaluates the lightweight ResMLP head across K stratified flow timesteps.
        Returns 0-D tensors on the active device to avoid per-batch CPU-GPU sync stalls.
        """
        obs: dict[str, torch.Tensor] = batch["obs"]
        task_id: torch.Tensor = batch["task_id"]
        actions: torch.Tensor = batch["actions"]

        b = actions.shape[0]
        k = self.config.num_flow_samples
        device = actions.device

        # 1. Encode cameras + proprio + task once per batch: (B, 192)
        z_fused = policy.extract_obs_features(obs, task_id, return_aux=False)

        # 2. Normalize target action chunk x_1: (B, 1, 16, 6)
        x_1 = policy.normalize_actions(actions).unsqueeze(1)

        # 3. Sample K stratified timesteps tau (B, K) and Gaussian noise x_0 (B, K, 16, 6)
        tau = self.sample_stratified_timesteps(b, k, device=device)
        x_0 = torch.randn(
            b,
            k,
            self.config.chunk_size,
            self.config.action_dim,
            device=device,
            dtype=torch.float32,
        )

        # 4. Straight-line optimal-transport interpolation and constant velocity target
        tau_4d = tau.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        x_tau = (1.0 - tau_4d) * x_0 + tau_4d * x_1
        u_target = x_1 - x_0  # (B, K, 16, 6)

        # 5. Predict velocity field v_theta(x_tau, tau | z_fused): (B, K, 16, 6)
        v_pred = policy.predict_velocity(z_fused, x_tau, tau)

        # 6. Dimension-weighted MSE (gripper_weight = 2.5 on dimension 5)
        sq_err = (v_pred - u_target).pow(2)
        dim_w = self._dim_weights.to(device=device).view(1, 1, 1, -1)
        weighted_loss = torch.sum(sq_err * dim_w, dim=-1) / torch.sum(dim_w)
        loss = weighted_loss.mean()

        with torch.no_grad():
            arm_mse = sq_err[..., :5].mean().detach()
            gripper_mse = sq_err[..., 5].mean().detach()
            v_norm = v_pred.norm(dim=-1).mean().detach()

        return {
            "loss": loss,
            "arm_mse": arm_mse,
            "gripper_mse": gripper_mse,
            "v_norm": v_norm,
        }

    @torch.no_grad()
    def sample_action_chunk(
        self,
        policy: TaskConditionedVisionFlowPolicy,
        obs: dict[str, torch.Tensor],
        task_id: torch.Tensor | int,
        ode_steps: int | None = None,
        clip_to_limits: bool = True,
        return_aux: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Integrate v_theta from tau = 0.0 to 1.0 using Euler ODE steps and unnormalize."""
        was_training = policy.training
        policy.eval()
        try:
            proprio = obs["proprio"]
            b = proprio.shape[0]
            device = proprio.device
            steps = int(ode_steps if ode_steps is not None else self.config.ode_steps)
            dt = 1.0 / float(max(steps, 1))

            if isinstance(task_id, int):
                tid_tensor = torch.full((b,), task_id, dtype=torch.long, device=device)
            else:
                tid_tensor = task_id.to(device=device, dtype=torch.long)

            if return_aux:
                z_fused, aux = policy.extract_obs_features(obs, tid_tensor, return_aux=True)
            else:
                z_fused = policy.extract_obs_features(obs, tid_tensor, return_aux=False)
                aux = {}

            x_tau = torch.randn(
                (b, self.config.chunk_size, self.config.action_dim),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )

            for step_idx in range(steps):
                t_val = float(step_idx) * dt
                tau = torch.full((b,), t_val, device=device, dtype=torch.float32)
                v = policy.predict_velocity(z_fused, x_tau, tau)
                x_tau = x_tau + dt * v

            actions = policy.unnormalize_actions(x_tau)
            if clip_to_limits:
                low = torch.as_tensor(JOINT_LIMITS_LOW, device=device, dtype=torch.float32)
                high = torch.as_tensor(JOINT_LIMITS_HIGH, device=device, dtype=torch.float32)
                actions = torch.max(torch.min(actions, high), low)

            if return_aux:
                return actions, aux
            return actions
        finally:
            if was_training:
                policy.train()


class TemporalEnsembler:
    """Exponential sliding-window action chunk blender for 20 Hz closed-loop control."""

    def __init__(
        self,
        chunk_size: int = 16,
        action_dim: int = 6,
        decay: float = 0.05,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.decay = float(decay)
        self._chunks: deque[np.ndarray] = deque(maxlen=self.chunk_size)

    def reset(self) -> None:
        """Clear buffered action chunk predictions at episode reset."""
        self._chunks.clear()

    def update(self, chunk: np.ndarray | torch.Tensor) -> np.ndarray:
        """Add a newly predicted (H, 6) chunk and return the blended 6D action for step t.

        Older chunks are indexed at their corresponding horizon step h = age and weighted
        by w_age = exp(-decay * age), where age = 0 is the newest prediction.
        """
        if isinstance(chunk, torch.Tensor):
            arr = chunk.detach().cpu().numpy()
        else:
            arr = np.asarray(chunk, dtype=np.float32)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"Expected chunk shape ({self.chunk_size}, {self.action_dim}), got {arr.shape}."
            )

        self._chunks.appendleft(arr.astype(np.float32, copy=True))

        preds: list[np.ndarray] = []
        weights: list[float] = []
        for age, past_chunk in enumerate(self._chunks):
            preds.append(past_chunk[age])
            weights.append(float(np.exp(-self.decay * float(age))))

        w_arr = np.asarray(weights, dtype=np.float32)[:, None]
        p_arr = np.stack(preds, axis=0)
        blended = np.sum(p_arr * w_arr, axis=0) / np.sum(w_arr)
        return blended.astype(np.float32)
