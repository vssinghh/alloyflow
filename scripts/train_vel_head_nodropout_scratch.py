"""Train the 778K-param Flow Velocity Network from scratch with dropout=0.0 for 40 epochs.

Source checkpoint (Stage A frozen observation encoders + norm_stats):
  checkpoints/exp08_dart_full_900/best_policy.pt

Settings:
  - Stage A (task_embedding, proprio_mlp, camera_encoders) copied from Exp08 and frozen in .eval() mode
  - Stage B (predict_velocity: 778,592 params) randomly initialized (seed=42) with dropout=0.0
  - 40 epochs, batch_size=128, lr=5e-4 -> 2.5e-5 (cosine), weight_decay=1e-4, k_flow=4

Output path:
  checkpoints/exp08_nodropout_scratch/best_policy.pt

120-episode benchmark result (k=0, policy.eval()):
  112 / 120 (93.3% total: 59/60 train, 53/60 test; Task 0: 37/40, Task 1: 37/40, Task 2: 38/40; BLOCKED 36/120)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import time

import torch

from training.flow_matching import ConditionalFlowMatcher
from training.model import TaskConditionedVisionFlowPolicy
from training.trainer import load_policy_checkpoint, save_policy_checkpoint


CACHE_PATH = Path("checkpoints/exp08_z_fused_cache.pt")
CKPT_IN = Path("checkpoints/exp08_dart_full_900/best_policy.pt")
CKPT_OUT = Path("checkpoints/exp08_nodropout_scratch/best_policy.pt")


@torch.no_grad()
def eval_val_ode_mse(
    policy: TaskConditionedVisionFlowPolicy,
    matcher: ConditionalFlowMatcher,
    z_val: torch.Tensor,
    x1_val_norm: torch.Tensor,
    device: torch.device,
) -> float:
    policy.eval()
    b = z_val.shape[0]
    steps = 5
    dt = 1.0 / float(steps)

    gen = torch.Generator(device=device)
    gen.manual_seed(12345)
    x_tau = torch.randn(
        (b, matcher.config.chunk_size, matcher.config.action_dim),
        device=device,
        dtype=torch.float32,
        generator=gen,
    )
    for s in range(steps):
        tau = torch.full((b,), float(s) * dt, device=device, dtype=torch.float32)
        v = policy.predict_velocity(z_val, x_tau, tau)
        x_tau = x_tau + dt * v

    pred_unnorm = policy.unnormalize_actions(x_tau)
    gt_unnorm = policy.unnormalize_actions(x1_val_norm)
    sq_err = (pred_unnorm - gt_unnorm).pow(2)
    dim_w = matcher._dim_weights.to(device=device).view(1, 1, -1)
    weighted_mse = (torch.sum(sq_err * dim_w, dim=-1) / torch.sum(dim_w)).mean()
    return float(weighted_mse.item())


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading cached z_fused from {CACHE_PATH} on {device} ...")
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)

    ref_policy, ref_cfg, _ = load_policy_checkpoint(CKPT_IN, device=device)
    ref_policy.eval()

    # Build fresh policy with dropout=0.0 from step 0
    torch.manual_seed(42)
    cfg = dataclasses.replace(ref_cfg, dropout=0.0)
    policy = TaskConditionedVisionFlowPolicy(cfg).to(device)

    # Copy only Stage A (observation encoder + norm_stats) from Exp08; keep Stage B randomly initialized
    policy.task_embedding.load_state_dict(ref_policy.task_embedding.state_dict())
    policy.proprio_mlp.load_state_dict(ref_policy.proprio_mlp.state_dict())
    policy.camera_encoders.load_state_dict(ref_policy.camera_encoders.state_dict())
    policy.aux_cam_pos_heads.load_state_dict(ref_policy.aux_cam_pos_heads.state_dict())
    policy.set_norm_stats(ref_policy.get_norm_stats())
    policy.eval()

    matcher = ConditionalFlowMatcher(cfg)
    dim_w = matcher._dim_weights.to(device=device).view(1, 1, 1, -1)

    train_mask = ~cache["is_val"]
    val_mask = cache["is_val"]

    z_train = cache["z_fused"][train_mask].to(device=device, dtype=torch.float32)
    x1_train = cache["x1_norm"][train_mask].to(device=device, dtype=torch.float32)
    z_val = cache["z_fused"][val_mask][:1024].to(device=device, dtype=torch.float32)
    x1_val = cache["x1_norm"][val_mask][:1024].to(device=device, dtype=torch.float32)

    vel_params = (
        list(policy.obs_proj.parameters())
        + list(policy.act_proj.parameters())
        + list(policy.time_encoder.parameters())
        + list(policy.res_blocks.parameters())
        + list(policy.out_head.parameters())
    )
    num_vel_params = sum(p.numel() for p in vel_params)
    print(f"Training {num_vel_params:,d} velocity-head parameters from scratch with dropout=0.0 ...")

    epochs = 40
    batch_size = 128
    lr = 5e-4
    n_train = z_train.shape[0]
    steps_per_epoch = n_train // batch_size

    optimizer = torch.optim.AdamW(vel_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=lr * 0.05
    )

    best_val_mse = float("inf")
    k_flow = 4
    t0 = time.time()

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        perm = torch.randperm(n_train, device=device)
        ep_loss = 0.0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            z_b = z_train[idx]
            x1_b = x1_train[idx]
            b = z_b.shape[0]

            tau = matcher.sample_stratified_timesteps(b, k_flow, device)
            x0 = torch.randn(
                (b, k_flow, cfg.chunk_size, cfg.action_dim), device=device, dtype=torch.float32
            )
            x1_exp = x1_b.unsqueeze(1).expand(b, k_flow, cfg.chunk_size, cfg.action_dim)
            tau_exp = tau.unsqueeze(-1).unsqueeze(-1)
            x_tau = (1.0 - tau_exp) * x0 + tau_exp * x1_exp
            u_target = x1_exp - x0

            v_pred = policy.predict_velocity(z_b, x_tau, tau)
            sq_err = (v_pred - u_target).pow(2)
            loss = (torch.sum(sq_err * dim_w, dim=-1) / torch.sum(dim_w)).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vel_params, 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += float(loss.item())

        val_mse = eval_val_ode_mse(policy, matcher, z_val, x1_val, device)
        is_best = val_mse < best_val_mse
        if is_best:
            best_val_mse = val_mse
            save_policy_checkpoint(
                CKPT_OUT,
                policy=policy,
                config=cfg,
                epoch=ep,
                metrics={"val_ode_mse": val_mse, "train_loss": ep_loss / steps_per_epoch},
                best_val_mse=best_val_mse,
                include_rng=False,
            )
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            mark = " *BEST*" if is_best else ""
            print(
                f"  [Epoch {ep:02d}/{epochs}] train_loss={ep_loss / steps_per_epoch:.5f} | "
                f"val_ode_mse={val_mse:.6f}{mark} ({time.time() - t_ep:.1f}s)"
            )

    print(
        f"Done in {time.time() - t0:.1f}s! Best val_ode_mse={best_val_mse:.6f} saved to {CKPT_OUT}"
    )


if __name__ == "__main__":
    main()
