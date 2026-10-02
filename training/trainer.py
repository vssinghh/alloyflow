"""Unified 4-mode trainer and checkpoint manager for AlloyFlow policies."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from training.config import AlloyTrainConfig
from training.dataset import HDF5DemoDataset, MultiModeBatchLoader
from training.flow_matching import ConditionalFlowMatcher
from training.model import TaskConditionedVisionFlowPolicy


def save_policy_checkpoint(
    path: str | Path,
    policy: TaskConditionedVisionFlowPolicy,
    config: AlloyTrainConfig,
    epoch: int,
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Save policy weights, normalization buffers, config dict, and training metrics."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": policy.state_dict(),
        "norm_stats": policy.get_norm_stats(),
        "config": config.to_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, out_path)
    return out_path


def load_policy_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
) -> tuple[TaskConditionedVisionFlowPolicy, AlloyTrainConfig, dict[str, Any]]:
    """Load a trained TaskConditionedVisionFlowPolicy and its AlloyTrainConfig from disk."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    raw_ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    cfg = AlloyTrainConfig.from_dict(raw_ckpt["config"], strict=False)
    if device == "auto":
        dev = cfg.resolve_device()
    else:
        dev = torch.device(device)

    policy = TaskConditionedVisionFlowPolicy(cfg).to(dev)
    policy.load_state_dict(raw_ckpt["model_state_dict"])
    if "norm_stats" in raw_ckpt:
        policy.set_norm_stats(raw_ckpt["norm_stats"])
    policy.eval()
    return policy, cfg, raw_ckpt


class PolicyTrainer:
    """Orchestrates AlloyFlow policy training across sim_only, real_only, finetune, and cotrain."""

    def __init__(
        self,
        config: AlloyTrainConfig,
        sim_dataset: HDF5DemoDataset | None = None,
        real_dataset: HDF5DemoDataset | None = None,
    ) -> None:
        self.config = config
        self.device = config.resolve_device()
        self._set_seed(config.seed)

        self.loader = MultiModeBatchLoader(
            config=config,
            sim_dataset=sim_dataset,
            real_dataset=real_dataset,
        )
        self.policy = TaskConditionedVisionFlowPolicy(config).to(self.device)
        self.flow_matcher = ConditionalFlowMatcher(config)

        if config.train_mode == "finetune":
            if not config.pretrained_checkpoint:
                raise ValueError(
                    "finetune mode requires config.pretrained_checkpoint to be specified."
                )
            ckpt_path = Path(config.pretrained_checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Pretrained checkpoint for finetune not found: {ckpt_path}"
                )
            raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.policy.load_state_dict(raw_ckpt["model_state_dict"])
            # Keep normalization stats locked from the pretrained checkpoint
            if "norm_stats" in raw_ckpt:
                self.policy.set_norm_stats(raw_ckpt["norm_stats"])
            elif not bool(self.policy.stats_initialized.item()):
                self.policy.set_norm_stats(self.loader.compute_norm_stats())
        else:
            self.policy.set_norm_stats(self.loader.compute_norm_stats())

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.effective_lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(config.epochs, 1),
            eta_min=config.effective_lr * 0.05,
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(self, verbose: bool = True) -> dict[str, Any]:
        """Run full training schedule and persist best + latest checkpoints."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        config_json_path = save_dir / "train_config.json"
        config_json_path.write_text(json.dumps(self.config.to_dict(), indent=2) + "\n")

        if verbose:
            print(
                f"[AlloyTrainer] Mode={self.config.train_mode} | Device={self.device} | "
                f"Params={self.policy.count_parameters():,} | "
                f"Batches/Epoch={self.loader.num_batches_per_epoch} | "
                f"K={self.config.num_flow_samples} | LR={self.config.effective_lr:.1e}"
            )

        best_loss = float("inf")
        history: list[dict[str, float]] = []
        t_start = time.perf_counter()

        for epoch in range(1, self.config.epochs + 1):
            ep_t0 = time.perf_counter()
            self.policy.train()

            # Accumulate detached 0-D tensors on device to prevent per-batch CPU-GPU syncs
            loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            arm_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            grip_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            vnorm_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            num_batches = 0

            for batch in self.loader.iter_epoch(epoch=epoch):
                self.optimizer.zero_grad(set_to_none=True)
                out = self.flow_matcher.compute_loss(self.policy, batch)
                loss = out["loss"]
                loss.backward()
                if self.config.max_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.max_grad_norm
                    )
                self.optimizer.step()

                loss_sum = loss_sum + loss.detach()
                arm_sum = arm_sum + out["arm_mse"]
                grip_sum = grip_sum + out["gripper_mse"]
                vnorm_sum = vnorm_sum + out["v_norm"]
                num_batches += 1

            self.scheduler.step()
            denom = float(max(num_batches, 1))
            ep_metrics = {
                "epoch": float(epoch),
                "loss": float((loss_sum / denom).item()),
                "arm_mse": float((arm_sum / denom).item()),
                "gripper_mse": float((grip_sum / denom).item()),
                "v_norm": float((vnorm_sum / denom).item()),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "epoch_time_s": float(time.perf_counter() - ep_t0),
            }
            history.append(ep_metrics)

            save_policy_checkpoint(
                path=save_dir / "latest_policy.pt",
                policy=self.policy,
                config=self.config,
                epoch=epoch,
                metrics=ep_metrics,
                optimizer=self.optimizer,
            )
            if ep_metrics["loss"] < best_loss:
                best_loss = ep_metrics["loss"]
                save_policy_checkpoint(
                    path=save_dir / "best_policy.pt",
                    policy=self.policy,
                    config=self.config,
                    epoch=epoch,
                    metrics=ep_metrics,
                )

            if verbose:
                print(
                    f"[Epoch {epoch:02d}/{self.config.epochs:02d}] "
                    f"loss={ep_metrics['loss']:.5f} "
                    f"(arm={ep_metrics['arm_mse']:.5f}, grip={ep_metrics['gripper_mse']:.5f}) | "
                    f"lr={ep_metrics['lr']:.2e} | time={ep_metrics['epoch_time_s']:.2f}s"
                )

        total_time_s = float(time.perf_counter() - t_start)
        summary = {
            "train_mode": self.config.train_mode,
            "best_loss": float(best_loss),
            "final_loss": float(history[-1]["loss"]) if history else float("nan"),
            "total_time_s": total_time_s,
            "save_dir": str(save_dir),
            "best_checkpoint": str(save_dir / "best_policy.pt"),
            "latest_checkpoint": str(save_dir / "latest_policy.pt"),
            "history": history,
        }
        (save_dir / "training_history.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
