"""Unified 4-mode trainer and checkpoint manager for AlloyFlow policies."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any
import urllib.parse
import urllib.request

import numpy as np
import torch
from torch import nn

from envs.base import CAMERA_NAMES
from training.config import AlloyTrainConfig
from training.dataset import HDF5DemoDataset, MultiModeBatchLoader
from training.flow_matching import ConditionalFlowMatcher
from training.model import TaskConditionedVisionFlowPolicy
from training.pretrain_vision import (
    evaluate_policy_localization,
    load_demo_frame0_validation,
    pretrain_vision_encoders,
)


def _capture_rng_states() -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch CPU, and CUDA RNG states for deterministic resume."""
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            states["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return states


def _restore_rng_states(states: dict[str, Any] | None) -> None:
    """Restore Python, NumPy, PyTorch CPU, and CUDA RNG states from checkpoint."""
    if not states:
        return
    try:
        if "python" in states:
            random.setstate(states["python"])
        if "numpy" in states:
            np.random.set_state(states["numpy"])
        if "torch" in states:
            torch_st = states["torch"]
            if isinstance(torch_st, torch.Tensor):
                torch.set_rng_state(torch_st.cpu().to(torch.uint8))
        if "cuda" in states and torch.cuda.is_available() and states["cuda"] is not None:
            cuda_st = [t.cpu().to(torch.uint8) for t in states["cuda"] if isinstance(t, torch.Tensor)]
            if len(cuda_st) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_st)
    except Exception:
        pass


def save_policy_checkpoint(
    path: str | Path,
    policy: TaskConditionedVisionFlowPolicy,
    config: AlloyTrainConfig,
    epoch: int,
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    best_val_mse: float | None = None,
    best_loss: float | None = None,
    history: list[dict[str, float]] | None = None,
    include_rng: bool = True,
) -> Path:
    """Save policy weights, normalization buffers, config dict, and resumable training state."""
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
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if best_val_mse is not None:
        payload["best_val_mse"] = float(best_val_mse)
    if best_loss is not None:
        payload["best_loss"] = float(best_loss)
    if history is not None:
        payload["history"] = list(history)
    if include_rng:
        payload["rng_state"] = _capture_rng_states()

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)
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
    policy.load_state_dict(raw_ckpt["model_state_dict"], strict=False)
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
            self.policy.load_state_dict(raw_ckpt["model_state_dict"], strict=False)
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

    def _load_loc_replay_tensors(self) -> dict[str, torch.Tensor] | None:
        """Load cached 3,000-layout localization dataset into CPU tensors for Stage 2 replay."""
        npz_path = Path(self.config.loc_data_path)
        if not npz_path.exists():
            return None
        raw = np.load(npz_path)
        out: dict[str, torch.Tensor] = {
            "task_id": torch.as_tensor(raw["task_id"], dtype=torch.long),
            "obj_xy": torch.as_tensor(raw["targets_12d"], dtype=torch.float32),
            "proprio": torch.as_tensor(raw["proprio"], dtype=torch.float32),
        }
        for cam in CAMERA_NAMES:
            out[f"rgb_{cam}"] = (
                torch.as_tensor(raw[f"rgb_{cam}"], dtype=torch.uint8)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        return out

    def _sample_loc_replay_batch(
        self,
        loc_tensors: dict[str, torch.Tensor],
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Sample a random minibatch from the 3,000-layout localization dataset."""
        n = len(loc_tensors["task_id"])
        idx = torch.randint(0, n, (batch_size,))
        obs: dict[str, torch.Tensor] = {
            "proprio": loc_tensors["proprio"][idx].to(self.device, non_blocking=True),
        }
        for cam in CAMERA_NAMES:
            obs[f"rgb_{cam}"] = (
                loc_tensors[f"rgb_{cam}"][idx]
                .to(self.device, non_blocking=True)
                .float()
                .div_(255.0)
            )
        return {
            "obs": obs,
            "task_id": loc_tensors["task_id"][idx].to(self.device, non_blocking=True),
            "obj_xy": loc_tensors["obj_xy"][idx].to(self.device, non_blocking=True),
        }

    def _get_drive_access_token(self) -> str | None:
        """Obtain a fresh Google Drive OAuth2 access token from env or auth JSON."""
        auth_json = os.environ.get("ALLOYFLOW_GDRIVE_AUTH_JSON", "")
        if auth_json and Path(auth_json).exists():
            try:
                raw = json.loads(Path(auth_json).read_text())
                tok_obj = raw.get("token", raw)
                refresh_tok = tok_obj.get("refresh_token")
                client_id = raw.get("client_id")
                client_secret = raw.get("client_secret")
                now = time.time()
                cached_tok = getattr(self, "_cached_drive_token", None)
                cached_exp = getattr(self, "_cached_drive_token_exp", 0.0)
                if cached_tok and now < cached_exp:
                    return cached_tok
                if refresh_tok and client_id and client_secret:
                    data = urllib.parse.urlencode({
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": refresh_tok,
                        "grant_type": "refresh_token",
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        "https://oauth2.googleapis.com/token",
                        data=data,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        new_tok = str(res["access_token"])
                        self._cached_drive_token = new_tok
                        self._cached_drive_token_exp = now + float(res.get("expires_in", 3600)) - 300.0
                        return new_tok
                if "access_token" in tok_obj:
                    return str(tok_obj["access_token"])
            except Exception:
                pass
        env_tok = os.environ.get("ALLOYFLOW_DRIVE_TOKEN", "")
        return env_tok if env_tok else None

    def _ensure_drive_run_folder(self, token: str, parent_id: str, run_folder_name: str) -> str | None:
        """Find or create a dedicated checkpoint subfolder in Google Drive."""
        cached_id = getattr(self, "_drive_run_folder_id", None)
        if cached_id:
            return cached_id
        try:
            q = (
                f"'{parent_id}' in parents and name = '{run_folder_name}' "
                f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            )
            url = f"https://www.googleapis.com/drive/v3/files?{urllib.parse.urlencode({'q': q, 'fields': 'files(id,name)'})}"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                files = json.loads(resp.read().decode("utf-8")).get("files", [])
                if files:
                    self._drive_run_folder_id = str(files[0]["id"])
                    return self._drive_run_folder_id
            meta = json.dumps({
                "name": run_folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            }).encode("utf-8")
            create_req = urllib.request.Request(
                "https://www.googleapis.com/drive/v3/files?fields=id",
                data=meta,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(create_req, timeout=15) as resp:
                fid = str(json.loads(resp.read().decode("utf-8"))["id"])
                self._drive_run_folder_id = fid
                return fid
        except Exception as exc:
            print(f"[DriveSync] Warning: could not ensure Drive folder '{run_folder_name}': {exc}", flush=True)
            return None

    def _upload_file_to_drive(self, token: str, folder_id: str, local_path: Path) -> None:
        """Create or update a single file inside the Google Drive checkpoint folder."""
        if not local_path.exists():
            return
        file_id_map: dict[str, str] = getattr(self, "_drive_file_ids", {})
        if not hasattr(self, "_drive_file_ids"):
            self._drive_file_ids = file_id_map
        fname = local_path.name
        fid = file_id_map.get(fname)
        if not fid:
            q = f"'{folder_id}' in parents and name = '{fname}' and trashed = false"
            url = f"https://www.googleapis.com/drive/v3/files?{urllib.parse.urlencode({'q': q, 'fields': 'files(id,name)'})}"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                files = json.loads(resp.read().decode("utf-8")).get("files", [])
                if files:
                    fid = str(files[0]["id"])
                    file_id_map[fname] = fid

        file_bytes = local_path.read_bytes()
        if fid:
            up_url = f"https://www.googleapis.com/upload/drive/v3/files/{fid}?uploadType=media"
            up_req = urllib.request.Request(
                up_url,
                data=file_bytes,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/octet-stream",
                },
                method="PATCH",
            )
            with urllib.request.urlopen(up_req, timeout=60) as _:
                pass
        else:
            boundary = f"===============af_boundary_{int(time.time() * 1000)}=="
            meta_bytes = json.dumps({"name": fname, "parents": [folder_id]}).encode("utf-8")
            body = (
                f"--{boundary}\r\n"
                "Content-Type: application/json; charset=UTF-8\r\n\r\n"
            ).encode("utf-8") + meta_bytes + (
                f"\r\n--{boundary}\r\n"
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
            up_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id"
            up_req = urllib.request.Request(
                up_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": f"multipart/related; boundary={boundary}",
                },
                method="POST",
            )
            with urllib.request.urlopen(up_req, timeout=60) as resp:
                new_id = str(json.loads(resp.read().decode("utf-8"))["id"])
                file_id_map[fname] = new_id

    def _sync_epoch_to_drive(
        self,
        save_dir: Path,
        epoch: int,
        is_new_best: bool,
        snapshot_path: Path | None = None,
    ) -> None:
        """Sync resumable checkpoints and history to Google Drive after each epoch."""
        parent_folder_id = os.environ.get("ALLOYFLOW_DRIVE_FOLDER_ID", "")
        if not parent_folder_id:
            return
        token = self._get_drive_access_token()
        if not token:
            return
        run_name = save_dir.name or self.config.train_mode
        run_folder_id = self._ensure_drive_run_folder(token, parent_folder_id, f"ckpt_{run_name}")
        if not run_folder_id:
            return
        try:
            self._upload_file_to_drive(token, run_folder_id, save_dir / "latest.pt")
            self._upload_file_to_drive(token, run_folder_id, save_dir / "training_history.json")
            if is_new_best:
                self._upload_file_to_drive(token, run_folder_id, save_dir / "best_policy.pt")
            if snapshot_path is not None and snapshot_path.exists():
                self._upload_file_to_drive(token, run_folder_id, snapshot_path)
        except Exception as exc:
            print(f"[DriveSync] Warning: epoch {epoch} Drive upload failed: {exc}", flush=True)

    def train(self, verbose: bool = True) -> dict[str, Any]:
        """Run full training schedule and persist best + latest + snapshot checkpoints."""
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

        resume_path: Path | None = None
        if self.config.resume_checkpoint:
            if self.config.resume_checkpoint == "auto":
                cand = save_dir / "latest.pt"
                if not cand.exists():
                    cand = save_dir / "latest_policy.pt"
                if cand.exists():
                    resume_path = cand
            else:
                cand = Path(self.config.resume_checkpoint)
                if cand.exists():
                    resume_path = cand
                else:
                    raise FileNotFoundError(f"Resume checkpoint not found: {cand}")

        pretrain_loc_summary: dict[str, Any] | None = None
        loc_replay_tensors: dict[str, torch.Tensor] | None = None
        start_epoch = 0
        best_val_mse = float("inf")
        best_loss = float("inf")
        history: list[dict[str, float]] = []

        if resume_path is not None:
            raw_res = torch.load(resume_path, map_location="cpu", weights_only=False)
            self.policy.load_state_dict(raw_res["model_state_dict"], strict=False)
            if "norm_stats" in raw_res:
                self.policy.set_norm_stats(raw_res["norm_stats"])
            self.policy.to(self.device)
            if (
                self.config.train_mode == "sim_only"
                and self.config.pretrain_loc_steps > 0
            ):
                loc_replay_tensors = self._load_loc_replay_tensors()
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=self.config.effective_lr,
                weight_decay=self.config.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(self.config.epochs, 1),
                eta_min=self.config.effective_lr * 0.05,
            )
            if "optimizer_state_dict" in raw_res:
                self.optimizer.load_state_dict(raw_res["optimizer_state_dict"])
            start_epoch = int(raw_res.get("epoch", 0))
            if "scheduler_state_dict" in raw_res:
                self.scheduler.load_state_dict(raw_res["scheduler_state_dict"])
            else:
                for _ in range(start_epoch):
                    self.scheduler.step()
            best_val_mse = float(raw_res.get("best_val_mse", raw_res.get("metrics", {}).get("val_chunk_mse", float("inf"))))
            best_loss = float(raw_res.get("best_loss", raw_res.get("metrics", {}).get("loss", float("inf"))))
            if isinstance(raw_res.get("history"), list):
                history = list(raw_res["history"])
            elif (save_dir / "training_history.json").exists():
                try:
                    prev_hist = json.loads((save_dir / "training_history.json").read_text())
                    if isinstance(prev_hist.get("history"), list):
                        history = [h for h in prev_hist["history"] if int(h.get("epoch", 0)) <= start_epoch]
                except Exception:
                    pass
            _restore_rng_states(raw_res.get("rng_state"))
            if verbose:
                cur_lr = float(self.optimizer.param_groups[0]["lr"])
                print(
                    f"[AlloyTrainer] Resumed from {resume_path} at epoch {start_epoch}/{self.config.epochs} | "
                    f"best_val_ode_mse={best_val_mse:.5f} | next_lr={cur_lr:.2e}",
                    flush=True,
                )
        elif (
            self.config.train_mode == "sim_only"
            and self.config.pretrain_loc_steps > 0
        ):
            if self.config.pretrained_checkpoint and Path(self.config.pretrained_checkpoint).exists():
                vis_ckpt_path = Path(self.config.pretrained_checkpoint)
                raw_vis = torch.load(vis_ckpt_path, map_location="cpu", weights_only=False)
                vis_sd = raw_vis.get("model_state_dict", raw_vis)
                vis_prefixes = ("task_embedding.", "camera_encoders.", "aux_cam_pos_heads.", "obj_xy_mean", "obj_xy_std")
                filtered_vis_sd = {k: v for k, v in vis_sd.items() if k.startswith(vis_prefixes)}
                self.policy.load_state_dict(filtered_vis_sd, strict=False)
                self.policy.to(self.device)
                val_data = load_demo_frame0_validation(self.config.sim_data_path)
                pretrain_loc_summary = evaluate_policy_localization(self.policy, val_data, self.device)
                ov = pretrain_loc_summary["overhead_cam"]
                tp = pretrain_loc_summary["third_person_cam"]
                wr = pretrain_loc_summary["wrist_cam"]
                print(
                    f"[PretrainVis] Reused Stage 1 weights from {vis_ckpt_path} | "
                    f"ov_src={ov['src_err_cm']:.2f}cm | tp_src={tp['src_err_cm']:.2f}cm | wr_src={wr['src_err_cm']:.2f}cm",
                    flush=True,
                )
                torch.save(
                    {"model_state_dict": self.policy.state_dict(), "val_localization": pretrain_loc_summary},
                    save_dir / "pretrained_vision.pt",
                )
                self._set_seed(self.config.seed)
            else:
                pretrain_loc_summary = pretrain_vision_encoders(
                    policy=self.policy,
                    loc_npz_path=self.config.loc_data_path,
                    demo_h5_path=self.config.sim_data_path,
                    steps=self.config.pretrain_loc_steps,
                    batch_size=64,
                    lr=1e-3,
                    device=self.device,
                    save_path=save_dir / "pretrained_vision.pt",
                )
            loc_replay_tensors = self._load_loc_replay_tensors()
            # Re-initialize Stage 2 AdamW optimizer after Stage 1 pretraining
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=self.config.effective_lr,
                weight_decay=self.config.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(self.config.epochs, 1),
                eta_min=self.config.effective_lr * 0.05,
            )

        t_start = time.perf_counter()
        val_batch = self.loader.get_validation_batch(
            num_samples=self.config.val_samples_per_epoch
        )

        for epoch in range(start_epoch + 1, self.config.epochs + 1):
            ep_t0 = time.perf_counter()
            self.policy.train()

            # Accumulate detached 0-D tensors on device to prevent per-batch CPU-GPU syncs
            loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            cfm_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            aux_pos_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            arm_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            grip_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            vnorm_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            num_batches = 0

            for batch in self.loader.iter_epoch(epoch=epoch):
                if loc_replay_tensors is not None and self.config.aux_pos_loss_weight > 0.0:
                    batch["loc_replay"] = self._sample_loc_replay_batch(
                        loc_replay_tensors, batch_size=32
                    )
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
                cfm_sum = cfm_sum + out.get("cfm_loss", loss.detach())
                aux_pos_sum = aux_pos_sum + out.get(
                    "aux_pos_loss", torch.zeros((), device=self.device)
                )
                arm_sum = arm_sum + out["arm_mse"]
                grip_sum = grip_sum + out["gripper_mse"]
                vnorm_sum = vnorm_sum + out["v_norm"]
                num_batches += 1

            self.scheduler.step()
            denom = float(max(num_batches, 1))
            val_chunk_mse = self.flow_matcher.compute_eval_chunk_mse(
                self.policy, val_batch
            )
            ep_metrics = {
                "epoch": float(epoch),
                "loss": float((loss_sum / denom).item()),
                "cfm_loss": float((cfm_sum / denom).item()),
                "aux_pos_loss": float((aux_pos_sum / denom).item()),
                "arm_mse": float((arm_sum / denom).item()),
                "gripper_mse": float((grip_sum / denom).item()),
                "v_norm": float((vnorm_sum / denom).item()),
                "val_chunk_mse": float(val_chunk_mse),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "epoch_time_s": float(time.perf_counter() - ep_t0),
            }
            history.append(ep_metrics)
            best_loss = min(best_loss, ep_metrics["loss"])
            is_new_best = val_chunk_mse < best_val_mse
            if is_new_best:
                best_val_mse = val_chunk_mse

            save_policy_checkpoint(
                path=save_dir / "latest.pt",
                policy=self.policy,
                config=self.config,
                epoch=epoch,
                metrics=ep_metrics,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                best_val_mse=best_val_mse,
                best_loss=best_loss,
                history=history,
                include_rng=True,
            )
            shutil.copyfile(save_dir / "latest.pt", save_dir / "latest_policy.pt")

            if is_new_best:
                save_policy_checkpoint(
                    path=save_dir / "best_policy.pt",
                    policy=self.policy,
                    config=self.config,
                    epoch=epoch,
                    metrics=ep_metrics,
                    best_val_mse=best_val_mse,
                    best_loss=best_loss,
                    include_rng=False,
                )

            snapshot_path: Path | None = None
            if epoch in (10, 20, 30, 40) or (epoch % 10 == 0):
                snapshot_path = save_dir / f"epoch_{epoch:03d}.pt"
                shutil.copyfile(save_dir / "latest.pt", snapshot_path)

            running_summary: dict[str, Any] = {
                "train_mode": self.config.train_mode,
                "best_loss": float(best_loss),
                "best_val_chunk_mse": float(best_val_mse),
                "final_loss": float(history[-1]["loss"]) if history else float("nan"),
                "total_time_s": float(time.perf_counter() - t_start),
                "save_dir": str(save_dir),
                "best_checkpoint": str(save_dir / "best_policy.pt"),
                "latest_checkpoint": str(save_dir / "latest.pt"),
                "pretrain_localization": pretrain_loc_summary,
                "history": history,
            }
            (save_dir / "training_history.json").write_text(json.dumps(running_summary, indent=2) + "\n")

            if verbose:
                print(
                    f"[Epoch {epoch:02d}/{self.config.epochs:02d}] "
                    f"loss={ep_metrics['loss']:.5f} "
                    f"(cfm={ep_metrics['cfm_loss']:.5f}, pos={ep_metrics['aux_pos_loss']:.5f}, "
                    f"arm={ep_metrics['arm_mse']:.5f}, grip={ep_metrics['gripper_mse']:.5f}) | "
                    f"val_ode_mse={val_chunk_mse:.5f} | "
                    f"lr={ep_metrics['lr']:.2e} | time={ep_metrics['epoch_time_s']:.2f}s",
                    flush=True,
                )

            self._sync_epoch_to_drive(
                save_dir=save_dir,
                epoch=epoch,
                is_new_best=is_new_best,
                snapshot_path=snapshot_path,
            )

        final_loc: dict[str, Any] | None = None
        if (
            self.config.train_mode == "sim_only"
            and self.config.pretrain_loc_steps > 0
            and Path(self.config.sim_data_path).exists()
        ):
            val_data = load_demo_frame0_validation(self.config.sim_data_path)
            final_loc = evaluate_policy_localization(self.policy, val_data, self.device)
            if verbose:
                ov = final_loc["overhead_cam"]
                tp = final_loc["third_person_cam"]
                wr = final_loc["wrist_cam"]
                print(
                    f"[PostTrainLoc] ov_src={ov['src_err_cm']:.2f}cm (tgt={ov['tgt_err_cm']:.2f}cm, all4={ov['all4_err_cm']:.2f}cm, R2={ov['r2_src_xy'][0]:+.2f},{ov['r2_src_xy'][1]:+.2f}) | "
                    f"tp_src={tp['src_err_cm']:.2f}cm | wr_src={wr['src_err_cm']:.2f}cm",
                    flush=True,
                )

        total_time_s = float(time.perf_counter() - t_start)
        summary: dict[str, Any] = {
            "train_mode": self.config.train_mode,
            "best_loss": float(best_loss),
            "best_val_chunk_mse": float(best_val_mse),
            "final_loss": float(history[-1]["loss"]) if history else float("nan"),
            "total_time_s": total_time_s,
            "save_dir": str(save_dir),
            "best_checkpoint": str(save_dir / "best_policy.pt"),
            "latest_checkpoint": str(save_dir / "latest.pt"),
            "pretrain_localization": pretrain_loc_summary,
            "final_localization": final_loc,
            "history": history,
        }
        (save_dir / "training_history.json").write_text(json.dumps(summary, indent=2) + "\n")
        self._sync_epoch_to_drive(save_dir=save_dir, epoch=self.config.epochs, is_new_best=False)
        return summary
