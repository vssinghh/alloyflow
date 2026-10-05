"""Unified-memory HDF5 dataset loader and 4-mode Sim/Real batch mixer."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from envs.base import OBJECT_NAMES, TASK_SPECS
from envs.sim_env import SimEnv
from training.config import DEFAULT_TRAIN_CONFIG, AlloyTrainConfig


def _compute_episode_layout_12d(
    spawns_xy: dict[str, tuple[float, float]],
    task_id: int,
) -> np.ndarray:
    """Build 12D tabletop XY target: [src_xy(2), tgt_xy(2), pen_xy(2), cup_xy(2), bowl_xy(2), cube_xy(2)]."""
    spec = TASK_SPECS[int(task_id)]
    src_xy = np.asarray(spawns_xy[spec.source_object][:2], dtype=np.float32)
    tgt_xy = np.asarray(spawns_xy[spec.target_object][:2], dtype=np.float32)
    all_xy = np.concatenate(
        [np.asarray(spawns_xy[name][:2], dtype=np.float32) for name in OBJECT_NAMES],
        axis=0,
    )
    return np.concatenate([src_xy, tgt_xy, all_xy], axis=0).astype(np.float32)


def build_action_chunks(
    actions: np.ndarray,
    chunk_size: int = DEFAULT_TRAIN_CONFIG.chunk_size,
) -> np.ndarray:
    """Slice (T, 6) episode actions into (T, chunk_size, 6) chunks with terminal hold padding.

    For steps near the end of an episode (t + h >= T), repeats the final action actions[T - 1]
    so the policy learns to hold its retracted terminal pose cleanly.
    """
    arr = np.asarray(actions, dtype=np.float32)
    t_len, act_dim = arr.shape
    chunks = np.empty((t_len, chunk_size, act_dim), dtype=np.float32)
    for t in range(t_len):
        end = t + chunk_size
        if end <= t_len:
            chunks[t] = arr[t:end]
        else:
            valid = t_len - t
            chunks[t, :valid] = arr[t:t_len]
            chunks[t, valid:] = arr[-1]
    return chunks


def filter_episode_frames(
    proprio: np.ndarray,
    actions: np.ndarray,
    chunk_size: int = DEFAULT_TRAIN_CONFIG.chunk_size,
    trim_stationary: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter out only truly motionless frames (arm still and gripper state/command unchanged).

    Preserves all grasp-closure, object-release, and post-placement retraction frames.
    """
    prop = np.asarray(proprio, dtype=np.float32).copy()
    act = np.asarray(actions, dtype=np.float32).copy()
    t_len = act.shape[0]

    if not trim_stationary or t_len <= max(2 * chunk_size, 32):
        keep_idx = np.arange(t_len, dtype=np.int64)
        return prop, act, build_action_chunks(act, chunk_size=chunk_size), keep_idx

    dq = np.linalg.norm(np.diff(prop[:, :5], axis=0, append=prop[-1:, :5]), axis=-1)
    dg = np.abs(np.diff(prop[:, 5], axis=0, append=prop[-1:, 5]))
    da = np.linalg.norm(np.diff(act, axis=0, append=act[-1:]), axis=-1)

    # Only trim truly motionless interior frames where arm, gripper state, and command are unchanged
    motionless = (
        (np.arange(t_len) > 0)
        & (np.arange(t_len) < t_len - 1)
        & (dq < 1e-3)
        & (dg < 1e-3)
        & (da < 1e-3)
    )
    active_idx = np.where(~motionless)[0].astype(np.int64)
    if len(active_idx) <= chunk_size:
        keep_idx = np.arange(t_len, dtype=np.int64)
        return prop, act, build_action_chunks(act, chunk_size=chunk_size), keep_idx

    prop_act = prop[active_idx]
    act_act = act[active_idx]
    chunks_act = build_action_chunks(act_act, chunk_size=chunk_size)
    return prop_act, act_act, chunks_act, active_idx


def build_proprio_history_step(
    proprio_seq: np.ndarray | list[np.ndarray],
    step_idx: int,
    lags: tuple[int, ...] = DEFAULT_TRAIN_CONFIG.proprio_history_lags,
) -> np.ndarray:
    """Build (num_lags, 6) causal proprioception history for a single timestep (shared train/eval)."""
    cur_t = int(step_idx)
    frames = [
        np.asarray(proprio_seq[max(0, cur_t - int(lag))], dtype=np.float32)
        for lag in lags
    ]
    return np.stack(frames, axis=0).astype(np.float32)


def build_proprio_history(
    proprio_raw: np.ndarray,
    anchor_raw_indices: np.ndarray,
    lags: tuple[int, ...] = DEFAULT_TRAIN_CONFIG.proprio_history_lags,
) -> np.ndarray:
    """Build (N_kept, num_lags, 6) causal proprioception history from raw 20 Hz episode steps."""
    prop = np.asarray(proprio_raw, dtype=np.float32)
    raw_idx = np.asarray(anchor_raw_indices, dtype=np.int64)
    frames = [prop[np.maximum(0, raw_idx - int(lag))] for lag in lags]
    return np.stack(frames, axis=1).astype(np.float32)


class HDF5DemoDataset:
    """Pre-allocated uint8 multi-camera HDF5 dataset with task-interleaved unified-memory layout."""

    def __init__(
        self,
        h5_path: str | Path,
        chunk_size: int = DEFAULT_TRAIN_CONFIG.chunk_size,
        camera_names: tuple[str, ...] = DEFAULT_TRAIN_CONFIG.camera_names,
        rolling_window_size: int = DEFAULT_TRAIN_CONFIG.rolling_window_size,
        trim_stationary: bool = True,
        proprio_history_lags: tuple[int, ...] = DEFAULT_TRAIN_CONFIG.proprio_history_lags,
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.h5_path}")

        self.chunk_size = int(chunk_size)
        self.camera_names = tuple(camera_names)
        self.rolling_window_size = int(rolling_window_size)
        self.trim_stationary = bool(trim_stationary)
        self.proprio_history_lags = tuple(int(x) for x in proprio_history_lags)

        dummy_sim_env: SimEnv | None = None
        with h5py.File(self.h5_path, "r") as f:
            root = f["data"] if "data" in f and isinstance(f["data"], h5py.Group) else f
            raw_keys = sorted(k for k in root if k.startswith("demo_"))
            if not raw_keys:
                raise ValueError(f"No 'demo_*' groups found in {self.h5_path}")

            # Group episode keys by (task_id, domain) and interleave round-robin so every
            # contiguous memory window contains a balanced mix of all tasks and domains.
            buckets: dict[tuple[int, str], list[str]] = defaultdict(list)
            ep_filtered_cache: dict[
                str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ] = {}
            total_steps = 0
            img_shape: tuple[int, int, int] | None = None

            for k in raw_keys:
                grp = root[k]
                tid = int(grp.attrs.get("task_id", 0))
                dom = str(grp.attrs.get("domain", "sim_clean"))
                traj_ver = str(grp.attrs.get("trajectory_version", root.attrs.get("trajectory_version", "v1")))
                buckets[(tid, dom)].append(k)
                act_np = grp["actions"][:].astype(np.float32)
                prop_np = grp["obs/proprio"][:].astype(np.float32)
                filt_p, filt_a, filt_c, filt_idx = filter_episode_frames(
                    prop_np,
                    act_np,
                    chunk_size=self.chunk_size,
                    trim_stationary=(self.trim_stationary and traj_ver != "v2"),
                )
                filt_h = build_proprio_history(
                    prop_np, filt_idx, lags=self.proprio_history_lags
                )
                ep_filtered_cache[k] = (filt_p, filt_h, filt_a, filt_c, filt_idx)
                total_steps += len(filt_idx)
                if img_shape is None:
                    cam0 = self.camera_names[0]
                    obs_grp = grp["obs"]
                    cam_key = f"rgb_{cam0}" if f"rgb_{cam0}" in obs_grp else cam0
                    img_shape = tuple(int(x) for x in obs_grp[cam_key].shape[1:])

            interleaved_keys: list[str] = []
            sorted_bucket_keys = sorted(buckets.keys())
            max_bucket_len = max(len(v) for v in buckets.values())
            for idx in range(max_bucket_len):
                for bkey in sorted_bucket_keys:
                    if idx < len(buckets[bkey]):
                        interleaved_keys.append(buckets[bkey][idx])

            assert img_shape is not None
            h, w, c = img_shape
            num_lags = len(self.proprio_history_lags)
            self.num_episodes = len(interleaved_keys)
            self.num_steps = total_steps

            # Pre-allocate single contiguous tensors to avoid 2x peak RAM from torch.cat
            self.proprio = torch.empty((total_steps, 6), dtype=torch.float32)
            self.proprio_history = torch.empty(
                (total_steps, num_lags, 6), dtype=torch.float32
            )
            self.ep_start_cursor = torch.empty((total_steps,), dtype=torch.long)
            self.ep_step = torch.empty((total_steps,), dtype=torch.long)
            self.actions = torch.empty((total_steps, 6), dtype=torch.float32)
            self.action_chunks = torch.empty(
                (total_steps, self.chunk_size, 6), dtype=torch.float32
            )
            self.task_ids = torch.empty((total_steps,), dtype=torch.long)
            self.obj_xy = torch.zeros((total_steps, 12), dtype=torch.float32)
            self.obj_xy_mask = torch.zeros((total_steps, 12), dtype=torch.float32)
            self.rgb_cache: dict[str, torch.Tensor] = {
                cam: torch.empty((total_steps, h, w, c), dtype=torch.uint8)
                for cam in self.camera_names
            }

            cursor = 0
            self.task_counts: dict[int, int] = defaultdict(int)
            self.domain_counts: dict[str, int] = defaultdict(int)

            for k in interleaved_keys:
                grp = root[k]
                tid = int(grp.attrs.get("task_id", 0))
                dom = str(grp.attrs.get("domain", "sim_clean"))
                filt_p, filt_h, filt_a, filt_c, filt_idx = ep_filtered_cache[k]
                t_ep = len(filt_idx)
                end = cursor + t_ep

                self.proprio[cursor:end] = torch.from_numpy(filt_p)
                self.proprio_history[cursor:end] = torch.from_numpy(filt_h)
                self.ep_start_cursor[cursor:end] = cursor
                self.ep_step[cursor:end] = torch.arange(t_ep, dtype=torch.long)
                self.actions[cursor:end] = torch.from_numpy(filt_a)
                self.action_chunks[cursor:end] = torch.from_numpy(filt_c)
                self.task_ids[cursor:end] = tid

                if "seed" in grp.attrs:
                    if dummy_sim_env is None:
                        dummy_sim_env = SimEnv(include_rgb=False)
                    ep_seed = int(grp.attrs["seed"])
                    spawns_xy = dummy_sim_env._sample_object_spawns(
                        np.random.default_rng(ep_seed)
                    )
                    layout_12d = _compute_episode_layout_12d(spawns_xy, tid)
                    self.obj_xy[cursor:end] = torch.from_numpy(layout_12d).unsqueeze(0)

                    # Target object and non-source objects remain at spawn XY for all steps;
                    # source object stays at spawn XY until the episode's actual lift_start_step
                    if "lift_start_step" in grp.attrs:
                        lift_step = int(grp.attrs["lift_start_step"])
                    else:
                        closed_steps = np.where(filt_a[:, 5] <= 0.25)[0]
                        lift_step = int(filt_idx[closed_steps[0]]) if len(closed_steps) > 0 else 45
                    mask_ep = np.ones((t_ep, 12), dtype=np.float32)
                    moved = filt_idx >= lift_step
                    mask_ep[moved, 0:2] = 0.0
                    src_name = TASK_SPECS[tid].source_object
                    src_slot = OBJECT_NAMES.index(src_name)
                    mask_ep[moved, 4 + 2 * src_slot : 6 + 2 * src_slot] = 0.0
                    self.obj_xy_mask[cursor:end] = torch.from_numpy(mask_ep)

                obs_grp = grp["obs"]
                for cam in self.camera_names:
                    cam_key = f"rgb_{cam}" if f"rgb_{cam}" in obs_grp else cam
                    raw_imgs = obs_grp[cam_key][:]
                    self.rgb_cache[cam][cursor:end] = torch.from_numpy(raw_imgs[filt_idx])

                self.task_counts[tid] += 1
                self.domain_counts[dom] += 1
                cursor = end

        if dummy_sim_env is not None:
            dummy_sim_env.close()

        self.norm_stats = self.compute_norm_stats()

    def __len__(self) -> int:
        return self.num_steps

    def compute_norm_stats(self) -> dict[str, torch.Tensor]:
        """Compute per-dimension mean and std (clamped >= 1e-2) across all steps."""
        valid_layout_rows = self.obj_xy_mask[:, 0] > 0.5
        if torch.any(valid_layout_rows):
            xy_valid = self.obj_xy[valid_layout_rows]
            obj_mean = xy_valid.mean(dim=0)
            obj_std = xy_valid.std(dim=0).clamp(min=1e-2)
        else:
            obj_mean = torch.zeros(12, dtype=torch.float32)
            obj_std = torch.ones(12, dtype=torch.float32)
        return {
            "proprio_mean": self.proprio.mean(dim=0),
            "proprio_std": self.proprio.std(dim=0).clamp(min=1e-2),
            "action_mean": self.actions.mean(dim=0),
            "action_std": self.actions.std(dim=0).clamp(min=1e-2),
            "obj_xy_mean": obj_mean,
            "obj_xy_std": obj_std,
        }

    def sample_epoch_permutation(
        self,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate a rolling-window permutation to prevent unified-memory page thrashing."""
        n = self.num_steps
        w = min(max(self.rolling_window_size, 256), n)
        if w >= n:
            return torch.randperm(n, generator=generator)

        shift = int(torch.randint(0, w, (1,), generator=generator).item())
        indices = torch.roll(torch.arange(n, dtype=torch.long), shifts=shift)

        num_full_windows = n // w
        remainder = n % w
        window_order = torch.randperm(num_full_windows, generator=generator)

        permuted_chunks: list[torch.Tensor] = []
        for win_idx in window_order.tolist():
            start = win_idx * w
            win_slice = indices[start : start + w]
            local_perm = torch.randperm(w, generator=generator)
            permuted_chunks.append(win_slice[local_perm])

        if remainder > 0:
            rem_slice = indices[num_full_windows * w :]
            rem_perm = torch.randperm(remainder, generator=generator)
            permuted_chunks.append(rem_slice[rem_perm])

        return torch.cat(permuted_chunks, dim=0)

    def get_batch(
        self,
        indices: torch.Tensor,
        device: torch.device,
    ) -> dict[str, Any]:
        """Gather a minibatch and convert uint8 NHWC camera frames to contiguous float32 NCHW."""
        idx_cpu = indices.detach().cpu().long()
        proprio_b = self.proprio[idx_cpu].to(device=device, non_blocking=True)
        proprio_hist_b = self.proprio_history[idx_cpu].to(device=device, non_blocking=True)
        actions_b = self.action_chunks[idx_cpu].to(device=device, non_blocking=True)
        task_id_b = self.task_ids[idx_cpu].to(device=device, non_blocking=True)
        obj_xy_b = self.obj_xy[idx_cpu].to(device=device, non_blocking=True)
        obj_xy_mask_b = self.obj_xy_mask[idx_cpu].to(device=device, non_blocking=True)

        obs_b: dict[str, torch.Tensor] = {
            "proprio": proprio_b,
            "proprio_history": proprio_hist_b,
        }
        for cam in self.camera_names:
            img_u8 = self.rgb_cache[cam][idx_cpu].to(device=device, non_blocking=True)
            # Materialize contiguous NCHW while still 1-byte uint8 before float32 cast on Metal
            img_f32 = img_u8.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            obs_b[f"rgb_{cam}"] = img_f32

        return {
            "obs": obs_b,
            "task_id": task_id_b,
            "actions": actions_b,
            "obj_xy": obj_xy_b,
            "obj_xy_mask": obj_xy_mask_b,
        }


def compute_combined_norm_stats(
    sim_dataset: HDF5DemoDataset,
    real_dataset: HDF5DemoDataset,
    real_ratio: float = DEFAULT_TRAIN_CONFIG.real_ratio,
) -> dict[str, torch.Tensor]:
    """Compute weighted normalization statistics across Sim and Real datasets for co-training."""
    w_real = float(np.clip(real_ratio, 0.0, 1.0))
    w_sim = 1.0 - w_real

    s_stats = sim_dataset.norm_stats
    r_stats = real_dataset.norm_stats

    p_mean = w_sim * s_stats["proprio_mean"] + w_real * r_stats["proprio_mean"]
    a_mean = w_sim * s_stats["action_mean"] + w_real * r_stats["action_mean"]

    # Combined variance: w_s * (std_s^2 + (mu_s - mu)^2) + w_r * (std_r^2 + (mu_r - mu)^2)
    p_var = w_sim * (
        s_stats["proprio_std"].pow(2) + (s_stats["proprio_mean"] - p_mean).pow(2)
    ) + w_real * (
        r_stats["proprio_std"].pow(2) + (r_stats["proprio_mean"] - p_mean).pow(2)
    )
    a_var = w_sim * (
        s_stats["action_std"].pow(2) + (s_stats["action_mean"] - a_mean).pow(2)
    ) + w_real * (
        r_stats["action_std"].pow(2) + (r_stats["action_mean"] - a_mean).pow(2)
    )

    return {
        "proprio_mean": p_mean,
        "proprio_std": torch.sqrt(p_var).clamp(min=1e-2),
        "action_mean": a_mean,
        "action_std": torch.sqrt(a_var).clamp(min=1e-2),
        "obj_xy_mean": s_stats["obj_xy_mean"],
        "obj_xy_std": s_stats["obj_xy_std"],
    }


class MultiModeBatchLoader:
    """Unified batch loader supporting sim_only, real_only, finetune, and cotrain modes."""

    def __init__(
        self,
        config: AlloyTrainConfig,
        sim_dataset: HDF5DemoDataset | None = None,
        real_dataset: HDF5DemoDataset | None = None,
    ) -> None:
        self.config = config
        self.device = config.resolve_device()
        self.mode = config.train_mode

        if self.mode == "sim_only":
            self.sim_dataset = sim_dataset or HDF5DemoDataset(
                config.sim_data_path,
                chunk_size=config.chunk_size,
                camera_names=config.camera_names,
                rolling_window_size=config.rolling_window_size,
                proprio_history_lags=config.proprio_history_lags,
            )
            self.real_dataset = None
        elif self.mode in ("real_only", "finetune"):
            self.sim_dataset = None
            self.real_dataset = real_dataset or HDF5DemoDataset(
                config.real_data_path,
                chunk_size=config.chunk_size,
                camera_names=config.camera_names,
                rolling_window_size=config.rolling_window_size,
                proprio_history_lags=config.proprio_history_lags,
            )
        elif self.mode == "cotrain":
            self.sim_dataset = sim_dataset or HDF5DemoDataset(
                config.sim_data_path,
                chunk_size=config.chunk_size,
                camera_names=config.camera_names,
                rolling_window_size=config.rolling_window_size,
                proprio_history_lags=config.proprio_history_lags,
            )
            self.real_dataset = real_dataset or HDF5DemoDataset(
                config.real_data_path,
                chunk_size=config.chunk_size,
                camera_names=config.camera_names,
                rolling_window_size=config.rolling_window_size,
                proprio_history_lags=config.proprio_history_lags,
            )
        else:
            raise ValueError(f"Unsupported train_mode: {self.mode}")

    @property
    def num_batches_per_epoch(self) -> int:
        """Return number of minibatches per epoch."""
        b = self.config.batch_size
        if self.mode == "sim_only":
            assert self.sim_dataset is not None
            return max(len(self.sim_dataset) // b, 1)
        if self.mode in ("real_only", "finetune"):
            assert self.real_dataset is not None
            return max(len(self.real_dataset) // b, 1)
        # In cotrain mode, epoch length is determined by covering the simulation dataset
        assert self.sim_dataset is not None
        n_real = max(1, min(b - 1, round(b * self.config.real_ratio)))
        n_sim = b - n_sim_complement(b, n_real)
        return max(len(self.sim_dataset) // n_sim, 1)

    def compute_norm_stats(self) -> dict[str, torch.Tensor]:
        """Return dataset normalization statistics for the active mode."""
        if self.mode == "sim_only":
            assert self.sim_dataset is not None
            return self.sim_dataset.norm_stats
        if self.mode in ("real_only", "finetune"):
            assert self.real_dataset is not None
            return self.real_dataset.norm_stats
        assert self.sim_dataset is not None and self.real_dataset is not None
        return compute_combined_norm_stats(
            self.sim_dataset, self.real_dataset, real_ratio=self.config.real_ratio
        )

    def iter_epoch(self, epoch: int = 0) -> Iterator[dict[str, Any]]:
        """Yield minibatches for one epoch."""
        gen = torch.Generator()
        gen.manual_seed(self.config.seed + epoch * 1009)
        b = self.config.batch_size

        if self.mode in ("sim_only", "real_only", "finetune"):
            ds = self.sim_dataset if self.mode == "sim_only" else self.real_dataset
            assert ds is not None
            perm = ds.sample_epoch_permutation(generator=gen)
            if len(perm) < b:
                reps = (b + len(perm) - 1) // len(perm)
                perm = perm.repeat(reps)
            n_batches = max(len(ds) // b, 1)
            for i in range(n_batches):
                batch_idx = perm[i * b : (i + 1) * b]
                yield ds.get_batch(batch_idx, device=self.device)
            return

        # Co-training mode: mix (1 - real_ratio) Sim + real_ratio Real in every minibatch
        assert self.sim_dataset is not None and self.real_dataset is not None
        n_real = max(1, min(b - 1, round(b * self.config.real_ratio)))
        n_sim = b - n_real

        sim_perm = self.sim_dataset.sample_epoch_permutation(generator=gen)
        if len(sim_perm) < n_sim:
            sim_perm = sim_perm.repeat((n_sim + len(sim_perm) - 1) // len(sim_perm))
        n_batches = max(len(self.sim_dataset) // n_sim, 1)

        real_cursor = 0
        real_perm = self.real_dataset.sample_epoch_permutation(generator=gen)

        for i in range(n_batches):
            sim_idx = sim_perm[i * n_sim : (i + 1) * n_sim]
            while real_cursor + n_real > len(real_perm):
                extra_perm = self.real_dataset.sample_epoch_permutation(generator=gen)
                real_perm = torch.cat([real_perm[real_cursor:], extra_perm], dim=0)
                real_cursor = 0
            real_idx = real_perm[real_cursor : real_cursor + n_real]
            real_cursor += n_real

            sim_batch = self.sim_dataset.get_batch(sim_idx, device=self.device)
            real_batch = self.real_dataset.get_batch(real_idx, device=self.device)

            merged_obs = {
                k: torch.cat([sim_batch["obs"][k], real_batch["obs"][k]], dim=0)
                for k in sim_batch["obs"]
            }
            yield {
                "obs": merged_obs,
                "task_id": torch.cat([sim_batch["task_id"], real_batch["task_id"]], dim=0),
                "actions": torch.cat([sim_batch["actions"], real_batch["actions"]], dim=0),
                "obj_xy": torch.cat([sim_batch["obj_xy"], real_batch["obj_xy"]], dim=0),
                "obj_xy_mask": torch.cat(
                    [sim_batch["obj_xy_mask"], real_batch["obj_xy_mask"]], dim=0
                ),
            }

    def get_validation_batch(self, num_samples: int = 512) -> dict[str, Any]:
        """Return a deterministic, evenly spaced validation batch across all tasks and episodes."""
        ds = self.sim_dataset if self.sim_dataset is not None else self.real_dataset
        assert ds is not None
        n_total = len(ds)
        k = min(max(1, int(num_samples)), n_total)
        val_idx = torch.linspace(0, n_total - 1, steps=k, dtype=torch.long)
        return ds.get_batch(val_idx, device=self.device)


def n_sim_complement(batch_size: int, n_real: int) -> int:
    """Helper returning n_real so n_sim = batch_size - n_real."""
    return n_real
