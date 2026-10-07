"""Unified HDF5 demonstration collector and schema validator for AlloyFlow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from collection.real_teleop import RealTeleopRecorder
from collection.sim_expert import SimExpertPlanner
from envs.base import (
    CAMERA_NAMES,
    HOME_PROPRIO_6D,
    JOINT_LIMITS_HIGH,
    JOINT_LIMITS_LOW,
    REAL_TRAIN_BASE_SEED,
    SIM_TRAIN_BASE_SEED,
    TASK_SPECS,
)
from envs.real_env import RealEnv
from envs.sim_env import SimEnv


def write_episode_to_hdf5(
    h5_file: h5py.File,
    episode_idx: int,
    episode_data: dict[str, Any],
    domain: str,
) -> None:
    """Write a single demonstration trajectory into an HDF5 group (`demo_<idx>`)."""
    grp_name = f"demo_{episode_idx:04d}"
    if grp_name in h5_file:
        del h5_file[grp_name]

    grp = h5_file.create_group(grp_name)
    task_id = int(episode_data["task_id"])
    num_steps = int(episode_data["num_steps"])
    grp.attrs["task_id"] = task_id
    grp.attrs["task_name"] = TASK_SPECS[task_id].name
    grp.attrs["domain"] = str(domain)
    grp.attrs["seed"] = int(episode_data.get("seed", episode_idx))
    grp.attrs["num_steps"] = num_steps
    grp.attrs["num_samples"] = num_steps
    grp.attrs["trajectory_version"] = str(episode_data.get("trajectory_version", "v1"))
    if "perturbation" in episode_data:
        grp.attrs["perturbation"] = str(episode_data["perturbation"])
    if "delta_mag_cm" in episode_data:
        grp.attrs["delta_mag_cm"] = float(episode_data["delta_mag_cm"])
    if "delta_xy" in episode_data:
        grp.attrs["delta_xy"] = np.asarray(episode_data["delta_xy"], dtype=np.float32)
    if "delta_tgt_mag_cm" in episode_data:
        grp.attrs["delta_tgt_mag_cm"] = float(episode_data["delta_tgt_mag_cm"])
    if "delta_tgt_xy" in episode_data:
        grp.attrs["delta_tgt_xy"] = np.asarray(episode_data["delta_tgt_xy"], dtype=np.float32)
    if "lift_start_step" in episode_data:
        grp.attrs["lift_start_step"] = int(episode_data["lift_start_step"])

    grp.create_dataset(
        "actions",
        data=np.asarray(episode_data["actions"], dtype=np.float32),
        compression="gzip",
        compression_opts=4,
    )

    obs_grp = grp.create_group("obs")
    obs_grp.create_dataset(
        "proprio",
        data=np.asarray(episode_data["proprio"], dtype=np.float32),
        compression="gzip",
        compression_opts=4,
    )
    for cam_name in CAMERA_NAMES:
        key = f"rgb_{cam_name}"
        obs_grp.create_dataset(
            key,
            data=np.asarray(episode_data[key], dtype=np.uint8),
            compression="gzip",
            compression_opts=4,
        )


def collect_sim_demos(
    output_path: str | Path,
    task_ids: list[int] | None = None,
    episodes_per_task: int = 100,
    domain_rand: bool = False,
    split_clean_and_dr: bool = False,
    base_seed: int = SIM_TRAIN_BASE_SEED,
    max_attempts_Factor: int = 4,
    trajectory_version: str = "v1",
    perturbation: str | None = None,
    reference_h5: str | Path | None = None,
    reuse_reference_groups: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Collect verified simulation demonstrations that pass all 4 constraint checks.

    Args:
        output_path: Target `.h5` file path.
        task_ids: List of task IDs to collect (defaults to `[0, 1, 2]`).
        episodes_per_task: Total episodes to save per task.
        domain_rand: Whether to enable visual + physical domain randomization.
        split_clean_and_dr: If True, collects the first half of `episodes_per_task`
            with `domain_rand=False` (`sim_clean`) and the second half with
            `domain_rand=True` (`sim_dr`), matching the 50 Clean + 50 DR design.
        base_seed: Starting RNG seed for reproducible spawns.
        max_attempts_Factor: Maximum seed attempts multiplier per required episode.
        trajectory_version: Expert trajectory version ('v1', 'v2', 'v2_dart', or 'v2_dart_full').
        perturbation: Optional perturbation mode (e.g. 'dart' or 'dart_full').
        reference_h5: Optional existing `.h5` dataset whose exact `(task_id, seed, domain)`
            schedule should be included first before any new seeds.
        reuse_reference_groups: If True and `reference_h5` already has matching
            `trajectory_version`, copy verified reference HDF5 groups directly.
        verbose: Whether to print progress updates.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tasks = [0, 1, 2] if task_ids is None else [int(t) for t in task_ids]

    env = SimEnv(domain_rand=domain_rand)
    planner = SimExpertPlanner(env, trajectory_version=trajectory_version)

    total_saved = 0
    total_attempts = 0
    per_task_counts: dict[int, int] = {t: 0 for t in tasks}
    failed_seeds: list[tuple[int, int, str, dict[str, Any]]] = []

    ref_schedule: list[tuple[str, int, int, str, str]] | None = None
    if reference_h5 is not None:
        with h5py.File(reference_h5, "r") as rf:
            rroot = rf.get("data", rf)
            root_tver = str(rroot.attrs.get("trajectory_version", "v1"))
            rkeys = sorted(k for k in rroot if k.startswith("demo_"))
            ref_schedule = [
                (
                    k,
                    int(rroot[k].attrs["task_id"]),
                    int(rroot[k].attrs["seed"]),
                    str(rroot[k].attrs.get("domain", "sim_clean")),
                    str(rroot[k].attrs.get("trajectory_version", root_tver)),
                )
                for k in rkeys
                if int(rroot[k].attrs["task_id"]) in tasks
            ]

    with h5py.File(out_file, "w") as h5f:
        h5f.attrs["num_cameras"] = len(CAMERA_NAMES)
        h5f.attrs["camera_names"] = list(CAMERA_NAMES)
        h5f.attrs["control_hz"] = env.control_hz
        h5f.attrs["trajectory_version"] = trajectory_version
        if perturbation is not None:
            h5f.attrs["perturbation"] = str(perturbation)

        if ref_schedule is not None:
            max_seed_by_task: dict[int, int] = {
                t: max((s for _, rt, s, _, _ in ref_schedule if rt == t), default=base_seed + t * 10_000)
                for t in tasks
            }
            ref_h5_handle = h5py.File(reference_h5, "r") if reuse_reference_groups else None
            try:
                for tid in tasks:
                    ref_task_items = [item for item in ref_schedule if item[1] == tid]
                    target_for_task = max(int(episodes_per_task), len(ref_task_items))
                    if split_clean_and_dr or any(item[3] == "sim_dr" for item in ref_task_items) and any(item[3] == "sim_clean" for item in ref_task_items):
                        clean_target = target_for_task // 2
                        dom_plan = [("sim_clean", clean_target), ("sim_dr", target_for_task - clean_target)]
                    else:
                        dom_tag = "sim_dr" if domain_rand else "sim_clean"
                        dom_plan = [(dom_tag, target_for_task)]

                    for domain_tag, dom_target in dom_plan:
                        env.domain_rand = (domain_tag == "sim_dr")
                        ref_dom_items = [item for item in ref_task_items if item[3] == domain_tag]
                        saved_for_dom = 0

                        for ref_key, _, seed, _, ref_tver in ref_dom_items:
                            if saved_for_dom >= dom_target:
                                break
                            if (
                                reuse_reference_groups
                                and ref_h5_handle is not None
                                and ref_tver == trajectory_version
                            ):
                                rroot = ref_h5_handle.get("data", ref_h5_handle)
                                grp_name = f"demo_{total_saved:04d}"
                                ref_h5_handle.copy(rroot[ref_key], h5f, name=grp_name)
                                total_attempts += 1
                                total_saved += 1
                                saved_for_dom += 1
                                per_task_counts[tid] = per_task_counts.get(tid, 0) + 1
                                continue

                            cur_seed = seed
                            while True:
                                total_attempts += 1
                                ep_data = planner.generate_episode(
                                    task_id=tid,
                                    seed=cur_seed,
                                    trim_dwell=(trajectory_version == "v1"),
                                    trajectory_version=trajectory_version,
                                )
                                constraints = ep_data["constraints"]
                                if constraints["all_constraints_passed"] and ep_data["num_steps"] > 0:
                                    write_episode_to_hdf5(
                                        h5_file=h5f,
                                        episode_idx=total_saved,
                                        episode_data=ep_data,
                                        domain=domain_tag,
                                    )
                                    total_saved += 1
                                    saved_for_dom += 1
                                    per_task_counts[tid] = per_task_counts.get(tid, 0) + 1
                                    if verbose and (per_task_counts[tid] % 25 == 0):
                                        print(
                                            f"[SimCollector-{trajectory_version}] Task {tid} ({TASK_SPECS[tid].name}): "
                                            f"saved {per_task_counts[tid]}/{target_for_task} (total={total_saved}, domain={domain_tag})",
                                            flush=True,
                                        )
                                    break
                                failed_seeds.append((tid, cur_seed, domain_tag, constraints))
                                max_seed_by_task[tid] += 1
                                cur_seed = max_seed_by_task[tid]

                        while saved_for_dom < dom_target:
                            max_seed_by_task[tid] += 1
                            cur_seed = max_seed_by_task[tid]
                            total_attempts += 1
                            ep_data = planner.generate_episode(
                                task_id=tid,
                                seed=cur_seed,
                                trim_dwell=(trajectory_version == "v1"),
                                trajectory_version=trajectory_version,
                            )
                            constraints = ep_data["constraints"]
                            if constraints["all_constraints_passed"] and ep_data["num_steps"] > 0:
                                write_episode_to_hdf5(
                                    h5_file=h5f,
                                    episode_idx=total_saved,
                                    episode_data=ep_data,
                                    domain=domain_tag,
                                )
                                total_saved += 1
                                saved_for_dom += 1
                                per_task_counts[tid] = per_task_counts.get(tid, 0) + 1
                                if verbose and (per_task_counts[tid] % 25 == 0 or per_task_counts[tid] == target_for_task):
                                    print(
                                        f"[SimCollector-{trajectory_version}] Task {tid} ({TASK_SPECS[tid].name}): "
                                        f"saved {per_task_counts[tid]}/{target_for_task} (total={total_saved}, domain={domain_tag})",
                                        flush=True,
                                    )
                            else:
                                failed_seeds.append((tid, cur_seed, domain_tag, constraints))
            finally:
                if ref_h5_handle is not None:
                    ref_h5_handle.close()
        else:
            for tid in tasks:
                saved_for_task = 0
                attempt = 0
                max_attempts = max(episodes_per_task * max_attempts_Factor, 10)
                clean_target = episodes_per_task // 2 if split_clean_and_dr else (
                    0 if domain_rand else episodes_per_task
                )

                while saved_for_task < episodes_per_task and attempt < max_attempts:
                    use_dr = (
                        (saved_for_task >= clean_target)
                        if split_clean_and_dr
                        else bool(domain_rand)
                    )
                    env.domain_rand = use_dr
                    domain_tag = "sim_dr" if use_dr else "sim_clean"

                    seed = base_seed + tid * 10_000 + attempt
                    attempt += 1
                    total_attempts += 1

                    ep_data = planner.generate_episode(
                        task_id=tid,
                        seed=seed,
                        trim_dwell=(trajectory_version == "v1"),
                        trajectory_version=trajectory_version,
                    )
                    constraints = ep_data["constraints"]

                    if constraints["all_constraints_passed"] and ep_data["num_steps"] > 0:
                        write_episode_to_hdf5(
                            h5_file=h5f,
                            episode_idx=total_saved,
                            episode_data=ep_data,
                            domain=domain_tag,
                        )
                        saved_for_task += 1
                        total_saved += 1
                        per_task_counts[tid] = saved_for_task
                        if verbose and (saved_for_task % 10 == 0 or saved_for_task == episodes_per_task):
                            print(
                                f"[SimCollector] Task {tid} ({TASK_SPECS[tid].name}): "
                                f"saved {saved_for_task}/{episodes_per_task} "
                                f"(attempts={attempt}, domain={domain_tag})"
                            )

                if saved_for_task < episodes_per_task:
                    env.close()
                    raise RuntimeError(
                        f"Failed to collect {episodes_per_task} verified episodes for task {tid} "
                        f"within {max_attempts} attempts (collected {saved_for_task})."
                    )

        h5f.attrs["num_episodes"] = total_saved

    env.close()
    return {
        "output_path": str(out_file),
        "total_saved": total_saved,
        "total_attempts": total_attempts,
        "per_task_counts": per_task_counts,
        "failed_seeds": failed_seeds,
    }


def collect_real_demos(
    output_path: str | Path,
    task_ids: list[int] | None = None,
    episodes_per_task: int = 20,
    follower_port: str = "/dev/tty.usbmodem58760431551",
    leader_port: str = "/dev/tty.usbmodem58760431552",
    max_steps: int = 160,
    mock_hardware: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Collect physical SO-ARM101 leader-follower demonstrations at 20 Hz."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tasks = [0, 1, 2] if task_ids is None else [int(t) for t in task_ids]

    follower_env = RealEnv(serial_port=follower_port, mock_hardware=mock_hardware)
    recorder = RealTeleopRecorder(
        follower_env=follower_env,
        leader_port=leader_port,
        mock_hardware=mock_hardware,
    )

    total_saved = 0
    per_task_counts: dict[int, int] = {t: 0 for t in tasks}

    with h5py.File(out_file, "w") as h5f:
        h5f.attrs["num_cameras"] = len(CAMERA_NAMES)
        h5f.attrs["camera_names"] = list(CAMERA_NAMES)
        h5f.attrs["control_hz"] = follower_env.control_hz

        for tid in tasks:
            saved_for_task = 0
            while saved_for_task < episodes_per_task:
                seed = REAL_TRAIN_BASE_SEED + tid * 1000 + saved_for_task
                ep_data = recorder.record_episode(
                    task_id=tid,
                    episode_seed=seed,
                    max_steps=max_steps,
                    trim_dwell=True,
                )
                write_episode_to_hdf5(
                    h5_file=h5f,
                    episode_idx=total_saved,
                    episode_data=ep_data,
                    domain="real",
                )
                saved_for_task += 1
                total_saved += 1
                per_task_counts[tid] = saved_for_task
                if verbose:
                    print(
                        f"[RealCollector] Task {tid} ({TASK_SPECS[tid].name}): "
                        f"saved {saved_for_task}/{episodes_per_task}"
                    )

        h5f.attrs["num_episodes"] = total_saved

    recorder.close()
    follower_env.close()
    return {
        "output_path": str(out_file),
        "total_saved": total_saved,
        "per_task_counts": per_task_counts,
    }


def inspect_hdf5_dataset(h5_path: str | Path) -> dict[str, Any]:
    """Validate an AlloyFlow HDF5 dataset against the Section 4.3 schema contract."""
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    task_counts: dict[int, int] = {}
    domain_counts: dict[str, int] = {}
    total_steps = 0

    with h5py.File(path, "r") as h5f:
        demo_keys = sorted(k for k in h5f if k.startswith("demo_"))
        for key in demo_keys:
            grp = h5f[key]
            tid = int(grp.attrs["task_id"])
            dom = str(grp.attrs["domain"])
            T = int(grp.attrs["num_steps"])

            actions = grp["actions"]
            proprio = grp["obs/proprio"]
            if actions.shape != (T, 6) or actions.dtype != np.float32:
                raise ValueError(f"{key}/actions invalid shape/dtype: {actions.shape}, {actions.dtype}")
            if proprio.shape != (T, 6) or proprio.dtype != np.float32:
                raise ValueError(f"{key}/obs/proprio invalid shape/dtype: {proprio.shape}, {proprio.dtype}")

            for cam_name in CAMERA_NAMES:
                cam_ds = grp[f"obs/rgb_{cam_name}"]
                if cam_ds.shape != (T, 128, 128, 3) or cam_ds.dtype != np.uint8:
                    raise ValueError(
                        f"{key}/obs/rgb_{cam_name} invalid shape/dtype: {cam_ds.shape}, {cam_ds.dtype}"
                    )

            task_counts[tid] = task_counts.get(tid, 0) + 1
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
            total_steps += T

    return {
        "num_episodes": len(demo_keys),
        "total_steps": total_steps,
        "task_counts": task_counts,
        "domain_counts": domain_counts,
    }


def verify_hdf5_action_replay(
    h5_path: str | Path,
    check_rgb_pixels: bool = True,
) -> dict[str, Any]:
    """Replay every saved simulation episode open-loop in SimEnv and audit all data invariants.

    Verifies:
      1. Unique `(task_id, seed)` pairs across episodes (no duplicate seeds).
      2. Valid joint radian limits and normalized `[0.0, 1.0]` gripper range on `proprio` and `actions`.
      3. Pre-action temporal alignment: `proprio[0] == HOME_PROPRIO_6D`, `||actions[t] - proprio[t]|| >= 1e-3`,
         and `||proprio[t+1] - proprio[t]|| >= 1e-3` at every step `t`.
      4. Zero open-loop replay drift: `max |replayed_proprio - saved_proprio| < 1e-6` and
         exact `0` pixel difference across all 3 cameras (`third_person_cam`, `overhead_cam`, `wrist_cam`).
      5. Physical task goal and bystander/target constraints pass after 10 post-rollout settling steps.
    """
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    env = SimEnv(domain_rand=False, include_rgb=check_rgb_pixels)
    seen_task_seeds: set[tuple[int, int]] = set()
    max_proprio_err = 0.0
    max_rgb_err = 0

    try:
        with h5py.File(path, "r") as h5f:
            demo_keys = sorted(k for k in h5f if k.startswith("demo_"))
            for key in demo_keys:
                grp = h5f[key]
                tid = int(grp.attrs["task_id"])
                seed = int(grp.attrs["seed"])
                dom = str(grp.attrs["domain"])
                T = int(grp.attrs["num_steps"])

                if (tid, seed) in seen_task_seeds:
                    raise ValueError(f"Duplicate (task_id={tid}, seed={seed}) in {key}")
                seen_task_seeds.add((tid, seed))

                actions = grp["actions"][:]
                proprio = grp["obs/proprio"][:]

                # Check joint & gripper unit bounds
                if np.any(proprio[:, :5] < JOINT_LIMITS_LOW[:5] - 0.05) or np.any(
                    proprio[:, :5] > JOINT_LIMITS_HIGH[:5] + 0.05
                ):
                    raise ValueError(f"{key}: proprio arm joints out of radian limits")
                if np.any(actions[:, :5] < JOINT_LIMITS_LOW[:5] - 0.05) or np.any(
                    actions[:, :5] > JOINT_LIMITS_HIGH[:5] + 0.05
                ):
                    raise ValueError(f"{key}: action arm joints out of radian limits")
                if np.any(proprio[:, 5] < -1e-4) or np.any(proprio[:, 5] > 1.0 + 1e-4):
                    raise ValueError(f"{key}: proprio gripper outside normalized [0.0, 1.0]")
                if np.any(actions[:, 5] < -1e-4) or np.any(actions[:, 5] > 1.0 + 1e-4):
                    raise ValueError(f"{key}: action gripper outside normalized [0.0, 1.0]")

                # Check pre-action temporal alignment & zero stationary frames
                if float(np.max(np.abs(proprio[0] - HOME_PROPRIO_6D))) > 1e-2:
                    raise ValueError(f"{key}: initial proprio[0] does not match HOME_PROPRIO_6D")
                dq = np.linalg.norm(np.diff(proprio, axis=0), axis=-1)
                da = np.linalg.norm(actions - proprio, axis=-1)
                if float(np.min(dq)) < 1e-3 or float(np.min(da)) < 1e-3:
                    raise ValueError(
                        f"{key}: stationary frame detected (min_dq={dq.min():.6f}, min_da={da.min():.6f})"
                    )

                # Replay open-loop in SimEnv
                env.domain_rand = dom == "sim_dr"
                obs = env.reset(task_id=tid, seed=seed)
                saved_rgb = (
                    {cam: grp[f"obs/rgb_{cam}"][:] for cam in CAMERA_NAMES}
                    if check_rgb_pixels
                    else {}
                )

                for t in range(T):
                    p_err = float(np.max(np.abs(obs["proprio"] - proprio[t])))
                    max_proprio_err = max(max_proprio_err, p_err)
                    if p_err > 1e-6:
                        raise ValueError(
                            f"{key} step {t}: proprio replay mismatch {p_err:.8f}"
                        )
                    if check_rgb_pixels:
                        for cam in CAMERA_NAMES:
                            c_err = int(
                                np.max(
                                    np.abs(
                                        obs[f"rgb_{cam}"].astype(int)
                                        - saved_rgb[cam][t].astype(int)
                                    )
                                )
                            )
                            max_rgb_err = max(max_rgb_err, c_err)
                            if c_err > 0:
                                raise ValueError(
                                    f"{key} step {t} {cam}: RGB pixel replay mismatch (max diff={c_err})"
                                )
                    obs = env.step(actions[t])

                # Step 10 extra settling frames holding final action to verify physical stability
                for _ in range(SimExpertPlanner.SETTLE_STEPS):
                    env.step(actions[-1])

                constraints = env.check_constraints(task_id=tid)
                if not constraints["all_constraints_passed"]:
                    raise ValueError(
                        f"{key}: open-loop replay failed constraints after settling: {constraints}"
                    )
    finally:
        env.close()

    return {
        "verified_episodes": len(seen_task_seeds),
        "max_proprio_err": max_proprio_err,
        "max_rgb_err": max_rgb_err,
        "all_replays_exact": True,
    }

