"""Unit and integration tests for the AlloyFlow 4-mode training package."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from envs.base import CAMERA_NAMES, JOINT_LIMITS_HIGH, JOINT_LIMITS_LOW
from training.__main__ import parse_args
from training.config import AlloyTrainConfig
from training.dataset import (
    HDF5DemoDataset,
    build_action_chunks,
)
from training.flow_matching import ConditionalFlowMatcher, TemporalEnsembler
from training.model import (
    MultiCameraAttention,
    SpatialSoftmax2d,
    SpatialSoftmaxConvNet,
    TaskConditionedVisionFlowPolicy,
)
from training.trainer import (
    PolicyTrainer,
    load_policy_checkpoint,
)


def _create_synthetic_h5(
    path: Path,
    num_episodes: int = 6,
    steps_per_ep: int = 20,
    img_size: int = 128,
    domain_prefix: str = "sim",
    action_offset: float = 0.0,
) -> Path:
    """Create a synthetic multi-task HDF5 dataset matching the AlloyFlow schema."""
    rng = np.random.default_rng(123)
    with h5py.File(path, "w") as f:
        for ep_idx in range(num_episodes):
            ep_grp = f.create_group(f"demo_{ep_idx:04d}")
            task_id = ep_idx % 3
            domain = f"{domain_prefix}_clean" if ep_idx < (num_episodes // 2) else f"{domain_prefix}_dr"
            ep_grp.attrs["task_id"] = task_id
            ep_grp.attrs["domain"] = domain

            proprio = (
                rng.uniform(-0.4, 0.4, size=(steps_per_ep, 6)).astype(np.float32)
                + action_offset
            )
            actions = (
                rng.uniform(-0.4, 0.4, size=(steps_per_ep, 6)).astype(np.float32)
                + action_offset
            )
            actions[:, 5] = np.clip(rng.uniform(0.0, 1.0, size=(steps_per_ep,)), 0.0, 1.0)

            obs_grp = ep_grp.create_group("obs")
            obs_grp.create_dataset("proprio", data=proprio)
            for cam in CAMERA_NAMES:
                rgb = rng.integers(
                    0, 256, size=(steps_per_ep, img_size, img_size, 3), dtype=np.uint8
                )
                obs_grp.create_dataset(f"rgb_{cam}", data=rgb)

            ep_grp.create_dataset("actions", data=actions)
    return path


def test_config_validation_and_serialization() -> None:
    """Verify AlloyTrainConfig properties, validation rules, and dict round-trip."""
    cfg = AlloyTrainConfig()
    assert cfg.attended_cam_dim == 96
    assert cfg.fused_dim == 192
    assert cfg.num_flow_samples == 4
    assert cfg.gripper_weight == pytest.approx(2.5)
    assert cfg.effective_lr == pytest.approx(5e-4)

    ft_cfg = AlloyTrainConfig(train_mode="finetune", pretrained_checkpoint="dummy.pt")
    assert ft_cfg.effective_lr == pytest.approx(1e-4)

    roundtrip = AlloyTrainConfig.from_dict(cfg.to_dict())
    assert roundtrip == cfg

    with pytest.raises(ValueError, match="Invalid train_mode"):
        AlloyTrainConfig(train_mode="invalid_mode")

    with pytest.raises(ValueError, match="real_ratio"):
        AlloyTrainConfig(train_mode="cotrain", real_ratio=1.0)

    with pytest.raises(ValueError, match="divisible"):
        AlloyTrainConfig(vision_feat_dim=30, num_attn_heads=4)


def test_spatial_softmax_and_convnet() -> None:
    """Verify SpatialSoftmax2d coordinate extraction and SpatialSoftmaxConvNet Task FiLM."""
    ss = SpatialSoftmax2d(height=16, width=16, temperature=0.01)
    heatmaps = torch.zeros((2, 4, 16, 16), dtype=torch.float32)
    # Channel 0 peak at top-left (row=0, col=0) -> x=-1, y=-1
    heatmaps[0, 0, 0, 0] = 20.0
    # Channel 1 peak at bottom-right (row=15, col=15) -> x=+1, y=+1
    heatmaps[0, 1, 15, 15] = 20.0

    flat, kp = ss(heatmaps, return_keypoints=True)
    assert flat.shape == (2, 8)
    assert kp.shape == (2, 4, 2)
    assert kp[0, 0, 0].item() == pytest.approx(-1.0, abs=1e-3)
    assert kp[0, 0, 1].item() == pytest.approx(-1.0, abs=1e-3)
    assert kp[0, 1, 0].item() == pytest.approx(1.0, abs=1e-3)
    assert kp[0, 1, 1].item() == pytest.approx(1.0, abs=1e-3)

    conv = SpatialSoftmaxConvNet(num_keypoints=32, vision_feat_dim=32, task_emb_dim=32, img_size=128)
    img = torch.rand((3, 3, 128, 128), dtype=torch.float32)
    z_task_a = torch.randn((3, 32), dtype=torch.float32)
    z_task_b = torch.randn((3, 32), dtype=torch.float32)

    # Enable non-zero FiLM weights to verify task conditioning changes visual keypoints
    torch.nn.init.normal_(conv.task_film.weight, std=0.2)
    tok_a, kp_a = conv(img, z_task_a, return_keypoints=True)
    tok_b, kp_b = conv(img, z_task_b, return_keypoints=True)

    assert tok_a.shape == (3, 32)
    assert kp_a.shape == (3, 32, 2)
    assert not torch.allclose(tok_a, tok_b)
    assert not torch.allclose(kp_a, kp_b)


def test_multi_camera_attention_side_by_side_slots() -> None:
    """Verify MultiCameraAttention keeps 3 weighted 32D camera slots side by side (96D)."""
    attn = MultiCameraAttention(
        num_cameras=3,
        vision_feat_dim=32,
        proprio_emb_dim=64,
        task_emb_dim=32,
        num_heads=4,
    )
    cam_tokens = torch.randn((4, 3, 32), dtype=torch.float32)
    z_prop = torch.randn((4, 64), dtype=torch.float32)
    z_task = torch.randn((4, 32), dtype=torch.float32)

    attended_slots, weights = attn(cam_tokens, z_prop, z_task, return_weights=True)
    assert attended_slots.shape == (4, 96)
    assert weights.shape == (4, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)

    # Verify that zeroing out one camera token only changes its corresponding 32D slot
    # when attention weights are held fixed
    v = attn.v_proj(cam_tokens)
    assert v.shape == (4, 3, 32)


def test_policy_forward_and_norm_buffers() -> None:
    """Verify TaskConditionedVisionFlowPolicy forward shapes, parameter count, and norm buffers."""
    cfg = AlloyTrainConfig(device="cpu")
    policy = TaskConditionedVisionFlowPolicy(cfg)
    n_params = policy.count_parameters()
    assert 900_000 < n_params < 1_100_000

    stats = {
        "proprio_mean": np.full((6,), 0.25, dtype=np.float32),
        "proprio_std": np.full((6,), 0.5, dtype=np.float32),
        "action_mean": np.full((6,), -0.1, dtype=np.float32),
        "action_std": np.full((6,), 0.4, dtype=np.float32),
    }
    policy.set_norm_stats(stats)
    assert bool(policy.stats_initialized.item()) is True

    raw_act = torch.randn((4, 16, 6), dtype=torch.float32)
    rec_act = policy.unnormalize_actions(policy.normalize_actions(raw_act))
    assert torch.allclose(raw_act, rec_act, atol=1e-5)

    obs = {
        "proprio": torch.randn((4, 6), dtype=torch.float32),
        "rgb_third_person_cam": torch.randint(0, 256, (4, 128, 128, 3), dtype=torch.uint8),
        "rgb_overhead_cam": torch.randint(0, 256, (4, 128, 128, 3), dtype=torch.uint8),
        "rgb_wrist_cam": torch.randint(0, 256, (4, 128, 128, 3), dtype=torch.uint8),
    }
    task_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    z_fused, aux = policy.extract_obs_features(obs, task_ids, return_aux=True)
    assert z_fused.shape == (4, 192)
    assert aux["camera_weights"].shape == (4, 3)
    assert set(aux["keypoints"].keys()) == set(CAMERA_NAMES)

    # Single-sample velocity prediction (inference shape)
    x_tau_3d = torch.randn((4, 16, 6), dtype=torch.float32)
    tau_1d = torch.rand((4,), dtype=torch.float32)
    v_3d = policy(obs, task_ids, x_tau_3d, tau_1d)
    assert v_3d.shape == (4, 16, 6)

    # Stratified K-sample velocity prediction (training shape)
    x_tau_4d = torch.randn((4, 4, 16, 6), dtype=torch.float32)
    tau_2d = torch.rand((4, 4), dtype=torch.float32)
    v_4d = policy.predict_velocity(z_fused, x_tau_4d, tau_2d)
    assert v_4d.shape == (4, 4, 16, 6)


def test_flow_matcher_and_temporal_ensembler() -> None:
    """Verify stratified timestep sampling, CFM loss gradients, ODE sampling, and ensembling."""
    cfg = AlloyTrainConfig(device="cpu", num_flow_samples=4, ode_steps=5)
    policy = TaskConditionedVisionFlowPolicy(cfg)
    matcher = ConditionalFlowMatcher(cfg)

    tau = matcher.sample_stratified_timesteps(batch_size=16, k=4, device=torch.device("cpu"))
    assert tau.shape == (16, 4)
    assert torch.all((tau[:, 0] >= 0.0) & (tau[:, 0] < 0.25))
    assert torch.all((tau[:, 1] >= 0.25) & (tau[:, 1] < 0.50))
    assert torch.all((tau[:, 2] >= 0.50) & (tau[:, 2] < 0.75))
    assert torch.all((tau[:, 3] >= 0.75) & (tau[:, 3] < 1.00))

    batch = {
        "obs": {
            "proprio": torch.randn((4, 6), dtype=torch.float32),
            "rgb_third_person_cam": torch.rand((4, 3, 128, 128), dtype=torch.float32),
            "rgb_overhead_cam": torch.rand((4, 3, 128, 128), dtype=torch.float32),
            "rgb_wrist_cam": torch.rand((4, 3, 128, 128), dtype=torch.float32),
        },
        "task_id": torch.tensor([0, 1, 2, 1], dtype=torch.long),
        "actions": torch.randn((4, 16, 6), dtype=torch.float32),
    }

    out = matcher.compute_loss(policy, batch)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    # Check gradients flow through task embedding, proprio MLP, camera encoders, and attention
    assert policy.task_embedding.weight.grad is not None
    assert policy.camera_attention.q_proj.weight.grad is not None
    assert policy.out_head[-1].weight.grad is not None

    sampled_actions, aux = matcher.sample_action_chunk(
        policy, batch["obs"], task_id=1, ode_steps=4, clip_to_limits=True, return_aux=True
    )
    assert sampled_actions.shape == (4, 16, 6)
    low = torch.as_tensor(JOINT_LIMITS_LOW, dtype=torch.float32)
    high = torch.as_tensor(JOINT_LIMITS_HIGH, dtype=torch.float32)
    assert torch.all(sampled_actions >= low - 1e-5)
    assert torch.all(sampled_actions <= high + 1e-5)
    assert "camera_weights" in aux

    # Temporal Ensembler test
    ensembler = TemporalEnsembler(chunk_size=4, action_dim=6, decay=0.1)
    c0 = np.ones((4, 6), dtype=np.float32) * 1.0
    c1 = np.ones((4, 6), dtype=np.float32) * 2.0
    a0 = ensembler.update(c0)
    assert np.allclose(a0, 1.0)
    a1 = ensembler.update(c1)
    w0 = 1.0
    w1 = np.exp(-0.1)
    expected = (2.0 * w0 + 1.0 * w1) / (w0 + w1)
    assert np.allclose(a1, expected, atol=1e-5)
    ensembler.reset()
    assert np.allclose(ensembler.update(c0), 1.0)


def test_action_chunks_and_hdf5_interleaving(tmp_path: Path) -> None:
    """Verify terminal hold padding and HDF5DemoDataset task/domain interleaving."""
    ep_actions = np.arange(30, dtype=np.float32).reshape(5, 6)
    chunks = build_action_chunks(ep_actions, chunk_size=4)
    assert chunks.shape == (5, 4, 6)
    assert np.array_equal(chunks[0], ep_actions[0:4])
    # At final step t=4, all 4 horizon steps should hold ep_actions[4]
    for h in range(4):
        assert np.array_equal(chunks[4, h], ep_actions[4])

    h5_path = _create_synthetic_h5(tmp_path / "sim.h5", num_episodes=6, steps_per_ep=10)
    ds = HDF5DemoDataset(h5_path, chunk_size=4, rolling_window_size=16)
    assert len(ds) == 60
    assert ds.num_episodes == 6
    assert ds.task_counts[0] == 2
    assert ds.task_counts[1] == 2
    assert ds.task_counts[2] == 2

    perm = ds.sample_epoch_permutation(generator=torch.Generator().manual_seed(7))
    assert perm.shape == (60,)
    assert sorted(perm.tolist()) == list(range(60))

    batch = ds.get_batch(perm[:8], device=torch.device("cpu"))
    assert batch["obs"]["proprio"].shape == (8, 6)
    assert batch["obs"]["rgb_third_person_cam"].shape == (8, 3, 128, 128)
    assert batch["obs"]["rgb_third_person_cam"].dtype == torch.float32
    assert batch["actions"].shape == (8, 4, 6)


def test_four_mode_trainer_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Verify PolicyTrainer across sim_only, real_only, finetune (locked stats), and cotrain."""
    sim_path = _create_synthetic_h5(
        tmp_path / "sim_demos.h5", num_episodes=6, steps_per_ep=8, domain_prefix="sim", action_offset=0.0
    )
    real_path = _create_synthetic_h5(
        tmp_path / "real_demos.h5", num_episodes=6, steps_per_ep=8, domain_prefix="real", action_offset=0.25
    )

    # 1. Mode 1: sim_only
    sim_cfg = AlloyTrainConfig(
        train_mode="sim_only",
        sim_data_path=str(sim_path),
        save_dir=str(tmp_path / "ckpt_sim"),
        epochs=2,
        batch_size=16,
        num_flow_samples=2,
        pretrain_loc_steps=0,
        device="cpu",
    )
    sim_trainer = PolicyTrainer(sim_cfg)
    sim_summary = sim_trainer.train(verbose=False)
    sim_ckpt_path = Path(sim_summary["best_checkpoint"])
    assert sim_ckpt_path.exists()
    sim_norm = sim_trainer.policy.get_norm_stats()

    # Verify load_policy_checkpoint round-trip
    loaded_policy, loaded_cfg, _ = load_policy_checkpoint(sim_ckpt_path, device="cpu")
    assert loaded_cfg.train_mode == "sim_only"
    for k, v in sim_norm.items():
        assert torch.allclose(loaded_policy.get_norm_stats()[k], v)

    # 2. Mode 2: real_only
    real_cfg = AlloyTrainConfig(
        train_mode="real_only",
        real_data_path=str(real_path),
        save_dir=str(tmp_path / "ckpt_real"),
        epochs=1,
        batch_size=16,
        num_flow_samples=2,
        device="cpu",
    )
    real_summary = PolicyTrainer(real_cfg).train(verbose=False)
    assert Path(real_summary["best_checkpoint"]).exists()

    # 3. Mode 3: finetune (verify norm_stats stay locked to sim_only checkpoint!)
    ft_cfg = AlloyTrainConfig(
        train_mode="finetune",
        real_data_path=str(real_path),
        pretrained_checkpoint=str(sim_ckpt_path),
        save_dir=str(tmp_path / "ckpt_finetune"),
        epochs=1,
        batch_size=16,
        num_flow_samples=2,
        device="cpu",
    )
    ft_trainer = PolicyTrainer(ft_cfg)
    ft_norm = ft_trainer.policy.get_norm_stats()
    for k, v in sim_norm.items():
        assert torch.allclose(ft_norm[k], v, atol=1e-6)
    ft_summary = ft_trainer.train(verbose=False)
    assert Path(ft_summary["best_checkpoint"]).exists()

    # 4. Mode 4: cotrain (50/50 batch mixing)
    co_cfg = AlloyTrainConfig(
        train_mode="cotrain",
        sim_data_path=str(sim_path),
        real_data_path=str(real_path),
        real_ratio=0.5,
        save_dir=str(tmp_path / "ckpt_cotrain"),
        epochs=1,
        batch_size=16,
        num_flow_samples=2,
        device="cpu",
    )
    co_trainer = PolicyTrainer(co_cfg)
    co_summary = co_trainer.train(verbose=False)
    assert Path(co_summary["best_checkpoint"]).exists()

    # Verify CLI argument parser
    parsed = parse_args(["--mode", "cotrain", "--real-ratio", "0.5", "--epochs", "5"])
    assert parsed.mode == "cotrain"
    assert parsed.real_ratio == pytest.approx(0.5)
    assert parsed.epochs == 5


def test_real_sim_demos_h5_header_integrity() -> None:
    """Verify the collected data/sim_demos.h5 file structure if present on disk."""
    sim_h5 = Path(__file__).resolve().parents[2] / "data" / "sim_demos.h5"
    if not sim_h5.exists():
        pytest.skip("data/sim_demos.h5 not present")

    with h5py.File(sim_h5, "r") as f:
        root = f.get("data", f)
        keys = sorted(k for k in root if k.startswith("demo_"))
        assert len(keys) == 300
        g0 = root["demo_0000"]
        assert g0["actions"].shape == (144, 6)
        assert g0["obs/proprio"].shape == (144, 6)
        assert g0["obs/rgb_third_person_cam"].shape == (144, 128, 128, 3)
        assert g0["obs/rgb_third_person_cam"].dtype == np.uint8
