"""CLI entry point for AlloyFlow 4-mode policy training (python -m training)."""

from __future__ import annotations

import argparse

from training.config import VALID_TRAIN_MODES, AlloyTrainConfig
from training.trainer import PolicyTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m training",
        description="Train AlloyFlow 3-task Vision Flow Matching policy across 4 modes.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=VALID_TRAIN_MODES,
        default="sim_only",
        help="Training regime: sim_only, real_only, finetune, or cotrain.",
    )
    parser.add_argument(
        "--sim-data",
        type=str,
        default="data/sim_demos.h5",
        help="Path to simulation demonstrations HDF5 file.",
    )
    parser.add_argument(
        "--real-data",
        type=str,
        default="data/real_demos.h5",
        help="Path to real-world demonstrations HDF5 file.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Pretrained checkpoint path (required when --mode finetune).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (defaults to checkpoints/<mode>).",
    )
    parser.add_argument(
        "--real-ratio",
        type=float,
        default=0.5,
        help="Fraction of each minibatch drawn from real data in cotrain mode.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Minibatch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override (defaults to 5e-4, or 1e-4 for finetune).",
    )
    parser.add_argument(
        "--flow-samples",
        type=int,
        default=4,
        help="Number of stratified flow timesteps K evaluated per CNN pass.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, mps, cuda, or cpu.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    save_dir = args.save_dir or f"checkpoints/{args.mode}"
    pretrained = args.pretrained_checkpoint
    if args.mode == "finetune" and pretrained is None:
        pretrained = "checkpoints/sim_only/best_policy.pt"

    lr = args.lr if args.lr is not None else (1e-4 if args.mode == "finetune" else 5e-4)

    config = AlloyTrainConfig(
        train_mode=args.mode,
        sim_data_path=args.sim_data,
        real_data_path=args.real_data,
        pretrained_checkpoint=pretrained,
        save_dir=save_dir,
        real_ratio=args.real_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=lr,
        num_flow_samples=args.flow_samples,
        seed=args.seed,
        device=args.device,
    )
    trainer = PolicyTrainer(config=config)
    summary = trainer.train(verbose=True)
    print(
        f"[AlloyTrainer] Finished mode={summary['train_mode']} | "
        f"best_loss={summary['best_loss']:.5f} | "
        f"saved to {summary['best_checkpoint']}"
    )


if __name__ == "__main__":
    main()
