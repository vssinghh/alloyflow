"""CLI entry point for AlloyFlow 4-mode policy training (python -m training)."""

from __future__ import annotations

import argparse

from training.config import DEFAULT_TRAIN_CONFIG, VALID_TRAIN_MODES, AlloyTrainConfig
from training.trainer import PolicyTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = DEFAULT_TRAIN_CONFIG
    parser = argparse.ArgumentParser(
        prog="python -m training",
        description="Train AlloyFlow 3-task Vision Flow Matching policy across 4 modes.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=VALID_TRAIN_MODES,
        default=d.train_mode,
        help="Training regime: sim_only, real_only, finetune, or cotrain.",
    )
    parser.add_argument(
        "--sim-data",
        type=str,
        default=d.sim_data_path,
        help="Path to simulation demonstrations HDF5 file.",
    )
    parser.add_argument(
        "--real-data",
        type=str,
        default=d.real_data_path,
        help="Path to real-world demonstrations HDF5 file.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=d.pretrained_checkpoint,
        help="Pretrained checkpoint path (defaults to default_finetune_checkpoint in finetune mode).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from <save-dir>/latest.pt if present.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Explicit checkpoint path (.pt) to resume training from.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Directory to save checkpoints (defaults to checkpoints/<mode>).",
    )
    parser.add_argument(
        "--real-ratio",
        type=float,
        default=d.real_ratio,
        help="Fraction of each minibatch drawn from real data in cotrain mode.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=d.epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=d.batch_size,
        help="Minibatch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override (defaults to config.effective_lr).",
    )
    parser.add_argument(
        "--flow-samples",
        type=int,
        default=d.num_flow_samples,
        help="Number of stratified flow timesteps K evaluated per CNN pass.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=d.seed,
        help="Random seed for reproducible training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=d.device,
        help="Compute device: auto, mps, cuda, or cpu.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    d = DEFAULT_TRAIN_CONFIG
    pretrained = args.pretrained_checkpoint
    if args.mode == "finetune" and pretrained is None:
        pretrained = d.default_finetune_checkpoint

    resume_ckpt = args.resume_from if args.resume_from else ("auto" if args.resume else None)

    config_kwargs = {
        "train_mode": args.mode,
        "sim_data_path": args.sim_data,
        "real_data_path": args.real_data,
        "pretrained_checkpoint": pretrained,
        "resume_checkpoint": resume_ckpt,
        "save_dir": args.save_dir,
        "real_ratio": args.real_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_flow_samples": args.flow_samples,
        "seed": args.seed,
        "device": args.device,
    }
    if args.lr is not None:
        config_kwargs["lr"] = args.lr

    config = AlloyTrainConfig(**config_kwargs)
    trainer = PolicyTrainer(config=config)
    summary = trainer.train(verbose=True)
    print(
        f"[AlloyTrainer] Finished mode={summary['train_mode']} | "
        f"best_loss={summary['best_loss']:.5f} | "
        f"saved to {summary['best_checkpoint']}"
    )


if __name__ == "__main__":
    main()
