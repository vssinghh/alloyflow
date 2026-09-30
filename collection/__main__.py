"""CLI entrypoint for collecting simulated or real SO-ARM101 demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

from collection.collector import (
    collect_real_demos,
    collect_sim_demos,
    inspect_hdf5_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect multi-task SO-ARM101 demonstrations in HDF5 format."
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["sim", "real"],
        default="sim",
        help="Collection domain: scripted MuJoCo IK ('sim') or leader-follower ('real').",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help="Task ID ('0', '1', '2') or 'all' for all 3 tasks.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of demonstrations to collect per task.",
    )
    parser.add_argument(
        "--dr",
        action="store_true",
        help="Collect 100%% Domain-Randomized episodes (overrides default 50/50 split).",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Collect 100%% Clean studio episodes (overrides default 50/50 split).",
    )
    parser.add_argument(
        "--split-clean-dr",
        action="store_true",
        default=True,
        help="Collect 50%% Clean + 50%% Domain-Randomized episodes per task (default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 path (defaults to data/sim_demos.h5 or data/real_demos.h5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base random seed for reproducible simulation spawns.",
    )
    parser.add_argument(
        "--follower-port",
        type=str,
        default="/dev/tty.usbmodem58760431551",
        help="USB serial port for the follower SO-ARM101.",
    )
    parser.add_argument(
        "--leader-port",
        type=str,
        default="/dev/tty.usbmodem58760431552",
        help="USB serial port for the leader SO-ARM101.",
    )
    parser.add_argument(
        "--mock-hardware",
        action="store_true",
        help="Use synthetic hardware signals when testing real teleop without USB servos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_ids = [0, 1, 2] if args.task == "all" else [int(args.task)]

    if args.output is None:
        default_name = "sim_demos.h5" if args.domain == "sim" else "real_demos.h5"
        output_path = Path("data") / default_name
    else:
        output_path = args.output

    if args.domain == "sim":
        use_split = bool(args.split_clean_dr and not args.dr and not args.clean_only)
        summary = collect_sim_demos(
            output_path=output_path,
            task_ids=task_ids,
            episodes_per_task=args.episodes,
            domain_rand=args.dr,
            split_clean_and_dr=use_split,
            base_seed=args.seed,
            verbose=True,
        )
    else:
        summary = collect_real_demos(
            output_path=output_path,
            task_ids=task_ids,
            episodes_per_task=args.episodes,
            follower_port=args.follower_port,
            leader_port=args.leader_port,
            mock_hardware=args.mock_hardware,
            verbose=True,
        )

    stats = inspect_hdf5_dataset(summary["output_path"])
    print(f"Saved dataset to {summary['output_path']}: {stats}")


if __name__ == "__main__":
    main()
