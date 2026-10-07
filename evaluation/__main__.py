"""CLI entrypoint for closed-loop evaluation and per-episode diagnostic GIF generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from evaluation.evaluator import EpisodeEvalResult, SimPolicyEvaluator


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Run closed-loop AlloyFlow evaluation and save per-episode diagnostic GIFs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/exp08b_cooldown_k0/best_policy.pt",
        help="Path to trained policy checkpoint (.pt).",
    )
    parser.add_argument(
        "--demos",
        type=str,
        default=None,
        help="Comma-separated HDF5 demo keys (e.g. 'demo_0000,demo_0101,demo_0200') to evaluate with GT overlay.",
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/sim_demos_v2_dart_full_900.h5",
        help="Path to HDF5 dataset when --demos is specified.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help="Task ID ('0', '1', '2') or 'all' when running seed evaluation.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes per task when --demos is not specified.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=2000,
        help="Base seed for held-out evaluation episodes.",
    )
    parser.add_argument(
        "--dr",
        action="store_true",
        help="Enable Sim-to-Real Domain Randomization during evaluation.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=280,
        help="Maximum control steps per episode.",
    )
    parser.add_argument(
        "--exec-horizon",
        type=int,
        default=1,
        help="Open-loop execution horizon when --no-ensemble is passed.",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable TemporalEnsembler and execute action chunks open-loop for --exec-horizon steps.",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        default=True,
        help="Save multi-camera diagnostic GIF and 6-keyframe strip for each episode (default: True).",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Disable GIF rendering for fast batch evaluation.",
    )
    parser.add_argument(
        "--gif-dir",
        type=str,
        default="checkpoints/eval_gifs",
        help="Output directory for diagnostic GIFs and keyframe strips.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run strict benchmark evaluation (>=20 seeds/task) on both training and test splits.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset added to per-step torch.manual_seed during evaluation (default: 0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for --benchmark evaluation (default: 1).",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Optional output path for benchmark JSON report.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('auto', 'mps', 'cuda', or 'cpu').",
    )
    return parser.parse_args(argv)


def _print_result(res: EpisodeEvalResult) -> None:
    print(
        f"[{res.episode_label}] task={res.task_id} ({res.task_name}) | "
        f"{res.status_text} | "
        f"min_src_xy={res.min_pinch_src_xy_cm:.2f}cm | "
        f"max_lift={res.max_src_lift_cm:.2f}cm | "
        f"final_tgt_xy={res.final_src_tgt_xy_cm:.2f}cm | "
        f"latency={res.mean_latency_ms:.1f}ms"
    )
    if res.gif_path:
        print(f"  -> GIF:   {res.gif_path}")
    if res.strip_path:
        print(f"  -> Strip: {res.strip_path}")


def main(argv: Sequence[str] | None = None) -> list[EpisodeEvalResult]:
    args = parse_args(argv)
    save_gif = bool(args.save_gif and not args.no_gif)
    use_ensemble = not bool(args.no_ensemble)

    evaluator = SimPolicyEvaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    results: list[EpisodeEvalResult] = []
    try:
        if args.benchmark:
            n_eps = max(args.episodes, 20)
            evaluator.evaluate_benchmark(
                episodes_per_task=n_eps,
                train_h5_path=args.h5_path,
                test_base_seed=9000,
                domain_rand=args.dr,
                max_steps=args.max_steps,
                seed_offset=args.seed_offset,
                num_workers=args.workers,
                exec_horizon=args.exec_horizon,
                use_temporal_ensemble=use_ensemble,
                save_report_path=args.report_out,
                verbose=True,
            )
            return results
        if args.demos:
            demo_keys = [k.strip() for k in args.demos.split(",") if k.strip()]
            for key in demo_keys:
                res = evaluator.run_demo_case(
                    demo_key=key,
                    h5_path=args.h5_path,
                    max_steps=args.max_steps,
                    seed_offset=args.seed_offset,
                    exec_horizon=args.exec_horizon,
                    use_temporal_ensemble=use_ensemble,
                    save_gif=save_gif,
                    gif_dir=args.gif_dir,
                )
                results.append(res)
                _print_result(res)
        else:
            task_ids = [0, 1, 2] if args.task == "all" else [int(args.task)]
            for tid in task_ids:
                for ep_idx in range(args.episodes):
                    seed = args.base_seed + tid * 100 + ep_idx
                    res = evaluator.run_episode(
                        task_id=tid,
                        seed=seed,
                        domain_rand=args.dr,
                        max_steps=args.max_steps,
                        seed_offset=args.seed_offset,
                        exec_horizon=args.exec_horizon,
                        use_temporal_ensemble=use_ensemble,
                        save_gif=save_gif,
                        gif_dir=args.gif_dir,
                    )
                    results.append(res)
                    _print_result(res)
    finally:
        evaluator.close()

    passed = sum(int(r.all_constraints_passed) for r in results)
    print(f"\n[EvalSummary] Passed: {passed}/{len(results)} ({100.0 * passed / max(len(results), 1):.1f}%)")
    return results


if __name__ == "__main__":
    main()
