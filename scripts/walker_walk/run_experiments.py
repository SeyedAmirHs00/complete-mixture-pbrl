#!/usr/bin/env python3
"""Unified runner for Walker-Walk mixture experiments.

Supports ``train_PEBBLE_mixture.py`` and ``train_PEBBLE_mixture_zero_last_wk_sgd.py``
across feedback budgets 100 / 500 / 1000 / 5000.  ``reward_batch`` and ``feed_type``
are chosen from the existing per-budget scripts (``500/``, ``1000/``, ``5000/``,
``run_pebble.sh``).

Examples
--------
  # Standard mixture, 500 labels (matches scripts/walker_walk/500/)
  python scripts/walker_walk/run_experiments.py --method mixture --max-feedback 500

  # Zero-last w_k + SGD TTP, 5000 labels, 10 seeds
  python scripts/walker_walk/run_experiments.py --method zero_last_wk_sgd --max-feedback 5000

  # Custom betas, dry-run
  python scripts/walker_walk/run_experiments.py --method mixture --max-feedback 1000 \\
      --teacher-betas 1 1 1 0 -1 --dry-run

  python scripts/walker_walk/run_experiments.py --method zero_last_wk_sgd \\
      --max-feedback 5000 --seeds 12345 23451 --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Sequence


DEFAULT_SEEDS: Sequence[int] = (
    12345,
    23451,
    34512,
    45123,
    51234,
    67890,
    78906,
    89067,
    90678,
    6789,
)

DEFAULT_TEACHER_BETAS: Sequence[int] = (1, 1, 1, -1)

ENTRYPOINTS = {
    "mixture": "train_PEBBLE_mixture.py",
    "zero_last_wk_sgd": "train_PEBBLE_mixture_zero_last_wk_sgd.py",
}

LOG_ROOTS = {
    "mixture": "exp_pebble_mixture",
    "zero_last_wk_sgd": "exp_pebble_mixture_zero_last_wk_sgd",
}


@dataclass(frozen=True)
class FeedbackPreset:
    max_feedback: int
    reward_batch: int
    feed_type: int


# Matches scripts/walker_walk/{500,1000,5000}/ and run_pebble.sh (100).
FEEDBACK_PRESETS: dict[int, FeedbackPreset] = {
    500: FeedbackPreset(max_feedback=500, reward_batch=5, feed_type=6),
    1000: FeedbackPreset(max_feedback=1000, reward_batch=10, feed_type=6),
    2500: FeedbackPreset(max_feedback=2500, reward_batch=25, feed_type=6),
    5000: FeedbackPreset(max_feedback=5000, reward_batch=50, feed_type=6),
    7500: FeedbackPreset(max_feedback=7500, reward_batch=75, feed_type=6),
    10000: FeedbackPreset(max_feedback=5000, reward_batch=50, feed_type=6),
}

WALKER_BASE: Sequence[str] = (
    "env=walker_walk",
    "agent.params.actor_lr=0.0005",
    "agent.params.critic_lr=0.0005",
    "agent.params.batch_size=1024",
    "double_q_critic.params.hidden_dim=1024",
    "double_q_critic.params.hidden_depth=2",
    "diag_gaussian_actor.params.hidden_dim=1024",
    "diag_gaussian_actor.params.hidden_depth=2",
    "num_unsup_steps=9000",
    "num_interact=20000",
    "reward_update=50",
    "reset_update=100",
)

WK_SGD_EXTRA: Sequence[str] = (
    "reward_lr=0.001",
    "alpha_lr=0.0005",
)


def format_betas(betas: Sequence[int]) -> str:
    return "[" + ",".join(str(b) for b in betas) + "]"


def build_cmd(
    *,
    method: str,
    seed: int,
    device: str,
    preset: FeedbackPreset,
    teacher_betas: Sequence[int],
    reward_batch: int | None,
    num_train_steps: int | None,
) -> List[str]:
    rb = preset.reward_batch if reward_batch is None else reward_batch
    steps = 500_000 if num_train_steps is None else num_train_steps
    extra: List[str] = [
        *WALKER_BASE,
        f"num_train_steps={steps}",
        f"reward_batch={rb}",
        f"max_feedback={preset.max_feedback}",
        f"feed_type={preset.feed_type}",
        f"teacher_betas={format_betas(teacher_betas)}",
    ]
    if method == "zero_last_wk_sgd":
        extra.extend(WK_SGD_EXTRA)

    return [
        sys.executable,
        ENTRYPOINTS[method],
        f"seed={seed}",
        f"device={device}",
        *extra,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=sorted(ENTRYPOINTS),
        default="zero_last_wk_sgd",
        help="Training entrypoint (default: mixture)",
    )
    parser.add_argument(
        "--max-feedback",
        type=int,
        choices=sorted(FEEDBACK_PRESETS),
        required=True,
        help="Total preference-query budget (100, 500, 1000, or 5000)",
    )
    parser.add_argument(
        "--teacher-betas",
        nargs="+",
        type=int,
        default=list(DEFAULT_TEACHER_BETAS),
        metavar="B",
        help=f"Expert rationalities (default: {list(DEFAULT_TEACHER_BETAS)})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Random seeds (default: {list(DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--reward-batch",
        type=int,
        default=None,
        help="Override preset reward_batch (default: auto from --max-feedback)",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help="Override num_train_steps (default: 500000)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preset = FEEDBACK_PRESETS[args.max_feedback]
    rb = preset.reward_batch if args.reward_batch is None else args.reward_batch

    print(f"Walker-Walk experiments — {args.method}")
    print(f"  entrypoint     : {ENTRYPOINTS[args.method]}")
    print(f"  max_feedback   : {preset.max_feedback}")
    print(f"  reward_batch   : {rb}")
    print(f"  feed_type      : {preset.feed_type}")
    print(f"  teacher_betas  : {list(args.teacher_betas)}")
    print(f"  seeds          : {args.seeds}")
    print(f"  device         : {args.device}")
    print(f"  logs           : {LOG_ROOTS[args.method]}/walker_walk/...")

    failures: List[tuple[int, int]] = []
    for seed in args.seeds:
        cmd = build_cmd(
            method=args.method,
            seed=seed,
            device=args.device,
            preset=preset,
            teacher_betas=args.teacher_betas,
            reward_batch=args.reward_batch,
            num_train_steps=args.num_train_steps,
        )
        print("\n" + "=" * 88)
        print(f"[walker_walk] seed={seed}  {' '.join(cmd)}")
        print("=" * 88)
        if args.dry_run:
            continue
        result = subprocess.run(
            cmd, preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None
        )
        if result.returncode != 0:
            failures.append((seed, result.returncode))
            print(f"[FAILED] walker_walk seed={seed} (exit {result.returncode})")

    if failures:
        print("\nFailed runs:")
        for seed, code in failures:
            print(f"  seed={seed}  exit={code}")
        return 1

    n = len(args.seeds)
    print(f"\nAll {n} walker_walk {args.method} run(s) completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
