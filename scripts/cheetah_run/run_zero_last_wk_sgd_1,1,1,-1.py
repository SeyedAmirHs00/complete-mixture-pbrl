#!/usr/bin/env python3
"""Run zero-last TTP with SGD + two-path w_k on Cheetah-Run (10 seeds).

Entrypoint: ``train_PEBBLE_mixture_zero_last_wk_sgd.py``
  - Stabilized init (zero last Linear of reward MLP)
  - SGD reward optimizer: network lr=0.001, alpha lr=0.005
  - Two-path w_k loss (fig6): detached w_k on reward CE; trust path on α

Hyperparameters match ``scripts/cheetah_run/run_pebble_mixture_b[1,1,1,-1].sh``
and ``scripts/run_zero_last_no_wk.py`` (cheetah_run block).

Examples
--------
  python scripts/cheetah_run/run_zero_last_wk_sgd_1,1,1,-1.py --dry-run
  python scripts/cheetah_run/run_zero_last_wk_sgd_1,1,1,-1.py --device cuda
  python scripts/cheetah_run/run_zero_last_wk_sgd_1,1,1,-1.py --seeds 12345 23451
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
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

CHEETAH_EXTRA: Sequence[str] = (
    "env=cheetah_run",
    "agent.params.actor_lr=0.0005",
    "agent.params.critic_lr=0.0005",
    "num_train_steps=1000000",
    "agent.params.batch_size=1024",
    "double_q_critic.params.hidden_dim=1024",
    "double_q_critic.params.hidden_depth=2",
    "diag_gaussian_actor.params.hidden_dim=1024",
    "diag_gaussian_actor.params.hidden_depth=2",
    "num_unsup_steps=9000",
    "reward_batch=100",
    "num_interact=20000",
    "max_feedback=4000",
    "feed_type=6",
    "reward_update=50",
    "reset_update=100",
    "teacher_betas=[1,1,1,-1]",
    "reward_lr=0.001",
    "alpha_lr=0.0005",
)


def build_cmd(seed: int, device: str) -> List[str]:
    return [
        sys.executable,
        "train_PEBBLE_mixture_zero_last_wk_sgd.py",
        f"seed={seed}",
        f"device={device}",
        *CHEETAH_EXTRA,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Random seeds (default: {list(DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Zero-last / w_k + SGD TTP — cheetah_run  β=[1,1,1,-1]")
    print(f"  device : {args.device}")
    print(f"  seeds  : {args.seeds}")
    print("  logs   : exp_pebble_mixture_zero_last_wk_sgd/cheetah_run/...")

    failures: List[tuple[int, int]] = []
    for seed in args.seeds:
        cmd = build_cmd(seed, args.device)
        print("\n" + "=" * 88)
        print(f"[cheetah_run] seed={seed}  {' '.join(cmd)}")
        print("=" * 88)
        if args.dry_run:
            continue
        result = subprocess.run(
            cmd, preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None
        )
        if result.returncode != 0:
            failures.append((seed, result.returncode))
            print(f"[FAILED] cheetah_run seed={seed} (exit {result.returncode})")

    if failures:
        print("\nFailed runs:")
        for seed, code in failures:
            print(f"  seed={seed}  exit={code}")
        return 1

    print(f"\nAll {len(args.seeds)} cheetah_run wk_sgd runs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
