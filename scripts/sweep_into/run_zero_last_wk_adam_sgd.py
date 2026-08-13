#!/usr/bin/env python3
"""Run zero-last TTP with Adam reward + SGD α + two-path w_k on Sweep-Into.

Entrypoint: ``train_PEBBLE_mixture_zero_last_wk_adam_sgd.py``
  - Stabilized init (zero last Linear of reward MLP)
  - Reward ensemble: Adam lr=0.0003
  - Trust α: SGD lr=0.005
  - Two-path w_k loss (fig6)

Examples
--------
  python scripts/sweep_into/run_zero_last_wk_adam_sgd.py --dry-run
  python scripts/sweep_into/run_zero_last_wk_adam_sgd.py --device cuda
  python scripts/sweep_into/run_zero_last_wk_adam_sgd.py --seeds 12345 23451
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Sequence


DEFAULT_SEEDS: Sequence[int] = (
    34512,
    78906,
    12345,
    23451,
    45123,
    51234,
    67890,
    89067,
    90678,
    6789,
)

SWEEP_INTO_EXTRA: Sequence[str] = (
    "env=metaworld_sweep-into-v2",
    "agent.params.actor_lr=0.0003",
    "agent.params.critic_lr=0.0003",
    "activation=tanh",
    "num_unsup_steps=9000",
    "num_train_steps=1000000",
    "agent.params.batch_size=512",
    "double_q_critic.params.hidden_dim=256",
    "double_q_critic.params.hidden_depth=3",
    "diag_gaussian_actor.params.hidden_dim=256",
    "diag_gaussian_actor.params.hidden_depth=3",
    "reward_update=10",
    "num_interact=5000",
    "max_feedback=40000",
    "reward_batch=50",
    "feed_type=6",
    "teacher_betas=[1,1,1,-1]",
    "reward_lr=0.0003",
    "alpha_lr=0.005",
)


def build_cmd(seed: int, device: str) -> List[str]:
    return [
        sys.executable,
        "train_PEBBLE_mixture_zero_last_wk_adam_sgd.py",
        f"seed={seed}",
        f"device={device}",
        *SWEEP_INTO_EXTRA,
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
    print("Zero-last / w_k + Adam(reward)/SGD(α) TTP — metaworld_sweep-into-v2")
    print(f"  device : {args.device}")
    print(f"  seeds  : {args.seeds}")
    print("  logs   : exp_pebble_mixture_zero_last_wk_adam_sgd/metaworld_sweep-into-v2/...")

    failures: List[tuple[int, int]] = []
    for seed in args.seeds:
        cmd = build_cmd(seed, args.device)
        print("\n" + "=" * 88)
        print(f"[sweep_into] seed={seed}  {' '.join(cmd)}")
        print("=" * 88)
        if args.dry_run:
            continue
        result = subprocess.run(
            cmd, preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None
        )
        if result.returncode != 0:
            failures.append((seed, result.returncode))
            print(f"[FAILED] sweep_into seed={seed} (exit {result.returncode})")

    if failures:
        print("\nFailed runs:")
        for seed, code in failures:
            print(f"  seed={seed}  exit={code}")
        return 1

    print(f"\nAll {len(args.seeds)} sweep_into wk_adam_sgd runs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
