#!/usr/bin/env python3
"""Rerun Door-Open group-B seeds with group-A learning rates.

Group B previously ran teacher_betas=[1,1,1,0] with:
  - reward_lr=0.001
  - alpha_lr=0.005

This runner keeps the same six group-B seeds but uses group-A LRs:
  - reward_lr=0.05
  - alpha_lr=0.005

Entrypoint: ``train_PEBBLE_mixture_zero_last_wk_sgd.py``

Examples
--------
  python scripts/door_open/run_zero_last_wk_sgd_1,1,1,0_groupB_lrA.py --dry-run
  python scripts/door_open/run_zero_last_wk_sgd_1,1,1,0_groupB_lrA.py --device cuda
  python scripts/door_open/run_zero_last_wk_sgd_1,1,1,0_groupB_lrA.py --seeds 51234 6789
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Sequence


# Group B seeds from b=[1,1,1,0] door-open runs that used reward_lr=0.001.
DEFAULT_SEEDS: Sequence[int] = (
    51234,
    6789,
    67890,
    78906,
    89067,
    90678,
)

DOOR_OPEN_EXTRA: Sequence[str] = (
    "env=metaworld_door-open-v2",
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
    "teacher_betas=[1,1,1,0]",
    "reward_lr=0.05",
    "alpha_lr=0.005",
)


def build_cmd(seed: int, device: str) -> List[str]:
    return [
        sys.executable,
        "train_PEBBLE_mixture_zero_last_wk_sgd.py",
        f"seed={seed}",
        f"device={device}",
        *DOOR_OPEN_EXTRA,
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
    print("Zero-last / w_k + SGD TTP — door-open group B with group A LRs")
    print(f"  device    : {args.device}")
    print(f"  seeds     : {args.seeds}")
    print("  betas     : [1,1,1,0]")
    print("  reward_lr : 0.05  (group A)")
    print("  alpha_lr  : 0.005 (group A)")
    print("  logs      : exp_pebble_mixture_zero_last_wk_sgd/metaworld_door-open-v2/...")

    failures: List[tuple[int, int]] = []
    for seed in args.seeds:
        cmd = build_cmd(seed, args.device)
        print("\n" + "=" * 88)
        print(f"[door_open] seed={seed}  {' '.join(cmd)}")
        print("=" * 88)
        if args.dry_run:
            continue
        result = subprocess.run(
            cmd, preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None
        )
        if result.returncode != 0:
            failures.append((seed, result.returncode))
            print(f"[FAILED] door_open seed={seed} (exit {result.returncode})")

    if failures:
        print("\nFailed runs:")
        for seed, code in failures:
            print(f"  seed={seed}  exit={code}")
        return 1

    print(f"\nAll {len(args.seeds)} door_open group-B (lr=A) runs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
