#!/usr/bin/env python3
"""Run zero-last / no-w_k TTP for cheetah_run with reward_update=100.

Single seed, same DMC hyperparams as run_zero_last_no_wk_part1/part2
except reward_update=100.

Entrypoint: ``train_PEBBLE_mixture_zero_last_no_wk.py``
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Sequence


SEED: int = 34512

EXTRA: Sequence[str] = (
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
)


def build_cmd(seed: int, device: str) -> List[str]:
    return [
        sys.executable,
        "train_PEBBLE_mixture_zero_last_no_wk.py",
        "env=cheetah_run",
        f"seed={seed}",
        f"device={device}",
        *EXTRA,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = build_cmd(args.seed, args.device)
    print("Zero-last / no-w_k TTP (cheetah_run, reward_update=100)")
    print(f"  device: {args.device}")
    print(f"  seed  : {args.seed}")
    print("\n" + "=" * 88)
    print(f"[cheetah_run] seed={args.seed}  {' '.join(cmd)}")
    print("=" * 88)
    if args.dry_run:
        return 0
    result = subprocess.run(
        cmd, preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None
    )
    if result.returncode != 0:
        print(f"[FAILED] cheetah_run seed={args.seed}")
        return 1
    print("\nCompleted successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
