#!/usr/bin/env python3
"""Run zero-last / no-w_k TTP for 3 envs on 4 seeds (part 1).

Runs:
  - cheetah_run
  - door_open
  - sweep_into

Seeds:
  34512 45123 51234 67890

Entrypoint: ``train_PEBBLE_mixture_zero_last_no_wk.py``
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class EnvRun:
    name: str
    env: str
    extra: Sequence[str]


SEEDS: Sequence[int] = (34512, 45123, 51234, 67890)

ENV_RUNS: Sequence[EnvRun] = (
    EnvRun(
        "cheetah_run",
        "cheetah_run",
        (
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
        ),
    ),
    EnvRun(
        "door_open",
        "metaworld_door-open-v2",
        (
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
        ),
    ),
    EnvRun(
        "sweep_into",
        "metaworld_sweep-into-v2",
        (
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
        ),
    ),
)


def build_cmd(run: EnvRun, seed: int, device: str) -> List[str]:
    return [
        sys.executable,
        "train_PEBBLE_mixture_zero_last_no_wk.py",
        f"env={run.env}",
        f"seed={seed}",
        f"device={device}",
        *run.extra,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Zero-last / no-w_k TTP (part 1)")
    print(f"  device: {args.device}")
    print(f"  seeds : {list(SEEDS)}")
    print(f"  envs   : {[r.env for r in ENV_RUNS]}")

    failures = []
    for seed in SEEDS:
        for run in ENV_RUNS:
            cmd = build_cmd(run, seed, args.device)
            print("\n" + "=" * 88)
            print(f"[{run.name}] seed={seed}  {' '.join(cmd)}")
            print("=" * 88)
            if args.dry_run:
                continue
            result = subprocess.run(
                cmd, preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None
            )
            if result.returncode != 0:
                failures.append((run.name, seed))
                print(f"[FAILED] {run.name} seed={seed}")

    if failures:
        print("\nFailed runs:")
        for name, seed in failures:
            print(f"  {name} / seed={seed}")
        return 1

    print("\nPart 1 completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
