#!/usr/bin/env python3
"""Rerun Sweep-Into zero-last wk_sgd seeds that were launched with wrong reward_lr.

Fixes seeds 89067 and 90678 for teacher_betas=[1,1,1,0], which previously ran
with reward_lr=0.001. This runner uses the correct settings:
  - reward_lr=0.05
  - alpha_lr=0.005

Entrypoint: ``train_PEBBLE_mixture_zero_last_wk_sgd.py``

Ctrl+C shows the current experiment details and asks for confirmation before
killing the training child and stopping the runner.

Examples
--------
  python scripts/sweep_into/run_zero_last_wk_sgd_1,1,1,0.py --dry-run
  python scripts/sweep_into/run_zero_last_wk_sgd_1,1,1,0.py --device cuda
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from typing import Dict, List, Optional, Sequence


DEFAULT_SEEDS: Sequence[int] = (
    89067,
    90678,
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
        *SWEEP_INTO_EXTRA,
    ]


def _overrides_dict(seed: int, device: str) -> Dict[str, str]:
    values = {"seed": str(seed), "device": device}
    for item in SWEEP_INTO_EXTRA:
        if "=" in item:
            key, value = item.split("=", 1)
            values[key] = value
    return values


def _format_experiment_details(seed: int, device: str, remaining: Sequence[int]) -> str:
    values = _overrides_dict(seed, device)
    keys = (
        "env",
        "seed",
        "device",
        "teacher_betas",
        "reward_lr",
        "alpha_lr",
        "max_feedback",
        "feed_type",
        "reward_batch",
        "num_train_steps",
        "num_interact",
        "reward_update",
    )
    lines = ["Experiment details", "  entrypoint     : train_PEBBLE_mixture_zero_last_wk_sgd.py"]
    for key in keys:
        if key in values:
            lines.append(f"  {key:14s}: {values[key]}")
    lines.append(f"  remaining_seeds: {list(remaining)}")
    return "\n".join(lines)


class InterruptGuard:
    """Confirm Ctrl+C at the runner level; child runs in its own process group."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.seed: Optional[int] = None
        self.device: str = "cuda"
        self.remaining: Sequence[int] = ()
        self.confirming = False
        self.cancel_requested = False

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handler)

    def set_current(
        self,
        proc: subprocess.Popen,
        seed: int,
        device: str,
        remaining: Sequence[int],
    ) -> None:
        self.proc = proc
        self.seed = seed
        self.device = device
        self.remaining = remaining

    def clear_current(self) -> None:
        self.proc = None
        self.seed = None

    def _kill_child(self) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    def _handler(self, signum, frame) -> None:
        if self.confirming:
            print("\nSecond Ctrl+C received — forcing exit.", flush=True)
            self._kill_child()
            os._exit(130)

        self.confirming = True
        try:
            print("\n" + "=" * 72, flush=True)
            print("Ctrl+C received — interrupt requested.", flush=True)
            if self.seed is not None:
                print(
                    _format_experiment_details(self.seed, self.device, self.remaining),
                    flush=True,
                )
            else:
                print("  (no active training child)", flush=True)
            print("=" * 72, flush=True)
            try:
                answer = input("Cancel this experiment (and stop the runner)? [y/N]: ")
                answer = answer.strip().lower()
            except EOFError:
                answer = "y"
            if answer in ("y", "yes"):
                print("Cancelling experiment and stopping runner.", flush=True)
                self.cancel_requested = True
                self._kill_child()
            else:
                print("Continuing current experiment...", flush=True)
        finally:
            self.confirming = False


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
    print("Zero-last / w_k + SGD TTP — metaworld_sweep-into-v2 (rerun wrong reward_lr)")
    print(f"  device    : {args.device}")
    print(f"  seeds     : {args.seeds}")
    print("  betas     : [1,1,1,0]")
    print("  reward_lr : 0.05")
    print("  alpha_lr  : 0.005")
    print("  logs      : exp_pebble_mixture_zero_last_wk_sgd/metaworld_sweep-into-v2/...")

    guard = InterruptGuard()
    guard.install()

    failures: List[tuple[int, int]] = []
    for idx, seed in enumerate(args.seeds):
        cmd = build_cmd(seed, args.device)
        print("\n" + "=" * 88)
        print(f"[sweep_into] seed={seed}  {' '.join(cmd)}")
        print("=" * 88)
        if args.dry_run:
            continue

        remaining = args.seeds[idx:]
        proc = subprocess.Popen(
            cmd,
            preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None,
        )
        guard.set_current(proc, seed, args.device, remaining)
        returncode = proc.wait()
        guard.clear_current()

        if guard.cancel_requested:
            print(f"\nRunner cancelled during seed={seed}.")
            return 130

        if returncode != 0:
            failures.append((seed, returncode))
            print(f"[FAILED] sweep_into seed={seed} (exit {returncode})")

    if failures:
        print("\nFailed runs:")
        for seed, code in failures:
            print(f"  seed={seed}  exit={code}")
        return 1

    print(f"\nAll {len(args.seeds)} sweep_into wk_sgd runs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
