#!/usr/bin/env python3
"""Evaluate initial reward scale of a PyTorch-default reward MLP.

Architecture (default helper):
    in_size -> 128 -> 128 -> 128 -> 1
with LeakyReLU between hidden layers and Tanh on the output.

Emits one table over each in_size under default PyTorch Linear initialization,
judged against the synthetic Stabilized / Random regimes from
    final_ttp_synthetic_compare_detached_w.py.
No training is performed --- only initialization-scale statistics.

Outputs (into --out_dir, default final_results/pytorch_reward_init):
    reward_init_comparison.csv
    README.txt

Usage:
    python eval_pytorch_reward_init.py
    python eval_pytorch_reward_init.py --in_size 48 --n_seeds 200 --use_torch
    python eval_pytorch_reward_init.py --out_dir final_results/pytorch_reward_init --overwrite

If PyTorch is unavailable, the script falls back to a pure-Python simulator
that uses the same nn.Linear default law documented by PyTorch:
    W, b ~ Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Reference baselines from final_ttp_synthetic_compare_detached_w.py
# ---------------------------------------------------------------------------

STABILIZED_STD_R = 0.0
STABILIZED_STD_DELTA = 0.0
RANDOM_STD_R = 1.0
RANDOM_STD_DELTA = math.sqrt(2.0)


@dataclass(frozen=True)
class InitStats:
    std_r: float
    std_delta: float
    mean_abs_r: float

    @property
    def delta_ratio_vs_random(self) -> float:
        return self.std_delta / RANDOM_STD_DELTA


def render_progress(label: str, current: int, total: int, *, width: int = 28) -> None:
    """In-place percent progress bar for the current experiment step."""
    total = max(total, 1)
    current = min(max(current, 0), total)
    pct = 100.0 * current / total
    filled = int(round(width * current / total))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {pct:5.1f}%  {current}/{total}  {label}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


ProgressFn = Optional[Callable[[int, int], None]]


def _seed_progress(progress: ProgressFn, seed_i: int, n_seeds: int) -> None:
    if progress is not None:
        progress(seed_i + 1, n_seeds)


def lrelu(x: float, negative_slope: float = 0.01) -> float:
    return x if x >= 0.0 else negative_slope * x


def pytorch_linear_bound(fan_in: int) -> float:
    """Bound for PyTorch default nn.Linear init (documented equivalent)."""
    return 1.0 / math.sqrt(fan_in)


def sample_pytorch_linear(fan_in: int, fan_out: int, rng: random.Random) -> Tuple[List[List[float]], List[float]]:
    """Sample W, b with PyTorch-default Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    bound = pytorch_linear_bound(fan_in)
    W = [[rng.uniform(-bound, bound) for _ in range(fan_in)] for _ in range(fan_out)]
    b = [rng.uniform(-bound, bound) for _ in range(fan_out)]
    return W, b


def linear_forward(x: Sequence[float], W: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    return [b[j] + sum(x[k] * W[j][k] for k in range(len(x))) for j in range(len(W))]


def build_reward_mlp_torch(in_size: int, hidden: int = 128, n_layers: int = 3):
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc

    layers: List[nn.Module] = []
    d = in_size
    for _ in range(n_layers):
        layers.append(nn.Linear(d, hidden))
        layers.append(nn.LeakyReLU())
        d = hidden
    layers.append(nn.Linear(d, 1))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def evaluate_torch(
    in_size: int,
    *,
    hidden: int,
    n_layers: int,
    n_traj: int,
    n_pairs: int,
    n_seeds: int,
    seed_offset: int,
    progress: ProgressFn = None,
) -> InitStats:
    import torch

    stds_r: List[float] = []
    stds_d: List[float] = []
    means_abs: List[float] = []

    for seed in range(n_seeds):
        torch.manual_seed(seed_offset + seed)
        net = build_reward_mlp_torch(in_size, hidden=hidden, n_layers=n_layers)

        x = torch.randn(n_traj, in_size)
        with torch.no_grad():
            r = net(x).squeeze(-1).cpu().tolist()

        stds_r.append(statistics.pstdev(r))
        means_abs.append(sum(abs(v) for v in r) / len(r))

        rng = random.Random(seed_offset + seed)
        deltas = [r[rng.randrange(n_traj)] - r[rng.randrange(n_traj)] for _ in range(n_pairs)]
        stds_d.append(statistics.pstdev(deltas))
        _seed_progress(progress, seed, n_seeds)

    return InitStats(
        std_r=statistics.mean(stds_r),
        std_delta=statistics.mean(stds_d),
        mean_abs_r=statistics.mean(means_abs),
    )


def evaluate_pure_python(
    in_size: int,
    *,
    hidden: int,
    n_layers: int,
    n_traj: int,
    n_pairs: int,
    n_seeds: int,
    seed_offset: int,
    progress: ProgressFn = None,
) -> InitStats:
    stds_r: List[float] = []
    stds_d: List[float] = []
    means_abs: List[float] = []

    for seed in range(n_seeds):
        rng = random.Random(seed_offset + seed)

        # Trajectory features: i.i.d. N(0, 1), matching the analysis assumption.
        x = [[rng.gauss(0.0, 1.0) for _ in range(in_size)] for _ in range(n_traj)]

        h = x
        d = in_size
        for _ in range(n_layers):
            W, b = sample_pytorch_linear(d, hidden, rng)
            h = [linear_forward(row, W, b) for row in h]
            h = [[lrelu(v) for v in row] for row in h]
            d = hidden

        W_out, b_out = sample_pytorch_linear(d, 1, rng)
        rewards: List[float] = []
        for row in h:
            pre = linear_forward(row, W_out, b_out)[0]
            rewards.append(math.tanh(pre))

        stds_r.append(statistics.pstdev(rewards))
        means_abs.append(sum(abs(v) for v in rewards) / len(rewards))

        deltas = [rewards[rng.randrange(n_traj)] - rewards[rng.randrange(n_traj)] for _ in range(n_pairs)]
        stds_d.append(statistics.pstdev(deltas))
        _seed_progress(progress, seed, n_seeds)

    return InitStats(
        std_r=statistics.mean(stds_r),
        std_delta=statistics.mean(stds_d),
        mean_abs_r=statistics.mean(means_abs),
    )


def verdict_for(stats: InitStats) -> str:
    if stats.delta_ratio_vs_random < 0.1:
        return "Stabilized-like"
    if stats.delta_ratio_vs_random > 0.5:
        return "Random-like"
    return "Intermediate"


def evaluate_one(
    backend: str,
    in_size: int,
    *,
    hidden: int,
    n_layers: int,
    n_traj: int,
    n_pairs: int,
    n_seeds: int,
    seed_offset: int,
    progress: ProgressFn = None,
) -> InitStats:
    kwargs = dict(
        hidden=hidden,
        n_layers=n_layers,
        n_traj=n_traj,
        n_pairs=n_pairs,
        n_seeds=n_seeds,
        seed_offset=seed_offset,
        progress=progress,
    )
    if backend == "torch":
        return evaluate_torch(in_size, **kwargs)
    return evaluate_pure_python(in_size, **kwargs)


def _fmt_float(x: float, digits: int = 6) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def build_comparison_rows(
    backend: str,
    in_sizes: Sequence[int],
    *,
    hidden: int,
    n_layers: int,
    n_traj: int,
    n_pairs: int,
    n_seeds: int,
    seed_offset: int,
) -> List[Dict[str, object]]:
    n_experiments = len(in_sizes)
    rows: List[Dict[str, object]] = []

    for exp_i, in_size in enumerate(in_sizes, start=1):
        label = f"experiment {exp_i}/{n_experiments} | default init | in_size={in_size} | seeds"
        print(f"\nRunning: {label}", flush=True)
        render_progress(label, 0, n_seeds)

        def progress(current: int, total: int, _label: str = label) -> None:
            render_progress(_label, current, total)

        t0 = time.perf_counter()
        stats = evaluate_one(
            backend,
            in_size,
            hidden=hidden,
            n_layers=n_layers,
            n_traj=n_traj,
            n_pairs=n_pairs,
            n_seeds=n_seeds,
            seed_offset=seed_offset,
            progress=progress,
        )
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "init": "default init",
                "in_size": in_size,
                "std_R": stats.std_r,
                "std_Delta_R": stats.std_delta,
                "ratio_vs_random": stats.delta_ratio_vs_random,
                "verdict": verdict_for(stats),
                "runtime_sec": elapsed,
            }
        )
        print(
            f"  finished experiment {exp_i}/{n_experiments}: "
            f"default init in_size={in_size}  std(R)={stats.std_r:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )
    return rows


def print_comparison_table(rows: Sequence[Dict[str, object]]) -> None:
    headers = ("init", "in_size", "std(R)", "std(Delta R)", "ratio vs Random", "verdict")
    formatted: List[Tuple[str, ...]] = []
    for row in rows:
        formatted.append(
            (
                str(row["init"]),
                str(row["in_size"]),
                _fmt_float(float(row["std_R"])),
                _fmt_float(float(row["std_Delta_R"])),
                _fmt_float(float(row["ratio_vs_random"]), 4),
                str(row["verdict"]),
            )
        )

    widths = [len(h) for h in headers]
    for cells in formatted:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print("\n" + "=" * 88)
    print("PyTorch-default reward MLP initialization (table only)")
    print(
        f"Baselines — Stabilized: std(R)={STABILIZED_STD_R}, std(Delta R)={STABILIZED_STD_DELTA}; "
        f"Random: std(R)={RANDOM_STD_R}, std(Delta R)={RANDOM_STD_DELTA:.6f}"
    )
    print("=" * 88)
    print(fmt_row(headers))
    print(fmt_row(tuple("-" * w for w in widths)))
    for cells in formatted:
        print(fmt_row(cells))
    print("=" * 88)


def write_comparison_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = ["init", "in_size", "std_R", "std_Delta_R", "ratio_vs_random", "verdict"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_size", type=int, nargs="+", default=[31, 48, 64], help="Input dimensions to test")
    p.add_argument("--hidden", type=int, default=128, help="Hidden width H")
    p.add_argument("--n_layers", type=int, default=3, help="Number of hidden Linear layers")
    p.add_argument("--n_traj", type=int, default=256, help="Trajectories per seed")
    p.add_argument("--n_pairs", type=int, default=1000, help="Random pairs sampled per seed")
    p.add_argument("--n_seeds", type=int, default=100, help="Random seeds (increase for tighter estimates)")
    p.add_argument("--seed_offset", type=int, default=0, help="Base seed offset")
    p.add_argument("--use_torch", action="store_true", help="Use PyTorch nn.Linear default init")
    p.add_argument(
        "--out_dir",
        type=str,
        default="final_results/pytorch_reward_init",
        help="Output directory for CSV table and README",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace an existing --out_dir")
    p.add_argument("--fast", action="store_true", help="Small quick run (n_seeds=20, n_traj=64)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.fast:
        args.n_seeds = min(args.n_seeds, 20)
        args.n_traj = min(args.n_traj, 64)
        args.n_pairs = min(args.n_pairs, 300)

    if os.path.exists(args.out_dir):
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {args.out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    backend = "torch" if args.use_torch else "pure-python"
    if args.use_torch:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("PyTorch not found; falling back to pure-python simulator.", file=sys.stderr)
            backend = "pure-python"

    print("=" * 72)
    print("PyTorch-default reward MLP initialization scale")
    print(f"Architecture: in_size -> {args.hidden} x{args.n_layers} -> 1 (LeakyReLU, Tanh)")
    print(f"Backend: {backend}")
    print(f"Seeds={args.n_seeds}, trajectories/seed={args.n_traj}, pairs/seed={args.n_pairs}")
    print(f"out_dir={args.out_dir}")
    print("=" * 72)

    t0 = time.perf_counter()
    rows = build_comparison_rows(
        backend,
        args.in_size,
        hidden=args.hidden,
        n_layers=args.n_layers,
        n_traj=args.n_traj,
        n_pairs=args.n_pairs,
        n_seeds=args.n_seeds,
        seed_offset=args.seed_offset,
    )
    print_comparison_table(rows)

    csv_path = os.path.join(args.out_dir, "reward_init_comparison.csv")
    write_comparison_csv(csv_path, rows)

    readme_path = os.path.join(args.out_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "PyTorch-default reward MLP initialization (table only)\n"
            f"runtime_sec={time.perf_counter() - t0:.2f}\n"
            f"backend={backend}\n"
            f"seeds={args.n_seeds}, n_traj={args.n_traj}, n_pairs={args.n_pairs}\n"
            f"in_size={list(args.in_size)}, hidden={args.hidden}, n_layers={args.n_layers}\n"
            "Columns: init, in_size, std_R, std_Delta_R, ratio_vs_random, verdict.\n"
            "Outputs:\n"
            "  reward_init_comparison.csv\n"
        )

    print(f"\nOUT: {args.out_dir}")
    print(f"  {csv_path}")
    print(f"  {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
