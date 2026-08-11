#!/usr/bin/env python3
"""Monte Carlo check: PyTorch-default reward MLP has near-zero init scale.

Reproduces the claim in Appendix (app:pytorch-deltaR) of main_v2.tex:

    A Monte Carlo check with the exact PyTorch uniform law on
    d_in in {17, 24, 39, 78} yields std(r_hat) ≈ 0.025.

Architecture matches production PEBBLE ``gen_net`` (default H=256, 3 layers):
    d_in -> Linear(H) -> LeakyReLU -> ... -> Linear(1) -> Tanh

Inputs are i.i.d. N(0, 1). Weights use PyTorch's default nn.Linear init
(equivalent to Uniform(-1/sqrt(d_in), 1/sqrt(d_in))).

Outputs (default: final_results/pytorch_reward_init/):
    per_din_summary.csv   — mean±std of std(r), std(Δr) for each d_in
    per_seed.csv          — raw std(r), std(Δr) for every (d_in, seed)
    summary.json          — overall means + T=50 preference extrapolation
    README.txt            — short human-readable summary

Example:
    python exp_pytorch_reward_init.py --overwrite
    python exp_pytorch_reward_init.py --n_seeds 100 --n_traj 256 --overwrite
    python exp_pytorch_reward_init.py --fast --overwrite
"""

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import time
from typing import Dict, List, Sequence, Tuple

import torch

from synthetic_shared_core import (
    DEFAULT_HIDDEN,
    DEFAULT_N_LAYERS,
    build_reward_mlp,
    get_device,
    progress_range,
    status_print,
)


# DM Control / Meta-World observation sizes used in the paper.
DEFAULT_DIN = (17, 24, 39, 78)
DEFAULT_OUT_DIR = os.path.join("final_results", "pytorch_reward_init")


@torch.no_grad()
def eval_one_seed(
    d_in: int,
    *,
    n_traj: int,
    seed: int,
    hidden: int,
    n_hidden: int,
    device=None,
) -> Tuple[float, float]:
    """Return (std of step rewards, std of pairwise differences)."""
    device = get_device(device)
    torch.manual_seed(seed)
    net = build_reward_mlp(d_in, hidden=hidden, n_layers=n_hidden).to(device)
    x = torch.randn(n_traj, d_in, device=device)
    r = net(x).squeeze(-1)  # (n_traj,)

    std_r = float(r.std(unbiased=False).item())
    # All unordered pairs would be O(n^2); use random pairs with replacement.
    i = torch.randint(0, n_traj, (n_traj,), device=device)
    j = torch.randint(0, n_traj, (n_traj,), device=device)
    delta = r[i] - r[j]
    std_delta = float(delta.std(unbiased=False).item())
    return std_r, std_delta


def run_din(
    d_in: int,
    *,
    n_seeds: int,
    n_traj: int,
    seed_offset: int,
    hidden: int,
    n_hidden: int,
    device=None,
) -> Tuple[List[float], List[float]]:
    """Return per-seed lists of std(r) and std(Δr)."""
    stds_r: List[float] = []
    stds_d: List[float] = []
    for s in progress_range(n_seeds, desc=f"d_in={d_in}", leave=False):
        sr, sd = eval_one_seed(
            d_in,
            n_traj=n_traj,
            seed=seed_offset + s,
            hidden=hidden,
            n_hidden=n_hidden,
            device=device,
        )
        stds_r.append(sr)
        stds_d.append(sd)
    return stds_r, stds_d


def sigmoid(u: float) -> float:
    return 1.0 / (1.0 + math.exp(-u))


def ensure_out_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d_in", type=int, nargs="+", default=list(DEFAULT_DIN), help="Input widths")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--n_hidden", type=int, default=DEFAULT_N_LAYERS, help="Number of Linear->LReLU hidden blocks")
    p.add_argument("--n_seeds", type=int, default=100)
    p.add_argument("--n_traj", type=int, default=256, help="Samples (trajectories / steps) per seed")
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--T", type=int, default=50, help="Segment length for extrapolated std(ΔR)")
    p.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory under final_results/",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace an existing --out_dir")
    p.add_argument("--fast", action="store_true", help="Quick run: 20 seeds, 64 samples")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.fast:
        args.n_seeds = min(args.n_seeds, 20)
        args.n_traj = min(args.n_traj, 64)

    ensure_out_dir(args.out_dir, args.overwrite)
    t0 = time.perf_counter()
    device = get_device()

    print("=" * 78)
    print("PyTorch-default reward MLP init scale (Monte Carlo)")
    print(f"arch: d_in -> ({args.hidden} LReLU) x{args.n_hidden} -> 1 Tanh")
    print(f"seeds={args.n_seeds}, samples/seed={args.n_traj}, d_in={list(args.d_in)}")
    print(f"device={device}, out_dir={args.out_dir}")
    print("=" * 78)
    print(f"{'d_in':>6}  {'mean std(r)':>12}  {'±':>6}  {'mean std(Δr)':>12}  {'±':>6}")
    print("-" * 78)

    per_seed_rows: List[Dict[str, object]] = []
    per_din_rows: List[Dict[str, object]] = []
    all_std_r: List[float] = []
    all_std_d: List[float] = []

    for d_in in args.d_in:
        stds_r, stds_d = run_din(
            d_in,
            n_seeds=args.n_seeds,
            n_traj=args.n_traj,
            seed_offset=args.seed_offset,
            hidden=args.hidden,
            n_hidden=args.n_hidden,
            device=device,
        )
        mean_r = statistics.mean(stds_r)
        sd_r = statistics.pstdev(stds_r) if len(stds_r) > 1 else 0.0
        mean_d = statistics.mean(stds_d)
        sd_d = statistics.pstdev(stds_d) if len(stds_d) > 1 else 0.0

        all_std_r.append(mean_r)
        all_std_d.append(mean_d)
        print(f"{d_in:6d}  {mean_r:12.6f}  {sd_r:6.4f}  {mean_d:12.6f}  {sd_d:6.4f}")
        status_print(f"d_in={d_in} mean_std(r)={mean_r:.4f} mean_std(Δr)={mean_d:.4f}")

        per_din_rows.append(
            {
                "d_in": d_in,
                "mean_std_r": mean_r,
                "std_of_std_r": sd_r,
                "mean_std_delta_r": mean_d,
                "std_of_std_delta_r": sd_d,
                "n_seeds": args.n_seeds,
                "n_traj": args.n_traj,
            }
        )
        for s, (sr, sd) in enumerate(zip(stds_r, stds_d)):
            per_seed_rows.append(
                {
                    "d_in": d_in,
                    "seed": args.seed_offset + s,
                    "std_r": sr,
                    "std_delta_r": sd,
                }
            )

    overall_r = statistics.mean(all_std_r)
    overall_d = statistics.mean(all_std_d)
    # Segment return gap under i.i.d. steps: std(ΔR) = std(r) * sqrt(2T)
    std_delta_R = overall_r * math.sqrt(2.0 * args.T)
    pref = sigmoid(std_delta_R)
    elapsed = time.perf_counter() - t0

    print("-" * 78)
    print(f"overall mean std(r̂_θ)     ≈ {overall_r:.4f}   (paper: ≈ 0.025)")
    print(f"overall mean std(Δr̂)      ≈ {overall_d:.4f}   (≈ √2 · std(r))")
    print(f"extrapolated std(ΔR), T={args.T}: {std_delta_R:.4f}")
    print(f"σ(std(ΔR))                  ≈ {pref:.4f}   (paper T=50: ≈ 0.56)")
    print("=" * 78)

    summary = {
        "architecture": f"d_in -> ({args.hidden} LReLU) x{args.n_hidden} -> 1 Tanh",
        "d_in": list(args.d_in),
        "hidden": args.hidden,
        "n_hidden": args.n_hidden,
        "n_seeds": args.n_seeds,
        "n_traj": args.n_traj,
        "seed_offset": args.seed_offset,
        "T": args.T,
        "overall_mean_std_r": overall_r,
        "overall_mean_std_delta_r": overall_d,
        "extrapolated_std_Delta_R": std_delta_R,
        "sigma_of_std_Delta_R": pref,
        "paper_std_r": 0.025,
        "paper_sigma_T50": 0.56,
        "runtime_sec": elapsed,
        "torch_version": torch.__version__,
    }

    per_din_path = os.path.join(args.out_dir, "per_din_summary.csv")
    per_seed_path = os.path.join(args.out_dir, "per_seed.csv")
    summary_path = os.path.join(args.out_dir, "summary.json")
    readme_path = os.path.join(args.out_dir, "README.txt")

    write_csv(
        per_din_path,
        [
            "d_in",
            "mean_std_r",
            "std_of_std_r",
            "mean_std_delta_r",
            "std_of_std_delta_r",
            "n_seeds",
            "n_traj",
        ],
        per_din_rows,
    )
    write_csv(per_seed_path, ["d_in", "seed", "std_r", "std_delta_r"], per_seed_rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("PyTorch-default reward MLP initialization scale (Monte Carlo)\n")
        f.write(f"architecture: {summary['architecture']}\n")
        f.write(f"d_in={summary['d_in']}, seeds={args.n_seeds}, n_traj={args.n_traj}\n")
        f.write(f"overall mean std(r) ≈ {overall_r:.6f}  (paper ≈ 0.025)\n")
        f.write(f"overall mean std(Δr) ≈ {overall_d:.6f}\n")
        f.write(f"extrapolated std(ΔR), T={args.T}: {std_delta_R:.6f}\n")
        f.write(f"σ(std(ΔR)) ≈ {pref:.6f}  (paper T=50 ≈ 0.56)\n")
        f.write(f"runtime_sec={elapsed:.2f}, torch={torch.__version__}\n")
        f.write("\nFiles:\n")
        f.write("  per_din_summary.csv\n")
        f.write("  per_seed.csv\n")
        f.write("  summary.json\n")

    print(f"\nSaved to {args.out_dir}/")
    print(f"  {per_din_path}")
    print(f"  {per_seed_path}")
    print(f"  {summary_path}")
    print(f"  {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
