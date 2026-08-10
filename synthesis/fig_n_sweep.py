"""
Sensitivity to number of trajectories n (n_seg) under disjoint feedback (q=0).

Shared PEBBLE gen_net head; Standard vs Stabilized on 3R1N / 3R1A / 1R3A.
Block size is n // K (K=4), so each expert's private pairs are drawn within a
block of that size. Larger n => larger private pools and denser coverage of R*.

Produces: final_results/synthetic_n_sweep/

Example:
  python fig_n_sweep.py --seeds 120 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthetic_shared_core import (
    SHARED_BRANCH_VARIANTS,
    build_k4_configs,
    run_shared_variant,
)


def ensure_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    os.makedirs(path)


def plot_n_figure(df: pd.DataFrame, out_root: str, ns) -> None:
    colors = {"standard": "#4c72b0", "stabilized": "#dd8452"}
    markers = {"standard": "o", "stabilized": "s"}
    order = ["3R1N", "3R1A", "1R3A"]
    letters = ["a", "b", "c"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
    for ax, cfg in zip(axes, order):
        for v in SHARED_BRANCH_VARIANTS:
            sub = df[(df.config == cfg) & (df.variant == v.name)].sort_values("n")
            ax.plot(
                sub["n"],
                100.0 * sub["correct_branch_rate"],
                color=colors[v.name],
                marker=markers[v.name],
                lw=2.0,
                ms=7,
                label=v.label,
            )
        ax.axhline(50, color="gray", ls=":", lw=0.9, alpha=0.8)
        ax.set_title(cfg, fontsize=11)
        ax.set_xlabel(r"Number of trajectories $n$")
        ax.set_xscale("log")
        ax.set_xticks(list(ns))
        ax.set_xticklabels([str(n) for n in ns])
        ax.grid(True, ls=":", alpha=0.45)
        ax.set_ylim(-2, 105)
    axes[0].set_ylabel("Correct-branch rate (%)")
    axes[-1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "n_sweep.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    for cfg, letter in zip(order, letters):
        fig, ax = plt.subplots(figsize=(3.9, 3.2))
        for v in SHARED_BRANCH_VARIANTS:
            sub = df[(df.config == cfg) & (df.variant == v.name)].sort_values("n")
            ax.plot(
                sub["n"],
                100.0 * sub["correct_branch_rate"],
                color=colors[v.name],
                marker=markers[v.name],
                lw=2.0,
                ms=7,
                label=v.label,
            )
        ax.axhline(50, color="gray", ls=":", lw=0.9, alpha=0.8)
        ax.set_xlabel(r"Number of trajectories $n$")
        ax.set_ylabel("Correct-branch rate (%)")
        ax.set_xscale("log")
        ax.set_xticks(list(ns))
        ax.set_xticklabels([str(n) for n in ns])
        ax.grid(True, ls=":", alpha=0.45)
        ax.set_ylim(-2, 105)
        if letter == "c":
            ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_root, f"n_sweep_{letter}_{cfg}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Sensitivity to n (trajectory count), q=0")
    p.add_argument("--out_root", default="final_results/synthetic_n_sweep")
    p.add_argument("--seeds", type=int, default=120)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument(
        "--ns",
        type=int,
        nargs="+",
        # Cover old default (48), mid range, and current default (500).
        # Min n=16 => block size 4 for K=4 (sample_expert_pairs requires >=2).
        default=[16, 32, 48, 64, 100, 200, 500, 1000],
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ensure_dir(args.out_root, args.overwrite)
    configs = build_k4_configs()
    order = ["3R1N", "3R1A", "1R3A"]
    rows = []
    idx = 0

    for n in args.ns:
        if n // 4 < 2:
            raise ValueError(f"n={n} too small for K=4 disjoint blocks (need n//4 >= 2)")
        for cfg in order:
            betas = configs[cfg]
            for v in SHARED_BRANCH_VARIANTS:
                idx += 1
                rho, abar, rms0 = run_shared_variant(
                    betas,
                    v,
                    seeds=args.seeds,
                    steps=args.steps,
                    n_seg=n,
                    pairs=args.pairs,
                    q=args.q,
                    seed=9017 + 31 * idx,
                )
                correct = float((rho > 0.05).mean())
                flipped = float((rho < -0.05).mean())
                rows.append(
                    {
                        "n": n,
                        "block_size": n // 4,
                        "q": args.q,
                        "config": cfg,
                        "variant": v.name,
                        "label": v.label,
                        "correct_branch_rate": correct,
                        "flipped_branch_rate": flipped,
                        "mean_rho": float(rho.mean()),
                        "init_rms": rms0,
                        "mean_abar_R": float(abar[:, 0].mean())
                        if cfg == "1R3A"
                        else float(abar[:, :3].mean()),
                        "mean_abar_A": float(abar[:, -1].mean())
                        if "A" in cfg
                        else np.nan,
                    }
                )
                print(
                    f"n={n:4d} blk={n // 4:3d} {cfg:6s} {v.name:10s} "
                    f"correct={correct:.3f} rho={rho.mean():+.3f} rms0={rms0:.3f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_root, "n_sweep.csv"), index=False)
    plot_n_figure(df, args.out_root, args.ns)
    print(f"OUT: {os.path.join(args.out_root, 'n_sweep.png')}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--replot":
        out = sys.argv[2] if len(sys.argv) > 2 else "final_results/synthetic_n_sweep"
        df = pd.read_csv(os.path.join(out, "n_sweep.csv"))
        ns = sorted(df["n"].unique().tolist())
        plot_n_figure(df, out, ns)
        print(f"replot OK: {out}")
    else:
        main()
