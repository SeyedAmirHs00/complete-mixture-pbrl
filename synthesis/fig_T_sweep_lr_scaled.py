"""
Trajectory-length (T) sweep with lr_theta scaled as 1/T so sum-over-T
reward gradients keep a comparable effective step size.

Baseline: T0=50, lr0=0.05  =>  lr(T) = lr0 * (T0 / T)

Produces: final_results/synthetic_T_sweep_lr_scaled/

Example:
  python fig_T_sweep_lr_scaled.py --seeds 120 --overwrite
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
    status_print,
)

T0 = 50
LR0 = 0.05


def lr_theta_for_T(T: int) -> float:
    return LR0 * (T0 / float(T))


def ensure_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    os.makedirs(path)


def main() -> None:
    p = argparse.ArgumentParser(description="T sweep with lr_theta ∝ 1/T")
    p.add_argument("--out_root", default="final_results/synthetic_T_sweep_lr_scaled")
    p.add_argument("--seeds", type=int, default=120)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument(
        "--Ts",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 200],
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ensure_dir(args.out_root, args.overwrite)
    configs = build_k4_configs()
    order = ["3R1N", "3R1A", "1R3A"]
    rows = []
    idx = 0

    for T in args.Ts:
        lr = lr_theta_for_T(T)
        for cfg in order:
            betas = configs[cfg]
            for v in SHARED_BRANCH_VARIANTS:
                idx += 1
                rho, abar, rms0 = run_shared_variant(
                    betas,
                    v,
                    seeds=args.seeds,
                    steps=args.steps,
                    T=T,
                    lr_theta=lr,
                    seed=9017 + 31 * idx,
                    progress_desc=f"T={T} {cfg}/{v.name}",
                )
                correct = float((rho > 0.05).mean())
                flipped = float((rho < -0.05).mean())
                rows.append(
                    {
                        "T": T,
                        "lr_theta": lr,
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
                status_print(
                    f"T={T:3d} lr={lr:.4f} {cfg:6s} {v.name:10s} "
                    f"correct={correct:.3f} rho={rho.mean():+.3f} rms0={rms0:.3f}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_root, "T_sweep_lr_scaled.csv"), index=False)

    plot_T_figure(df, args.out_root, args.Ts)
    print(f"OUT: {os.path.join(args.out_root, 'T_sweep_lr_scaled.png')}")


def plot_T_figure(df: pd.DataFrame, out_root: str, Ts) -> None:
    colors = {"standard": "#4c72b0", "stabilized": "#dd8452"}
    markers = {"standard": "o", "stabilized": "s"}
    order = ["3R1N", "3R1A", "1R3A"]
    letters = ["a", "b", "c"]

    # Combined figure (mixture names only; letters come from LaTeX subcaptions)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
    for ax, cfg in zip(axes, order):
        for v in SHARED_BRANCH_VARIANTS:
            sub = df[(df.config == cfg) & (df.variant == v.name)].sort_values("T")
            ax.plot(
                sub["T"],
                100.0 * sub["correct_branch_rate"],
                color=colors[v.name],
                marker=markers[v.name],
                lw=2.0,
                ms=7,
                label=v.label,
            )
        ax.axhline(50, color="gray", ls=":", lw=0.9, alpha=0.8)
        ax.set_title(cfg, fontsize=11)
        ax.set_xlabel(r"Trajectory length $T$")
        ax.set_xscale("log")
        ax.set_xticks(list(Ts))
        ax.set_xticklabels([str(t) for t in Ts])
        ax.grid(True, ls=":", alpha=0.45)
        ax.set_ylim(-2, 105)
    axes[0].set_ylabel("Correct-branch rate (%)")
    axes[-1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "T_sweep_lr_scaled.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Per-panel PNGs for LaTeX subcaptions under each image
    for cfg, letter in zip(order, letters):
        fig, ax = plt.subplots(figsize=(3.9, 3.2))
        for v in SHARED_BRANCH_VARIANTS:
            sub = df[(df.config == cfg) & (df.variant == v.name)].sort_values("T")
            ax.plot(
                sub["T"],
                100.0 * sub["correct_branch_rate"],
                color=colors[v.name],
                marker=markers[v.name],
                lw=2.0,
                ms=7,
                label=v.label,
            )
        ax.axhline(50, color="gray", ls=":", lw=0.9, alpha=0.8)
        ax.set_xlabel(r"Trajectory length $T$")
        ax.set_ylabel("Correct-branch rate (%)")
        ax.set_xscale("log")
        ax.set_xticks(list(Ts))
        ax.set_xticklabels([str(t) for t in Ts])
        ax.grid(True, ls=":", alpha=0.45)
        ax.set_ylim(-2, 105)
        if letter == "c":
            ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_root, f"T_sweep_lr_scaled_{letter}_{cfg}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--replot":
        out = sys.argv[2] if len(sys.argv) > 2 else "final_results/synthetic_T_sweep_lr_scaled"
        df = pd.read_csv(os.path.join(out, "T_sweep_lr_scaled.csv"))
        Ts = sorted(df["T"].unique().tolist())
        plot_T_figure(df, out, Ts)
        print(f"replot OK: {out}")
    else:
        main()
