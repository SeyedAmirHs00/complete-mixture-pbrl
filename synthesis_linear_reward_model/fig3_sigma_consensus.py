"""
Fig. 3 (main_v2): init-scale sweep (no consensus).
Label: fig:synthetic-sigma-consensus

Produces: final_results/synthetic_shared_all/sigma_consensus_sweep/

Example:
  python fig3_sigma_consensus.py --seeds 200 --overwrite
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
    SharedVariant,
    build_k4_configs,
    calibrate_theta_scale,
    run_shared_variant,
)


def ensure_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    os.makedirs(path)


def run_sigma(out_root: str, seeds: int, steps: int, overwrite: bool) -> None:
    out = os.path.join(out_root, "sigma_consensus_sweep")
    ensure_dir(out, overwrite)
    configs = build_k4_configs()
    order = ["3R1A", "2R1A1N", "3R1N", "1R3A"]
    targets = [0.0, 0.05, 0.1, 0.25, 0.5, 1.4, 6.0]
    T = 50
    n_seg = 500
    methods = [
        ("no_cons", 0.0),
    ]
    cal_rng = np.random.default_rng(9201)
    theta_scales = {
        t: calibrate_theta_scale(t, seeds=40, n_seg=n_seg, T=T, d=16, rng=cal_rng) for t in targets
    }

    rows = []
    idx = 0
    for cfg in order:
        betas = configs[cfg]
        for t in targets:
            for mname, ccoef in methods:
                idx += 1
                v = SharedVariant(mname, mname, target_rms=t, consensus_coef=ccoef)
                rho, abar, rms0 = run_shared_variant(
                    betas,
                    v,
                    seeds=seeds,
                    steps=steps,
                    n_seg=n_seg,
                    q=0.0,
                    seed=9201 + 37 * idx,
                    theta_scale=theta_scales[t],
                )
                row = {
                    "config": cfg,
                    "method": mname,
                    "target_rms": t,
                    "step_sigma": t / np.sqrt(2 * T) if t > 0 else 0.0,
                    "init_rms_deltaR": rms0,
                    "correct_branch_rate": float((rho > 0.05).mean()),
                    "flipped_branch_rate": float((rho < -0.05).mean()),
                    "mean_rho": float(rho.mean()),
                }
                rows.append(row)
                print(
                    f"[sigma] {cfg:7s} {mname:10s} rms={t:<4g} correct={row['correct_branch_rate']:.3f} "
                    f"rho={row['mean_rho']:+.3f}"
                )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "sigma_consensus_sweep.csv"), index=False)
    plot_sigma_figure(table, out, T=T, seeds=seeds)

    band_rows = []
    for t in targets:
        n = table[(table.config == "3R1N") & (table.method == "no_cons") & (table.target_rms == t)].iloc[
            0
        ]
        a = table[(table.config == "3R1A") & (table.method == "no_cons") & (table.target_rms == t)].iloc[
            0
        ]
        band_rows.append(
            {
                "target_rms": t,
                "3R1N_no": n.correct_branch_rate,
                "3R1A_no": a.correct_branch_rate,
                "both_no_ge_0.9": (n.correct_branch_rate >= 0.9) and (a.correct_branch_rate >= 0.9),
            }
        )
    pd.DataFrame(band_rows).to_csv(os.path.join(out, "init_band_both_NA.csv"), index=False)
    print(f"OUT sigma: {out}")


def plot_sigma_figure(table: pd.DataFrame, out: str, *, T: int = 50, seeds: int = 200) -> None:
    order = ["3R1A", "2R1A1N", "3R1N", "1R3A"]
    letters = ["a", "b", "c", "d"]
    methods = [
        ("no_cons", 0.0),
    ]
    legend_names = {"no_cons": "no cons."}
    colors = {"no_cons": "#4c72b0"}

    # Combined figure (no in-image letter titles; letters come from LaTeX subcaptions)
    fig, axes = plt.subplots(1, len(order), figsize=(4.0 * len(order), 3.2), sharey=True)
    for ax, cfg, letter in zip(axes, order, letters):
        for mname, _ in methods:
            sub = table[(table["config"] == cfg) & (table["method"] == mname)].sort_values(
                "init_rms_deltaR"
            )
            x = np.maximum(sub["init_rms_deltaR"].to_numpy(), 1e-4)
            ax.plot(
                x,
                sub["correct_branch_rate"],
                "o-",
                color=colors[mname],
                label=legend_names[mname],
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xlabel(r"Init $\mathrm{rms}|\Delta R|$")
        ax.set_title(cfg, fontsize=11)
        ax.grid(True, which="both", ls=":", alpha=0.45)
        if ax is axes[0]:
            ax.set_ylabel("Correct-branch rate")
        if ax is axes[-1]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "shared_sigma_consensus_sweep.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Per-panel PNGs for LaTeX subcaptions under each image
    for cfg, letter in zip(order, letters):
        fig, ax = plt.subplots(figsize=(3.9, 3.2))
        for mname, _ in methods:
            sub = table[(table["config"] == cfg) & (table["method"] == mname)].sort_values(
                "init_rms_deltaR"
            )
            x = np.maximum(sub["init_rms_deltaR"].to_numpy(), 1e-4)
            ax.plot(
                x,
                sub["correct_branch_rate"],
                "o-",
                color=colors[mname],
                label=legend_names[mname],
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xlabel(r"Init $\mathrm{rms}|\Delta R|$")
        ax.set_ylabel("Correct-branch rate")
        ax.grid(True, which="both", ls=":", alpha=0.45)
        if letter == "d":
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out, f"shared_sigma_consensus_{letter}_{cfg}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fig. 4: init/sigma sweep, no consensus (fig:synthetic-sigma-consensus)"
    )
    p.add_argument("--out_root", default="final_results/synthetic_shared_all")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()

    if args.replot:
        out = os.path.join(args.out_root, "sigma_consensus_sweep")
        table = pd.read_csv(os.path.join(out, "sigma_consensus_sweep.csv"))
        plot_sigma_figure(table, out)
        print(f"replot OK: {out}")
        return

    os.makedirs(args.out_root, exist_ok=True)
    run_sigma(args.out_root, args.seeds, args.steps, args.overwrite)


if __name__ == "__main__":
    main()
