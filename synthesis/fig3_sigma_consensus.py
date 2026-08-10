"""
Fig. 3 (main_v2): init-kind × consensus sweep (PEBBLE gen_net head).
Label: fig:synthetic-sigma-consensus

Standard = PyTorch-default MLP init; Stabilized = zero last Linear.
Optional consensus majority anchor on the reward path.

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
    run_shared_variant,
    status_print,
)


def ensure_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    os.makedirs(path)


INIT_KINDS = (
    ("stabilized", "Stabilized"),
    ("standard", "Standard"),
)

METHODS = (
    ("no_cons", 0.0),
    ("consensus", 0.5),
)


def run_sigma(out_root: str, seeds: int, steps: int, overwrite: bool) -> None:
    out = os.path.join(out_root, "sigma_consensus_sweep")
    ensure_dir(out, overwrite)
    configs = build_k4_configs()
    order = ["3R1A", "2R1A1N", "3R1N", "1R3A"]
    n_seg = 500

    rows = []
    idx = 0
    for cfg in order:
        betas = configs[cfg]
        for init_kind, init_label in INIT_KINDS:
            for mname, ccoef in METHODS:
                idx += 1
                v = SharedVariant(
                    f"{init_kind}_{mname}",
                    f"{init_label} / {mname}",
                    init_kind=init_kind,
                    consensus_coef=ccoef,
                )
                rho, abar, rms0 = run_shared_variant(
                    betas,
                    v,
                    seeds=seeds,
                    steps=steps,
                    n_seg=n_seg,
                    q=0.0,
                    seed=9201 + 37 * idx,
                    progress_desc=f"sigma {cfg}/{init_kind}/{mname}",
                )
                row = {
                    "config": cfg,
                    "method": mname,
                    "init_kind": init_kind,
                    "init_rms_deltaR": rms0,
                    "correct_branch_rate": float((rho > 0.05).mean()),
                    "flipped_branch_rate": float((rho < -0.05).mean()),
                    "mean_rho": float(rho.mean()),
                }
                rows.append(row)
                status_print(
                    f"[sigma] {cfg:7s} {init_kind:10s} {mname:10s} "
                    f"rms0={rms0:.4g} correct={row['correct_branch_rate']:.3f} "
                    f"rho={row['mean_rho']:+.3f}"
                )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "sigma_consensus_sweep.csv"), index=False)
    plot_sigma_figure(table, out, seeds=seeds)

    band_rows = []
    for init_kind, _ in INIT_KINDS:
        n = table[
            (table.config == "3R1N")
            & (table.method == "no_cons")
            & (table.init_kind == init_kind)
        ].iloc[0]
        a = table[
            (table.config == "3R1A")
            & (table.method == "no_cons")
            & (table.init_kind == init_kind)
        ].iloc[0]
        band_rows.append(
            {
                "init_kind": init_kind,
                "init_rms_deltaR": n.init_rms_deltaR,
                "3R1N_no": n.correct_branch_rate,
                "3R1A_no": a.correct_branch_rate,
                "both_no_ge_0.9": (n.correct_branch_rate >= 0.9)
                and (a.correct_branch_rate >= 0.9),
            }
        )
    pd.DataFrame(band_rows).to_csv(os.path.join(out, "init_band_both_NA.csv"), index=False)
    print(f"OUT sigma: {out}")


def plot_sigma_figure(table: pd.DataFrame, out: str, *, seeds: int = 200) -> None:
    order = ["3R1A", "2R1A1N", "3R1N", "1R3A"]
    letters = ["a", "b", "c", "d"]
    init_order = [k for k, _ in INIT_KINDS]
    method_styles = {
        ("no_cons", "stabilized"): ("#4c72b0", "o"),
        ("no_cons", "standard"): ("#4c72b0", "s"),
        ("consensus", "stabilized"): ("#dd8452", "o"),
        ("consensus", "standard"): ("#dd8452", "s"),
    }
    legend_names = {
        ("no_cons", "stabilized"): "no cons. / Stab.",
        ("no_cons", "standard"): "no cons. / Std.",
        ("consensus", "stabilized"): "cons. / Stab.",
        ("consensus", "standard"): "cons. / Std.",
    }

    x_pos = {k: i for i, k in enumerate(init_order)}

    fig, axes = plt.subplots(1, len(order), figsize=(4.0 * len(order), 3.2), sharey=True)
    for ax, cfg, letter in zip(axes, order, letters):
        for mname, _ in METHODS:
            for init_kind in init_order:
                sub = table[
                    (table["config"] == cfg)
                    & (table["method"] == mname)
                    & (table["init_kind"] == init_kind)
                ]
                if sub.empty:
                    continue
                color, marker = method_styles[(mname, init_kind)]
                ax.plot(
                    [x_pos[init_kind]],
                    [sub.iloc[0]["correct_branch_rate"]],
                    marker=marker,
                    color=color,
                    linestyle="None",
                    markersize=9,
                    label=legend_names[(mname, init_kind)],
                )
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(["Stab.", "Std."])
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Init kind")
        ax.set_title(cfg, fontsize=11)
        ax.grid(True, axis="y", ls=":", alpha=0.45)
        if ax is axes[0]:
            ax.set_ylabel("Correct-branch rate")
        if ax is axes[-1]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "shared_sigma_consensus_sweep.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    for cfg, letter in zip(order, letters):
        fig, ax = plt.subplots(figsize=(3.9, 3.2))
        for mname, _ in METHODS:
            for init_kind in init_order:
                sub = table[
                    (table["config"] == cfg)
                    & (table["method"] == mname)
                    & (table["init_kind"] == init_kind)
                ]
                if sub.empty:
                    continue
                color, marker = method_styles[(mname, init_kind)]
                ax.plot(
                    [x_pos[init_kind]],
                    [sub.iloc[0]["correct_branch_rate"]],
                    marker=marker,
                    color=color,
                    linestyle="None",
                    markersize=9,
                    label=legend_names[(mname, init_kind)],
                )
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(["Stab.", "Std."])
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Init kind")
        ax.set_ylabel("Correct-branch rate")
        ax.grid(True, axis="y", ls=":", alpha=0.45)
        if letter == "d":
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out, f"shared_sigma_consensus_{letter}_{cfg}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fig. 3: init-kind × consensus sweep (fig:synthetic-sigma-consensus)"
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
