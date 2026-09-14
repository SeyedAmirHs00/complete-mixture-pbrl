"""Plot branch-symmetry trust bars from CSVs written by branch_symmetry_data."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthetic_shared_core import SHARED_BRANCH_VARIANTS, build_k4_configs


def plot_branch_bars(out: str) -> None:
    stats = pd.read_csv(os.path.join(out, "alpha_bar_stats.csv"))
    configs = build_k4_configs()
    order = ["3R1N", "3R1A", "1R3A"]
    colors = {"standard": "#4c72b0", "stabilized": "#dd8452"}

    for cfg in order:
        betas = configs[cfg]
        k = len(betas)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        x = np.arange(k)
        n_v = len(SHARED_BRANCH_VARIANTS)
        width = 0.35 if n_v == 2 else 0.25
        for vi, v in enumerate(SHARED_BRANCH_VARIANTS):
            sub = stats[(stats.config == cfg) & (stats.variant == v.name)].sort_values("expert_idx")
            mean = sub["mean_abar"].to_numpy()
            std = sub["std_abar"].to_numpy()
            ax.bar(
                x + (vi - (n_v - 1) / 2) * width,
                mean,
                width,
                yerr=std,
                label=v.label,
                color=colors[v.name],
                alpha=0.9,
            )
        labels = []
        for b in betas:
            if b > 0:
                labels.append("R")
            elif b < 0:
                labels.append("A")
            else:
                labels.append("N")
        ax.set_xticks(x)
        ax.set_xticklabels([f"E{i} ({lab})" for i, lab in enumerate(labels)])
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel(r"$\bar\alpha_k$")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"alpha_bar_{cfg}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot branch-symmetry bars from CSV")
    p.add_argument(
        "--out_dir",
        default="results/synthetic_branch_symmetry",
        help="Directory containing alpha_bar_stats.csv",
    )
    args = p.parse_args()
    plot_branch_bars(args.out_dir)
    print(f"OUT branch plots: {args.out_dir}")


if __name__ == "__main__":
    main()
