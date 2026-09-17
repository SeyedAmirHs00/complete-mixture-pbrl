"""Plot enhancement ablation figures from enhancement_ablation.csv.

Generates:
1. enhancement_ablation_metrics.png / .pdf (correct branch rate and mean signed correlation)
2. enhancement_ablation_trust.png / .pdf (recovered trust alpha_bar across expert types)
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import savefig_png_pdf

CSV_NAME = "enhancement_ablation.csv"


def plot_enhancement_ablation_figures(table: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    labels = table["label"].tolist()
    x = np.arange(len(labels))
    width = 0.55

    # -----------------------------------------------------------------------
    # Figure 1: Correct-branch rate and Mean Signed Pearson Correlation
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    correct_vals = table["correct"].to_numpy(dtype=float)
    bars1 = ax1.bar(x, correct_vals, width, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Correct-branch rate", fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, axis="y", ls=":", alpha=0.4)
    ax1.set_title("(a) Correct-branch rate", fontsize=11)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    corr_col = "mean_signed_corr" if "mean_signed_corr" in table.columns else "corr"
    corr_vals = table[corr_col].to_numpy(dtype=float)
    colors2 = ["#2ca02c" if c >= 0 else "#d62728" for c in corr_vals]
    bars2 = ax2.bar(x, corr_vals, width, color=colors2, edgecolor="black", alpha=0.85)
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Mean signed corr with $r^*$", fontsize=10)
    ax2.set_ylim(min(-0.1, float(corr_vals.min()) - 0.1), 1.1)
    ax2.grid(True, axis="y", ls=":", alpha=0.4)
    ax2.set_title("(b) Signed Correlation with $r^*$", fontsize=11)
    for bar in bars2:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.02 if h >= 0 else -0.05
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + offset,
            f"{h:+.2f}",
            ha="center",
            va=va,
            fontsize=8,
        )

    fig.suptitle("Enhancement Ablation (2R1N1A)", fontsize=12, y=1.02)
    fig.tight_layout()
    metrics_path = os.path.join(out_dir, "enhancement_ablation_metrics.png")
    savefig_png_pdf(fig, metrics_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {metrics_path}")

    # -----------------------------------------------------------------------
    # Figure 2: Recovered Trust (\bar{\alpha}_R, \bar{\alpha}_N, \bar{\alpha}_A)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    bar_w = 0.24
    aR = table["abar_R"].to_numpy(dtype=float)
    aN = table["abar_N"].to_numpy(dtype=float)
    aA = table["abar_A"].to_numpy(dtype=float)

    ax.bar(x - bar_w, aR, bar_w, label=r"Reliable $\bar{\alpha}_R$ (true: +1)", color="#2ca02c", edgecolor="black", alpha=0.85)
    ax.bar(x, aN, bar_w, label=r"Neutral $\bar{\alpha}_N$ (true: 0)", color="#7f7f7f", edgecolor="black", alpha=0.75)
    ax.bar(x + bar_w, aA, bar_w, label=r"Adversary $\bar{\alpha}_A$ (true: -1)", color="#d62728", edgecolor="black", alpha=0.85)

    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(r"Recovered Trust $\bar{\alpha}$", fontsize=10)
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.set_title(r"Recovered Trust $\bar{\alpha}$ across Enhancements", fontsize=11)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    trust_path = os.path.join(out_dir, "enhancement_ablation_trust.png")
    savefig_png_pdf(fig, trust_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {trust_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot enhancement ablation figures")
    p.add_argument("--out_dir", default="results/synthetic_enhancement_ablation")
    p.add_argument("--csv", default=None, help="Path to enhancement_ablation.csv")
    args = p.parse_args()

    csv_path = args.csv or os.path.join(args.out_dir, CSV_NAME)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = args.out_dir if args.out_dir else os.path.dirname(os.path.abspath(csv_path))
    table = pd.read_csv(csv_path)
    plot_enhancement_ablation_figures(table, out_dir)
    print(f"OUT enhancement ablation plots: {out_dir}")


if __name__ == "__main__":
    main()
