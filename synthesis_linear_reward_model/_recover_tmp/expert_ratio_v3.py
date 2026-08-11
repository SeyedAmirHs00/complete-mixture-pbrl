"""Expert-ratio sweep (K=10) with shared reward head."""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_ttp_synthetic_shared_core import SHARED_BRANCH_VARIANTS, run_shared_variant


def _annotate(ax, g: np.ndarray, k: int, signed: bool = False) -> None:
    for ni in range(k + 1):
        for na in range(k + 1):
            val = g[ni, na]
            if not np.isfinite(val):
                continue
            if signed:
                txt = f"{val:+.2f}"
            else:
                txt = f"{val:.2f}"
            ax.text(
                na,
                ni,
                txt,
                ha="center",
                va="center",
                fontsize=5.2,
                color="black",
                zorder=3,
            )


def _panel(
    ax,
    g: np.ndarray,
    k: int,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    xlabel: bool,
    ylabel: bool,
    signed: bool,
):
    im = ax.imshow(
        np.ma.masked_invalid(g),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[-0.5, k + 0.5, -0.5, k + 0.5],
        aspect="equal",
    )
    ax.plot([0, k / 2], [k, 0], color="black", lw=1.15, ls="--", zorder=2)
    _annotate(ax, g, k, signed=signed)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlim(-0.5, k + 0.5)
    ax.set_ylim(-0.5, k + 0.5)
    ax.set_xticks(range(0, k + 1, 2))
    ax.set_yticks(range(0, k + 1, 2))
    if xlabel:
        ax.set_xlabel("# adversarial experts", fontsize=9)
    if ylabel:
        ax.set_ylabel("# noisy experts", fontsize=9)
    return im


def plot_heatmaps(
    out_dir: str,
    k: int,
    grids_c: Dict[str, np.ndarray],
    grids_a: Dict[str, np.ndarray],
) -> None:
    """Reference style: RdYlGn + RdBu_r, plain black digits, combined 2x3."""
    labels = [v.label for v in SHARED_BRANCH_VARIANTS]
    names = [v.name for v in SHARED_BRANCH_VARIANTS]

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.6))

    im_c = None
    for j, name in enumerate(names):
        im_c = _panel(
            axes[0, j],
            grids_c[name],
            k,
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            title=labels[j],
            xlabel=False,
            ylabel=(j == 0),
            signed=False,
        )
    cbar_c = fig.colorbar(im_c, ax=axes[0, :].tolist(), fraction=0.046, pad=0.02)
    cbar_c.set_label("correct-branch rate", fontsize=9)

    im_a = None
    for j, name in enumerate(names):
        im_a = _panel(
            axes[1, j],
            grids_a[name],
            k,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            title=labels[j],
            xlabel=True,
            ylabel=(j == 0),
            signed=True,
        )
    cbar_a = fig.colorbar(im_a, ax=axes[1, :].tolist(), fraction=0.046, pad=0.02)
    cbar_a.set_label(r"median $\bar\alpha_A$", fontsize=9)

    # Row tags matching reference (a)/(b)
    axes[0, 0].annotate(
        "(a)",
        xy=(0.0, 1.02),
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    axes[1, 0].annotate(
        "(b)",
        xy=(0.0, 1.02),
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.18, hspace=0.28, right=0.90)
    combined = os.path.join(out_dir, "expert_ratio_sweep.png")
    fig.savefig(combined, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Also keep per-row PNGs (same style) for any external use
    def row_fig(grids, fname, cmap, vmin, vmax, cbar_label, signed: bool):
        f, axs = plt.subplots(1, 3, figsize=(11.8, 3.6))
        im = None
        for j, name in enumerate(names):
            im = _panel(
                axs[j],
                grids[name],
                k,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                title=labels[j],
                xlabel=True,
                ylabel=(j == 0),
                signed=signed,
            )
        cb = f.colorbar(im, ax=axs.tolist(), fraction=0.046, pad=0.02)
        cb.set_label(cbar_label, fontsize=9)
        f.tight_layout()
        f.savefig(os.path.join(out_dir, fname), dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(f)

    row_fig(grids_c, "expert_ratio_correct_branch.png", "RdYlGn", 0.0, 1.0, "correct-branch rate", False)
    row_fig(
        grids_a,
        "expert_ratio_adv_trust.png",
        "RdBu_r",
        -1.0,
        1.0,
        r"median $\bar\alpha_A$",
        True,
    )


def grids_from_table(table: pd.DataFrame, k: int):
    grids_c = {v.name: np.full((k + 1, k + 1), np.nan) for v in SHARED_BRANCH_VARIANTS}
    grids_a = {v.name: np.full((k + 1, k + 1), np.nan) for v in SHARED_BRANCH_VARIANTS}
    for _, r in table.iterrows():
        grids_c[r["variant"]][int(r["n_N"]), int(r["n_A"])] = float(r["correct_branch_rate"])
        grids_a[r["variant"]][int(r["n_N"]), int(r["n_A"])] = float(r["median_adv_trust"])
    return grids_c, grids_a


def replot_from_csv(out_dir: str = "final_results/synthetic_expert_ratio_sweep_K10") -> None:
    table = pd.read_csv(os.path.join(out_dir, "expert_ratio_shared.csv"))
    k = int((table["n_R"] + table["n_N"] + table["n_A"]).iloc[0])
    grids_c, grids_a = grids_from_table(table, k)
    plot_heatmaps(out_dir, k, grids_c, grids_a)
    print(f"replot OK: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="final_results/synthetic_expert_ratio_sweep_K10")
    p.add_argument("--n_experts", type=int, default=10)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--replot", action="store_true", help="Only regenerate figures from CSV")
    args = p.parse_args()

    if args.replot:
        replot_from_csv(args.out_dir)
        return

    if os.path.exists(args.out_dir):
        if not args.overwrite:
            raise FileExistsError(args.out_dir)
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    k = args.n_experts
    rows = []
    idx = 0
    for n_a in range(k + 1):
        for n_n in range(k - n_a + 1):
            n_r = k - n_a - n_n
            betas = tuple([1.0] * n_r + [0.0] * n_n + [-1.0] * n_a)
            name = f"{n_r}R{n_n}N{n_a}A"
            for v in SHARED_BRANCH_VARIANTS:
                idx += 1
                rho, abar, _ = run_shared_variant(
                    betas, v, seeds=args.seeds, steps=args.steps, seed=8000 + idx
                )
                adv_trust = float(np.median(abar[:, n_r + n_n :])) if n_a > 0 else np.nan
                rows.append(
                    {
                        "n_R": n_r,
                        "n_N": n_n,
                        "n_A": n_a,
                        "variant": v.name,
                        "correct_branch_rate": float((rho > 0.05).mean()),
                        "median_adv_trust": adv_trust,
                        "mean_rho": float(rho.mean()),
                    }
                )
            sub = [r for r in rows if r["n_R"] == n_r and r["n_N"] == n_n and r["n_A"] == n_a]
            msg = " | ".join(f"{r['variant'][:3]}={r['correct_branch_rate']:.2f}" for r in sub)
            print(f"  {name:12s} {msg}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "expert_ratio_shared.csv"), index=False)
    grids_c, grids_a = grids_from_table(table, k)
    plot_heatmaps(args.out_dir, k, grids_c, grids_a)
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
