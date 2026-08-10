"""
Fig. 5 (main_v2): K=10 expert-ratio heatmaps.
Label: fig:synthetic-ratio-sweep

Example:
  python fig4_expert_ratio_K10.py --seeds 100 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthetic_shared_core import SHARED_BRANCH_VARIANTS, run_shared_variant, status_print


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


def _col_spec() -> List[Tuple[str, str, str, float, float, bool, str]]:
    """(grid_key, col_subcaption, cbar_label, vmin, vmax, signed, cmap)."""
    return [
        ("correct", r"(a) Correct-branch rate", "correct-branch rate", 0.0, 1.0, False, "RdYlGn"),
        ("adv", r"(b) Mean $\bar\alpha_A$", r"mean $\bar\alpha_A$", -1.0, 1.0, True, "RdBu"),
        ("noisy", r"(c) Mean $\bar\alpha_N$", r"mean $\bar\alpha_N$", -1.0, 1.0, True, "RdBu"),
        ("rel", r"(d) Mean $\bar\alpha_R$", r"mean $\bar\alpha_R$", -1.0, 1.0, True, "RdBu"),
    ]


def plot_heatmaps(
    out_dir: str,
    k: int,
    grids: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Combined 2x4: rows = Standard / Stabilized; columns = correct + trusts."""
    from matplotlib.gridspec import GridSpec

    row_labels = {
        "standard": "(i) Standard",
        "stabilized": "(ii) Stabilized",
    }
    names = [v.name for v in SHARED_BRANCH_VARIANTS]
    cols = _col_spec()
    n_rows = len(names)
    n_cols = len(cols)

    fig = plt.figure(figsize=(3.6 * n_cols + 0.8, 3.5 * n_rows + 1.15))
    gs = GridSpec(
        n_rows + 1,
        n_cols,
        figure=fig,
        height_ratios=[1.0] * n_rows + [0.06],
        wspace=0.22,
        hspace=0.35,
        left=0.07,
        right=0.98,
        top=0.92,
        bottom=0.06,
    )
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_cols)] for i in range(n_rows)])

    for j, (key, col_caption, cbar_label, vmin, vmax, signed, cmap) in enumerate(cols):
        im = None
        for i, name in enumerate(names):
            im = _panel(
                axes[i, j],
                grids[key][name],
                k,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                title=col_caption if i == 0 else "",
                xlabel=(i == n_rows - 1),
                ylabel=(j == 0),
                signed=signed,
            )
            if j == 0:
                axes[i, j].annotate(
                    row_labels.get(name, name),
                    xy=(-0.32, 0.5),
                    xycoords="axes fraction",
                    fontsize=11,
                    fontweight="bold",
                    ha="right",
                    va="center",
                    rotation=90,
                )
        cax = fig.add_subplot(gs[n_rows, j])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    combined = os.path.join(out_dir, "expert_ratio_sweep.png")
    fig.savefig(combined, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Keep per-metric two-panel exports for optional use.
    for key, col_caption, cbar_label, vmin, vmax, signed, cmap in cols:
        fname = {
            "correct": "expert_ratio_correct_branch.png",
            "adv": "expert_ratio_adv_trust.png",
            "noisy": "expert_ratio_noisy_trust.png",
            "rel": "expert_ratio_rel_trust.png",
        }[key]
        f = plt.figure(figsize=(3.55 * n_rows + 0.8, 3.9))
        gs1 = GridSpec(
            1,
            n_rows + 1,
            figure=f,
            width_ratios=[1.0] * n_rows + [0.045],
            wspace=0.22,
            left=0.06,
            right=0.94,
            top=0.88,
            bottom=0.18,
        )
        axs = [f.add_subplot(gs1[0, j]) for j in range(n_rows)]
        cax = f.add_subplot(gs1[0, n_rows])
        im = None
        for j, name in enumerate(names):
            im = _panel(
                axs[j],
                grids[key][name],
                k,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                title=row_labels.get(name, name),
                xlabel=True,
                ylabel=(j == 0),
                signed=signed,
            )
        cb = f.colorbar(im, cax=cax)
        cb.set_label(cbar_label, fontsize=9)
        axs[n_rows // 2].annotate(
            col_caption,
            xy=(0.5, -0.28),
            xycoords="axes fraction",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top",
        )
        f.savefig(os.path.join(out_dir, fname), dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(f)

def _nan_grid(k: int) -> Dict[str, np.ndarray]:
    return {v.name: np.full((k + 1, k + 1), np.nan) for v in SHARED_BRANCH_VARIANTS}


def grids_from_table(table: pd.DataFrame, k: int) -> Dict[str, Dict[str, np.ndarray]]:
    grids = {
        "correct": _nan_grid(k),
        "adv": _nan_grid(k),
        "noisy": _nan_grid(k),
        "rel": _nan_grid(k),
    }
    for _, r in table.iterrows():
        ni, na = int(r["n_N"]), int(r["n_A"])
        name = r["variant"]
        grids["correct"][name][ni, na] = float(r["correct_branch_rate"])
        grids["adv"][name][ni, na] = float(r["mean_adv_trust"])
        grids["noisy"][name][ni, na] = float(r["mean_noisy_trust"])
        grids["rel"][name][ni, na] = float(r["mean_rel_trust"])
    return grids


def replot_from_csv(out_dir: str = "final_results/synthetic_expert_ratio_sweep_K10") -> None:
    table = pd.read_csv(os.path.join(out_dir, "expert_ratio_shared.csv"))
    required = {"mean_noisy_trust", "mean_rel_trust"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"CSV missing {sorted(missing)}; re-run without --replot to recompute trusts."
        )
    k = int((table["n_R"] + table["n_N"] + table["n_A"]).iloc[0])
    grids = grids_from_table(table, k)
    plot_heatmaps(out_dir, k, grids)
    print(f"replot OK: {out_dir}")


def _mean_slice(abar: np.ndarray, start: int, stop: int) -> float:
    if stop <= start:
        return float(np.nan)
    return float(np.mean(abar[:, start:stop]))


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
                    betas,
                    v,
                    seeds=args.seeds,
                    steps=args.steps,
                    seed=8000 + idx,
                    progress_desc=f"ratio {name}/{v.name}",
                )
                # Expert order in abar: reliable | noisy | adversarial
                rel_trust = _mean_slice(abar, 0, n_r)
                noisy_trust = _mean_slice(abar, n_r, n_r + n_n)
                adv_trust = _mean_slice(abar, n_r + n_n, n_r + n_n + n_a)
                rows.append(
                    {
                        "n_R": n_r,
                        "n_N": n_n,
                        "n_A": n_a,
                        "variant": v.name,
                        "correct_branch_rate": float((rho > 0.05).mean()),
                        "mean_adv_trust": adv_trust,
                        "mean_noisy_trust": noisy_trust,
                        "mean_rel_trust": rel_trust,
                        "mean_rho": float(rho.mean()),
                    }
                )
            sub = [r for r in rows if r["n_R"] == n_r and r["n_N"] == n_n and r["n_A"] == n_a]
            msg = " | ".join(f"{r['variant'][:3]}={r['correct_branch_rate']:.2f}" for r in sub)
            status_print(f"  {name:12s} {msg}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "expert_ratio_shared.csv"), index=False)
    grids = grids_from_table(table, k)
    plot_heatmaps(args.out_dir, k, grids)
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
