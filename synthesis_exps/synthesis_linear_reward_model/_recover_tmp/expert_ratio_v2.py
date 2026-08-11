"""Expert-ratio sweep (K=10) with shared reward head."""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_ttp_synthetic_shared_core import SHARED_BRANCH_VARIANTS, run_shared_variant


def plot_heatmaps(
    out_dir: str,
    k: int,
    grids_c: Dict[str, np.ndarray],
    grids_a: Dict[str, np.ndarray],
) -> None:
    def plot(grids, fname, vmin, vmax, cmap, label):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        roman = ("i", "ii", "iii")
        cmap_obj = plt.get_cmap(cmap)
        im = None
        for ax, v, idx in zip(axes, SHARED_BRANCH_VARIANTS, range(3)):
            g = grids[v.name]
            im = ax.imshow(
                np.ma.masked_invalid(g),
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=[-0.5, k + 0.5, -0.5, k + 0.5],
            )
            ax.plot([0, k / 2], [k, 0], color="black", lw=1.3, ls="--")
            for ni in range(k + 1):
                for na in range(k + 1):
                    val = g[ni, na]
                    if not np.isfinite(val):
                        continue
                    norm = float(np.clip((val - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0))
                    rr, gg, bb, _ = cmap_obj(norm)
                    # Relative luminance: black on yellow/light, white on dark
                    lum = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
                    color = "black" if lum > 0.45 else "white"
                    ax.text(
                        na,
                        ni,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=color,
                        fontweight="semibold",
                    )
            ax.set_title(f"({roman[idx]}) {v.label}")
            ax.set_xlabel("# adversarial")
            ax.set_ylabel("# noisy")
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label=label)
        fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)

    plot(grids_c, "expert_ratio_correct_branch.png", 0, 1, "viridis", "correct-branch rate")
    plot(grids_a, "expert_ratio_adv_trust.png", -1, 1, "coolwarm", r"median $\bar\alpha_A$")


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
