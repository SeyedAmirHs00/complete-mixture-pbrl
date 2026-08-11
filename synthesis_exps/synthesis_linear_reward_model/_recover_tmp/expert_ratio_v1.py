"""Expert-ratio sweep (K=10) with shared reward head."""

from __future__ import annotations

import argparse
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_ttp_synthetic_shared_core import SHARED_BRANCH_VARIANTS, run_shared_variant


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="final_results/synthetic_expert_ratio_sweep_K10")
    p.add_argument("--n_experts", type=int, default=10)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

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
                row = {
                    "n_R": n_r,
                    "n_N": n_n,
                    "n_A": n_a,
                    "variant": v.name,
                    "correct_branch_rate": float((rho > 0.05).mean()),
                    "median_adv_trust": adv_trust,
                    "mean_rho": float(rho.mean()),
                }
                rows.append(row)
            print(
                f"{name:12s} std={(rho>0.05).mean():.2f}"  # last rho is consensus; print all below
            )
            # clearer print
            sub = [r for r in rows if r["n_R"] == n_r and r["n_N"] == n_n and r["n_A"] == n_a]
            msg = " | ".join(f"{r['variant'][:3]}={r['correct_branch_rate']:.2f}" for r in sub)
            print(f"  {name:12s} {msg}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "expert_ratio_shared.csv"), index=False)

    # heatmaps
    grids_c = {v.name: np.full((k + 1, k + 1), np.nan) for v in SHARED_BRANCH_VARIANTS}
    grids_a = {v.name: np.full((k + 1, k + 1), np.nan) for v in SHARED_BRANCH_VARIANTS}
    for r in rows:
        grids_c[r["variant"]][r["n_N"], r["n_A"]] = r["correct_branch_rate"]
        grids_a[r["variant"]][r["n_N"], r["n_A"]] = r["median_adv_trust"]

    def plot(grids, fname, vmin, vmax, cmap, label):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        roman = ("i", "ii", "iii")
        im = None
        for ax, v, idx in zip(axes, SHARED_BRANCH_VARIANTS, range(3)):
            g = grids[v.name]
            im = ax.imshow(np.ma.masked_invalid(g), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=[-0.5, k + 0.5, -0.5, k + 0.5])
            ax.plot([0, k / 2], [k, 0], color="black", lw=1.3, ls="--")
            ax.set_title(f"({roman[idx]}) {v.label}")
            ax.set_xlabel("# adversarial")
            ax.set_ylabel("# noisy")
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label=label)
        fig.savefig(os.path.join(args.out_dir, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)

    plot(grids_c, "expert_ratio_correct_branch.png", 0, 1, "viridis", "correct-branch rate")
    plot(grids_a, "expert_ratio_adv_trust.png", -1, 1, "coolwarm", r"median $\bar\alpha_A$")
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
