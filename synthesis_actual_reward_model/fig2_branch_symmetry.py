"""
Fig. 2 (main_v2): branch-symmetry trust bars + correct-branch table.
Label: fig:synthetic-branch

Produces: final_results/synthetic_shared_all/branch_symmetry/

Example:
  python fig2_branch_symmetry.py --seeds 200 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, Tuple

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


def ensure_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    os.makedirs(path)


def run_branch(out_root: str, seeds: int, steps: int, overwrite: bool) -> None:
    out = os.path.join(out_root, "branch_symmetry")
    ensure_dir(out, overwrite)
    configs = build_k4_configs()
    order = ["3R1N", "3R1A", "1R3A"]
    rows = []
    run_store: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {c: {} for c in order}
    idx = 0
    for cfg in order:
        betas = configs[cfg]
        for v in SHARED_BRANCH_VARIANTS:
            idx += 1
            rho, abar, rms0 = run_shared_variant(
                betas,
                v,
                seeds=seeds,
                steps=steps,
                seed=9017 + 31 * idx,
                progress_desc=f"branch {cfg}/{v.name}",
            )
            correct = rho > 0.05
            flipped = rho < -0.05
            if float(correct.mean()) >= float(flipped.mean()):
                mask = correct
                branch_tag = "correct"
            else:
                mask = flipped
                branch_tag = "flipped"
            abar_m = abar[mask] if mask.any() else abar
            rows.append(
                {
                    "config": cfg,
                    "variant": v.name,
                    "init_rms": rms0,
                    "correct_branch_rate": float(correct.mean()),
                    "flipped_branch_rate": float(flipped.mean()),
                    "mean_rho": float(rho.mean()),
                    "trust_conditioned_on": branch_tag,
                    "n_trust_seeds": int(mask.sum()),
                    "mean_abar_R": float(abar_m[:, :3].mean())
                    if cfg != "1R3A"
                    else float(abar_m[:, 0].mean()),
                    "mean_abar_N": float(abar_m[:, 3].mean()) if "N" in cfg else np.nan,
                    "mean_abar_A": float(abar_m[:, -1].mean()) if "A" in cfg else np.nan,
                }
            )
            run_store[cfg][v.name] = (rho, abar)
            status_print(
                f"[branch] {cfg:6s} {v.name:10s} correct={rows[-1]['correct_branch_rate']:.3f} "
                f"rho={rows[-1]['mean_rho']:+.3f} rms0={rms0:.3f} "
                f"trust@{branch_tag} aR={rows[-1]['mean_abar_R']:+.2f}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "paper_table_synthetic_symmetry_fix.csv"), index=False)

    wide = table.pivot(index="config", columns="variant", values="correct_branch_rate")
    wide = wide.reindex(columns=[v.name for v in SHARED_BRANCH_VARIANTS])
    wide.to_csv(os.path.join(out, "branch_correct_wide.csv"))

    colors = {"standard": "#4c72b0", "stabilized": "#dd8452"}
    for cfg in order:
        betas = configs[cfg]
        k = len(betas)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        x = np.arange(k)
        n_v = len(SHARED_BRANCH_VARIANTS)
        width = 0.35 if n_v == 2 else 0.25
        for vi, v in enumerate(SHARED_BRANCH_VARIANTS):
            rho, abar = run_store[cfg][v.name]
            correct = rho > 0.05
            flipped = rho < -0.05
            mask = correct if float(correct.mean()) >= float(flipped.mean()) else flipped
            abar_m = abar[mask] if mask.any() else abar
            mean = abar_m.mean(0)
            std = abar_m.std(0)
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
    print(f"OUT branch: {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fig. 2: branch-symmetry (fig:synthetic-branch)")
    p.add_argument("--out_root", default="final_results/synthetic_shared_all")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    run_branch(args.out_root, args.seeds, args.steps, args.overwrite)


if __name__ == "__main__":
    main()
