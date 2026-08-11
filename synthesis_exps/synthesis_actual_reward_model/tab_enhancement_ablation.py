"""
Enhancement ablation on shared PEBBLE gen_net head (Stabilized init).
Label: tab:enhancement-ablation

Example:
  python tab_enhancement_ablation.py --seeds 200 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import List, Tuple

import numpy as np
import pandas as pd

from synthetic_shared_core import SharedVariant, run_shared_variant, status_print

# Production MLP reward head. Columns ablate α-tanh / max-norm / w_k.
ABLATION_VARIANTS: Tuple[SharedVariant, ...] = (
    SharedVariant(
        "raw",
        "Raw",
        init_kind="stabilized",
        use_alpha_tanh=False,
        use_maxnorm=False,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "tanh",
        "+Tanh",
        init_kind="stabilized",
        use_alpha_tanh=True,
        use_maxnorm=False,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "maxnorm",
        "+Max-norm",
        init_kind="stabilized",
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "full",
        "Full (detached $w_k$)",
        init_kind="stabilized",
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=True,
        detach_weights=True,
    ),
    SharedVariant(
        "no_w",
        "w/o $w_k$",
        init_kind="stabilized",
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "attach_w",
        "Attached $w_k$",
        init_kind="stabilized",
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=True,
        detach_weights=False,
    ),
)

BETAS = (1.0, 1.0, 0.0, -1.0)  # 2R1N1A
CFG = "2R1N1A"


def summarize(variant: SharedVariant, rho: np.ndarray, abar: np.ndarray) -> dict:
    return {
        "config": CFG,
        "variant": variant.name,
        "label": variant.label,
        "use_alpha_tanh": variant.use_alpha_tanh,
        "use_maxnorm": variant.use_maxnorm,
        "use_w": variant.use_confidence_weights,
        "detach_w": variant.detach_weights,
        "correct": float((rho > 0.05).mean()),
        "mean_abs_corr": float(np.abs(rho).mean()),
        "mean_signed_corr": float(rho.mean()),
        "abar_R": float(abar[:, :2].mean()),
        "abar_N": float(abar[:, 2].mean()),
        "abar_A": float(abar[:, 3].mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="final_results/synthetic_wk_ablation")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.out_dir):
        if not args.overwrite:
            raise FileExistsError(args.out_dir)
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    rows: List[dict] = []
    for idx, v in enumerate(ABLATION_VARIANTS):
        rho, abar, _ = run_shared_variant(
            BETAS,
            v,
            seeds=args.seeds,
            steps=args.steps,
            n_seg=args.n,
            pairs=args.pairs,
            q=args.q,
            seed=9100 + 17 * idx,
            progress_desc=f"ablation {v.name}",
        )
        row = summarize(v, rho, abar)
        rows.append(row)
        status_print(
            f"[{CFG}] {v.name:10s} correct={row['correct']:.3f} "
            f"|corr|={row['mean_abs_corr']:.3f} "
            f"aR={row['abar_R']:+.3f} aN={row['abar_N']:+.3f} aA={row['abar_A']:+.3f}"
        )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "shared_wk_ablation.csv"), index=False)
    table.to_csv(os.path.join(args.out_dir, "shared_wk_ablation_2R1N1A.csv"), index=False)
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
