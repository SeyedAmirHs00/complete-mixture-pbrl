"""
Enhancement ablation table data (no plotting).
Label: tab:enhancement-ablation

Example:
  python enhancement_ablation_data.py --seeds 200 --overwrite
  python enhancement_ablation_data.py --reward-model mlp --seeds 200 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from synthetic_shared_core import (
    DEFAULT_COEF_MAX_DELTA,
    SharedVariant,
    run_shared_variant,
)

ABLATION_VARIANTS: Tuple[SharedVariant, ...] = (
    SharedVariant(
        "raw",
        "Raw",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=False,
        use_maxnorm=False,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "tanh",
        "+Tanh",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=True,
        use_maxnorm=False,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "maxnorm",
        "+Max-norm",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "full",
        "Full (detached $w_k$)",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=True,
        detach_weights=True,
    ),
    SharedVariant(
        "no_w",
        "w/o $w_k$",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "attach_w",
        "Attached $w_k$",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=True,
        detach_weights=False,
    ),
)

BETAS = (1.0, 1.0, 0.0, -1.0)
CFG = "2R1N1A"


def summarize(
    variant: SharedVariant,
    rho: np.ndarray,
    abar: np.ndarray,
    *,
    reward_model: str = "linear",
) -> dict:
    signed_corr = float(rho.mean())
    return {
        "config": CFG,
        "variant": variant.name,
        "label": variant.label,
        "reward_model": reward_model,
        "use_alpha_tanh": variant.use_alpha_tanh,
        "use_maxnorm": variant.use_maxnorm,
        "use_w": variant.use_confidence_weights,
        "detach_w": variant.detach_weights,
        "correct": float((rho > 0.5).mean()),
        # Signed (real) Pearson corr with r*; not |ρ|.
        "mean_signed_corr": signed_corr,
        "corr": signed_corr,
        "abar_R": float(abar[:, :2].mean()),
        "abar_N": float(abar[:, 2].mean()),
        "abar_A": float(abar[:, 3].mean()),
    }


def run_ablation_data(
    out_dir: Optional[str] = None,
    *,
    seeds: int = 200,
    steps: int = 400,
    n: int = 500,
    pairs: int = 256,
    q: float = 0.0,
    overwrite: bool = False,
    coef_max_delta: float = DEFAULT_COEF_MAX_DELTA,
    reward_model: str = "linear",
    optimizer: str = "sgd",
    lr_model: Optional[float] = None,
    lr_alpha: Optional[float] = None,
    hidden: int = 128,
    n_layers: int = 3,
) -> str:
    if out_dir is None:
        suffix = "_mlp" if reward_model == "mlp" else ""
        out_dir = f"results/synthetic_enhancement_ablation{suffix}"

    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    rows: List[dict] = []
    print(
        f"Running enhancement ablation: reward_model={reward_model} "
        f"optimizer={optimizer} seeds={seeds} steps={steps} -> {out_dir}"
    )
    for idx, v in enumerate(ABLATION_VARIANTS):
        rho, abar, _ = run_shared_variant(
            BETAS,
            v,
            seeds=seeds,
            steps=steps,
            n_seg=n,
            pairs=pairs,
            q=q,
            seed=9100 + 17 * idx,
            coef_max_delta=coef_max_delta,
            reward_model=reward_model,
            optimizer=optimizer,
            lr_theta=lr_model,
            lr_alpha=lr_alpha,
            hidden=hidden,
            n_layers=n_layers,
        )
        row = summarize(v, rho, abar, reward_model=reward_model)
        rows.append(row)
        print(
            f"[{CFG}] {v.name:10s} correct={row['correct']:.3f} "
            f"corr={row['mean_signed_corr']:+.3f} "
            f"aR={row['abar_R']:+.3f} aN={row['abar_N']:+.3f} aA={row['abar_A']:+.3f}"
        )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "enhancement_ablation.csv"), index=False)
    table.to_csv(os.path.join(out_dir, f"enhancement_ablation_{CFG}.csv"), index=False)
    print(f"OUT ablation data: {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Enhancement ablation table data generation")
    p.add_argument("--out_dir", default=None, help="Default depends on --reward-model")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--reward-model",
        choices=["linear", "mlp"],
        default="linear",
        help="Reward model: linear (default) or mlp (PEBBLE-style neural network).",
    )
    p.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw", "adam_sgd"],
        default="sgd",
    )
    p.add_argument("--lr-model", type=float, default=None)
    p.add_argument("--lr-alpha", type=float, default=None)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    args = p.parse_args()

    run_ablation_data(
        args.out_dir,
        seeds=args.seeds,
        steps=args.steps,
        n=args.n,
        pairs=args.pairs,
        q=args.q,
        overwrite=args.overwrite,
        coef_max_delta=args.coef_max_delta,
        reward_model=args.reward_model,
        optimizer=args.optimizer,
        lr_model=args.lr_model,
        lr_alpha=args.lr_alpha,
        hidden=args.hidden,
        n_layers=args.n_layers,
    )


if __name__ == "__main__":
    main()
