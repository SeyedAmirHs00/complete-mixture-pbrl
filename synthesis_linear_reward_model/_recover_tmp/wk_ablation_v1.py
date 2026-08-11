"""
Tabular free-R enhancement ablation — designed so detached w_k helps.

Why this synthetic (vs shared linear head)
-----------------------------------------
With a free score R_i per trajectory, the reward model can overfit noisy
labels. Max-norm unlocks signed trust magnitudes, but noisy experts often
retain non-trivial |bar alpha_N|, so they still pull nabla R. Detached
confidence weights w_k ∝ |tilde alpha_k| cut that residual and raise
|corr(R_hat, R*)|.

(On a low-capacity shared linear head with equal pairs / mean-per-expert
loss, max-norm alone usually drives bar alpha_N → 0 and w_k is redundant;
that is a different regime from tabular / MLP / PEBBLE.)

Primary mixture: 2R2N (half the crowd is noise). Also report 1R3N / 3R1N.

Example:
  python final_ttp_synthetic_wk_ablation.py --seeds 200 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from final_ttp_synthetic_shared_core import rowwise_corr, sigmoid_np


@dataclass(frozen=True)
class AblationVariant:
    name: str
    label: str
    use_tanh: bool
    use_maxnorm: bool
    use_w: bool
    detach_w: bool = True


VARIANTS: Tuple[AblationVariant, ...] = (
    AblationVariant("raw", "Raw", False, False, False),
    AblationVariant("tanh", "+Tanh", True, False, False),
    AblationVariant("maxnorm", "+Max-norm", True, True, False),
    AblationVariant("full", "Full (detached $w_k$)", True, True, True, True),
    AblationVariant("no_w", "w/o $w_k$", True, True, False),
    AblationVariant("attach_w", "Attached $w_k$", True, True, True, False),
)

MIXTURES: Dict[str, Tuple[float, ...]] = {
    "2R2N": (1.0, 1.0, 0.0, 0.0),
    "1R3N": (1.0, 0.0, 0.0, 0.0),
    "3R1N": (1.0, 1.0, 1.0, 0.0),
}


def run_tabular(
    betas: Tuple[float, ...],
    variant: AblationVariant,
    *,
    seeds: int,
    n: int,
    pairs: int,
    steps: int,
    seed: int,
    lr_r: float = 0.08,
    lr_alpha: float = 0.005,
    alpha_init: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stabilized tabular free-R_i (init at 0)."""
    rng = np.random.default_rng(seed)
    k = len(betas)
    b = np.asarray(betas, dtype=np.float64)

    r_star = rng.normal(size=(seeds, n))
    r_star = (r_star - r_star.mean(1, keepdims=True)) / (r_star.std(1, keepdims=True) + 1e-12)

    i_np = rng.integers(0, n, size=(seeds, pairs))
    j_np = rng.integers(0, n, size=(seeds, pairs))
    same = i_np == j_np
    while same.any():
        j_np[same] = rng.integers(0, n, size=int(same.sum()))
        same = i_np == j_np

    d_star = np.take_along_axis(r_star, i_np, 1) - np.take_along_axis(r_star, j_np, 1)
    y_np = np.zeros((seeds, k, pairs), dtype=np.float64)
    for e in range(k):
        y_np[:, e] = (rng.random((seeds, pairs)) < sigmoid_np(b[e] * d_star)).astype(np.float64)

    R = torch.nn.Parameter(torch.zeros(seeds, n))
    alpha = torch.nn.Parameter(torch.full((seeds, k), float(alpha_init)))
    i = torch.as_tensor(i_np, dtype=torch.long)
    j = torch.as_tensor(j_np, dtype=torch.long)
    y = torch.as_tensor(y_np, dtype=torch.float32)

    for _ in range(steps):
        if R.grad is not None:
            R.grad = None
        if alpha.grad is not None:
            alpha.grad = None

        delta = R.gather(1, i) - R.gather(1, j)
        trust = torch.tanh(alpha) if variant.use_tanh else alpha
        abs_t = trust.abs()
        if variant.use_maxnorm:
            coef = trust / abs_t.amax(1, keepdim=True).clamp_min(1e-12).detach()
        else:
            coef = trust

        if variant.use_w:
            w = k * abs_t / abs_t.sum(1, keepdim=True).clamp_min(1e-12)
            if variant.detach_w:
                w = w.detach()
        else:
            w = torch.ones(seeds, k)

        logits_R = coef.detach().unsqueeze(2) * delta.unsqueeze(1)
        bce_R = F.binary_cross_entropy_with_logits(logits_R, y, reduction="none")
        loss_R = (w.unsqueeze(2) * bce_R).mean(dim=(1, 2)).sum()

        logits_A = coef.unsqueeze(2) * delta.detach().unsqueeze(1)
        bce_A = F.binary_cross_entropy_with_logits(logits_A, y, reduction="none")
        if variant.use_w and not variant.detach_w:
            loss_A = (w.unsqueeze(2) * bce_A).mean(dim=(1, 2)).sum()
        else:
            loss_A = bce_A.mean(dim=(1, 2)).sum()

        (loss_R + loss_A).backward()
        with torch.no_grad():
            g = R.grad
            gn = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
            g.mul_(torch.clamp(10.0 / gn, max=1.0))
            R.data.sub_(lr_r * g)
            alpha.data.sub_(lr_alpha * alpha.grad)

    with torch.no_grad():
        R_hat = R.detach().cpu().numpy()
        trust = torch.tanh(alpha) if variant.use_tanh else alpha
        if variant.use_maxnorm:
            abar = (trust / trust.abs().amax(1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
        else:
            abar = trust.cpu().numpy()
    return rowwise_corr(R_hat, r_star), abar


def summarize_row(
    cfg: str, betas: Tuple[float, ...], variant: AblationVariant, rho: np.ndarray, abar: np.ndarray
) -> dict:
    n_r = sum(1 for b in betas if b > 0.5)
    n_n = sum(1 for b in betas if abs(b) < 0.5)
    return {
        "config": cfg,
        "variant": variant.name,
        "label": variant.label,
        "use_tanh": variant.use_tanh,
        "use_maxnorm": variant.use_maxnorm,
        "use_w": variant.use_w,
        "detach_w": variant.detach_w,
        "correct": float((rho > 0.05).mean()),
        "mean_abs_corr": float(np.abs(rho).mean()),
        "mean_signed_corr": float(rho.mean()),
        "abar_R": float(abar[:, :n_r].mean()) if n_r else np.nan,
        "abar_N": float(abar[:, n_r : n_r + n_n].mean()) if n_n else np.nan,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="final_results/synthetic_wk_ablation")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.out_dir):
        if not args.overwrite:
            raise FileExistsError(args.out_dir)
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    rows: List[dict] = []
    idx = 0
    for cfg, betas in MIXTURES.items():
        for v in VARIANTS:
            rho, abar = run_tabular(
                betas,
                v,
                seeds=args.seeds,
                n=args.n,
                pairs=args.pairs,
                steps=args.steps,
                seed=9100 + 17 * idx,
            )
            idx += 1
            row = summarize_row(cfg, betas, v, rho, abar)
            rows.append(row)
            print(
                f"[{cfg:5s}] {v.name:10s} correct={row['correct']:.3f} "
                f"|corr|={row['mean_abs_corr']:.3f} "
                f"aR={row['abar_R']:+.3f} aN={row['abar_N']:+.3f}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "tabular_wk_ablation.csv"), index=False)

    # Paper-facing primary table: 2R2N
    primary = table[table.config == "2R2N"].copy()
    primary.to_csv(os.path.join(args.out_dir, "tabular_wk_ablation_2R2N.csv"), index=False)
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
