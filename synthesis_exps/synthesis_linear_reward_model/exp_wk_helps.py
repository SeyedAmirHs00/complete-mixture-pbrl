"""
When does detached w_k help? (MLP reward head)

Hypothesis
----------
Max-norm already shrinks bar-alpha for noisy experts. Detached w_k ∝ |tilde alpha|
helps when residual noisy gradients still pollute nabla R:
  (A) many noisy experts (noise dominates the mean-over-experts loss);
  (B) a high-capacity reward MLP that can fit label noise.

On a low-capacity shared linear head, bar-alpha_N → 0 and w_k is nearly redundant.

Design
------
- Heads: PEBBLE-style MLP (high-cap) vs shared linear R = sum tanh(θ^T s) (low-cap).
- Mixtures: 2 reliable + N_noisy in {0,1,2,4,8,16}.
- Variants: max-norm only vs max-norm + detached w_k.
- MLP uses shared pairs across experts (same comparisons, conflicting labels).
- Shared linear uses q=0 disjoint blocks (standard robotics regime).

Example:
  python exp_wk_helps.py --seeds 50 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic_shared_core import SharedVariant, get_device, rowwise_corr, run_shared_variant, sigmoid_np


@dataclass(frozen=True)
class WkVariant:
    name: str
    label: str
    use_w: bool


WK_VARIANTS: Tuple[WkVariant, ...] = (
    WkVariant("no_w", "max-norm only", False),
    WkVariant("with_w", "max-norm + detached $w_k$", True),
)

N_NOISY_GRID_DEFAULT: Tuple[int, ...] = (0, 1, 2, 4, 8, 16)
N_RELIABLE = 2


def betas_for(n_noisy: int) -> Tuple[float, ...]:
    return tuple([1.0] * N_RELIABLE + [0.0] * n_noisy)


class RewardMLP(nn.Module):
    """PEBBLE-style reward: d -> H -> H -> H -> 1, Tanh out; R = sum_t r(s_t)."""

    def __init__(self, d: int, hidden: int = 128, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        din = d
        for _ in range(n_layers):
            layers.append(nn.Linear(din, hidden))
            layers.append(nn.LeakyReLU(0.01))
            din = hidden
        layers.append(nn.Linear(din, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # states: [N, T, D] -> returns [N]
        n, t, d = states.shape
        r = self.net(states.reshape(n * t, d)).reshape(n, t)
        return r.sum(dim=1)


def run_mlp_seed(
    betas: Tuple[float, ...],
    use_w: bool,
    *,
    states: np.ndarray,
    r_star: np.ndarray,
    i_np: np.ndarray,
    j_np: np.ndarray,
    y_np: np.ndarray,
    steps: int,
    hidden: int,
    lr_theta: float,
    lr_alpha: float,
    alpha_init: float,
    seed: int,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    """Train one seed; returns (corr, abar[K])."""
    torch.manual_seed(seed)
    k = len(betas)
    n, t, d = states.shape

    net = RewardMLP(d, hidden=hidden).to(device)
    alpha = torch.nn.Parameter(torch.full((k,), float(alpha_init), device=device))

    st = torch.as_tensor(states, dtype=torch.float32, device=device)
    i = torch.as_tensor(i_np, dtype=torch.long, device=device)  # [P]
    j = torch.as_tensor(j_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)  # [K, P]

    opt = torch.optim.SGD(
        [
            {"params": net.parameters(), "lr": lr_theta},
            {"params": [alpha], "lr": lr_alpha},
        ]
    )

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        R = net(st)  # [N]
        delta = R[i] - R[j]  # [P]

        trust = torch.tanh(alpha)
        abs_t = trust.abs()
        coef = trust / abs_t.amax().clamp_min(1e-12).detach()

        if use_w:
            w = (k * abs_t / abs_t.sum().clamp_min(1e-12)).detach()
        else:
            w = torch.ones(k, device=device)

        # Reward path: stop grad through coef
        logits_R = coef.detach().unsqueeze(1) * delta.unsqueeze(0)  # [K, P]
        bce_R = F.binary_cross_entropy_with_logits(logits_R, y, reduction="none")
        loss_R = (w.unsqueeze(1) * bce_R).mean()

        # Trust path: stop grad through delta; unweighted
        logits_A = coef.unsqueeze(1) * delta.detach().unsqueeze(0)
        loss_A = F.binary_cross_entropy_with_logits(logits_A, y, reduction="none").mean()

        (loss_R + loss_A).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()

    with torch.no_grad():
        R_hat = net(st).cpu().numpy()
        trust = torch.tanh(alpha)
        abar = (trust / trust.abs().amax().clamp_min(1e-12)).cpu().numpy()

    rho = float(rowwise_corr(R_hat[None, :], r_star[None, :])[0])
    return rho, abar


def run_mlp(
    betas: Tuple[float, ...],
    use_w: bool,
    *,
    seeds: int,
    n_seg: int,
    T: int,
    d: int,
    pairs: int,
    steps: int,
    hidden: int,
    seed: int,
    lr_theta: float = 0.05,
    lr_alpha: float = 0.005,
    alpha_init: float = 0.01,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-cap MLP; shared (i,j) across experts so many noisy votes hit the same pairs.
    """
    device = get_device(device)
    rng = np.random.default_rng(seed)
    k = len(betas)
    b = np.asarray(betas, dtype=np.float64)

    rhos = np.zeros(seeds)
    abars = np.zeros((seeds, k))

    for s in range(seeds):
        # Ground-truth linear teacher (same as shared synthetics)
        theta_star = rng.normal(size=d)
        theta_star /= np.linalg.norm(theta_star) + 1e-12
        states = rng.normal(size=(n_seg, T, d))
        r_star = np.tanh(states @ theta_star).sum(axis=1)
        r_star = (r_star - r_star.mean()) / (r_star.std() + 1e-12)

        i_np = rng.integers(0, n_seg, size=pairs)
        j_np = rng.integers(0, n_seg, size=pairs)
        same = i_np == j_np
        while same.any():
            j_np[same] = rng.integers(0, n_seg, size=int(same.sum()))
            same = i_np == j_np

        d_star = r_star[i_np] - r_star[j_np]
        y_np = np.zeros((k, pairs), dtype=np.float64)
        for e in range(k):
            y_np[e] = (rng.random(pairs) < sigmoid_np(b[e] * d_star)).astype(np.float64)

        rho, abar = run_mlp_seed(
            betas,
            use_w,
            states=states,
            r_star=r_star,
            i_np=i_np,
            j_np=j_np,
            y_np=y_np,
            steps=steps,
            hidden=hidden,
            lr_theta=lr_theta,
            lr_alpha=lr_alpha,
            alpha_init=alpha_init,
            seed=seed + 1000 * s + 7,
            device=device,
        )
        rhos[s] = rho
        abars[s] = abar
        if (s + 1) % max(1, seeds // 5) == 0:
            print(f"    seed {s+1}/{seeds}", flush=True)

    return rhos, abars


def summarize(
    head: str,
    n_noisy: int,
    variant: WkVariant,
    rho: np.ndarray,
    abar: np.ndarray,
) -> dict:
    n_r = N_RELIABLE
    return {
        "head": head,
        "n_reliable": n_r,
        "n_noisy": n_noisy,
        "K": n_r + n_noisy,
        "variant": variant.name,
        "label": variant.label,
        "use_w": variant.use_w,
        "correct": float((rho > 0.05).mean()),
        "mean_abs_corr": float(np.abs(rho).mean()),
        "mean_signed_corr": float(rho.mean()),
        "abar_R": float(abar[:, :n_r].mean()),
        "abar_N": float(abar[:, n_r:].mean()) if n_noisy > 0 else float("nan"),
        "abar_N_abs": float(np.abs(abar[:, n_r:]).mean()) if n_noisy > 0 else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="final_results/synthetic_wk_helps")
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--n_seg", type=int, default=256)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--n_noisy",
        type=int,
        nargs="+",
        default=list(N_NOISY_GRID_DEFAULT),
        help="noisy-expert counts to sweep",
    )
    p.add_argument("--skip_shared", action="store_true", help="only run MLP head")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.out_dir):
        if not args.overwrite:
            raise FileExistsError(args.out_dir)
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    device = torch.device(args.device)
    n_noisy_grid = tuple(args.n_noisy)
    rows: List[dict] = []
    idx = 0

    # --- (A)+(B): MLP high-cap, many-N sweep ---
    for n_noisy in n_noisy_grid:
        betas = betas_for(n_noisy)
        for v in WK_VARIANTS:
            print(f"\n[mlp 2R{n_noisy}N] {v.name}", flush=True)
            rho, abar = run_mlp(
                betas,
                v.use_w,
                seeds=args.seeds,
                n_seg=args.n_seg,
                T=args.T,
                d=args.d,
                pairs=args.pairs,
                steps=args.steps,
                hidden=args.hidden,
                seed=9200 + 19 * idx,
                device=device,
            )
            row = summarize("mlp", n_noisy, v, rho, abar)
            rows.append(row)
            print(
                f"[mlp      2R{n_noisy}N] {v.name:7s} "
                f"correct={row['correct']:.3f} |corr|={row['mean_abs_corr']:.3f} "
                f"aR={row['abar_R']:+.3f} |aN|={row['abar_N_abs']:.3f}",
                flush=True,
            )
            idx += 1

    # --- Control: low-cap shared linear (expect little w_k gap) ---
    if not args.skip_shared:
        shared_map = {
            "no_w": SharedVariant(
                "no_w",
                "max-norm only",
                target_rms=0.0,
                use_tanh=True,
                use_alpha_tanh=True,
                use_maxnorm=True,
                use_confidence_weights=False,
            ),
            "with_w": SharedVariant(
                "with_w",
                "max-norm + detached $w_k$",
                target_rms=0.0,
                use_tanh=True,
                use_alpha_tanh=True,
                use_maxnorm=True,
                use_confidence_weights=True,
                detach_weights=True,
            ),
        }

        for n_noisy in n_noisy_grid:
            betas = betas_for(n_noisy)
            n_seg = max(args.n_seg, 2 * (N_RELIABLE + n_noisy) * 8)
            for v in WK_VARIANTS:
                print(f"\n[shared 2R{n_noisy}N] {v.name}", flush=True)
                rho, abar, _ = run_shared_variant(
                    betas,
                    shared_map[v.name],
                    seeds=args.seeds,
                    steps=args.steps,
                    n_seg=n_seg,
                    T=args.T,
                    d=args.d,
                    pairs=args.pairs,
                    q=0.0,
                    seed=9300 + 23 * idx,
                )
                row = summarize("shared", n_noisy, v, rho, abar)
                rows.append(row)
                print(
                    f"[shared   2R{n_noisy}N] {v.name:7s} "
                    f"correct={row['correct']:.3f} |corr|={row['mean_abs_corr']:.3f} "
                    f"aR={row['abar_R']:+.3f} |aN|={row['abar_N_abs']:.3f}",
                    flush=True,
                )
                idx += 1

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "wk_helps_sweep.csv"), index=False)

    pivots = []
    for head in sorted(table["head"].unique()):
        sub = table[table["head"] == head]
        for n_noisy in n_noisy_grid:
            a = sub[(sub.n_noisy == n_noisy) & (sub.variant == "no_w")].iloc[0]
            b = sub[(sub.n_noisy == n_noisy) & (sub.variant == "with_w")].iloc[0]
            pivots.append(
                {
                    "head": head,
                    "n_noisy": n_noisy,
                    "correct_no_w": a.correct,
                    "correct_with_w": b.correct,
                    "delta_correct": b.correct - a.correct,
                    "abs_corr_no_w": a.mean_abs_corr,
                    "abs_corr_with_w": b.mean_abs_corr,
                    "delta_abs_corr": b.mean_abs_corr - a.mean_abs_corr,
                    "abs_aN_no_w": a.abar_N_abs,
                    "abs_aN_with_w": b.abar_N_abs,
                }
            )
    pivot = pd.DataFrame(pivots)
    pivot.to_csv(os.path.join(args.out_dir, "wk_helps_delta.csv"), index=False)

    print("\n=== Delta |corr| from adding detached w_k ===")
    print(pivot.to_string(index=False))
    print(f"\nOUT: {args.out_dir}")


if __name__ == "__main__":
    main()
