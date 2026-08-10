"""
When does detached w_k help? (PEBBLE gen_net reward head)

Hypothesis
----------
Max-norm already shrinks bar-alpha for noisy experts. Detached w_k ∝ |tilde alpha|
helps when residual noisy gradients still pollute nabla R:
  (A) many noisy experts (noise dominates the mean-over-experts loss);
  (B) a high-capacity reward MLP that can fit label noise.

Design
------
- Head: production PEBBLE gen_net (H=256, 3 layers, Tanh).
- Teacher R*: independently initialized frozen gen_net.
- Mixtures: 2 reliable + N_noisy in {0,1,2,4,8,16}.
- Variants: max-norm only vs max-norm + detached w_k.
- Shared pairs across experts (same comparisons, conflicting labels).

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
import torch.nn.functional as F

from synthetic_shared_core import (
    DEFAULT_HIDDEN,
    DEFAULT_N_LAYERS,
    apply_init_kind,
    build_reward_mlp,
    get_device,
    progress_range,
    rowwise_corr,
    segment_returns,
    sigmoid_np,
    status_print,
    teacher_segment_returns,
)


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
    lr_theta: float,
    lr_alpha: float,
    alpha_init: float,
    seed: int,
    device: torch.device,
    hidden: int,
    n_layers: int,
) -> Tuple[float, np.ndarray]:
    """Train one seed; returns (corr, abar[K])."""
    torch.manual_seed(seed)
    k = len(betas)
    n, t, d = states.shape

    net = build_reward_mlp(d, hidden=hidden, n_layers=n_layers).to(device)
    apply_init_kind(net, "stabilized")
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
        opt.zero_grad()
        R = segment_returns(net, st)  # [N]
        # Broadcast the same (i,j) across experts
        delta = R[i] - R[j]  # [P]

        trust = torch.tanh(alpha)
        abs_t = trust.abs()
        coef = trust / torch.max(abs_t).clamp_min(1e-12).detach()

        if use_w:
            w = (k * abs_t / abs_t.sum().clamp_min(1e-12)).detach()
        else:
            w = torch.ones(k, device=device)

        logits_R = coef.detach().unsqueeze(1) * delta.unsqueeze(0)  # [K, P]
        bce_R = F.binary_cross_entropy_with_logits(logits_R, y, reduction="none")
        loss_R = (w.unsqueeze(1) * bce_R).mean()

        logits_A = coef.unsqueeze(1) * delta.detach().unsqueeze(0)
        loss_A = F.binary_cross_entropy_with_logits(logits_A, y, reduction="none").mean()

        (loss_R + loss_A).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()

    with torch.no_grad():
        R_hat = segment_returns(net, st).cpu().numpy()
        trust = torch.tanh(alpha)
        abar = (trust / torch.max(trust.abs()).clamp_min(1e-12)).cpu().numpy()

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
    seed: int,
    lr_theta: float = 0.05,
    lr_alpha: float = 0.005,
    alpha_init: float = 0.01,
    device: torch.device | None = None,
    hidden: int = DEFAULT_HIDDEN,
    n_layers: int = DEFAULT_N_LAYERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """High-cap gen_net; shared (i,j) across experts so many noisy votes hit the same pairs."""
    device = get_device(device)
    rng = np.random.default_rng(seed)
    k = len(betas)
    b = np.asarray(betas, dtype=np.float64)

    rhos = np.zeros(seeds)
    abars = np.zeros((seeds, k))

    status_print(
        f"wk_helps 2R{k - N_RELIABLE}N use_w={use_w} | device={device} "
        f"seeds={seeds} steps={steps} n={n_seg} T={T} d={d}"
    )
    for s in progress_range(seeds, desc=f"wk 2R{k - N_RELIABLE}N w={int(use_w)}"):
        states = rng.normal(size=(n_seg, T, d)).astype(np.float32)
        r_star = teacher_segment_returns(
            states,
            d=d,
            torch_seed=seed + 10_000 + 97 * s,
            device=device,
            hidden=hidden,
            n_layers=n_layers,
        )
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
            lr_theta=lr_theta,
            lr_alpha=lr_alpha,
            alpha_init=alpha_init,
            seed=seed + 1000 * s + 7,
            device=device,
            hidden=hidden,
            n_layers=n_layers,
        )
        rhos[s] = rho
        abars[s] = abar

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
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument(
        "--device",
        default="auto",
        help="torch device, or 'auto' for cuda if available",
    )
    p.add_argument(
        "--n_noisy",
        type=int,
        nargs="+",
        default=list(N_NOISY_GRID_DEFAULT),
        help="noisy-expert counts to sweep",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.out_dir):
        if not args.overwrite:
            raise FileExistsError(args.out_dir)
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    device = get_device(None if args.device == "auto" else args.device)
    status_print(f"wk_helps | device={device}")
    n_noisy_grid = tuple(args.n_noisy)
    rows: List[dict] = []
    idx = 0

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
                seed=9200 + 19 * idx,
                device=device,
                hidden=args.hidden,
                n_layers=args.n_layers,
            )
            row = summarize("mlp", n_noisy, v, rho, abar)
            rows.append(row)
            status_print(
                f"[mlp      2R{n_noisy}N] {v.name:7s} "
                f"correct={row['correct']:.3f} |corr|={row['mean_abs_corr']:.3f} "
                f"aR={row['abar_R']:+.3f} |aN|={row['abar_N_abs']:.3f}"
            )
            idx += 1

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "wk_helps_sweep.csv"), index=False)

    pivots = []
    for n_noisy in n_noisy_grid:
        a = table[(table.n_noisy == n_noisy) & (table.variant == "no_w")].iloc[0]
        b = table[(table.n_noisy == n_noisy) & (table.variant == "with_w")].iloc[0]
        pivots.append(
            {
                "head": "mlp",
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
