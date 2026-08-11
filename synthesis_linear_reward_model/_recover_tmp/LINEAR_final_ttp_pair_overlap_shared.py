"""Overlap sweep (shared head): TTP vs No-α vs Dawid–Skene (DS-Sym)."""

from __future__ import annotations

import argparse
import os
import shutil
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from final_ttp_synthetic_shared_core import rowwise_corr, sigmoid_np


def _returns(states: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return torch.tanh(torch.einsum("sntd,sd->snt", states, theta)).sum(dim=2)


def train_ttp(
    states: torch.Tensor,
    i_all: torch.Tensor,
    j_all: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr_theta: float = 0.05,
    lr_alpha: float = 0.005,
    alpha_init: float = 0.01,
    fix_alpha: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stabilized shared-head TTP (or No-α if fix_alpha)."""
    seeds, n_total, T, d = states.shape
    k, m = y.shape[1], y.shape[2]
    theta = torch.nn.Parameter(torch.zeros(seeds, d, device=states.device))
    if fix_alpha:
        alpha = torch.full((seeds, k), 1.0, device=states.device)  # fixed equal trust before tanh/maxnorm
        # Use raw ones after max-norm: all coef=1. Keep as buffer.
        alpha_param = None
    else:
        alpha_param = torch.nn.Parameter(torch.full((seeds, k), alpha_init, device=states.device))

    for _ in range(steps):
        if theta.grad is not None:
            theta.grad = None
        if alpha_param is not None and alpha_param.grad is not None:
            alpha_param.grad = None

        R = _returns(states, theta)
        if fix_alpha:
            coef = torch.ones(seeds, k, device=states.device)
            w = torch.ones(seeds, k, device=states.device)
            trust = coef
        else:
            trust = torch.tanh(alpha_param)
            denom = trust.abs().amax(1, keepdim=True).clamp_min(1e-12).detach()
            coef = trust / denom
            w = (k * trust.abs() / trust.abs().sum(1, keepdim=True).clamp_min(1e-12)).detach()

        loss_R = 0.0
        loss_A = 0.0
        for e in range(k):
            ie, je = i_all[:, e], j_all[:, e]
            delta = R.gather(1, ie) - R.gather(1, je)
            logits_R = coef[:, e : e + 1].detach() * delta
            bce = F.binary_cross_entropy_with_logits(logits_R, y[:, e], reduction="none")
            loss_R = loss_R + (w[:, e : e + 1] * bce).mean(dim=1).sum() / k
            if not fix_alpha:
                logits_A = coef[:, e : e + 1] * delta.detach()
                loss_A = loss_A + F.binary_cross_entropy_with_logits(
                    logits_A, y[:, e], reduction="none"
                ).mean(dim=1).sum() / k

        (loss_R + loss_A).backward()
        with torch.no_grad():
            g = theta.grad
            gn = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
            g.mul_(torch.clamp(10.0 / gn, max=1.0))
            theta.data.sub_(lr_theta * g)
            if alpha_param is not None:
                alpha_param.data.sub_(lr_alpha * alpha_param.grad)

    with torch.no_grad():
        R = _returns(states, theta).cpu().numpy()
        if fix_alpha:
            abar = np.ones((seeds, k))
        else:
            trust = torch.tanh(alpha_param)
            abar = (trust / trust.abs().amax(1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
    return R, abar


def train_ds_sym(
    states: torch.Tensor,
    i_all: torch.Tensor,
    j_all: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr: float = 0.05,
) -> np.ndarray:
    """Shared-head DS-Sym: slope s, per-expert flip probs q_k, shared theta."""
    seeds, _, _, d = states.shape
    k, m = y.shape[1], y.shape[2]
    # Small random init — zero init makes R≡0 and std-normalization NaN
    theta = torch.nn.Parameter(0.01 * torch.randn(seeds, d, device=states.device))
    s = torch.nn.Parameter(torch.ones(seeds, device=states.device))
    q_raw = torch.nn.Parameter(torch.zeros(seeds, k, device=states.device))
    opt = torch.optim.SGD([theta, s, q_raw], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        R = _returns(states, theta)
        # center/scale per seed for DS identifiability convenience
        std = R.std(1, keepdim=True).clamp_min(1e-6)
        R = (R - R.mean(1, keepdim=True)) / std
        q = torch.sigmoid(q_raw)
        loss = 0.0
        for e in range(k):
            ie, je = i_all[:, e], j_all[:, e]
            delta = R.gather(1, ie) - R.gather(1, je)
            p_z = torch.sigmoid(s.unsqueeze(1) * delta)
            p_y = q[:, e : e + 1] * p_z + (1.0 - q[:, e : e + 1]) * (1.0 - p_z)
            p_y = p_y.clamp(1e-6, 1.0 - 1e-6)
            loss = loss + F.binary_cross_entropy(p_y, y[:, e], reduction="none").mean(dim=1).sum() / k
        loss.backward()
        with torch.no_grad():
            if theta.grad is not None:
                gn = theta.grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
                theta.grad.mul_(torch.clamp(10.0 / gn, max=1.0))
        opt.step()

    with torch.no_grad():
        R = _returns(states, theta)
        std = R.std(1, keepdim=True).clamp_min(1e-6)
        R = (R - R.mean(1, keepdim=True)) / std
        return R.cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="final_results/synthetic_overlap_sweep")
    p.add_argument("--seeds", type=int, default=120)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.out_dir):
        if not args.overwrite:
            raise FileExistsError(args.out_dir)
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    k, n_total, T, d, m = 4, 48, 50, 16, 256
    bs = n_total // k
    qs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    methods = ("ttp", "no_alpha", "ds_sym")
    rows = []

    for qi, q in enumerate(qs):
        rng = np.random.default_rng(7000 + qi)
        seeds = args.seeds
        theta_star = rng.normal(size=(seeds, d))
        theta_star /= np.linalg.norm(theta_star, axis=1, keepdims=True) + 1e-12
        states_np = rng.normal(size=(seeds, n_total, T, d))
        r_star = np.tanh(np.einsum("sntd,sd->snt", states_np, theta_star)).sum(2)
        r_star = (r_star - r_star.mean(1, keepdims=True)) / (r_star.std(1, keepdims=True) + 1e-12)

        blocks = [np.arange(b * bs, (b + 1) * bs) for b in range(k)]
        n_shared = int(round(q * m))
        n_priv = m - n_shared

        i_all_np = np.zeros((seeds, k, m), dtype=np.int64)
        j_all_np = np.zeros((seeds, k, m), dtype=np.int64)
        if n_shared > 0:
            i_s = rng.integers(0, n_total, (seeds, n_shared))
            j_s = rng.integers(0, n_total, (seeds, n_shared))
            same = i_s == j_s
            while same.any():
                j_s[same] = rng.integers(0, n_total, int(same.sum()))
                same = i_s == j_s
            i_all_np[:, :, :n_shared] = i_s[:, None, :]
            j_all_np[:, :, :n_shared] = j_s[:, None, :]
        for e, blk in enumerate(blocks):
            if n_priv <= 0:
                continue
            i_p = rng.choice(blk, size=(seeds, n_priv), replace=True)
            j_p = rng.choice(blk, size=(seeds, n_priv), replace=True)
            same = i_p == j_p
            while same.any():
                j_p[same] = rng.choice(blk, size=int(same.sum()), replace=True)
                same = i_p == j_p
            i_all_np[:, e, n_shared:] = i_p
            j_all_np[:, e, n_shared:] = j_p

        betas = np.array([1.0, 1.0, 1.0, -1.0])
        y_np = np.zeros((seeds, k, m))
        for e in range(k):
            dstar = np.take_along_axis(r_star, i_all_np[:, e], 1) - np.take_along_axis(
                r_star, j_all_np[:, e], 1
            )
            y_np[:, e] = (rng.random((seeds, m)) < sigmoid_np(betas[e] * dstar)).astype(float)

        states = torch.as_tensor(states_np, dtype=torch.float32)
        i_all = torch.as_tensor(i_all_np, dtype=torch.long)
        j_all = torch.as_tensor(j_all_np, dtype=torch.long)
        y = torch.as_tensor(y_np, dtype=torch.float32)

        for method in methods:
            if method == "ttp":
                R, abar = train_ttp(states, i_all, j_all, y, steps=args.steps, fix_alpha=False)
                aA = float(abar[:, -1].mean())
            elif method == "no_alpha":
                R, abar = train_ttp(states, i_all, j_all, y, steps=args.steps, fix_alpha=True)
                aA = float("nan")
            else:
                R = train_ds_sym(states, i_all, j_all, y, steps=args.steps)
                aA = float("nan")

            rho = rowwise_corr(R, r_star)
            glob = np.abs(rho)
            locals_ = [np.abs(rowwise_corr(R[:, blk], r_star[:, blk])) for blk in blocks]
            loc = np.mean(np.stack(locals_, 0), 0)
            row = {
                "q": q,
                "method": method,
                "global_med": float(np.median(glob)),
                "global_q25": float(np.percentile(glob, 25)),
                "global_q75": float(np.percentile(glob, 75)),
                "signed_med": float(np.median(rho)),
                "signed_q25": float(np.percentile(rho, 25)),
                "signed_q75": float(np.percentile(rho, 75)),
                "local_med": float(np.median(loc)),
                "correct": float((rho > 0.05).mean()),
                "mean_abar_A": aA,
            }
            rows.append(row)
            print(
                f"q={q:<4g} {method:8s} |rho|_med={row['global_med']:.3f} "
                f"rho_med={row['signed_med']:+.3f} correct={row['correct']:.3f}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "overlap_shared.csv"), index=False)
    plot_overlap_figure(table, args.out_dir)
    print(f"OUT: {args.out_dir}")


def plot_overlap_figure(table: pd.DataFrame, out_dir: str) -> None:
    colors = {"ttp": "#4c72b0", "no_alpha": "#dd8452", "ds_sym": "#55a868"}
    labels = {"ttp": "TTP", "no_alpha": r"No-$\alpha$", "ds_sym": "DS-Sym"}
    methods = ("ttp", "no_alpha", "ds_sym")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    for method in methods:
        sub = table[table.method == method].sort_values("q")
        ax.fill_between(
            sub.q, sub.global_q25, sub.global_q75, color=colors[method], alpha=0.15
        )
        ax.plot(sub.q, sub.global_med, "o-", color=colors[method], label=labels[method])
    sub_t = table[table.method == "ttp"].sort_values("q")
    ax.plot(sub_t.q, sub_t.local_med, "s--", color="#8172b3", label=r"TTP local $|\mathrm{corr}|$")
    ax.set_xlabel("shared-pair fraction $q$")
    ax.set_ylabel(r"median $|\mathrm{corr}|$")
    ax.set_ylim(0, 1.05)
    ax.set_title(r"(a) Sign-blind $|\mathrm{corr}|$")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1]
    for method in methods:
        sub = table[table.method == method].sort_values("q")
        ax.fill_between(
            sub.q, sub.signed_q25, sub.signed_q75, color=colors[method], alpha=0.15
        )
        ax.plot(sub.q, sub.signed_med, "o-", color=colors[method], label=labels[method])
    ax.axhline(0.0, color="gray", ls=":", lw=0.9)
    ax.set_xlabel("shared-pair fraction $q$")
    ax.set_ylabel(r"median signed $\mathrm{corr}(\hat R,R^*)$")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(r"(b) Signed $\mathrm{corr}$")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, ls=":", alpha=0.4)

    fig.suptitle("Overlap counterfactual (shared head, 3R1A, Stabilized)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "3R1A_overlap_global_vs_local.png"),
        dpi=200,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--replot":
        out = sys.argv[2] if len(sys.argv) > 2 else "final_results/synthetic_overlap_sweep"
        plot_overlap_figure(pd.read_csv(os.path.join(out, "overlap_shared.csv")), out)
        print(f"replot OK: {out}")
    else:
        main()
