"""
Fig. 5 (main_v2): overlap counterfactual (TTP / No-alpha / DS-Sym).
Label: fig:synthetic-overlap-sweep

Uses the production PEBBLE gen_net shared reward head.

Example:
  python fig5_overlap.py --seeds 120 --overwrite
"""


from __future__ import annotations

import argparse
import os
import shutil
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from synthetic_shared_core import (
    apply_init_kind,
    build_reward_mlp,
    progress_range,
    rowwise_corr,
    segment_returns,
    sigmoid_np,
    status_print,
    teacher_segment_returns,
)


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
    init_kind: str = "stabilized",
    torch_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stabilized shared-head TTP (or No-α if fix_alpha). One seed: states [N,T,D]."""
    n, t, d = states.shape
    k, m = y.shape[0], y.shape[1]
    device = states.device

    torch.manual_seed(torch_seed)
    net = build_reward_mlp(d).to(device)
    apply_init_kind(net, init_kind)

    if fix_alpha:
        alpha_param = None
    else:
        alpha_param = torch.nn.Parameter(torch.full((k,), alpha_init, device=device))

    params = list(net.parameters())
    if alpha_param is not None:
        opt = torch.optim.SGD(
            [
                {"params": params, "lr": lr_theta},
                {"params": [alpha_param], "lr": lr_alpha},
            ]
        )
    else:
        opt = torch.optim.SGD(params, lr=lr_theta)

    for _ in range(steps):
        opt.zero_grad()
        R = segment_returns(net, states)
        if fix_alpha:
            coef = torch.ones(k, device=device)
            w = torch.ones(k, device=device)
        else:
            trust = torch.tanh(alpha_param)
            denom = torch.max(trust.abs()).clamp_min(1e-12).detach()
            coef = trust / denom
            w = (k * trust.abs() / trust.abs().sum().clamp_min(1e-12)).detach()

        loss_R = 0.0
        loss_A = 0.0
        for e in range(k):
            delta = R[i_all[e]] - R[j_all[e]]
            logits_R = coef[e] * delta
            bce = F.binary_cross_entropy_with_logits(logits_R, y[e], reduction="none")
            loss_R = loss_R + (w[e] * bce).mean() / k
            if not fix_alpha:
                logits_A = coef[e] * delta.detach()
                loss_A = loss_A + F.binary_cross_entropy_with_logits(
                    logits_A, y[e], reduction="none"
                ).mean() / k

        (loss_R + loss_A).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()

    with torch.no_grad():
        R = segment_returns(net, states).cpu().numpy()
        if fix_alpha:
            abar = np.ones(k)
        else:
            trust = torch.tanh(alpha_param)
            abar = (trust / torch.max(trust.abs()).clamp_min(1e-12)).cpu().numpy()
    return R, abar


def train_ds_sym(
    states: torch.Tensor,
    i_all: torch.Tensor,
    j_all: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr: float = 0.05,
    torch_seed: int = 0,
) -> np.ndarray:
    """Shared-head DS-Sym: slope s, per-expert flip probs q_k, shared gen_net."""
    n, t, d = states.shape
    k, m = y.shape[0], y.shape[1]
    device = states.device

    # Default PyTorch init (nonzero) — zero last layer would make R≡0.
    torch.manual_seed(torch_seed)
    net = build_reward_mlp(d).to(device)
    s = torch.nn.Parameter(torch.ones((), device=device))
    q_raw = torch.nn.Parameter(torch.zeros(k, device=device))
    opt = torch.optim.SGD(
        [
            {"params": net.parameters(), "lr": lr},
            {"params": [s, q_raw], "lr": lr},
        ]
    )

    for _ in range(steps):
        opt.zero_grad()
        R = segment_returns(net, states)
        std = R.std().clamp_min(1e-6)
        R = (R - R.mean()) / std
        q = torch.sigmoid(q_raw)
        loss = 0.0
        for e in range(k):
            delta = R[i_all[e]] - R[j_all[e]]
            p_z = torch.sigmoid(s * delta)
            p_y = q[e] * p_z + (1.0 - q[e]) * (1.0 - p_z)
            p_y = p_y.clamp(1e-6, 1.0 - 1e-6)
            loss = loss + F.binary_cross_entropy(p_y, y[e], reduction="none").mean() / k
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()

    with torch.no_grad():
        R = segment_returns(net, states)
        std = R.std().clamp_min(1e-6)
        R = (R - R.mean()) / std
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

    k, n_total, T, d, m = 4, 500, 50, 16, 256
    bs = n_total // k
    qs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    methods = ("ttp", "no_alpha", "ds_sym")
    rows = []
    device = torch.device("cpu")

    for qi, q in enumerate(qs):
        rng = np.random.default_rng(7000 + qi)
        seeds = args.seeds
        states_np = rng.normal(size=(seeds, n_total, T, d)).astype(np.float32)

        r_star = np.zeros((seeds, n_total), dtype=np.float64)
        for s in progress_range(seeds, desc=f"q={q:g} teacher", leave=False):
            r = teacher_segment_returns(
                states_np[s],
                d=d,
                torch_seed=7000 + 10_000 + 97 * s + qi,
                device=device,
            )
            r_star[s] = (r - r.mean()) / (r.std() + 1e-12)

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

        for method in methods:
            R_all = np.zeros((seeds, n_total))
            abar_all = np.zeros((seeds, k))
            for s in progress_range(
                seeds, desc=f"q={q:g} {method}", leave=True
            ):
                states = torch.as_tensor(states_np[s], dtype=torch.float32, device=device)
                i_all = torch.as_tensor(i_all_np[s], dtype=torch.long, device=device)
                j_all = torch.as_tensor(j_all_np[s], dtype=torch.long, device=device)
                y = torch.as_tensor(y_np[s], dtype=torch.float32, device=device)
                ts = 7000 + 1_000 * qi + 31 * s
                if method == "ttp":
                    R, abar = train_ttp(
                        states,
                        i_all,
                        j_all,
                        y,
                        steps=args.steps,
                        fix_alpha=False,
                        init_kind="stabilized",
                        torch_seed=ts,
                    )
                    abar_all[s] = abar
                elif method == "no_alpha":
                    R, abar = train_ttp(
                        states,
                        i_all,
                        j_all,
                        y,
                        steps=args.steps,
                        fix_alpha=True,
                        init_kind="stabilized",
                        torch_seed=ts,
                    )
                    abar_all[s] = abar
                else:
                    R = train_ds_sym(
                        states, i_all, j_all, y, steps=args.steps, torch_seed=ts
                    )
                R_all[s] = R

            rho = rowwise_corr(R_all, r_star)
            glob = np.abs(rho)
            locals_ = [np.abs(rowwise_corr(R_all[:, blk], r_star[:, blk])) for blk in blocks]
            loc = np.mean(np.stack(locals_, 0), 0)
            if method == "ttp":
                aA = abar_all[:, -1]
                aA_mean = float(np.mean(aA))
                aA_med = float(np.median(aA))
                aA_q25 = float(np.percentile(aA, 25))
                aA_q75 = float(np.percentile(aA, 75))
            else:
                aA_mean = aA_med = aA_q25 = aA_q75 = float("nan")
            row = {
                "q": q,
                "method": method,
                "global_med": float(np.mean(glob)),
                "global_q25": float(np.percentile(glob, 25)),
                "global_q75": float(np.percentile(glob, 75)),
                "signed_med": float(np.mean(rho)),
                "signed_q25": float(np.percentile(rho, 25)),
                "signed_q75": float(np.percentile(rho, 75)),
                "local_med": float(np.mean(loc)),
                "correct": float((rho > 0.05).mean()),
                "mean_abar_A": aA_mean,
                "med_abar_A": aA_med,
                "abar_A_q25": aA_q25,
                "abar_A_q75": aA_q75,
            }
            rows.append(row)
            status_print(
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

    qs = sorted(table.q.unique())
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    if len(qs) == 1:
        x = np.arange(len(methods))
        for mi, method in enumerate(methods):
            sub = table[table.method == method].iloc[0]
            ax.bar(
                mi,
                sub.signed_med,
                color=colors[method],
                yerr=[[sub.signed_med - sub.signed_q25], [sub.signed_q75 - sub.signed_med]],
                capsize=4,
                label=labels[method],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in methods])
        ax.set_xlabel(rf"method ($q={qs[0]:g}$, $n=500$)")
    else:
        for method in methods:
            sub = table[table.method == method].sort_values("q")
            ax.fill_between(
                sub.q, sub.signed_q25, sub.signed_q75, color=colors[method], alpha=0.15
            )
            ax.plot(sub.q, sub.signed_med, "o-", color=colors[method], label=labels[method])
        ax.set_xlabel("shared-pair fraction $q$")
        ax.legend(fontsize=8, loc="lower right")
    ax.axhline(0.0, color="gray", ls=":", lw=0.9)
    ax.set_ylabel(r"median signed $\mathrm{corr}(\hat R,R^*)$")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "3R1A_overlap_global_vs_local.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--replot":
        out = sys.argv[2] if len(sys.argv) > 2 else "final_results/synthetic_overlap_sweep"
        plot_overlap_figure(pd.read_csv(os.path.join(out, "overlap_shared.csv")), out)
        print(f"replot OK: {out}")
    else:
        main()
