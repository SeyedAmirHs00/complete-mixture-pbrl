"""
Fig. 6 (main_v2): partial / stochastic adversaries.
Label: fig:partial-adversary

Example:
  python fig6_partial_adversary.py --seeds 200 --overwrite
"""


from __future__ import annotations

import argparse
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthetic_shared_core import sample_expert_pairs, sigmoid


def run_with_stochastic_adv(
    *,
    seeds: int,
    steps: int,
    target_rms: float,
    consensus_coef: float,
    beta_adv: float | None,
    flip_prob: float | None,
    seed: int,
    n_seg: int = 500,
    q: float = 0.0,
    pairs: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Custom label generation for partial/stochastic adversary, then shared train."""
    from synthetic_shared_core import calibrate_theta_scale

    rng = np.random.default_rng(seed)
    k, T, d = 4, 50, 16

    theta_scale = calibrate_theta_scale(target_rms, seeds=40, n_seg=n_seg, T=T, d=d, rng=rng)
    theta_star = rng.normal(size=(seeds, d))
    theta_star /= np.linalg.norm(theta_star, axis=1, keepdims=True) + 1e-12
    states = rng.normal(size=(seeds, n_seg, T, d))
    r_star = np.tanh(np.einsum("sntd,sd->snt", states, theta_star)).sum(2)
    r_star = (r_star - r_star.mean(1, keepdims=True)) / (r_star.std(1, keepdims=True) + 1e-12)

    i, j = sample_expert_pairs(rng, seeds=seeds, k=k, n_seg=n_seg, pairs=pairs, q=q)

    y = np.zeros((seeds, k, pairs))
    for e in range(3):
        d_star = np.take_along_axis(r_star, i[:, e], 1) - np.take_along_axis(r_star, j[:, e], 1)
        y[:, e] = (rng.random((seeds, pairs)) < sigmoid(d_star)).astype(float)
    d_adv = np.take_along_axis(r_star, i[:, 3], 1) - np.take_along_axis(r_star, j[:, 3], 1)
    if flip_prob is not None:
        # stochastic: with prob p anti-oracle, else reliable
        anti = (rng.random((seeds, pairs)) < sigmoid(-d_adv)).astype(float)
        rel = (rng.random((seeds, pairs)) < sigmoid(d_adv)).astype(float)
        use_anti = rng.random((seeds, pairs)) < flip_prob
        y[:, 3] = np.where(use_anti, anti, rel)
    else:
        ba = float(beta_adv)
        y[:, 3] = (rng.random((seeds, pairs)) < sigmoid(ba * d_adv)).astype(float)

    consensus_target = y.mean(1)
    if theta_scale == 0:
        theta0 = np.zeros((seeds, d))
    else:
        theta0 = rng.normal(scale=theta_scale / np.sqrt(d), size=(seeds, d))

    import torch
    import torch.nn.functional as F
    from synthetic_shared_core import _segment_returns, rowwise_corr

    device = torch.device("cpu")
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    i_t = torch.as_tensor(i, dtype=torch.long, device=device)
    j_t = torch.as_tensor(j, dtype=torch.long, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    y_bar = torch.as_tensor(consensus_target, dtype=torch.float32, device=device)
    theta = torch.nn.Parameter(torch.as_tensor(theta0, dtype=torch.float32, device=device))
    alpha = torch.nn.Parameter(torch.full((seeds, k), 0.01, device=device))

    for _ in range(steps):
        if theta.grad is not None:
            theta.grad = None
        if alpha.grad is not None:
            alpha.grad = None
        R = _segment_returns(states_t, theta, True)
        delta = R.gather(1, i_t.reshape(seeds, -1)).reshape(seeds, k, pairs) - R.gather(
            1, j_t.reshape(seeds, -1)
        ).reshape(seeds, k, pairs)
        trust = torch.tanh(alpha)
        denom = trust.abs().amax(1, keepdim=True).clamp_min(1e-12).detach()
        coef = trust / denom
        w = (k * trust.abs() / trust.abs().sum(1, keepdim=True).clamp_min(1e-12)).detach()
        logits_R = coef.detach().unsqueeze(2) * delta
        bce_R = F.binary_cross_entropy_with_logits(logits_R, y_t, reduction="none")
        loss_R = (w.unsqueeze(2) * bce_R).mean(dim=(1, 2)).sum()
        if consensus_coef > 0:
            loss_R = loss_R + consensus_coef * F.binary_cross_entropy_with_logits(
                delta.mean(dim=1), y_bar, reduction="none"
            ).mean(dim=1).sum()
        logits_A = coef.unsqueeze(2) * delta.detach()
        loss_A = F.binary_cross_entropy_with_logits(logits_A, y_t, reduction="none").mean(
            dim=(1, 2)
        ).sum()
        (loss_R + loss_A).backward()
        with torch.no_grad():
            g = theta.grad
            gn = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
            g.mul_(torch.clamp(10.0 / gn, max=1.0))
            theta.data.sub_(0.05 * g)
            alpha.data.sub_(0.005 * alpha.grad)

    with torch.no_grad():
        R = _segment_returns(states_t, theta, True).cpu().numpy()
        trust = torch.tanh(alpha)
        abar = (trust / trust.abs().amax(1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
    rho = rowwise_corr(R, r_star)
    return rho, abar


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="final_results/synthetic_partial_adversary")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.out_dir):
        if not args.overwrite:
            raise FileExistsError(args.out_dir)
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    settings = [
        ("perfect_flip", dict(beta_adv=-1.0, flip_prob=None)),
        ("beta_-0.5", dict(beta_adv=-0.5, flip_prob=None)),
        ("beta_-0.25", dict(beta_adv=-0.25, flip_prob=None)),
        ("stoch_p0.5", dict(beta_adv=None, flip_prob=0.5)),
        ("stoch_p0.25", dict(beta_adv=None, flip_prob=0.25)),
    ]
    methods = [
        ("stabilized", dict(target_rms=0.0, consensus_coef=0.0)),
        ("standard", dict(target_rms=1.4, consensus_coef=0.0)),
    ]

    rows = []
    idx = 0
    for sname, skw in settings:
        for mname, mkw in methods:
            idx += 1
            rho, abar = run_with_stochastic_adv(
                seeds=args.seeds,
                steps=args.steps,
                seed=9400 + idx,
                n_seg=500,
                q=0.0,
                **skw,
                **mkw,
            )
            rows.append(
                {
                    "setting": sname,
                    "method": mname,
                    "correct": float((rho > 0.05).mean()),
                    "mean_rho": float(rho.mean()),
                    "abar_R": float(abar[:, :3].mean()),
                    "abar_A": float(abar[:, 3].mean()),
                }
            )
            print(
                f"[partial] {sname:12s} {mname:10s} correct={rows[-1]['correct']:.3f} "
                f"aR={rows[-1]['abar_R']:+.2f} aA={rows[-1]['abar_A']:+.2f}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "partial_adversary_shared.csv"), index=False)

    # plots
    settings_order = [s for s, _ in settings]
    x = np.arange(len(settings_order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for mi, (mname, _) in enumerate(methods):
        vals = [table[(table.setting == s) & (table.method == mname)].iloc[0].correct for s in settings_order]
        ax.bar(x + (mi - 0.5) * width, vals, width, label=mname)
    ax.set_xticks(x)
    ax.set_xticklabels(settings_order, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Correct-branch rate")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "partial_adversary_correct_branch.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for mi, (mname, _) in enumerate(methods):
        aA = [table[(table.setting == s) & (table.method == mname)].iloc[0].abar_A for s in settings_order]
        aR = [table[(table.setting == s) & (table.method == mname)].iloc[0].abar_R for s in settings_order]
        ax.bar(x + (mi - 0.5) * width, aA, width, label=f"{mname} A")
        ax.scatter(x + (mi - 0.5) * width, aR, marker="D", color="black", zorder=3, s=20)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(settings_order, rotation=20, ha="right")
    ax.set_ylabel(r"mean $\bar\alpha$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "partial_adversary_recovered_trust.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
