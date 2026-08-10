"""
Fig. 6 (main_v2): partial / stochastic adversaries.
Label: fig:partial-adversary

Uses the production PEBBLE gen_net shared reward head.

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
import torch

from synthetic_shared_core import (
    SharedVariant,
    rowwise_corr,
    sample_expert_pairs,
    sigmoid,
    teacher_segment_returns,
    train_one_seed_ttp,
)


def run_with_stochastic_adv(
    *,
    seeds: int,
    steps: int,
    init_kind: str,
    consensus_coef: float,
    beta_adv: float | None,
    flip_prob: float | None,
    seed: int,
    n_seg: int = 500,
    q: float = 0.0,
    pairs: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Custom label generation for partial/stochastic adversary, then shared MLP train."""
    rng = np.random.default_rng(seed)
    k, T, d = 4, 50, 16
    device = torch.device("cpu")

    states = rng.normal(size=(seeds, n_seg, T, d)).astype(np.float32)
    r_star = np.zeros((seeds, n_seg), dtype=np.float64)
    for s in range(seeds):
        r = teacher_segment_returns(
            states[s],
            d=d,
            torch_seed=seed + 10_000 + 97 * s,
            device=device,
        )
        r_star[s] = (r - r.mean()) / (r.std() + 1e-12)

    i, j = sample_expert_pairs(rng, seeds=seeds, k=k, n_seg=n_seg, pairs=pairs, q=q)

    y = np.zeros((seeds, k, pairs))
    for e in range(3):
        d_star = np.take_along_axis(r_star, i[:, e], 1) - np.take_along_axis(r_star, j[:, e], 1)
        y[:, e] = (rng.random((seeds, pairs)) < sigmoid(d_star)).astype(float)
    d_adv = np.take_along_axis(r_star, i[:, 3], 1) - np.take_along_axis(r_star, j[:, 3], 1)
    if flip_prob is not None:
        anti = (rng.random((seeds, pairs)) < sigmoid(-d_adv)).astype(float)
        rel = (rng.random((seeds, pairs)) < sigmoid(d_adv)).astype(float)
        use_anti = rng.random((seeds, pairs)) < flip_prob
        y[:, 3] = np.where(use_anti, anti, rel)
    else:
        ba = float(beta_adv)
        y[:, 3] = (rng.random((seeds, pairs)) < sigmoid(ba * d_adv)).astype(float)

    consensus_target = y.mean(1)
    variant = SharedVariant(
        init_kind,
        init_kind,
        init_kind=init_kind,
        consensus_coef=consensus_coef,
    )

    rhos = np.zeros(seeds)
    abars = np.zeros((seeds, k))
    for s in range(seeds):
        states_t = torch.as_tensor(states[s], dtype=torch.float32, device=device)
        i_t = torch.as_tensor(i[s], dtype=torch.long, device=device)
        j_t = torch.as_tensor(j[s], dtype=torch.long, device=device)
        y_t = torch.as_tensor(y[s], dtype=torch.float32, device=device)
        y_bar = torch.as_tensor(consensus_target[s], dtype=torch.float32, device=device)
        R_hat, abar, _ = train_one_seed_ttp(
            states_t,
            i_t,
            j_t,
            y_t,
            y_bar,
            variant,
            steps=steps,
            lr_theta=0.05,
            lr_alpha=0.005,
            alpha_init=0.01,
            torch_seed=seed + 1_000 + 31 * s,
        )
        rhos[s] = float(rowwise_corr(R_hat[None, :], r_star[s : s + 1])[0])
        abars[s] = abar
    return rhos, abars


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
        ("stabilized", dict(init_kind="stabilized", consensus_coef=0.0)),
        ("standard", dict(init_kind="standard", consensus_coef=0.0)),
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
