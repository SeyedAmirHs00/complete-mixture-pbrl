"""Partial/stochastic adversaries on shared head (3R1A; Stabilized vs Consensus)."""

from __future__ import annotations

import argparse
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_ttp_synthetic_shared_core import SharedVariant, run_shared_variant, sigmoid


def run_with_stochastic_adv(
    *,
    seeds: int,
    steps: int,
    target_rms: float,
    consensus_coef: float,
    beta_adv: float | None,
    flip_prob: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Custom label generation for partial/stochastic adversary, then shared train."""
    from final_ttp_synthetic_shared_core import calibrate_theta_scale

    rng = np.random.default_rng(seed)
    k, n_seg, T, d, pairs = 4, 48, 50, 16, 256
    betas = np.array([1.0, 1.0, 1.0, -1.0])  # placeholder; labels overridden for expert 3

    theta_scale = calibrate_theta_scale(target_rms, seeds=40, n_seg=n_seg, T=T, d=d, rng=rng)
    theta_star = rng.normal(size=(seeds, d))
    theta_star /= np.linalg.norm(theta_star, axis=1, keepdims=True) + 1e-12
    states = rng.normal(size=(seeds, n_seg, T, d))
    r_star = np.tanh(np.einsum("sntd,sd->snt", states, theta_star)).sum(2)
    r_star = (r_star - r_star.mean(1, keepdims=True)) / (r_star.std(1, keepdims=True) + 1e-12)

    i = rng.integers(0, n_seg, (seeds, pairs))
    j = rng.integers(0, n_seg, (seeds, pairs))
    d_star = np.take_along_axis(r_star, i, 1) - np.take_along_axis(r_star, j, 1)

    y = np.zeros((seeds, k, pairs))
    for e in range(3):
        y[:, e] = (rng.random((seeds, pairs)) < sigmoid(d_star)).astype(float)
    if flip_prob is not None:
        # stochastic: with prob p anti-oracle, else reliable
        anti = (rng.random((seeds, pairs)) < sigmoid(-d_star)).astype(float)
        rel = (rng.random((seeds, pairs)) < sigmoid(d_star)).astype(float)
        use_anti = rng.random((seeds, pairs)) < flip_prob
        y[:, 3] = np.where(use_anti, anti, rel)
    else:
        ba = float(beta_adv)
        y[:, 3] = (rng.random((seeds, pairs)) < sigmoid(ba * d_star)).astype(float)

    consensus_target = y.mean(1)
    if theta_scale == 0:
        theta = np.zeros((seeds, d))
    else:
        theta = rng.normal(scale=theta_scale / np.sqrt(d), size=(seeds, d))
    alpha = np.full((seeds, k), 0.01)
    rows = np.repeat(np.arange(seeds), pairs)

    for _ in range(steps):
        pre = np.einsum("sntd,sd->snt", states, theta)
        r = np.tanh(pre)
        R = r.sum(2)
        trust = np.tanh(alpha)
        denom = np.maximum(np.abs(trust).max(1, keepdims=True), 1e-12)
        coef = trust / denom
        w = k * np.abs(trust) / np.maximum(np.abs(trust).sum(1, keepdims=True), 1e-12)
        delta = np.take_along_axis(R, i, 1) - np.take_along_axis(R, j, 1)
        err = sigmoid(coef[:, :, None] * delta[:, None, :]) - y
        coeff = (w[:, :, None] * err * coef[:, :, None]).mean(1) / pairs
        dL = np.zeros_like(R)
        np.add.at(dL, (rows, i.ravel()), coeff.ravel())
        np.add.at(dL, (rows, j.ravel()), -coeff.ravel())
        if consensus_coef > 0:
            anchor = (sigmoid(delta) - consensus_target) * (consensus_coef / pairs)
            np.add.at(dL, (rows, i.ravel()), anchor.ravel())
            np.add.at(dL, (rows, j.ravel()), -anchor.ravel())
        sech2 = 1 - r * r
        g = np.einsum("sn,snt,sntd->sd", dL, sech2, states)
        gn = np.linalg.norm(g, axis=1, keepdims=True)
        g *= np.minimum(1.0, 10 / (gn + 1e-12))
        ga = (err * delta[:, None, :]).mean(2) * (1 - trust * trust) / denom
        theta -= 0.05 * g
        alpha -= 0.005 * ga

    R = np.tanh(np.einsum("sntd,sd->snt", states, theta)).sum(2)
    from final_ttp_synthetic_shared_core import rowwise_corr

    rho = rowwise_corr(R, r_star)
    abar = np.tanh(alpha) / np.maximum(np.abs(np.tanh(alpha)).max(1, keepdims=True), 1e-12)
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
        ("consensus", dict(target_rms=1.4, consensus_coef=0.5)),
    ]

    rows = []
    idx = 0
    for sname, skw in settings:
        for mname, mkw in methods:
            idx += 1
            rho, abar = run_with_stochastic_adv(
                seeds=args.seeds, steps=args.steps, seed=9400 + idx, **skw, **mkw
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
    ax.set_title("Partial adversaries (shared head)")
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
    ax.set_title("Recovered trust (bars=A, diamonds=R)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "partial_adversary_recovered_trust.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
