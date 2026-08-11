"""Overlap sweep with shared head (3R1A). Shared theta couples blocks even at q=0."""

from __future__ import annotations

import argparse
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_ttp_synthetic_shared_core import rowwise_corr, sigmoid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="final_results/synthetic_overlap_sweep")
    p.add_argument("--seeds", type=int, default=120)
    p.add_argument("--steps", type=int, default=800)
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
    rows = []

    for qi, q in enumerate(qs):
        rng = np.random.default_rng(7000 + qi)
        seeds = args.seeds
        # shared features for all trajectories
        theta_star = rng.normal(size=(seeds, d))
        theta_star /= np.linalg.norm(theta_star, axis=1, keepdims=True) + 1e-12
        # n_total segments (trajectories)
        states = rng.normal(size=(seeds, n_total, T, d))
        r_star = np.tanh(np.einsum("sntd,sd->snt", states, theta_star)).sum(2)
        r_star = (r_star - r_star.mean(1, keepdims=True)) / (r_star.std(1, keepdims=True) + 1e-12)

        blocks = [np.arange(b * bs, (b + 1) * bs) for b in range(k)]
        n_shared = int(round(q * m))
        n_priv = m - n_shared

        # pair indices per expert: (seeds, k, m)
        i_all = np.zeros((seeds, k, m), dtype=int)
        j_all = np.zeros((seeds, k, m), dtype=int)
        if n_shared > 0:
            i_s = rng.integers(0, n_total, (seeds, n_shared))
            j_s = rng.integers(0, n_total, (seeds, n_shared))
            same = i_s == j_s
            while same.any():
                j_s[same] = rng.integers(0, n_total, int(same.sum()))
                same = i_s == j_s
            i_all[:, :, :n_shared] = i_s[:, None, :]
            j_all[:, :, :n_shared] = j_s[:, None, :]
        for e, blk in enumerate(blocks):
            if n_priv <= 0:
                continue
            i_p = rng.choice(blk, size=(seeds, n_priv), replace=True)
            j_p = rng.choice(blk, size=(seeds, n_priv), replace=True)
            same = i_p == j_p
            while same.any():
                j_p[same] = rng.choice(blk, size=int(same.sum()), replace=True)
                same = i_p == j_p
            i_all[:, e, n_shared:] = i_p
            j_all[:, e, n_shared:] = j_p

        betas = np.array([1.0, 1.0, 1.0, -1.0])
        y = np.zeros((seeds, k, m))
        for e in range(k):
            dstar = np.take_along_axis(r_star, i_all[:, e], 1) - np.take_along_axis(r_star, j_all[:, e], 1)
            y[:, e] = (rng.random((seeds, m)) < sigmoid(betas[e] * dstar)).astype(float)

        # Stabilized shared init
        theta = np.zeros((seeds, d))
        alpha = np.full((seeds, k), 0.01)

        for _ in range(args.steps):
            pre = np.einsum("sntd,sd->snt", states, theta)
            r = np.tanh(pre)
            R = r.sum(2)
            trust = np.tanh(alpha)
            denom = np.maximum(np.abs(trust).max(1, keepdims=True), 1e-12)
            coef = trust / denom
            w = k * np.abs(trust) / np.maximum(np.abs(trust).sum(1, keepdims=True), 1e-12)

            dL = np.zeros_like(R)
            g_alpha = np.zeros_like(alpha)
            for e in range(k):
                ie, je = i_all[:, e], j_all[:, e]
                delta = np.take_along_axis(R, ie, 1) - np.take_along_axis(R, je, 1)
                err = sigmoid(coef[:, e : e + 1] * delta) - y[:, e]
                coeff = (w[:, e : e + 1] * err * coef[:, e : e + 1]).mean(1) / m
                rows = np.arange(seeds)
                np.add.at(dL, (rows[:, None], ie), coeff)
                np.add.at(dL, (rows[:, None], je), -coeff)
                g_alpha[:, e] = (err * delta).mean(1) * (1 - trust[:, e] ** 2) / denom[:, 0]

            sech2 = 1 - r * r
            g = np.einsum("sn,snt,sntd->sd", dL, sech2, states)
            gn = np.linalg.norm(g, axis=1, keepdims=True)
            g *= np.minimum(1.0, 10 / (gn + 1e-12))
            theta -= 0.05 * g
            alpha -= 0.005 * g_alpha

        R = np.tanh(np.einsum("sntd,sd->snt", states, theta)).sum(2)
        glob = np.abs(rowwise_corr(R, r_star))
        locals_ = []
        for blk in blocks:
            locals_.append(np.abs(rowwise_corr(R[:, blk], r_star[:, blk])))
        loc = np.mean(np.stack(locals_, 0), 0)
        correct = float((rowwise_corr(R, r_star) > 0.05).mean())
        rows.append(
            {
                "q": q,
                "global_med": float(np.median(glob)),
                "global_q25": float(np.percentile(glob, 25)),
                "global_q75": float(np.percentile(glob, 75)),
                "local_med": float(np.median(loc)),
                "correct": correct,
            }
        )
        print(f"q={q:<4g} global_med={rows[-1]['global_med']:.3f} local_med={rows[-1]['local_med']:.3f} correct={correct:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "overlap_shared.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.fill_between(table.q, table.global_q25, table.global_q75, color="#4c72b0", alpha=0.2)
    ax.plot(table.q, table.global_med, "o-", color="#4c72b0", label=r"global $|\mathrm{corr}|$")
    ax.plot(table.q, table.local_med, "s-", color="#dd8452", label=r"local $|\mathrm{corr}|$")
    ax.set_xlabel("shared-pair fraction $q$")
    ax.set_ylabel("median alignment")
    ax.set_ylim(0, 1.05)
    ax.set_title("Overlap counterfactual (shared head, 3R1A, Stabilized)")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "3R1A_overlap_global_vs_local.png"), dpi=200, bbox_inches="tight")
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
