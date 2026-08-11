"""
ALL paper synthetic diagnostics on a shared reward head.

Produces:
  final_results/synthetic_shared_all/
    branch_symmetry/          (table + alpha bar figs)
    sigma_consensus_sweep/    (replaces sum-trajectory + why-real)
    ...

Example:
  python final_ttp_synthetic_shared_all.py --mode branch --seeds 200 --overwrite
  python final_ttp_synthetic_shared_all.py --mode sigma --seeds 200 --overwrite
  python final_ttp_synthetic_shared_all.py --mode all --seeds 200 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_ttp_synthetic_shared_core import (
    SHARED_BRANCH_VARIANTS,
    SharedVariant,
    build_k4_configs,
    calibrate_theta_scale,
    run_shared_variant,
)


def ensure_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    os.makedirs(path)


def run_branch(out_root: str, seeds: int, steps: int, overwrite: bool) -> None:
    out = os.path.join(out_root, "branch_symmetry")
    ensure_dir(out, overwrite)
    configs = build_k4_configs()
    # paper table uses 3R1N, 3R1A, 1R3A (keep 2R1A1N in sigma sweep)
    order = ["3R1N", "3R1A", "1R3A"]
    rows = []
    # Store (rho, abar) so trust bars can condition on the modal branch
    # (mixing correct+flipped seeds washes out 3R1N Stabilized/Consensus).
    run_store: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {c: {} for c in order}
    cal_rng = np.random.default_rng(9017)
    idx = 0
    for cfg in order:
        betas = configs[cfg]
        for v in SHARED_BRANCH_VARIANTS:
            idx += 1
            rho, abar, rms0 = run_shared_variant(
                betas,
                v,
                seeds=seeds,
                steps=steps,
                seed=9017 + 31 * idx,
                cal_rng=cal_rng,
            )
            correct = rho > 0.05
            flipped = rho < -0.05
            # Trust stats on the modal branch (matches Table~\ref{tab:synthetic-branch} story)
            if float(correct.mean()) >= float(flipped.mean()):
                mask = correct
                branch_tag = "correct"
            else:
                mask = flipped
                branch_tag = "flipped"
            abar_m = abar[mask] if mask.any() else abar
            rows.append(
                {
                    "config": cfg,
                    "variant": v.name,
                    "init_rms": rms0,
                    "correct_branch_rate": float(correct.mean()),
                    "flipped_branch_rate": float(flipped.mean()),
                    "mean_rho": float(rho.mean()),
                    "trust_conditioned_on": branch_tag,
                    "n_trust_seeds": int(mask.sum()),
                    "mean_abar_R": float(abar_m[:, :3].mean()) if cfg != "1R3A" else float(abar_m[:, 0].mean()),
                    "mean_abar_N": float(abar_m[:, 3].mean()) if "N" in cfg else np.nan,
                    "mean_abar_A": float(abar_m[:, -1].mean()) if "A" in cfg else np.nan,
                }
            )
            run_store[cfg][v.name] = (rho, abar)
            print(
                f"[branch] {cfg:6s} {v.name:10s} correct={rows[-1]['correct_branch_rate']:.3f} "
                f"rho={rows[-1]['mean_rho']:+.3f} rms0={rms0:.3f} "
                f"trust@{branch_tag} aR={rows[-1]['mean_abar_R']:+.2f}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "paper_table_synthetic_symmetry_fix.csv"), index=False)

    # wide table like paper
    wide = table.pivot(index="config", columns="variant", values="correct_branch_rate")
    wide = wide.reindex(columns=["standard", "stabilized", "consensus"])
    wide.to_csv(os.path.join(out, "branch_correct_wide.csv"))

    # trust bar plots: mean±std on the modal branch only
    colors = {"standard": "#4c72b0", "stabilized": "#dd8452", "consensus": "#55a868"}
    for cfg in order:
        betas = configs[cfg]
        k = len(betas)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        x = np.arange(k)
        width = 0.25
        for vi, v in enumerate(SHARED_BRANCH_VARIANTS):
            rho, abar = run_store[cfg][v.name]
            correct = rho > 0.05
            flipped = rho < -0.05
            mask = correct if float(correct.mean()) >= float(flipped.mean()) else flipped
            abar_m = abar[mask] if mask.any() else abar
            mean = abar_m.mean(0)
            std = abar_m.std(0)
            ax.bar(x + (vi - 1) * width, mean, width, yerr=std, label=v.label, color=colors[v.name], alpha=0.9)
        labels = []
        for b in betas:
            if b > 0:
                labels.append("R")
            elif b < 0:
                labels.append("A")
            else:
                labels.append("N")
        ax.set_xticks(x)
        ax.set_xticklabels([f"E{i} ({lab})" for i, lab in enumerate(labels)])
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel(r"$\bar\alpha_k$")
        ax.set_title(f"{cfg} (shared head; modal branch)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"alpha_bar_{cfg}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"OUT branch: {out}")


def run_sigma(out_root: str, seeds: int, steps: int, overwrite: bool) -> None:
    out = os.path.join(out_root, "sigma_consensus_sweep")
    ensure_dir(out, overwrite)
    configs = build_k4_configs()
    order = ["3R1A", "2R1A1N", "3R1N", "1R3A"]
    targets = [0.0, 0.05, 0.1, 0.25, 0.5, 1.4, 6.0]
    # map to sigma = rms/sqrt(2T)
    T = 50
    methods = [
        ("no_cons", 0.0),
        ("consensus", 0.5),
    ]
    cal_rng = np.random.default_rng(9201)
    theta_scales = {
        t: calibrate_theta_scale(t, seeds=40, n_seg=48, T=T, d=16, rng=cal_rng) for t in targets
    }

    rows = []
    idx = 0
    for cfg in order:
        betas = configs[cfg]
        for t in targets:
            for mname, ccoef in methods:
                idx += 1
                v = SharedVariant(mname, mname, target_rms=t, consensus_coef=ccoef)
                rho, abar, rms0 = run_shared_variant(
                    betas,
                    v,
                    seeds=seeds,
                    steps=steps,
                    seed=9201 + 37 * idx,
                    theta_scale=theta_scales[t],
                )
                row = {
                    "config": cfg,
                    "method": mname,
                    "target_rms": t,
                    "step_sigma": t / np.sqrt(2 * T) if t > 0 else 0.0,
                    "init_rms_deltaR": rms0,
                    "correct_branch_rate": float((rho > 0.05).mean()),
                    "flipped_branch_rate": float((rho < -0.05).mean()),
                    "mean_rho": float(rho.mean()),
                }
                rows.append(row)
                print(
                    f"[sigma] {cfg:7s} {mname:10s} rms={t:<4g} correct={row['correct_branch_rate']:.3f} "
                    f"rho={row['mean_rho']:+.3f}"
                )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "sigma_consensus_sweep.csv"), index=False)

    # Plot: 4 panels, two curves each
    fig, axes = plt.subplots(1, len(order), figsize=(4.0 * len(order), 3.6), sharey=True)
    colors = {"no_cons": "#4c72b0", "consensus": "#dd8452"}
    for ax, cfg in zip(axes, order):
        for mname, _ in methods:
            sub = table[(table["config"] == cfg) & (table["method"] == mname)].sort_values("init_rms_deltaR")
            x = np.maximum(sub["init_rms_deltaR"].to_numpy(), 1e-4)
            ax.plot(x, sub["correct_branch_rate"], "o-", color=colors[mname], label=mname)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xlabel(r"Init $\mathrm{rms}|\Delta R|$")
        ax.set_title(cfg)
        ax.grid(True, which="both", ls=":", alpha=0.45)
        if ax is axes[0]:
            ax.set_ylabel("Correct-branch rate")
        if ax is axes[-1]:
            ax.legend(fontsize=8)
    fig.suptitle(rf"Shared head $\pm$ consensus ($T={T}$, {seeds} seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "shared_sigma_consensus_sweep.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Identify joint-good band: 3R1N and 3R1A both >= 0.9 without consensus
    band_rows = []
    for t in targets:
        n = table[(table.config == "3R1N") & (table.method == "no_cons") & (table.target_rms == t)].iloc[0]
        a = table[(table.config == "3R1A") & (table.method == "no_cons") & (table.target_rms == t)].iloc[0]
        nc = table[(table.config == "3R1N") & (table.method == "consensus") & (table.target_rms == t)].iloc[0]
        ac = table[(table.config == "3R1A") & (table.method == "consensus") & (table.target_rms == t)].iloc[0]
        band_rows.append(
            {
                "target_rms": t,
                "3R1N_no": n.correct_branch_rate,
                "3R1A_no": a.correct_branch_rate,
                "both_no_ge_0.9": (n.correct_branch_rate >= 0.9) and (a.correct_branch_rate >= 0.9),
                "3R1N_cons": nc.correct_branch_rate,
                "3R1A_cons": ac.correct_branch_rate,
                "both_cons_ge_0.9": (nc.correct_branch_rate >= 0.9) and (ac.correct_branch_rate >= 0.9),
            }
        )
    pd.DataFrame(band_rows).to_csv(os.path.join(out, "init_band_both_NA.csv"), index=False)
    print(f"OUT sigma: {out}")


def run_ablation(out_root: str, seeds: int, steps: int, overwrite: bool) -> None:
    out = os.path.join(out_root, "enhancement_ablation")
    ensure_dir(out, overwrite)
    betas = (1.0, 1.0, 1.0, 0.0)  # 3R1N
    variants = [
        SharedVariant("raw", "Raw", 0.0, 0.0, use_tanh=False, use_maxnorm=False, use_confidence_weights=False),
        SharedVariant("tanh", "+Tanh", 0.0, 0.0, use_tanh=True, use_maxnorm=False, use_confidence_weights=False),
        SharedVariant("maxnorm", "+Max-norm", 0.0, 0.0, use_tanh=True, use_maxnorm=True, use_confidence_weights=False),
        SharedVariant("full", "Full (detached)", 0.0, 0.0, use_tanh=True, use_maxnorm=True, use_confidence_weights=True),
        SharedVariant("no_w", "w/o $w_k$", 0.0, 0.0, use_tanh=True, use_maxnorm=True, use_confidence_weights=False),
        SharedVariant(
            "attach_w",
            "Attached $w_k$",
            0.0,
            0.0,
            use_tanh=True,
            use_maxnorm=True,
            use_confidence_weights=True,
            detach_weights=False,
        ),
    ]
    rows = []
    for i, v in enumerate(variants):
        rho, abar, _ = run_shared_variant(betas, v, seeds=seeds, steps=steps, seed=9301 + 17 * i)
        # R: 0,1,2; N: 3
        rows.append(
            {
                "variant": v.name,
                "label": v.label,
                "correct": float((rho > 0.05).mean()),
                "mean_abs_corr": float(np.abs(rho).mean()),
                "abar_R": float(abar[:, :3].mean()),
                "abar_N": float(abar[:, 3].mean()),
            }
        )
        print(
            f"[ablation] {v.name:10s} correct={rows[-1]['correct']:.3f} "
            f"|corr|={rows[-1]['mean_abs_corr']:.3f} "
            f"aR={rows[-1]['abar_R']:+.2f} aN={rows[-1]['abar_N']:+.2f}"
        )
    pd.DataFrame(rows).to_csv(os.path.join(out, "enhancement_ablation_3R1N.csv"), index=False)
    print(f"OUT ablation: {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="final_results/synthetic_shared_all")
    p.add_argument("--mode", choices=["branch", "sigma", "ablation", "all"], default="all")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    if args.mode in ("branch", "all"):
        run_branch(args.out_root, args.seeds, args.steps, args.overwrite)
    if args.mode in ("sigma", "all"):
        run_sigma(args.out_root, args.seeds, args.steps, args.overwrite)
    if args.mode in ("ablation", "all"):
        run_ablation(args.out_root, args.seeds, args.steps, args.overwrite)


if __name__ == "__main__":
    main()
