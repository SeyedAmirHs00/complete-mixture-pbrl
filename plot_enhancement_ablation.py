#!/usr/bin/env python3
"""Publication-style plots for TriTrust-PBRL enhancement ablations.

Reads ``exp_pebble_mixture_ablation/<env>/ablation_t*_m*_w*/**/eval.csv``
(and optionally reward/train CSVs) and writes:

  - learning curves (mean ± SEM across seeds)
  - final-return bar chart
  - late-training window summary table (CSV)
  - optional expert-coefficient trajectories from reward.csv

Example
-------
  python plot_enhancement_ablation.py
  python plot_enhancement_ablation.py --root exp_pebble_mixture_ablation --env walker_walk
  python plot_enhancement_ablation.py --metric episode_reward --ci std --smooth 3
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Variant metadata (matches scripts/walker_walk/run_enhancement_ablation.py)
# ---------------------------------------------------------------------------

FOLDER_RE = re.compile(
    r"ablation_t(?P<tanh>True|False)_m(?P<maxn>True|False)_w(?P<wk>True|False)$"
)


@dataclass(frozen=True)
class VariantMeta:
    key: str  # folder suffix flags, e.g. tTrue_mTrue_wTrue
    label: str
    short: str
    color: str
    linestyle: str
    order: int
    use_tanh: bool
    use_max_norm: bool
    use_confidence_weight: bool


# Colour-blind friendly palette; Full TTP highlighted.
VARIANT_META: Dict[Tuple[bool, bool, bool], VariantMeta] = {
    (False, False, False): VariantMeta(
        "tFalse_mFalse_wFalse", "Raw", "Raw", "#7A7A7A", "--", 0, False, False, False
    ),
    (True, False, False): VariantMeta(
        "tTrue_mFalse_wFalse", "+Tanh", "+Tanh", "#4C78A8", "-", 1, True, False, False
    ),
    (True, True, False): VariantMeta(
        "tTrue_mTrue_wFalse", "+Tanh, +Max-norm", "+Tanh+Max", "#F58518", "-", 2, True, True, False
    ),
    (True, True, True): VariantMeta(
        "tTrue_mTrue_wTrue", "Full TTP", "Full TTP", "#E45756", "-", 3, True, True, True
    ),
    (True, False, True): VariantMeta(
        "tTrue_mFalse_wTrue", "w/o Max-norm", "w/o Max", "#54A24B", "-.", 4, True, False, True
    ),
    (False, True, True): VariantMeta(
        "tFalse_mTrue_wTrue", "w/o Tanh", "w/o Tanh", "#B279A2", "-.", 5, False, True, True
    ),
}


def parse_variant_folder(name: str) -> Optional[VariantMeta]:
    m = FOLDER_RE.match(name)
    if not m:
        return None
    flags = (
        m.group("tanh") == "True",
        m.group("maxn") == "True",
        m.group("wk") == "True",
    )
    return VARIANT_META.get(flags)


# ---------------------------------------------------------------------------
# Data loading / aggregation
# ---------------------------------------------------------------------------


def find_csv_files(variant_dir: str, csv_name: str) -> List[str]:
    pattern = os.path.join(glob.escape(variant_dir), "**", f"{csv_name}.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    return [f for f in files if os.path.getsize(f) > 0]


def load_seed_series(
    csv_files: Sequence[str], metric: str, x_col: str = "step"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, Y) where Y has shape (n_seeds, n_steps), aligned on shared x."""
    series = []
    xs = []
    for path in csv_files:
        df = pd.read_csv(path)
        if x_col not in df.columns or metric not in df.columns:
            continue
        xs.append(df[x_col].to_numpy(dtype=float))
        series.append(df[metric].to_numpy(dtype=float))

    if not series:
        raise ValueError(f"No usable series for metric={metric!r}")

    # Align on intersection of x grids (eval logs are usually identical).
    common_x = xs[0]
    for x in xs[1:]:
        common_x = np.intersect1d(common_x, x)
    if common_x.size == 0:
        raise ValueError("Seed runs have no overlapping x values")

    Y = np.stack(
        [s[np.isin(x, common_x)] for s, x in zip(series, xs)],
        axis=0,
    )
    return common_x, Y


def aggregate(Y: np.ndarray, ci: str = "sem") -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(Y, axis=0)
    std = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)
    if ci == "std":
        band = std
    elif ci == "sem":
        band = std / np.sqrt(max(Y.shape[0], 1))
    elif ci == "none":
        band = np.zeros_like(mean)
    else:
        raise ValueError(f"Unknown ci={ci!r}; use sem|std|none")
    return mean, band


def smooth_curve(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y
    kernel = np.ones(window, dtype=float) / window
    # Reflect-pad to keep length.
    pad = window // 2
    yp = np.pad(y, (pad, window - 1 - pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
        }
    )


def plot_learning_curves(
    curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    metric: str,
    title: str,
    out_path: str,
    xlabel: str = "Environment steps",
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for meta in sorted(curves.keys(), key=lambda m: m.order):
        x, mean, band = curves[meta]
        ax.plot(x, mean, color=meta.color, linestyle=meta.linestyle, label=meta.label)
        ax.fill_between(
            x, mean - band, mean + band, color=meta.color, alpha=0.18, linewidth=0
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(_pretty_metric(metric))
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    # Compact scientific x ticks for large step counts.
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_final_bars(
    finals: Dict[VariantMeta, np.ndarray],
    metric: str,
    title: str,
    out_path: str,
    last_n: int,
) -> None:
    metas = sorted(finals.keys(), key=lambda m: m.order)
    means = [float(np.mean(finals[m])) for m in metas]
    sems = [
        float(np.std(finals[m], ddof=1) / np.sqrt(len(finals[m])))
        if len(finals[m]) > 1
        else 0.0
        for m in metas
    ]
    labels = [m.label for m in metas]
    colors = [m.color for m in metas]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(metas))
    bars = ax.bar(
        x,
        means,
        yerr=sems,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        alpha=0.9,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    # Mark Full TTP.
    for i, m in enumerate(metas):
        if m.short == "Full TTP":
            bars[i].set_linewidth(1.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(_pretty_metric(metric))
    ax.set_title(title)
    # Annotate mean values.
    y_max = max(means[i] + sems[i] for i in range(len(means)))
    for i, (mu, se) in enumerate(zip(means, sems)):
        ax.text(
            i,
            mu + se + 0.01 * y_max,
            f"{mu:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}  (last {last_n} eval points averaged per seed)")


def plot_expert_coefs(
    reward_curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray]],
    title: str,
    out_path: str,
    n_experts: int = 5,
) -> None:
    """One subplot per variant: mean expert_coef_k over steps."""
    metas = sorted(reward_curves.keys(), key=lambda m: m.order)
    n = len(metas)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True, sharey=True
    )
    axes_flat = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("tab10")

    for ax, meta in zip(axes_flat, metas):
        x, Y = reward_curves[meta]  # Y: (n_seeds, n_steps, n_experts)
        mean = np.nanmean(Y, axis=0)
        for k in range(n_experts):
            ax.plot(x, mean[:, k], color=cmap(k), label=f"expert {k}", linewidth=1.6)
        ax.set_title(meta.label, color=meta.color)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_xlabel("steps")
        ax.set_ylabel("expert coef")

    for ax in axes_flat[len(metas) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_experts, frameon=False)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _pretty_metric(metric: str) -> str:
    return {
        "true_episode_reward": "True episode return",
        "episode_reward": "Episode return (learned reward)",
        "actor_loss": "Actor loss",
        "critic_loss": "Critic loss",
    }.get(metric, metric.replace("_", " ").title())


def _backbone_label(meta: VariantMeta) -> str:
    parts = []
    if meta.use_tanh:
        parts.append("Tanh")
    if meta.use_max_norm:
        parts.append("Max-norm")
    return "+".join(parts) if parts else "Raw"


def matched_w_pairs(
    metas: Sequence[VariantMeta],
) -> List[Tuple[str, VariantMeta, VariantMeta]]:
    """Pairs that share tanh/max-norm and differ only in confidence weight w_k."""
    by_backbone: Dict[Tuple[bool, bool], Dict[bool, VariantMeta]] = {}
    for m in metas:
        key = (m.use_tanh, m.use_max_norm)
        by_backbone.setdefault(key, {})[m.use_confidence_weight] = m

    pairs = []
    for (tanh, maxn), d in sorted(by_backbone.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if False in d and True in d:
            without_w, with_w = d[False], d[True]
            pairs.append((_backbone_label(without_w), without_w, with_w))
    return pairs


def plot_w_comparison(
    curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    finals: Dict[VariantMeta, np.ndarray],
    metric: str,
    env: str,
    out_dir: str,
    last_n: int,
) -> Optional[pd.DataFrame]:
    """Learning-curve pairs + grouped bars for with vs without confidence weight w_k."""
    pairs = matched_w_pairs(list(curves.keys()))
    if not pairs:
        print("No matched with/without-w_k pairs found; skipping w comparison.")
        return None

    color_wo, color_w = "#4C78A8", "#E45756"

    # --- Side-by-side learning curves (one panel per backbone) ---
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), sharey=True)
    axes_list = np.atleast_1d(axes).ravel()
    for ax, (backbone, wo, wi) in zip(axes_list, pairs):
        for meta, color, ls, tag in (
            (wo, color_wo, "--", "without w_k"),
            (wi, color_w, "-", "with w_k"),
        ):
            x, mean, band = curves[meta]
            ax.plot(x, mean, color=color, linestyle=ls, label=f"{tag} ({meta.label})")
            ax.fill_between(
                x, mean - band, mean + band, color=color, alpha=0.18, linewidth=0
            )
        ax.set_title(f"Backbone: {backbone}")
        ax.set_xlabel("Environment steps")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes_list[0].set_ylabel(_pretty_metric(metric))
    fig.suptitle(f"Confidence weight w_k — with vs without ({env})", y=1.02)
    fig.tight_layout()
    curve_path = os.path.join(out_dir, f"w_comparison_curves_{metric}.png")
    fig.savefig(curve_path, bbox_inches="tight")
    fig.savefig(curve_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {curve_path}")

    # --- Grouped final-return bars ---
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x = np.arange(len(pairs))
    width = 0.36
    means_wo, sems_wo, means_w, sems_w = [], [], [], []
    rows = []
    for backbone, wo, wi in pairs:
        yw, yi = finals[wo], finals[wi]
        m_wo, m_w = float(np.mean(yw)), float(np.mean(yi))
        s_wo = float(np.std(yw, ddof=1) / np.sqrt(len(yw))) if len(yw) > 1 else 0.0
        s_w = float(np.std(yi, ddof=1) / np.sqrt(len(yi))) if len(yi) > 1 else 0.0
        means_wo.append(m_wo)
        sems_wo.append(s_wo)
        means_w.append(m_w)
        sems_w.append(s_w)
        delta = m_w - m_wo
        rows.append(
            {
                "backbone": backbone,
                "without_w_label": wo.label,
                "with_w_label": wi.label,
                "without_w_mean": m_wo,
                "without_w_sem": s_wo,
                "with_w_mean": m_w,
                "with_w_sem": s_w,
                "delta_with_minus_without": delta,
                "n_seeds_without": len(yw),
                "n_seeds_with": len(yi),
                "last_n_eval_points": last_n,
            }
        )

    bars_wo = ax.bar(
        x - width / 2,
        means_wo,
        width,
        yerr=sems_wo,
        color=color_wo,
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        label="without w_k",
        error_kw={"elinewidth": 1.1},
    )
    bars_w = ax.bar(
        x + width / 2,
        means_w,
        width,
        yerr=sems_w,
        color=color_w,
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        label="with w_k",
        error_kw={"elinewidth": 1.1},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([b for b, _, _ in pairs])
    ax.set_ylabel(_pretty_metric(metric))
    ax.set_title(f"Final return: with vs without w_k (last {last_n} evals)")
    ax.legend(frameon=False, loc="lower right")
    y_top = max(
        max(means_wo[i] + sems_wo[i], means_w[i] + sems_w[i]) for i in range(len(pairs))
    )
    for i, (m_wo, m_w, s_wo, s_w) in enumerate(
        zip(means_wo, means_w, sems_wo, sems_w)
    ):
        ax.text(
            i - width / 2,
            m_wo + s_wo + 0.008 * y_top,
            f"{m_wo:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            i + width / 2,
            m_w + s_w + 0.008 * y_top,
            f"{m_w:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.annotate(
            f"Δ={m_w - m_wo:+.0f}",
            xy=(i, max(m_wo + s_wo, m_w + s_w)),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#333333",
        )
    del bars_wo, bars_w
    fig.tight_layout()
    bar_path = os.path.join(out_dir, f"w_comparison_bars_{metric}.png")
    fig.savefig(bar_path, bbox_inches="tight")
    fig.savefig(bar_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {bar_path}")

    # --- Delta-only chart ---
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    deltas = [r["delta_with_minus_without"] for r in rows]
    colors = ["#54A24B" if d >= 0 else "#E45756" for d in deltas]
    ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([b for b, _, _ in pairs])
    ax.set_ylabel(f"Δ {_pretty_metric(metric)} (with − without w_k)")
    ax.set_title("Effect of confidence weight w_k")
    offset = 0.03 * (max(abs(d) for d in deltas) or 1.0)
    for i, d in enumerate(deltas):
        ax.text(
            i,
            d + offset if d >= 0 else d - offset,
            f"{d:+.1f}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            fontsize=10,
        )
    fig.tight_layout()
    delta_path = os.path.join(out_dir, f"w_comparison_delta_{metric}.png")
    fig.savefig(delta_path, bbox_inches="tight")
    fig.savefig(delta_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {delta_path}")

    table = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"w_comparison_{metric}.csv")
    table.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved {csv_path}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    return table


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def collect_variants(env_dir: str) -> List[Tuple[VariantMeta, str]]:
    found = []
    for name in sorted(os.listdir(env_dir)):
        path = os.path.join(env_dir, name)
        if not os.path.isdir(path):
            continue
        meta = parse_variant_folder(name)
        if meta is None:
            print(f"  skip unrecognized folder: {name}")
            continue
        found.append((meta, path))
    return found


def build_summary_table(
    curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    finals: Dict[VariantMeta, np.ndarray],
    last_n: int,
) -> pd.DataFrame:
    rows = []
    for meta in sorted(curves.keys(), key=lambda m: m.order):
        x, mean, band = curves[meta]
        seed_finals = finals[meta]
        rows.append(
            {
                "variant": meta.label,
                "tanh": meta.use_tanh,
                "max_norm": meta.use_max_norm,
                "w_k": meta.use_confidence_weight,
                "n_seeds": len(seed_finals),
                "final_mean": float(np.mean(seed_finals)),
                "final_std": float(np.std(seed_finals, ddof=1))
                if len(seed_finals) > 1
                else 0.0,
                "final_sem": float(np.std(seed_finals, ddof=1) / np.sqrt(len(seed_finals)))
                if len(seed_finals) > 1
                else 0.0,
                "curve_last_mean": float(mean[-1]),
                "auc_mean": float(
                    (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(mean, x)
                    / max(x[-1] - x[0], 1.0)
                ),
                "last_n_eval_points": last_n,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="exp_pebble_mixture_ablation")
    p.add_argument("--env", default="walker_walk")
    p.add_argument(
        "--metric",
        default="true_episode_reward",
        help="Column in eval.csv (default: true_episode_reward)",
    )
    p.add_argument(
        "--ci",
        choices=("sem", "std", "none"),
        default="sem",
        help="Shaded band: standard error (default), std, or none",
    )
    p.add_argument(
        "--last-n",
        type=int,
        default=5,
        help="Average last N eval points per seed for final bar chart",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window on mean curves (1 = off)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: results/exp_pebble_mixture_ablation/<env>)",
    )
    p.add_argument(
        "--skip-reward",
        action="store_true",
        help="Skip expert-coefficient plots from reward.csv",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_style()

    repo = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(repo, args.root, args.env)
    if not os.path.isdir(env_dir):
        print(f"Environment directory not found: {env_dir}")
        return 1

    out_dir = args.out_dir or os.path.join(
        repo, "results", args.root, args.env, "enhancement_ablation"
    )
    os.makedirs(out_dir, exist_ok=True)

    variants = collect_variants(env_dir)
    if not variants:
        print(f"No ablation_* folders under {env_dir}")
        return 1

    print(f"Found {len(variants)} ablation variants in {env_dir}")
    curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    finals: Dict[VariantMeta, np.ndarray] = {}
    reward_curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray]] = {}

    for meta, vdir in variants:
        eval_files = find_csv_files(vdir, "eval")
        if not eval_files:
            print(f"  [{meta.label}] no eval.csv — skipping")
            continue
        x, Y = load_seed_series(eval_files, args.metric)
        mean, band = aggregate(Y, ci=args.ci)
        if args.smooth > 1:
            mean = smooth_curve(mean, args.smooth)
            band = smooth_curve(band, args.smooth)
        curves[meta] = (x, mean, band)

        # Per-seed late average for bar chart.
        n = min(args.last_n, Y.shape[1])
        seed_scores = np.nanmean(Y[:, -n:], axis=1)
        finals[meta] = seed_scores
        print(
            f"  [{meta.label:18s}] seeds={Y.shape[0]}  "
            f"last-{n} mean={seed_scores.mean():.1f} ± {seed_scores.std(ddof=1):.1f}"
        )

        if not args.skip_reward:
            reward_files = find_csv_files(vdir, "reward")
            coef_cols = None
            seed_mats = []
            xs_r = []
            for rf in reward_files:
                df = pd.read_csv(rf)
                cols = [c for c in df.columns if re.fullmatch(r"expert_coef_\d+", c)]
                if not cols or "step" not in df.columns:
                    continue
                cols = sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))
                if coef_cols is None:
                    coef_cols = cols
                xs_r.append(df["step"].to_numpy(dtype=float))
                seed_mats.append(df[cols].to_numpy(dtype=float))
            if seed_mats and coef_cols:
                common_x = xs_r[0]
                for xr in xs_r[1:]:
                    common_x = np.intersect1d(common_x, xr)
                # Subsample reward logs (often dense) for lighter plots.
                if common_x.size > 400:
                    idx = np.linspace(0, common_x.size - 1, 400).astype(int)
                    common_x = common_x[idx]
                mats = []
                for mat, xr in zip(seed_mats, xs_r):
                    mask = np.isin(xr, common_x)
                    mats.append(mat[mask])
                reward_curves[meta] = (common_x, np.stack(mats, axis=0))

    if not curves:
        print("Nothing to plot.")
        return 1

    # --- Learning curves ---
    plot_learning_curves(
        curves,
        metric=args.metric,
        title=f"Enhancement ablation — {args.env}",
        out_path=os.path.join(out_dir, f"learning_curve_{args.metric}.png"),
    )

    # --- Final performance bars ---
    plot_final_bars(
        finals,
        metric=args.metric,
        title=f"Final return (last {args.last_n} evals) — {args.env}",
        out_path=os.path.join(out_dir, f"final_bar_{args.metric}.png"),
        last_n=args.last_n,
    )

    # --- With vs without confidence weight w_k ---
    print("\n=== With vs without w_k ===")
    plot_w_comparison(
        curves,
        finals,
        metric=args.metric,
        env=args.env,
        out_dir=out_dir,
        last_n=args.last_n,
    )

    # --- Summary CSV ---
    table = build_summary_table(curves, finals, args.last_n)
    table_path = os.path.join(out_dir, f"summary_{args.metric}.csv")
    table.to_csv(table_path, index=False, float_format="%.4f")
    print(f"Saved {table_path}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # --- Expert coefficients ---
    if reward_curves:
        plot_expert_coefs(
            reward_curves,
            title=f"Expert coefficients — {args.env}",
            out_path=os.path.join(out_dir, "expert_coefficients.png"),
        )

    print(f"\nAll figures → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
