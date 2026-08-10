#!/usr/bin/env python3
"""General plotting for TriTrust-PBRL / PEBBLE mixture experiment trees.

Expected on-disk layout (Hydra run folders)::

    <root>/
      <env>/
        max_feedbackN_feed_type..._b[1, 1, 1, -1]_m0_s0_e0/
          seedS/
            test/eval.csv
            reward/reward.csv   # optional (alphas, expert coefs)

Works for ``exp_pebble_mixture_zero_last`` and any similarly structured tree.

Examples
--------
  # Plot everything under zero_last
  python plot_experiments.py --root exp_pebble_mixture_zero_last

  # One environment
  python plot_experiments.py --root exp_pebble_mixture_zero_last --env walker_walk

  # Paper envs (aliases + Hydra folder names both accepted)
  python plot_experiments.py --root exp_pebble_mixture_zero_last \\
      --envs walker_walk cheetah_run door_open sweep_into

  # Compare two experiment roots as named series
  python plot_experiments.py \\
      --series zero_last:exp_pebble_mixture_zero_last \\
      --series baseline:exp_pebble_mixture \\
      --env walker_walk

See ``PLOTTING.md`` for the full guide.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants / aliases
# ---------------------------------------------------------------------------

FEEDBACK_RE = re.compile(r"^max_feedback(?P<fb>\d+)_")
TEACHER_BETAS_RE = re.compile(r"_b\[(?P<betas>[^\]]+)\]_")
SEED_RE = re.compile(r"seed(?P<seed>\d+)")

ENV_ALIASES: Dict[str, str] = {
    "door_open": "metaworld_door-open-v2",
    "sweep_into": "metaworld_sweep-into-v2",
    "walker": "walker_walk",
    "cheetah": "cheetah_run",
}

DEFAULT_METRIC: Dict[str, str] = {
    "walker_walk": "true_episode_reward",
    "cheetah_run": "true_episode_reward",
    "metaworld_door-open-v2": "success_rate",
    "metaworld_sweep-into-v2": "success_rate",
}

SERIES_COLORS = [
    "#E45756",
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#72B7B2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#EACA2B",
]


@dataclass(frozen=True)
class RunConfig:
    """One Hydra config folder under ``<root>/<env>/``."""

    path: str
    env: str
    max_feedback: int
    teacher_betas: Tuple[float, ...]
    label: str


@dataclass(frozen=True)
class SeriesSpec:
    """A named experiment root to overlay."""

    name: str
    root: str
    color: str
    linestyle: str = "-"


# ---------------------------------------------------------------------------
# Path / config parsing
# ---------------------------------------------------------------------------


def resolve_env(env: str) -> str:
    return ENV_ALIASES.get(env, env)


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def abs_under_repo(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root(), path)


def normalize_teacher_betas(betas: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(b) for b in betas)


def parse_teacher_betas_from_name(name: str) -> Optional[Tuple[float, ...]]:
    m = TEACHER_BETAS_RE.search(name)
    if not m:
        return None
    parts = [p.strip() for p in m.group("betas").split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def teacher_betas_match(name: str, teacher_betas: Sequence[float]) -> bool:
    parsed = parse_teacher_betas_from_name(name)
    if parsed is None:
        return False
    return parsed == normalize_teacher_betas(teacher_betas)


def _format_beta_value(b: float) -> str:
    ib = int(b)
    return str(ib) if b == ib else str(b)


def format_teacher_betas(betas: Sequence[float]) -> str:
    return "[" + ", ".join(_format_beta_value(b) for b in betas) + "]"


def parse_seed_from_path(path: str) -> Optional[int]:
    m = SEED_RE.search(path)
    return int(m.group("seed")) if m else None


def default_metric_for_env(env: str) -> str:
    folder = resolve_env(env)
    if folder in DEFAULT_METRIC:
        return DEFAULT_METRIC[folder]
    if "metaworld" in folder.lower():
        return "success_rate"
    return "true_episode_reward"


def pretty_metric(metric: str) -> str:
    return {
        "true_episode_reward": "True episode return",
        "episode_reward": "Episode return (learned reward)",
        "success_rate": "Success rate",
        "actor_loss": "Actor loss",
        "critic_loss": "Critic loss",
    }.get(metric, metric.replace("_", " ").title())


def run_label(max_feedback: int, teacher_betas: Sequence[float]) -> str:
    return f"fb={max_feedback}, β={format_teacher_betas(teacher_betas)}"


def discover_envs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and not name.startswith(".")
    )


def list_run_configs(
    env_dir: str,
    env: str,
    *,
    teacher_betas: Optional[Sequence[float]] = None,
    max_feedback: Optional[int] = None,
) -> List[RunConfig]:
    """List Hydra config folders directly under an environment directory."""
    configs: List[RunConfig] = []
    if not os.path.isdir(env_dir):
        return configs
    for name in sorted(os.listdir(env_dir)):
        path = os.path.join(env_dir, name)
        if not os.path.isdir(path):
            continue
        m = FEEDBACK_RE.match(name)
        if not m:
            continue
        fb = int(m.group("fb"))
        if max_feedback is not None and fb != max_feedback:
            continue
        betas = parse_teacher_betas_from_name(name)
        if betas is None:
            continue
        if teacher_betas is not None and betas != normalize_teacher_betas(teacher_betas):
            continue
        configs.append(
            RunConfig(
                path=path,
                env=env,
                max_feedback=fb,
                teacher_betas=betas,
                label=run_label(fb, betas),
            )
        )
    return configs


def find_csv_files(
    run_dir: str,
    csv_name: str,
    seeds: Optional[Sequence[int]] = None,
) -> List[str]:
    pattern = os.path.join(glob.escape(run_dir), "**", f"{csv_name}.csv")
    files = sorted(f for f in glob.glob(pattern, recursive=True) if os.path.getsize(f) > 0)
    if not seeds:
        return files
    seed_set = {int(s) for s in seeds}
    return [f for f in files if parse_seed_from_path(f) in seed_set]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def load_seed_series(
    csv_files: Sequence[str], metric: str, x_col: str = "step"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(x, Y)`` with ``Y`` shape ``(n_seeds, n_steps)`` on a shared x grid."""
    series: List[np.ndarray] = []
    xs: List[np.ndarray] = []
    for path in csv_files:
        df = pd.read_csv(path)
        if x_col not in df.columns or metric not in df.columns:
            continue
        xs.append(df[x_col].to_numpy(dtype=float))
        series.append(df[metric].to_numpy(dtype=float))

    if not series:
        raise ValueError(f"No usable series for metric={metric!r}")

    common_x = xs[0]
    for x in xs[1:]:
        common_x = np.intersect1d(common_x, x)
    if common_x.size == 0:
        raise ValueError("Seed runs have no overlapping x values")

    Y = np.stack([s[np.isin(x, common_x)] for s, x in zip(series, xs)], axis=0)
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
    pad = window // 2
    yp = np.pad(y, (pad, window - 1 - pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")


def load_reward_channels(
    reward_files: Sequence[str],
    col_regex: str,
    max_points: int = 400,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Load aligned reward CSV channels → ``(x, Y[n_seeds,n_steps,n_ch], col_names)``."""
    seed_mats: List[np.ndarray] = []
    xs: List[np.ndarray] = []
    cols_ref: Optional[List[str]] = None
    for path in reward_files:
        df = pd.read_csv(path)
        if "step" not in df.columns:
            continue
        cols = [c for c in df.columns if re.fullmatch(col_regex, c)]
        if not cols:
            continue
        cols = sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))
        if cols_ref is None:
            cols_ref = cols
        elif cols != cols_ref:
            cols = [c for c in cols_ref if c in cols]
            if not cols:
                continue
        xs.append(df["step"].to_numpy(dtype=float))
        seed_mats.append(df[cols].to_numpy(dtype=float))

    if not seed_mats or not cols_ref:
        return None

    common_x = xs[0]
    for xr in xs[1:]:
        common_x = np.intersect1d(common_x, xr)
    if common_x.size == 0:
        return None
    if common_x.size > max_points:
        idx = np.linspace(0, common_x.size - 1, max_points).astype(int)
        common_x = common_x[idx]

    mats = []
    for mat, xr in zip(seed_mats, xs):
        mats.append(mat[np.isin(xr, common_x)])
    return common_x, np.stack(mats, axis=0), cols_ref


def load_reward_scalar(
    reward_files: Sequence[str],
    column: str,
    max_points: int = 400,
    ci: str = "sem",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    try:
        x, Y = load_seed_series(reward_files, column)
    except ValueError:
        return None
    if x.size > max_points:
        idx = np.linspace(0, x.size - 1, max_points).astype(int)
        x = x[idx]
        Y = Y[:, idx]
    mean, band = aggregate(Y, ci=ci)
    return x, mean, band


# ---------------------------------------------------------------------------
# Plotting
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


def save_fig(fig: plt.Figure, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


Curve = Tuple[np.ndarray, np.ndarray, np.ndarray]  # x, mean, band


def plot_learning_curves(
    curves: Dict[str, Curve],
    *,
    metric: str,
    title: str,
    out_path: str,
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    xlabel: str = "Environment steps",
) -> None:
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for i, (label, (x, mean, band)) in enumerate(curves.items()):
        color = (colors or {}).get(label, SERIES_COLORS[i % len(SERIES_COLORS)])
        ls = (linestyles or {}).get(label, "-")
        ax.plot(x, mean, color=color, linestyle=ls, label=label)
        if np.any(band > 0):
            ax.fill_between(x, mean - band, mean + band, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(pretty_metric(metric))
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    save_fig(fig, out_path)


def plot_final_bars(
    finals: Dict[str, np.ndarray],
    *,
    metric: str,
    title: str,
    out_path: str,
    last_n: int,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    if not finals:
        return
    labels = list(finals.keys())
    means = [float(np.mean(finals[k])) for k in labels]
    sems = [
        float(np.std(finals[k], ddof=1) / np.sqrt(len(finals[k])))
        if len(finals[k]) > 1
        else 0.0
        for k in labels
    ]
    bar_colors = [
        (colors or {}).get(k, SERIES_COLORS[i % len(SERIES_COLORS)])
        for i, k in enumerate(labels)
    ]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(labels)), 4.6))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=sems,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        alpha=0.9,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(pretty_metric(metric))
    ax.set_title(title)
    y_max = max(means[i] + sems[i] for i in range(len(means))) if means else 1.0
    for i, (mu, se) in enumerate(zip(means, sems)):
        ax.text(
            i,
            mu + se + 0.01 * max(y_max, 1e-8),
            f"{mu:.2g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_fig(fig, out_path)
    print(f"  (final bars average last {last_n} eval points per seed)")


def plot_cross_env_curves(
    env_curves: Dict[str, Dict[str, Curve]],
    *,
    metric_by_env: Dict[str, str],
    title: str,
    out_path: str,
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
) -> None:
    """One subplot per environment; overlay series within each."""
    envs = [e for e, curves in env_curves.items() if curves]
    if not envs:
        return
    n = len(envs)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.0 * ncols, 4.2 * nrows), squeeze=False
    )
    axes_flat = axes.ravel()

    for ax, env in zip(axes_flat, envs):
        curves = env_curves[env]
        for i, (label, (x, mean, band)) in enumerate(curves.items()):
            color = (colors or {}).get(label, SERIES_COLORS[i % len(SERIES_COLORS)])
            ls = (linestyles or {}).get(label, "-")
            ax.plot(x, mean, color=color, linestyle=ls, label=label)
            if np.any(band > 0):
                ax.fill_between(
                    x, mean - band, mean + band, color=color, alpha=0.15, linewidth=0
                )
        ax.set_title(env)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(pretty_metric(metric_by_env.get(env, "true_episode_reward")))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.legend(frameon=False, fontsize=8, loc="best")

    for ax in axes_flat[len(envs) :]:
        ax.axis("off")

    fig.suptitle(title, y=1.02)
    save_fig(fig, out_path)


def plot_channel_panels(
    channel_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    title: str,
    out_path: str,
    ylabel: str,
    channel_prefix: str,
) -> None:
    """One subplot per series label; lines are expert / alpha channels."""
    if not channel_curves:
        return
    labels = list(channel_curves.keys())
    n = len(labels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), squeeze=False, sharex=True
    )
    axes_flat = axes.ravel()
    cmap = plt.get_cmap("tab10")

    for ax, label in zip(axes_flat, labels):
        x, Y = channel_curves[label]  # (n_seeds, n_steps, n_ch)
        mean = np.nanmean(Y, axis=0)
        n_ch = mean.shape[1]
        for k in range(n_ch):
            ax.plot(x, mean[:, k], color=cmap(k % 10), label=f"{channel_prefix}_{k}", linewidth=1.6)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
        ax.set_title(label)
        ax.set_xlabel("steps")
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    for ax in axes_flat[len(labels) :]:
        ax.axis("off")

    handles, leg_labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            leg_labels,
            loc="upper center",
            ncol=min(6, len(leg_labels)),
            frameon=False,
        )
    fig.suptitle(title, y=1.02)
    save_fig(fig, out_path)


def plot_scalar_overlay(
    curves: Dict[str, Curve],
    *,
    title: str,
    out_path: str,
    ylabel: str,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for i, (label, (x, mean, band)) in enumerate(curves.items()):
        color = (colors or {}).get(label, SERIES_COLORS[i % len(SERIES_COLORS)])
        ax.plot(x, mean, color=color, label=label)
        if np.any(band > 0):
            ax.fill_between(x, mean - band, mean + band, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def parse_series_arg(raw: str, index: int) -> SeriesSpec:
    """Parse ``name:path`` or bare ``path`` (name = basename)."""
    if ":" in raw and not (raw.startswith("/") and raw.count(":") == 1 and "\\" not in raw):
        # Prefer split on first ':' that separates name from path.
        # Windows drive letters are rare here; still allow bare paths without name.
        name, path = raw.split(":", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Bad --series {raw!r}; expected name:path")
    else:
        path = raw
        name = os.path.basename(os.path.normpath(path))
    return SeriesSpec(
        name=name,
        root=path,
        color=SERIES_COLORS[index % len(SERIES_COLORS)],
        linestyle=["-", "--", "-.", ":"][index % 4],
    )


def collect_env_names(series_list: Sequence[SeriesSpec], requested: Sequence[str]) -> List[str]:
    if requested:
        return [resolve_env(e) for e in requested]
    envs: set = set()
    for spec in series_list:
        root = abs_under_repo(spec.root)
        envs.update(discover_envs(root))
    return sorted(envs)


def build_curve_from_files(
    eval_files: Sequence[str],
    metric: str,
    *,
    ci: str,
    smooth: int,
) -> Optional[Tuple[Curve, np.ndarray, int]]:
    """Return ``((x, mean, band), seed_finals, n_seeds)`` or None."""
    if not eval_files:
        return None
    try:
        x, Y = load_seed_series(eval_files, metric)
    except ValueError as exc:
        print(f"  skip eval: {exc}")
        return None
    mean, band = aggregate(Y, ci=ci)
    if smooth > 1:
        mean = smooth_curve(mean, smooth)
        band = smooth_curve(band, smooth)
    return (x, mean, band), Y, Y.shape[0]


def plot_env(
    env: str,
    series_list: Sequence[SeriesSpec],
    *,
    out_dir: str,
    metric: Optional[str],
    teacher_betas: Optional[Sequence[float]],
    max_feedback: Optional[int],
    seeds: Optional[Sequence[int]],
    ci: str,
    last_n: int,
    smooth: int,
    skip_reward: bool,
    group_by: str,
) -> Optional[Dict[str, Curve]]:
    """Plot one environment across series / configs. Returns learning curves dict."""
    metric = metric or default_metric_for_env(env)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'=' * 72}\n{env}  metric={metric}  → {out_dir}\n{'=' * 72}")

    curves: Dict[str, Curve] = {}
    finals: Dict[str, np.ndarray] = {}
    colors: Dict[str, str] = {}
    linestyles: Dict[str, str] = {}
    alpha_panels: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    coef_panels: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    abs_sum_curves: Dict[str, Curve] = {}
    summary_rows: List[dict] = []

    multi_series = len(series_list) > 1

    for spec in series_list:
        root = abs_under_repo(spec.root)
        env_dir = os.path.join(root, env)
        configs = list_run_configs(
            env_dir,
            env,
            teacher_betas=teacher_betas,
            max_feedback=max_feedback,
        )
        if not configs:
            print(f"  [{spec.name}] no matching runs under {env_dir}")
            continue

        for cfg in configs:
            if group_by == "series":
                label = spec.name
            elif group_by == "config":
                label = cfg.label if not multi_series else f"{spec.name} | {cfg.label}"
            else:  # auto
                root_base = os.path.basename(os.path.normpath(spec.root))
                named = spec.name != root_base  # user passed name:path
                if multi_series:
                    if len(configs) == 1:
                        label = spec.name
                    else:
                        label = f"{spec.name} | {cfg.label}"
                elif len(configs) > 1:
                    label = cfg.label
                else:
                    # One series, one config: prefer explicit series name, else fb/β tag.
                    label = spec.name if named else cfg.label

            eval_files = find_csv_files(cfg.path, "eval", seeds=seeds)
            built = build_curve_from_files(eval_files, metric, ci=ci, smooth=smooth)
            if built is None:
                print(f"  [{label}] no eval.csv")
            else:
                curve, Y, n_seeds = built
                curves[label] = curve
                n = min(last_n, Y.shape[1])
                seed_scores = np.nanmean(Y[:, -n:], axis=1)
                finals[label] = seed_scores
                colors[label] = spec.color
                linestyles[label] = spec.linestyle
                print(
                    f"  [{label:40s}] seeds={n_seeds}  "
                    f"last-{n} mean={seed_scores.mean():.3g} ± "
                    f"{(seed_scores.std(ddof=1) if n_seeds > 1 else 0.0):.3g}"
                )
                summary_rows.append(
                    {
                        "series": spec.name,
                        "env": env,
                        "label": label,
                        "max_feedback": cfg.max_feedback,
                        "teacher_betas": format_teacher_betas(cfg.teacher_betas),
                        "metric": metric,
                        "n_seeds": n_seeds,
                        "final_mean": float(seed_scores.mean()),
                        "final_std": float(
                            seed_scores.std(ddof=1) if n_seeds > 1 else 0.0
                        ),
                        "last_n": n,
                    }
                )

            if skip_reward:
                continue

            reward_files = find_csv_files(cfg.path, "reward", seeds=seeds)
            if not reward_files:
                print(f"  [{label}] no reward.csv")
                continue

            alphas = load_reward_channels(reward_files, r"alpha_\d+")
            if alphas is not None:
                x_a, Y_a, cols_a = alphas
                alpha_panels[label] = (x_a, Y_a)
                print(f"  [{label:40s}] alphas channels={cols_a}")

            coefs = load_reward_channels(reward_files, r"expert_coef_\d+")
            if coefs is not None:
                x_c, Y_c, _ = coefs
                coef_panels[label] = (x_c, Y_c)

            abs_sum = load_reward_scalar(reward_files, "alpha_abs_sum", ci=ci)
            if abs_sum is not None:
                abs_sum_curves[label] = abs_sum
                colors.setdefault(label, spec.color)

    if not curves and not alpha_panels:
        print(f"Nothing to plot for {env}.")
        return None

    title = env
    if teacher_betas is not None:
        title += f" ({format_teacher_betas(teacher_betas)})"
    if max_feedback is not None:
        title += f", max_feedback={max_feedback}"

    if curves:
        plot_learning_curves(
            curves,
            metric=metric,
            title=f"Learning curve — {title}",
            out_path=os.path.join(out_dir, f"learning_curve_{metric}.png"),
            colors=colors,
            linestyles=linestyles,
        )
        plot_final_bars(
            finals,
            metric=metric,
            title=f"Final performance (last {last_n} evals) — {title}",
            out_path=os.path.join(out_dir, f"final_bar_{metric}.png"),
            last_n=last_n,
            colors=colors,
        )

    if alpha_panels:
        plot_channel_panels(
            alpha_panels,
            title=rf"Trust parameters $\alpha_k$ — {title}",
            out_path=os.path.join(out_dir, "alphas.png"),
            ylabel=r"$\alpha_k$",
            channel_prefix=r"$\alpha$",
        )
    if coef_panels:
        plot_channel_panels(
            coef_panels,
            title=f"Expert coefficients — {title}",
            out_path=os.path.join(out_dir, "expert_coefficients.png"),
            ylabel="expert coef",
            channel_prefix="expert",
        )
    if abs_sum_curves:
        plot_scalar_overlay(
            abs_sum_curves,
            title=rf"$|\alpha|_1$ — {title}",
            out_path=os.path.join(out_dir, "alpha_abs_sum.png"),
            ylabel=r"$|\alpha|_1$",
            colors=colors,
        )

    if summary_rows:
        table = pd.DataFrame(summary_rows)
        table_path = os.path.join(out_dir, f"summary_{metric}.csv")
        table.to_csv(table_path, index=False, float_format="%.4f")
        print(f"Saved {table_path}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    return curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="General learning-curve / alpha plots for PEBBLE mixture runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Experiment root (e.g. exp_pebble_mixture_zero_last). "
        "Ignored if --series is set.",
    )
    p.add_argument(
        "--series",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help="Named experiment root to overlay (repeatable). "
        "Example: --series zero_last:exp_pebble_mixture_zero_last",
    )
    p.add_argument("--env", type=str, default=None, help="Single environment folder / alias")
    p.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Environment list (aliases ok). Default: discover all under root(s).",
    )
    p.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Eval CSV column. Default: success_rate for MetaWorld, "
        "true_episode_reward otherwise.",
    )
    p.add_argument(
        "--teacher-betas",
        type=float,
        nargs="+",
        default=None,
        help="Filter runs by teacher_betas encoded in the folder name",
    )
    p.add_argument(
        "--max-feedback",
        type=int,
        default=None,
        help="Filter runs by max_feedbackN prefix",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Only include these seeds",
    )
    p.add_argument("--ci", choices=["sem", "std", "none"], default="sem")
    p.add_argument(
        "--last-n",
        type=int,
        default=10,
        help="Number of trailing eval points averaged for final bars / summary",
    )
    p.add_argument("--smooth", type=int, default=1, help="Moving-average window (≥1)")
    p.add_argument(
        "--skip-reward",
        action="store_true",
        help="Skip alpha / expert-coefficient plots from reward.csv",
    )
    p.add_argument(
        "--group-by",
        choices=["auto", "series", "config"],
        default="auto",
        help="How to label overlapping curves within an env",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: results/<root_basename>/)",
    )
    p.add_argument(
        "--no-cross-env",
        action="store_true",
        help="Do not write the multi-environment overview figure",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_style()

    series_list: List[SeriesSpec] = []
    if args.series:
        for i, raw in enumerate(args.series):
            series_list.append(parse_series_arg(raw, i))
    elif args.root:
        series_list.append(parse_series_arg(args.root, 0))
    else:
        # Sensible default for this project.
        series_list.append(parse_series_arg("exp_pebble_mixture_zero_last", 0))

    for spec in series_list:
        root = abs_under_repo(spec.root)
        if not os.path.isdir(root):
            print(f"ERROR: experiment root not found: {root}")
            return 1

    requested: List[str] = []
    if args.env:
        requested.append(args.env)
    if args.envs:
        requested.extend(args.envs)
    env_names = collect_env_names(series_list, requested)
    if not env_names:
        print("No environments found.")
        return 1

    if args.out:
        out_root = abs_under_repo(args.out)
    elif len(series_list) == 1:
        out_root = os.path.join(
            repo_root(), "results", os.path.basename(os.path.normpath(series_list[0].root))
        )
    else:
        out_root = os.path.join(repo_root(), "results", "experiment_compare")

    print("Plot experiments")
    print(f"  series : {[(s.name, s.root) for s in series_list]}")
    print(f"  envs   : {env_names}")
    print(f"  out    : {out_root}")

    env_curves: Dict[str, Dict[str, Curve]] = {}
    metric_by_env: Dict[str, str] = {}
    any_ok = False

    for env in env_names:
        metric = args.metric or default_metric_for_env(env)
        metric_by_env[env] = metric
        curves = plot_env(
            env,
            series_list,
            out_dir=os.path.join(out_root, env),
            metric=metric,
            teacher_betas=args.teacher_betas,
            max_feedback=args.max_feedback,
            seeds=args.seeds,
            ci=args.ci,
            last_n=args.last_n,
            smooth=args.smooth,
            skip_reward=args.skip_reward,
            group_by=args.group_by,
        )
        if curves:
            env_curves[env] = curves
            any_ok = True

    if any_ok and not args.no_cross_env and len(env_curves) > 1:
        # Color map: reuse first series color per label when possible.
        colors: Dict[str, str] = {}
        linestyles: Dict[str, str] = {}
        for i, spec in enumerate(series_list):
            colors[spec.name] = spec.color
            linestyles[spec.name] = spec.linestyle
        # Also color any config labels that appear.
        seen_labels = []
        for curves in env_curves.values():
            for label in curves:
                if label not in seen_labels:
                    seen_labels.append(label)
        for i, label in enumerate(seen_labels):
            colors.setdefault(label, SERIES_COLORS[i % len(SERIES_COLORS)])

        plot_cross_env_curves(
            env_curves,
            metric_by_env=metric_by_env,
            title="Cross-environment overview",
            out_path=os.path.join(out_root, "cross_env_learning_curves.png"),
            colors=colors,
            linestyles=linestyles,
        )

    if not any_ok:
        print("No plots were generated.")
        return 1

    print(f"\nAll figures → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
