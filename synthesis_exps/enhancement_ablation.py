"""
Enhancement ablation on shared head (linear or MLP).
Label: tab:enhancement-ablation

Example:
  python enhancement_ablation.py --seeds 200 --overwrite
  python enhancement_ablation.py --reward-model mlp --seeds 200 --overwrite
  python enhancement_ablation.py --replot
  python enhancement_ablation.py --reward-model mlp --replot
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from enhancement_ablation_data import run_ablation_data
from enhancement_ablation_plot import plot_enhancement_ablation_figures
from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA


def _default_out_dir(reward_model: str) -> str:
    suffix = "_mlp" if reward_model == "mlp" else ""
    return f"results/synthetic_enhancement_ablation{suffix}"


def main() -> None:
    p = argparse.ArgumentParser(description="Enhancement ablation (linear or MLP)")
    p.add_argument("--out_dir", default=None, help="Default depends on --reward-model")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--reward-model",
        choices=["linear", "mlp"],
        default="linear",
        help="Reward model: linear (default) or mlp (PEBBLE-style neural network).",
    )
    p.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw", "adam_sgd"],
        default="sgd",
    )
    p.add_argument("--lr-model", type=float, default=None)
    p.add_argument("--lr-alpha", type=float, default=None)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate plots after data collection (default: True).",
    )
    p.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip figure generation.",
    )
    p.add_argument(
        "--replot",
        action="store_true",
        help="Only regenerate figures from existing CSV without retraining.",
    )
    args = p.parse_args()

    out_dir = args.out_dir or _default_out_dir(args.reward_model)
    csv_path = os.path.join(out_dir, "enhancement_ablation.csv")

    if args.replot:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found for replot: {csv_path}")
        table = pd.read_csv(csv_path)
        plot_enhancement_ablation_figures(table, out_dir)
        print(f"replot OK: {out_dir}")
        return

    run_ablation_data(
        out_dir,
        seeds=args.seeds,
        steps=args.steps,
        n=args.n,
        pairs=args.pairs,
        q=args.q,
        overwrite=args.overwrite,
        coef_max_delta=args.coef_max_delta,
        reward_model=args.reward_model,
        optimizer=args.optimizer,
        lr_model=args.lr_model,
        lr_alpha=args.lr_alpha,
        hidden=args.hidden,
        n_layers=args.n_layers,
    )

    if args.plot:
        table = pd.read_csv(csv_path)
        plot_enhancement_ablation_figures(table, out_dir)


if __name__ == "__main__":
    main()
