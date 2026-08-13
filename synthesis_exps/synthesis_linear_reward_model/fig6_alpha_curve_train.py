"""Train Fig. 6 partial-adversary α learning curves (writes CSVs; optional plot).

Linear (batched) and MLP (per-seed) TTP runs that record raw α, tanh(α), and
\\bar α at every logged step. Plot with ``--plot`` or separately via
``fig6_alpha_curve_plot.py``.

Methods (init)
--------------
  stabilized — near-zero reward init (linear: θ=0; MLP: zero last Linear)
  standard   — non-trivial init (linear: rms|ΔR|≈1.4; MLP: PyTorch default)
  both       — run stabilized and standard (default)

Reward model
------------
  linear — shared R = Σ_t tanh(θᵀ s_t)  (default; batched over seeds)
  mlp    — PEBBLE-style MLP reward head (higher capacity; per-seed train)

Examples
--------
  python fig6_alpha_curve_train.py --seeds 1 --overwrite
  python fig6_alpha_curve_train.py --methods stabilized --overwrite --plot
  python fig6_alpha_curve_train.py --methods both --reward-model mlp --overwrite --plot
  python fig6_alpha_curve_plot.py --run_dir results/synthetic_partial_adversary_alpha_curve
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic_shared_core import (
    _segment_returns,
    calibrate_theta_scale,
    get_device,
    rowwise_corr,
    sample_expert_pairs,
    sigmoid,
)
from fig6_alpha_curve_plot import write_alpha_curve_figures

HIST_CSV_NAME = "alpha_learning_curve_per_step.csv"
SUMMARY_CSV_NAME = "partial_adversary_alpha_curve_summary.csv"

METHOD_SPECS: Dict[str, Dict[str, float]] = {
    # Linear head: target_rms controls θ init scale (0 ⇒ θ=0).
    "stabilized": {"target_rms": 0.0, "consensus_coef": 0.0},
    "standard": {"target_rms": 1.4, "consensus_coef": 0.0},
}

PARTIAL_ADV_SETTINGS: List[Tuple[str, Dict[str, Any]]] = [
    ("perfect_flip", dict(beta_adv=-1.0, flip_prob=None)),
    ("beta_-0.5", dict(beta_adv=-0.5, flip_prob=None)),
    ("beta_-0.25", dict(beta_adv=-0.25, flip_prob=None)),
    ("stoch_p0.5", dict(beta_adv=None, flip_prob=0.5)),
    ("stoch_p0.25", dict(beta_adv=None, flip_prob=0.25)),
]


def resolve_methods(raw: Sequence[str]) -> List[str]:
    if "both" in raw:
        return ["stabilized", "standard"]
    out: List[str] = []
    for m in raw:
        if m not in out:
            out.append(m)
    return out


def make_partial_adv_labels(
    rng: np.random.Generator,
    *,
    r_star: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    beta_adv: float | None,
    flip_prob: float | None,
) -> np.ndarray:
    """Build y with shape (seeds, K=4, pairs) for 3R1A partial/stochastic adv."""
    seeds, k, pairs = i.shape
    assert k == 4
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
    return y


# ---------------------------------------------------------------------------
# Linear reward head (batched over seeds)
# ---------------------------------------------------------------------------


def run_linear_with_alpha_history(
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
    log_every: int = 1,
    lr_theta: float = 0.05,
    lr_alpha: float = 0.005,
) -> dict[str, Any]:
    """Train shared linear reward; return final metrics + per-step α history."""
    rng = np.random.default_rng(seed)
    k, T, d = 4, 50, 16

    theta_scale = calibrate_theta_scale(target_rms, seeds=40, n_seg=n_seg, T=T, d=d, rng=rng)
    theta_star = rng.normal(size=(seeds, d))
    theta_star /= np.linalg.norm(theta_star, axis=1, keepdims=True) + 1e-12
    states = rng.normal(size=(seeds, n_seg, T, d))
    r_star = np.tanh(np.einsum("sntd,sd->snt", states, theta_star)).sum(2)
    r_star = (r_star - r_star.mean(1, keepdims=True)) / (r_star.std(1, keepdims=True) + 1e-12)

    i, j = sample_expert_pairs(rng, seeds=seeds, k=k, n_seg=n_seg, pairs=pairs, q=q)
    y = make_partial_adv_labels(
        rng, r_star=r_star, i=i, j=j, beta_adv=beta_adv, flip_prob=flip_prob
    )
    consensus_target = y.mean(1)

    if theta_scale == 0:
        theta0 = np.zeros((seeds, d))
    else:
        theta0 = rng.normal(scale=theta_scale / np.sqrt(d), size=(seeds, d))

    device = get_device()
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    i_t = torch.as_tensor(i, dtype=torch.long, device=device)
    j_t = torch.as_tensor(j, dtype=torch.long, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    y_bar = torch.as_tensor(consensus_target, dtype=torch.float32, device=device)
    theta = torch.nn.Parameter(torch.as_tensor(theta0, dtype=torch.float32, device=device))
    alpha = torch.nn.Parameter(torch.full((seeds, k), 0.01, device=device))
    opt = torch.optim.Adam(
        [
            {"params": [theta], "lr": lr_theta},
            {"params": [alpha], "lr": lr_alpha},
        ]
    )

    hist_rows: list[dict[str, float | int]] = []

    def _record(step: int) -> None:
        with torch.no_grad():
            raw = alpha.detach().cpu().numpy()
            trust = torch.tanh(alpha).detach()
            tilde_np = trust.cpu().numpy()
            abar_np = (
                trust / trust.abs().amax(1, keepdim=True).clamp_min(1e-12)
            ).cpu().numpy()
        for s in range(seeds):
            for e in range(k):
                hist_rows.append(
                    {
                        "step": step,
                        "seed_idx": s,
                        "expert": e,
                        "expert_role": "A" if e == 3 else "R",
                        "alpha": float(raw[s, e]),
                        "tilde_alpha": float(tilde_np[s, e]),
                        "abar_alpha": float(abar_np[s, e]),
                    }
                )

    _record(0)
    for t in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        R = _segment_returns(states_t, theta, True)
        delta = R.gather(1, i_t.reshape(seeds, -1)).reshape(seeds, k, pairs) - R.gather(
            1, j_t.reshape(seeds, -1)
        ).reshape(seeds, k, pairs)
        trust = torch.tanh(alpha)
        denom = trust.abs().amax(1, keepdim=True).clamp_min(1e-12).detach()
        coef = trust / denom
        # Reward path: stop-grad through coef; no w_k expert reweighting.
        logits_R = coef.detach().unsqueeze(2) * delta
        loss_R = F.binary_cross_entropy_with_logits(logits_R, y_t, reduction="none").mean(
            dim=(1, 2)
        ).sum()
        if consensus_coef > 0:
            loss_R = loss_R + consensus_coef * F.binary_cross_entropy_with_logits(
                delta.mean(dim=1), y_bar, reduction="none"
            ).mean(dim=1).sum()
        logits_A = coef.unsqueeze(2) * delta.detach()
        loss_A = F.binary_cross_entropy_with_logits(logits_A, y_t, reduction="none").mean(
            dim=(1, 2)
        ).sum()
        (loss_R + loss_A).backward()
        torch.nn.utils.clip_grad_norm_([theta], 10.0)
        opt.step()
        if t % log_every == 0 or t == steps:
            _record(t)

    with torch.no_grad():
        R = _segment_returns(states_t, theta, True).cpu().numpy()
        trust = torch.tanh(alpha)
        tilde = trust.cpu().numpy()
        abar = (trust / trust.abs().amax(1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
    rho = rowwise_corr(R, r_star)
    return {
        "rho": rho,
        "abar": abar,
        "tilde": tilde,
        "hist": pd.DataFrame(hist_rows),
    }


# ---------------------------------------------------------------------------
# MLP reward head (higher capacity; train one seed at a time)
# ---------------------------------------------------------------------------


class RewardMLP(nn.Module):
    """PEBBLE-style reward: d → H×L → 1, Tanh out; R = Σ_t r(s_t)."""

    def __init__(self, d: int, hidden: int = 128, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        din = d
        for _ in range(n_layers):
            layers.append(nn.Linear(din, hidden))
            layers.append(nn.LeakyReLU(0.01))
            din = hidden
        # Final Linear is the Stabilized target (zero weight+bias at init).
        layers.append(nn.Linear(din, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        n, t, d = states.shape
        r = self.net(states.reshape(n * t, d)).reshape(n, t)
        return r.sum(dim=1)

    def last_linear(self) -> nn.Linear:
        """Return the final Linear (just before Tanh)."""
        for layer in reversed(list(self.net.children())):
            if isinstance(layer, nn.Linear):
                return layer
        raise ValueError("RewardMLP has no Linear layers")

    def zero_last_layer(self) -> None:
        """Stabilized init: set final Linear weight and bias to 0 (⇒ R≡0)."""
        last = self.last_linear()
        with torch.no_grad():
            last.weight.zero_()
            if last.bias is not None:
                last.bias.zero_()


def apply_mlp_init(net: RewardMLP, method: str) -> None:
    """Apply learner init. Stabilized zeros the last Linear; standard keeps PyTorch default."""
    if method == "standard":
        return
    if method == "stabilized":
        net.zero_last_layer()
        return
    raise ValueError(f"unknown MLP init method={method!r}")


@torch.no_grad()
def mlp_teacher_returns(
    states: np.ndarray,
    *,
    d: int,
    torch_seed: int,
    device: torch.device,
    hidden: int,
    n_layers: int,
) -> np.ndarray:
    torch.manual_seed(torch_seed)
    teacher = RewardMLP(d, hidden=hidden, n_layers=n_layers).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    st = torch.as_tensor(states, dtype=torch.float32, device=device)
    return teacher(st).cpu().numpy()


def train_mlp_seed_with_history(
    *,
    states: np.ndarray,
    i_np: np.ndarray,
    j_np: np.ndarray,
    y_np: np.ndarray,
    y_bar_np: np.ndarray,
    r_star: np.ndarray,
    method: str,
    steps: int,
    hidden: int,
    n_layers: int,
    consensus_coef: float,
    torch_seed: int,
    seed_idx: int,
    device: torch.device,
    log_every: int,
    lr_theta: float = 0.0003,
    lr_alpha: float = 0.0003,
    alpha_init: float = 0.01,
) -> Tuple[float, np.ndarray, np.ndarray, List[dict[str, float | int]]]:
    """One-seed MLP TTP; returns (rho, abar, tilde, hist_rows)."""
    torch.manual_seed(torch_seed)
    k = y_np.shape[0]
    n, t, d = states.shape

    net = RewardMLP(d, hidden=hidden, n_layers=n_layers).to(device)
    # Stabilized + MLP: zero last Linear of the *learner* at step 0 (teacher untouched).
    apply_mlp_init(net, method)
    if method == "stabilized":
        with torch.no_grad():
            last = net.last_linear()
            st_check = torch.as_tensor(states[: min(8, n)], dtype=torch.float32, device=device)
            r_abs = float(net(st_check).abs().mean().item())
            w_abs = float(last.weight.abs().sum().item())
            b_abs = float(last.bias.abs().sum().item()) if last.bias is not None else 0.0
        if w_abs > 0.0 or b_abs > 0.0 or r_abs > 1e-8:
            raise RuntimeError(
                f"stabilized MLP init failed (seed_idx={seed_idx}): "
                f"|W|_1={w_abs:.3g} |b|_1={b_abs:.3g} mean|R|={r_abs:.3g}"
            )

    alpha = torch.nn.Parameter(torch.full((k,), float(alpha_init), device=device))

    st = torch.as_tensor(states, dtype=torch.float32, device=device)
    i = torch.as_tensor(i_np, dtype=torch.long, device=device)
    j = torch.as_tensor(j_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    y_bar = torch.as_tensor(y_bar_np, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(
        [
            {"params": net.parameters(), "lr": lr_theta},
            {"params": [alpha], "lr": lr_alpha},
        ]
    )

    hist_rows: List[dict[str, float | int]] = []

    def _record(step: int) -> None:
        with torch.no_grad():
            raw = alpha.detach().cpu().numpy()
            trust = torch.tanh(alpha).detach()
            tilde_np = trust.cpu().numpy()
            abar_np = (trust / trust.abs().amax().clamp_min(1e-12)).cpu().numpy()
        for e in range(k):
            hist_rows.append(
                {
                    "step": step,
                    "seed_idx": seed_idx,
                    "expert": e,
                    "expert_role": "A" if e == 3 else "R",
                    "alpha": float(raw[e]),
                    "tilde_alpha": float(tilde_np[e]),
                    "abar_alpha": float(abar_np[e]),
                }
            )

    _record(0)
    for t_step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        R = net(st)
        delta = R[i] - R[j]  # [K, P]

        trust = torch.tanh(alpha)
        coef = trust / trust.abs().amax().clamp_min(1e-12).detach()

        # Reward path: stop-grad through coef; no w_k expert reweighting.
        logits_R = coef.detach().unsqueeze(1) * delta
        loss_R = F.binary_cross_entropy_with_logits(logits_R, y, reduction="none").mean()
        if consensus_coef > 0:
            loss_R = loss_R + consensus_coef * F.binary_cross_entropy_with_logits(
                delta.mean(dim=0), y_bar, reduction="none"
            ).mean()

        logits_A = coef.unsqueeze(1) * delta.detach()
        loss_A = F.binary_cross_entropy_with_logits(logits_A, y, reduction="none").mean()

        (loss_R + loss_A).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()

        if t_step % log_every == 0 or t_step == steps:
            _record(t_step)

    with torch.no_grad():
        R_hat = net(st).cpu().numpy()
        trust = torch.tanh(alpha)
        tilde = trust.cpu().numpy()
        abar = (trust / trust.abs().amax().clamp_min(1e-12)).cpu().numpy()
    rho = float(rowwise_corr(R_hat[None, :], r_star[None, :])[0])
    return rho, abar, tilde, hist_rows


def run_mlp_with_alpha_history(
    *,
    seeds: int,
    steps: int,
    method: str,
    consensus_coef: float,
    beta_adv: float | None,
    flip_prob: float | None,
    seed: int,
    n_seg: int = 500,
    q: float = 0.0,
    pairs: int = 256,
    log_every: int = 1,
    hidden: int = 128,
    n_layers: int = 3,
) -> dict[str, Any]:
    """Train MLP reward (teacher + learner); return metrics + α history."""
    rng = np.random.default_rng(seed)
    k, T, d = 4, 50, 16
    device = get_device()

    states = rng.normal(size=(seeds, n_seg, T, d)).astype(np.float32)
    r_star = np.zeros((seeds, n_seg), dtype=np.float64)
    for s in range(seeds):
        r = mlp_teacher_returns(
            states[s],
            d=d,
            torch_seed=seed + 10_000 + 97 * s,
            device=device,
            hidden=hidden,
            n_layers=n_layers,
        )
        r_star[s] = (r - r.mean()) / (r.std() + 1e-12)

    i, j = sample_expert_pairs(rng, seeds=seeds, k=k, n_seg=n_seg, pairs=pairs, q=q)
    y = make_partial_adv_labels(
        rng, r_star=r_star, i=i, j=j, beta_adv=beta_adv, flip_prob=flip_prob
    )
    consensus_target = y.mean(1)

    rhos = np.zeros(seeds)
    abars = np.zeros((seeds, k))
    tildes = np.zeros((seeds, k))
    hist_rows: List[dict[str, float | int]] = []

    for s in range(seeds):
        rho, abar, tilde, rows = train_mlp_seed_with_history(
            states=states[s],
            i_np=i[s],
            j_np=j[s],
            y_np=y[s],
            y_bar_np=consensus_target[s],
            r_star=r_star[s],
            method=method,
            steps=steps,
            hidden=hidden,
            n_layers=n_layers,
            consensus_coef=consensus_coef,
            torch_seed=seed + 1_000 + 31 * s,
            seed_idx=s,
            device=device,
            log_every=log_every,
        )
        rhos[s] = rho
        abars[s] = abar
        tildes[s] = tilde
        hist_rows.extend(rows)

    return {
        "rho": rhos,
        "abar": abars,
        "tilde": tildes,
        "hist": pd.DataFrame(hist_rows),
    }


def run_alpha_curve_experiment(
    *,
    methods: Sequence[str],
    reward_model: str,
    seeds: int,
    steps: int,
    log_every: int = 1,
    hidden: int = 128,
    n_layers: int = 3,
    settings: Sequence[Tuple[str, Dict[str, Any]]] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all (setting, method) jobs.

    Returns
    -------
    hist_df : per-step α trajectories
    summary_df : final correct-branch / trust summary
    """
    methods = resolve_methods(methods)
    settings = list(settings) if settings is not None else list(PARTIAL_ADV_SETTINGS)

    print(
        f"[alpha-curve] reward_model={reward_model} methods={methods} "
        f"seeds={seeds} steps={steps}",
        flush=True,
    )
    if reward_model == "mlp":
        print(f"  mlp hidden={hidden} n_layers={n_layers}", flush=True)

    all_hist: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str]] = []
    idx = 0
    for sname, skw in settings:
        for mname in methods:
            idx += 1
            print(f"[alpha-curve] {sname:12s} {mname:10s} ({reward_model}) ...", flush=True)
            mkw = METHOD_SPECS[mname]
            if reward_model == "linear":
                out = run_linear_with_alpha_history(
                    seeds=seeds,
                    steps=steps,
                    seed=9400 + idx,
                    n_seg=500,
                    q=0.0,
                    log_every=log_every,
                    **skw,
                    **mkw,
                )
            elif reward_model == "mlp":
                if mname == "stabilized" and idx == 1:
                    print(
                        "  [mlp/stabilized] learner last Linear weight+bias zeroed at init",
                        flush=True,
                    )
                out = run_mlp_with_alpha_history(
                    seeds=seeds,
                    steps=steps,
                    method=mname,
                    consensus_coef=float(mkw["consensus_coef"]),
                    seed=9400 + idx,
                    n_seg=500,
                    q=0.0,
                    log_every=log_every,
                    hidden=hidden,
                    n_layers=n_layers,
                    **skw,
                )
            else:
                raise ValueError(f"unknown reward_model={reward_model!r}")

            hist = out["hist"].copy()
            hist["setting"] = sname
            hist["method"] = mname
            hist["reward_model"] = reward_model
            all_hist.append(hist)

            rho, abar, tilde = out["rho"], out["abar"], out["tilde"]
            summary_rows.append(
                {
                    "setting": sname,
                    "method": mname,
                    "reward_model": reward_model,
                    "correct": float((rho > 0.05).mean()),
                    "mean_rho": float(rho.mean()),
                    "abar_R": float(abar[:, :3].mean()),
                    "abar_A": float(abar[:, 3].mean()),
                    "tilde_R": float(tilde[:, :3].mean()),
                    "tilde_A": float(tilde[:, 3].mean()),
                }
            )
            print(
                f"  correct={summary_rows[-1]['correct']:.3f} "
                f"tR={summary_rows[-1]['tilde_R']:+.3f} tA={summary_rows[-1]['tilde_A']:+.3f}",
                flush=True,
            )

    return pd.concat(all_hist, ignore_index=True), pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--out_dir",
        default="results/synthetic_partial_adversary_alpha_curve_mlp",
        help="Directory for CSV outputs.",
    )
    p.add_argument("--seeds", type=int, default=100, help="Number of MC seeds (batch size).")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=1, help="Record alpha every N steps.")
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["stabilized", "standard", "both"],
        default=["stabilized"],
        help="Init variants: stabilized, standard, or both (default: both).",
    )
    p.add_argument(
        "--reward-model",
        choices=["linear", "mlp"],
        default="linear",
        help="Reward head complexity: linear (default) or PEBBLE-style mlp.",
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="MLP hidden width (ignored for --reward-model linear).",
    )
    p.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="MLP hidden depth before the final Linear(1) (ignored for linear).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--plot",
        action="store_true",
        help="After training, write aggregated α curve figures to --out_dir.",
    )
    p.add_argument(
        "--plot-ci",
        choices=["std", "sem", "var", "none"],
        default="std",
        help="Seed band for --plot (default: std).",
    )
    p.add_argument(
        "--plot-per-seed",
        action="store_true",
        help="With --plot, also write one figure per (setting, method, seed).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    methods = resolve_methods(args.methods)

    if os.path.exists(args.out_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.out_dir} exists; pass --overwrite to replace it"
            )
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    hist_df, summary_df = run_alpha_curve_experiment(
        methods=methods,
        reward_model=args.reward_model,
        seeds=args.seeds,
        steps=args.steps,
        log_every=args.log_every,
        hidden=args.hidden,
        n_layers=args.n_layers,
    )

    hist_path = os.path.join(args.out_dir, HIST_CSV_NAME)
    summary_path = os.path.join(args.out_dir, SUMMARY_CSV_NAME)
    hist_df.to_csv(hist_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {hist_path}")
    print(f"Saved {summary_path}")
    print(f"OUT: {args.out_dir}")

    if args.plot:
        print(f"Plotting figures → {args.out_dir} (ci={args.plot_ci})")
        write_alpha_curve_figures(
            hist_df,
            args.out_dir,
            ci=args.plot_ci,
            per_seed=args.plot_per_seed,
        )
    else:
        print(f"  plot with: python fig6_alpha_curve_plot.py --run_dir {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
