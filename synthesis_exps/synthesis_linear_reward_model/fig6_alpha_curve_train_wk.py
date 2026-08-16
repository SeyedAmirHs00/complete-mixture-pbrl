"""Train Fig. 6 partial-adversary α curves **with detached w_k** on the reward loss.

Same CLI and layout as ``fig6_alpha_curve_train.py``, but the reward-path BCE is
reweighted by detached confidence weights::

    w_k = K · |\\tilde α_k| / Σ_j |\\tilde α_j|

Trust-path loss stays unweighted. Use ``fig6_alpha_curve_train.py`` for the
no-w_k ablation.

Methods (init)
--------------
  stabilized    — near-zero reward init (linear: θ=0; MLP: zero last Linear)
  standard      — non-trivial init (linear: rms|ΔR|≈1.4; MLP: PyTorch default)
  subtract_init — standard weight init with explicit zero functional init::

                    R(x) = f_θ(x) − stopgrad(f_θ₀(x))

                  so R≡0 at initialization while internal weights stay arbitrary.
  both          — run stabilized and standard

Optimizers (``--optimizer``)
----------------------------
  sgd      — SGD on reward + α  (default lrs 0.05 / 0.005)
  adam     — Adam on reward + α (default lrs 3e-4 / 1e-4; like adamw.py groups)
  adamw    — AdamW on reward + α (default lrs 3e-4 / 1e-4, wd=1e-2 on reward)
  adam_sgd — Adam on reward, SGD on α (default lrs 3e-4 / 5e-4)

Examples
--------
  python fig6_alpha_curve_train_wk.py --seeds 100 --overwrite --plot
  python fig6_alpha_curve_train_wk.py --optimizer adamw --methods subtract_init --overwrite --plot
  python fig6_alpha_curve_train_wk.py --optimizer adam_sgd --reward-model mlp --overwrite
  python fig6_alpha_curve_plot.py --run_dir results/synthetic_partial_adversary_alpha_curve_wk
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from fig6_alpha_curve_plot import write_alpha_curve_figures
from fig6_alpha_curve_train import (
    HIST_CSV_NAME,
    METHOD_SPECS,
    PARTIAL_ADV_SETTINGS,
    SUMMARY_CSV_NAME,
    RewardMLP,
    apply_mlp_init,
    make_partial_adv_labels,
    mlp_teacher_returns,
    resolve_methods,
)
from synthetic_shared_core import (
    _segment_returns,
    calibrate_theta_scale,
    get_device,
    rowwise_corr,
    sample_expert_pairs,
)

# Default (lr_model, lr_alpha) per optimizer when CLI does not override.
OPTIMIZER_DEFAULT_LRS: Dict[str, Tuple[float, float]] = {
    "sgd": (0.05, 0.005),
    "adam": (3e-4, 1e-4),
    "adamw": (3e-4, 1e-4),
    "adam_sgd": (3e-4, 5e-4),
}
DEFAULT_WEIGHT_DECAY_ADAMW = 1e-2


@dataclass
class OptimizerBundle:
    """One or more optimizers stepped together (needed for adam_sgd)."""

    opts: Tuple[torch.optim.Optimizer, ...]
    name: str
    lr_model: float
    lr_alpha: float

    def zero_grad(self) -> None:
        for opt in self.opts:
            opt.zero_grad(set_to_none=True)

    def step(self) -> None:
        for opt in self.opts:
            opt.step()


def make_optimizers(
    name: str,
    model_params: Iterable[nn.Parameter],
    alpha: nn.Parameter,
    *,
    lr_model: float,
    lr_alpha: float,
    weight_decay_model: float = DEFAULT_WEIGHT_DECAY_ADAMW,
) -> OptimizerBundle:
    """Build reward/α optimizers.

    Parameters
    ----------
    name :
        ``sgd`` | ``adam`` | ``adamw`` | ``adam_sgd``
    """
    params = list(model_params)
    if name == "sgd":
        opt = torch.optim.SGD(
            [
                {"params": params, "lr": lr_model},
                {"params": [alpha], "lr": lr_alpha},
            ]
        )
        return OptimizerBundle((opt,), name, lr_model, lr_alpha)
    if name == "adam":
        opt = torch.optim.Adam(
            [
                {"params": params, "lr": lr_model},
                {"params": [alpha], "lr": lr_alpha},
            ]
        )
        return OptimizerBundle((opt,), name, lr_model, lr_alpha)
    if name == "adamw":
        opt = torch.optim.AdamW(
            [
                {"params": params, "lr": lr_model, "weight_decay": weight_decay_model},
                {"params": [alpha], "lr": lr_alpha, "weight_decay": 0.0},
            ]
        )
        return OptimizerBundle((opt,), name, lr_model, lr_alpha)
    if name == "adam_sgd":
        opt_reward = torch.optim.Adam(params, lr=lr_model)
        opt_alpha = torch.optim.SGD([alpha], lr=lr_alpha)
        return OptimizerBundle((opt_reward, opt_alpha), name, lr_model, lr_alpha)
    raise ValueError(f"unknown optimizer={name!r}")


def resolve_lrs(
    optimizer: str,
    lr_model: Optional[float],
    lr_alpha: Optional[float],
) -> Tuple[float, float]:
    d_model, d_alpha = OPTIMIZER_DEFAULT_LRS[optimizer]
    return (
        d_model if lr_model is None else float(lr_model),
        d_alpha if lr_alpha is None else float(lr_alpha),
    )


def detached_wk(trust: torch.Tensor, k: int) -> torch.Tensor:
    """Detached confidence weights w_k ∝ |tanh(α_k)|, normalized to sum to K."""
    abs_t = trust.abs()
    if trust.dim() == 1:
        return (k * abs_t / abs_t.sum().clamp_min(1e-12)).detach()
    return (k * abs_t / abs_t.sum(dim=1, keepdim=True).clamp_min(1e-12)).detach()


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
    subtract_init: bool = False,
    optimizer: str = "sgd",
    weight_decay_model: float = DEFAULT_WEIGHT_DECAY_ADAMW,
) -> dict[str, Any]:
    """Train shared linear reward with detached w_k on the reward loss.

    If ``subtract_init``, keep a frozen θ₀ and use
    ``R = f(θ) − stopgrad(f(θ₀))`` so R≡0 at initialization.
    """
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
    # Frozen init snapshot for subtract_init (not optimized).
    theta_init = theta.detach().clone()
    alpha = torch.nn.Parameter(torch.full((seeds, k), 0.01, device=device))
    opt = make_optimizers(
        optimizer,
        [theta],
        alpha,
        lr_model=lr_theta,
        lr_alpha=lr_alpha,
        weight_decay_model=weight_decay_model,
    )

    def predict_R(th: torch.Tensor) -> torch.Tensor:
        R = _segment_returns(states_t, th, True)
        if subtract_init:
            R = R - _segment_returns(states_t, theta_init, True).detach()
        return R

    if subtract_init:
        with torch.no_grad():
            r0_abs = float(predict_R(theta).abs().mean().item())
        if r0_abs > 1e-6:
            raise RuntimeError(
                f"subtract_init linear failed: mean|R|={r0_abs:.3g} at init (expected ~0)"
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
        opt.zero_grad()
        R = predict_R(theta)
        delta = R.gather(1, i_t.reshape(seeds, -1)).reshape(seeds, k, pairs) - R.gather(
            1, j_t.reshape(seeds, -1)
        ).reshape(seeds, k, pairs)
        trust = torch.tanh(alpha)
        denom = trust.abs().amax(1, keepdim=True).clamp_min(1e-12).detach()
        coef = trust / denom
        w = detached_wk(trust, k)
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
        torch.nn.utils.clip_grad_norm_([theta], 10.0)
        opt.step()
        if t % log_every == 0 or t == steps:
            _record(t)

    with torch.no_grad():
        R = predict_R(theta).cpu().numpy()
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
    lr_theta: float = 0.05,
    lr_alpha: float = 0.005,
    alpha_init: float = 0.01,
    optimizer: str = "sgd",
    weight_decay_model: float = DEFAULT_WEIGHT_DECAY_ADAMW,
) -> Tuple[float, np.ndarray, np.ndarray, List[dict[str, float | int]]]:
    """One-seed MLP TTP with detached w_k on the reward loss.

    For ``method == "subtract_init"``, keep a frozen network copy θ₀ and use
    ``R = f_θ(x) − stopgrad(f_θ₀(x))``.
    """
    torch.manual_seed(torch_seed)
    k = y_np.shape[0]
    n, t, d = states.shape
    subtract_init = method == "subtract_init"

    net = RewardMLP(d, hidden=hidden, n_layers=n_layers).to(device)
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

    net_init = None
    if subtract_init:
        net_init = copy.deepcopy(net).eval()
        for p in net_init.parameters():
            p.requires_grad_(False)

    def predict_R(st_in: torch.Tensor) -> torch.Tensor:
        R = net(st_in)
        if net_init is not None:
            R = R - net_init(st_in).detach()
        return R

    alpha = torch.nn.Parameter(torch.full((k,), float(alpha_init), device=device))

    st = torch.as_tensor(states, dtype=torch.float32, device=device)
    i = torch.as_tensor(i_np, dtype=torch.long, device=device)
    j = torch.as_tensor(j_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    y_bar = torch.as_tensor(y_bar_np, dtype=torch.float32, device=device)

    if subtract_init:
        with torch.no_grad():
            r0_abs = float(predict_R(st).abs().mean().item())
        if r0_abs > 1e-6:
            raise RuntimeError(
                f"subtract_init MLP failed (seed_idx={seed_idx}): "
                f"mean|R|={r0_abs:.3g} at init (expected ~0)"
            )

    opt = make_optimizers(
        optimizer,
        net.parameters(),
        alpha,
        lr_model=lr_theta,
        lr_alpha=lr_alpha,
        weight_decay_model=weight_decay_model,
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
        opt.zero_grad()
        R = predict_R(st)
        delta = R[i] - R[j]  # [K, P]

        trust = torch.tanh(alpha)
        coef = trust / trust.abs().amax().clamp_min(1e-12).detach()
        w = detached_wk(trust, k)

        logits_R = coef.detach().unsqueeze(1) * delta
        bce_R = F.binary_cross_entropy_with_logits(logits_R, y, reduction="none")
        loss_R = (w.unsqueeze(1) * bce_R).mean()
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
        R_hat = predict_R(st).cpu().numpy()
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
    lr_theta: float = 0.05,
    lr_alpha: float = 0.005,
    optimizer: str = "sgd",
    weight_decay_model: float = DEFAULT_WEIGHT_DECAY_ADAMW,
) -> dict[str, Any]:
    """Train MLP reward (teacher + learner) with w_k; return metrics + α history."""
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
            lr_theta=lr_theta,
            lr_alpha=lr_alpha,
            optimizer=optimizer,
            weight_decay_model=weight_decay_model,
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
    optimizer: str = "sgd",
    lr_model: float = 0.05,
    lr_alpha: float = 0.005,
    weight_decay_model: float = DEFAULT_WEIGHT_DECAY_ADAMW,
    settings: Sequence[Tuple[str, Dict[str, Any]]] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all (setting, method) jobs with detached w_k on the reward loss."""
    methods = resolve_methods(methods)
    settings = list(settings) if settings is not None else list(PARTIAL_ADV_SETTINGS)

    print(
        f"[alpha-curve w_k] reward_model={reward_model} methods={methods} "
        f"optimizer={optimizer} lr_model={lr_model} lr_alpha={lr_alpha} "
        f"seeds={seeds} steps={steps}",
        flush=True,
    )
    if reward_model == "mlp":
        print(f"  mlp hidden={hidden} n_layers={n_layers}", flush=True)

    all_hist: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str]] = []
    idx = 0
    opt_kw = dict(
        lr_theta=lr_model,
        lr_alpha=lr_alpha,
        optimizer=optimizer,
        weight_decay_model=weight_decay_model,
    )
    for sname, skw in settings:
        for mname in methods:
            idx += 1
            print(
                f"[alpha-curve w_k] {sname:12s} {mname:10s} ({reward_model}/{optimizer}) ...",
                flush=True,
            )
            mkw = METHOD_SPECS[mname]
            if reward_model == "linear":
                out = run_linear_with_alpha_history(
                    seeds=seeds,
                    steps=steps,
                    seed=9400 + idx,
                    n_seg=500,
                    q=0.0,
                    log_every=log_every,
                    subtract_init=(mname == "subtract_init"),
                    **skw,
                    **mkw,
                    **opt_kw,
                )
            elif reward_model == "mlp":
                if mname == "stabilized" and idx == 1:
                    print(
                        "  [mlp/stabilized] learner last Linear weight+bias zeroed at init",
                        flush=True,
                    )
                if mname == "subtract_init" and idx == 1:
                    print(
                        "  [mlp/subtract_init] R = f_θ(x) − stopgrad(f_θ₀(x))",
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
                    **opt_kw,
                )
            else:
                raise ValueError(f"unknown reward_model={reward_model!r}")

            hist = out["hist"].copy()
            hist["setting"] = sname
            hist["method"] = mname
            hist["reward_model"] = reward_model
            hist["use_wk"] = True
            hist["optimizer"] = optimizer
            all_hist.append(hist)

            rho, abar, tilde = out["rho"], out["abar"], out["tilde"]
            summary_rows.append(
                {
                    "setting": sname,
                    "method": mname,
                    "reward_model": reward_model,
                    "use_wk": True,
                    "optimizer": optimizer,
                    "lr_model": lr_model,
                    "lr_alpha": lr_alpha,
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
        default=None,
        help="Directory for CSV outputs "
        "(default: results/synthetic_partial_adversary_alpha_curve_wk[_<optimizer>]).",
    )
    p.add_argument("--seeds", type=int, default=100, help="Number of MC seeds (batch size).")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=1, help="Record alpha every N steps.")
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["stabilized", "standard", "subtract_init", "both"],
        default=["stabilized"],
        help="Init variants: stabilized, standard, subtract_init, or both.",
    )
    p.add_argument(
        "--reward-model",
        choices=["linear", "mlp"],
        default="linear",
        help="Reward head: linear (default) or PEBBLE-style mlp.",
    )
    p.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw", "adam_sgd"],
        default="sgd",
        help="Optimizer: sgd | adam | adamw | adam_sgd (Adam on reward, SGD on α).",
    )
    p.add_argument(
        "--lr-model",
        type=float,
        default=None,
        help="LR for reward net / θ (default depends on --optimizer).",
    )
    p.add_argument(
        "--lr-alpha",
        type=float,
        default=None,
        help="LR for trust α (default depends on --optimizer).",
    )
    p.add_argument(
        "--weight-decay-model",
        type=float,
        default=DEFAULT_WEIGHT_DECAY_ADAMW,
        help="Weight decay on reward params for adamw (α uses 0).",
    )
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
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
    )
    p.add_argument("--plot-per-seed", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    methods = resolve_methods(args.methods)
    lr_model, lr_alpha = resolve_lrs(args.optimizer, args.lr_model, args.lr_alpha)

    if args.out_dir is None:
        suffix = "" if args.optimizer == "sgd" else f"_{args.optimizer}"
        args.out_dir = f"results/synthetic_partial_adversary_alpha_curve_wk{suffix}"

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
        optimizer=args.optimizer,
        lr_model=lr_model,
        lr_alpha=lr_alpha,
        weight_decay_model=args.weight_decay_model,
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
