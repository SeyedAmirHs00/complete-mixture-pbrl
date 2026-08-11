"""
Shared-head synthetic core (PyTorch autograd).

Mean TTP loss over experts and pairs (proper /K scaling via .mean()).
Detached max-norm denom + detached w_k on the reward path; trust path unweighted.
Optional consensus majority anchor on the reward path only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def rowwise_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    num = (x * y).sum(axis=1)
    den = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1)) + eps
    return num / den


# Back-compat aliases used by runners
sigmoid = sigmoid_np


@dataclass(frozen=True)
class SharedVariant:
    name: str
    label: str
    target_rms: float  # 0 => theta=0 (Stabilized); >0 => random theta calibrated to rms
    consensus_coef: float = 0.0
    use_tanh: bool = True
    use_maxnorm: bool = True
    use_confidence_weights: bool = True
    detach_maxnorm: bool = True
    detach_weights: bool = True


SHARED_BRANCH_VARIANTS = (
    SharedVariant("standard", "Standard", target_rms=1.4, consensus_coef=0.0),
    SharedVariant("stabilized", "Stabilized", target_rms=0.0, consensus_coef=0.0),
    SharedVariant("consensus", "Consensus", target_rms=1.4, consensus_coef=0.5),
)


def calibrate_theta_scale(
    target_rms: float,
    *,
    seeds: int,
    n_seg: int,
    T: int,
    d: int,
    rng: np.random.Generator,
) -> float:
    if target_rms <= 0:
        return 0.0

    def measure(scale: float) -> float:
        th = rng.normal(scale=scale / np.sqrt(d), size=(seeds, d))
        states = rng.normal(size=(seeds, n_seg, T, d))
        R = np.tanh(np.einsum("sntd,sd->snt", states, th)).sum(axis=2)
        i = rng.integers(0, n_seg, size=(seeds, 512))
        j = rng.integers(0, n_seg, size=(seeds, 512))
        dR = np.take_along_axis(R, i, axis=1) - np.take_along_axis(R, j, axis=1)
        return float(np.sqrt(np.mean(dR**2)))

    lo, hi = 1e-4, 20.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if measure(mid) < target_rms:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _segment_returns(states: torch.Tensor, theta: torch.Tensor, use_tanh: bool) -> torch.Tensor:
    # states: [S,N,T,D], theta: [S,D] -> R: [S,N]
    pre = torch.einsum("sntd,sd->snt", states, theta)
    r = torch.tanh(pre) if use_tanh else pre
    return r.sum(dim=2)


def run_shared_variant(
    betas: Tuple[float, ...],
    variant: SharedVariant,
    *,
    seeds: int,
    n_seg: int = 48,
    T: int = 50,
    d: int = 16,
    pairs: int = 256,
    steps: int = 400,
    lr_theta: float = 0.05,
    lr_alpha: float = 0.005,
    alpha_init: float = 0.01,
    seed: int = 0,
    theta_scale: Optional[float] = None,
    cal_rng: Optional[np.random.Generator] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns
    -------
    rho : (seeds,) corr(R_hat, R*)
    alpha_bar : (seeds, K)
    init_rms : float empirical init rms|Delta R|
    """
    device = device or torch.device("cpu")
    rng = np.random.default_rng(seed)
    k = len(betas)
    b = np.asarray(betas, dtype=np.float64)

    if theta_scale is None:
        crng = cal_rng if cal_rng is not None else np.random.default_rng(seed + 7)
        theta_scale = calibrate_theta_scale(
            variant.target_rms, seeds=min(40, seeds), n_seg=n_seg, T=T, d=d, rng=crng
        )

    theta_star = rng.normal(size=(seeds, d))
    theta_star /= np.linalg.norm(theta_star, axis=1, keepdims=True) + 1e-12
    states_np = rng.normal(size=(seeds, n_seg, T, d))

    r_star = np.tanh(np.einsum("sntd,sd->snt", states_np, theta_star)).sum(2)
    r_star = (r_star - r_star.mean(1, keepdims=True)) / (r_star.std(1, keepdims=True) + 1e-12)

    i_np = rng.integers(0, n_seg, size=(seeds, pairs))
    j_np = rng.integers(0, n_seg, size=(seeds, pairs))
    same = i_np == j_np
    while same.any():
        j_np[same] = rng.integers(0, n_seg, size=int(same.sum()))
        same = i_np == j_np

    d_star = np.take_along_axis(r_star, i_np, 1) - np.take_along_axis(r_star, j_np, 1)
    y_np = (rng.random((seeds, k, pairs)) < sigmoid_np(b[None, :, None] * d_star[:, None, :])).astype(
        np.float64
    )
    consensus_np = y_np.mean(1)

    if theta_scale == 0.0:
        theta0 = np.zeros((seeds, d), dtype=np.float64)
    else:
        theta0 = rng.normal(scale=theta_scale / np.sqrt(d), size=(seeds, d))

    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    i = torch.as_tensor(i_np, dtype=torch.long, device=device)
    j = torch.as_tensor(j_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    y_bar = torch.as_tensor(consensus_np, dtype=torch.float32, device=device)

    theta = torch.nn.Parameter(torch.as_tensor(theta0, dtype=torch.float32, device=device))
    alpha = torch.nn.Parameter(torch.full((seeds, k), float(alpha_init), device=device))

    with torch.no_grad():
        R0 = _segment_returns(states, theta, variant.use_tanh)
        d0 = R0.gather(1, i) - R0.gather(1, j)
        init_rms = float(torch.sqrt((d0**2).mean()).item())

    for _ in range(steps):
        if theta.grad is not None:
            theta.grad = None
        if alpha.grad is not None:
            alpha.grad = None

        R = _segment_returns(states, theta, variant.use_tanh)
        delta = R.gather(1, i) - R.gather(1, j)  # [S,P]

        if variant.use_tanh:
            trust = torch.tanh(alpha)
        else:
            trust = alpha

        abs_t = trust.abs()
        if variant.use_maxnorm:
            denom = abs_t.amax(dim=1, keepdim=True).clamp_min(1e-12)
            if variant.detach_maxnorm:
                denom = denom.detach()
            coef = trust / denom
        else:
            coef = trust

        if variant.use_confidence_weights:
            w = k * abs_t / abs_t.sum(dim=1, keepdim=True).clamp_min(1e-12)
            if variant.detach_weights:
                w = w.detach()
        else:
            w = torch.ones(seeds, k, device=device)

        # Reward path: hold coef (and usually w) fixed.
        # Mean over (K,P) per seed, then SUM over seeds — matches independent
        # per-seed numpy updates (do NOT .mean() over the seed axis).
        logits_R = coef.detach().unsqueeze(2) * delta.unsqueeze(1)  # [S,K,P]
        bce_R = F.binary_cross_entropy_with_logits(logits_R, y, reduction="none")
        loss_R = (w.unsqueeze(2) * bce_R).mean(dim=(1, 2)).sum()

        if variant.consensus_coef > 0.0:
            loss_R = loss_R + variant.consensus_coef * F.binary_cross_entropy_with_logits(
                delta, y_bar, reduction="none"
            ).mean(dim=1).sum()

        # Trust path: unweighted; hold delta fixed so only alpha gets this term
        logits_A = coef.unsqueeze(2) * delta.detach().unsqueeze(1)
        if variant.detach_weights or not variant.use_confidence_weights:
            loss_A = F.binary_cross_entropy_with_logits(logits_A, y, reduction="none").mean(
                dim=(1, 2)
            ).sum()
        else:
            # attached-w ablation: weight the alpha loss too
            bce_A = F.binary_cross_entropy_with_logits(logits_A, y, reduction="none")
            loss_A = (w.unsqueeze(2) * bce_A).mean(dim=(1, 2)).sum()

        (loss_R + loss_A).backward()

        with torch.no_grad():
            # Per-seed grad clip on theta (matches prior numpy recipe)
            g = theta.grad
            gn = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
            g.mul_(torch.clamp(10.0 / gn, max=1.0))
            theta.data.sub_(lr_theta * g)
            alpha.data.sub_(lr_alpha * alpha.grad)

    with torch.no_grad():
        R = _segment_returns(states, theta, variant.use_tanh).cpu().numpy()
        if variant.use_tanh:
            trust = torch.tanh(alpha)
        else:
            trust = alpha
        if variant.use_maxnorm:
            abar = (trust / trust.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
        else:
            abar = trust.cpu().numpy()

    return rowwise_corr(R, r_star), abar, init_rms


def build_k4_configs() -> Dict[str, Tuple[float, ...]]:
    return {
        "3R1N": (1.0, 1.0, 1.0, 0.0),
        "3R1A": (1.0, 1.0, 1.0, -1.0),
        "2R1A1N": (1.0, 1.0, -1.0, 0.0),
        "1R3A": (1.0, -1.0, -1.0, -1.0),
    }
