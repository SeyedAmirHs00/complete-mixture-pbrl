"""
Shared-head synthetic core for ALL paper TTP diagnostics.

R(tau) = sum_{t=1}^T tanh(theta · s_t)
max-norm trust + detached w_k (paper recipe).
Optional consensus majority anchor on the reward path only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from final_ttp_synthetic_compare_detached_w import rowwise_corr, sigmoid


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


# Paper branch-symmetry triad, on a shared head.
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
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns
    -------
    rho : (seeds,) corr(R_hat, R*)
    alpha_bar : (seeds, K)
    init_rms : float empirical init rms|Delta R|
    """
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
    states = rng.normal(size=(seeds, n_seg, T, d))

    r_star = np.tanh(np.einsum("sntd,sd->snt", states, theta_star)).sum(2)
    r_star = (r_star - r_star.mean(1, keepdims=True)) / (r_star.std(1, keepdims=True) + 1e-12)

    i = rng.integers(0, n_seg, size=(seeds, pairs))
    j = rng.integers(0, n_seg, size=(seeds, pairs))
    same = i == j
    while same.any():
        j[same] = rng.integers(0, n_seg, size=int(same.sum()))
        same = i == j

    d_star = np.take_along_axis(r_star, i, 1) - np.take_along_axis(r_star, j, 1)
    y = (rng.random((seeds, k, pairs)) < sigmoid(b[None, :, None] * d_star[:, None, :])).astype(np.float64)
    consensus_target = y.mean(1)  # soft majority

    if theta_scale == 0.0:
        theta = np.zeros((seeds, d))
    else:
        theta = rng.normal(scale=theta_scale / np.sqrt(d), size=(seeds, d))
    alpha = np.full((seeds, k), alpha_init)
    rows = np.repeat(np.arange(seeds), pairs)

    def returns(th):
        pre = np.einsum("sntd,sd->snt", states, th)
        r = np.tanh(pre) if variant.use_tanh else pre
        return r.sum(2)

    R0 = returns(theta)
    d0 = np.take_along_axis(R0, i, 1) - np.take_along_axis(R0, j, 1)
    init_rms = float(np.sqrt(np.mean(d0**2)))

    for _ in range(steps):
        pre = np.einsum("sntd,sd->snt", states, theta)
        r_step = np.tanh(pre) if variant.use_tanh else pre
        R = r_step.sum(2)

        if variant.use_tanh:
            trust_raw = np.tanh(alpha)
        else:
            trust_raw = alpha

        abs_t = np.abs(trust_raw)
        if variant.use_maxnorm:
            max_denom = np.maximum(abs_t.max(1, keepdims=True), 1e-12)
            if not variant.detach_maxnorm:
                # attached: still use same value numerically; synthetic has no autograd,
                # so "attach" is approximated by also routing alpha grads through denom
                # (handled below by not detaching in ga scale — keep simple: always detach denom
                # for stability unless explicitly requested; paper ablation used attach).
                pass
            coef = trust_raw / max_denom
            denom_for_ga = max_denom
        else:
            coef = trust_raw
            denom_for_ga = np.ones((seeds, 1))

        if variant.use_confidence_weights:
            w = k * abs_t / np.maximum(abs_t.sum(1, keepdims=True), 1e-12)
        else:
            w = np.ones((seeds, k))

        delta = np.take_along_axis(R, i, 1) - np.take_along_axis(R, j, 1)
        err = sigmoid(coef[:, :, None] * delta[:, None, :]) - y
        coeff = (w[:, :, None] * err * coef[:, :, None]).mean(1) / pairs
        dL = np.zeros_like(R)
        np.add.at(dL, (rows, i.ravel()), coeff.ravel())
        np.add.at(dL, (rows, j.ravel()), -coeff.ravel())

        if variant.consensus_coef > 0.0:
            anchor = (sigmoid(delta) - consensus_target) * (variant.consensus_coef / pairs)
            np.add.at(dL, (rows, i.ravel()), anchor.ravel())
            np.add.at(dL, (rows, j.ravel()), -anchor.ravel())

        if variant.use_tanh:
            sech2 = 1.0 - r_step * r_step
        else:
            sech2 = np.ones_like(r_step)
        g_theta = np.einsum("sn,snt,sntd->sd", dL, sech2, states)
        gn = np.linalg.norm(g_theta, axis=1, keepdims=True)
        g_theta *= np.minimum(1.0, 10.0 / (gn + 1e-12))

        # Trust gradient: unweighted (detached w), through coef
        dloss_dcoef = (err * delta[:, None, :]).mean(2) / k
        if variant.use_tanh:
            dcoef_dalpha = (1.0 - trust_raw * trust_raw) / denom_for_ga
        else:
            dcoef_dalpha = 1.0 / denom_for_ga
        g_alpha = dloss_dcoef * dcoef_dalpha
        if not variant.detach_weights and variant.use_confidence_weights:
            # crude attached-w: also scale alpha grad by w (matches ablation intent)
            g_alpha = g_alpha * w

        theta -= lr_theta * g_theta
        alpha -= lr_alpha * g_alpha

    R = returns(theta)
    trust = np.tanh(alpha) if variant.use_tanh else alpha
    if variant.use_maxnorm:
        abar = trust / np.maximum(np.abs(trust).max(1, keepdims=True), 1e-12)
    else:
        abar = trust
    return rowwise_corr(R, r_star), abar, init_rms


def build_k4_configs() -> Dict[str, Tuple[float, ...]]:
    return {
        "3R1N": (1.0, 1.0, 1.0, 0.0),
        "3R1A": (1.0, 1.0, 1.0, -1.0),
        "2R1A1N": (1.0, 1.0, -1.0, 0.0),
        "1R3A": (1.0, -1.0, -1.0, -1.0),
    }
