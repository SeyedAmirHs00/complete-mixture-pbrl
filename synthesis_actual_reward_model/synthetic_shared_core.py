"""
Shared-head synthetic core (PyTorch autograd).

Uses the production PEBBLE reward MLP (`gen_net` from reward_model):
  d -> Linear(H) -> LeakyReLU -> ... -> Linear(1) -> Tanh,  R = sum_t r(s_t).

Mean TTP loss over experts and pairs (proper /K scaling via .mean()).
Detached max-norm denom + detached w_k on the reward path; trust path unweighted.
Optional consensus majority anchor on the reward path only (disabled by default).

Init kinds
----------
standard   — PyTorch default nn.Linear init
stabilized — default init, then zero the last Linear weight and bias

Teacher R* is an independently initialized frozen gen_net (same architecture),
with per-seed segment-return normalization so preference difficulty is set by β.

Defaults match the disjoint robotics regime: n_seg=500, q=0 (no shared pairs).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from reward_model.vanilla_reward_model import gen_net  # noqa: E402

# Match production mixture / vanilla construct_ensemble defaults.
DEFAULT_HIDDEN = 256
DEFAULT_N_LAYERS = 3

_T = TypeVar("_T")


def status_print(msg: str) -> None:
    """Flush a status line; uses tqdm.write when a bar is active."""
    line = f"[synth] {msg}"
    try:
        from tqdm import tqdm

        tqdm.write(line)
    except ImportError:
        print(line, flush=True)


def progress_iter(
    iterable: Iterable[_T],
    *,
    total: Optional[int] = None,
    desc: str = "",
    leave: bool = True,
    disable: bool = False,
) -> Iterator[_T]:
    """Wrap an iterable with a tqdm bar when available."""
    if disable:
        return iter(iterable)
    try:
        from tqdm import tqdm

        return tqdm(
            iterable,
            total=total,
            desc=desc,
            leave=leave,
            dynamic_ncols=True,
            mininterval=0.3,
        )
    except ImportError:
        if desc:
            status_print(desc)
        return iter(iterable)


def progress_range(
    n: int,
    *,
    desc: str = "",
    leave: bool = True,
    disable: bool = False,
) -> Iterator[int]:
    return progress_iter(range(n), total=n, desc=desc, leave=leave, disable=disable)


def get_device(device: Optional[torch.device] = None) -> torch.device:
    """Resolve training device; prefer CUDA when available."""
    if device is not None:
        return torch.device(device) if not isinstance(device, torch.device) else device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    init_kind: str = "stabilized"  # "standard" | "stabilized"
    consensus_coef: float = 0.0
    use_alpha_tanh: bool = True  # trust path: α ↦ tanh(α)
    use_maxnorm: bool = True
    use_confidence_weights: bool = True
    detach_maxnorm: bool = True
    detach_weights: bool = True


SHARED_BRANCH_VARIANTS = (
    SharedVariant("standard", "Standard", init_kind="standard", consensus_coef=0.0),
    SharedVariant("stabilized", "Stabilized", init_kind="stabilized", consensus_coef=0.0),
)


def build_reward_mlp(
    d: int,
    *,
    hidden: int = DEFAULT_HIDDEN,
    n_layers: int = DEFAULT_N_LAYERS,
    activation: str = "tanh",
) -> nn.Sequential:
    """Production PEBBLE reward head (state-only input dim ``d``)."""
    return nn.Sequential(
        *gen_net(in_size=d, out_size=1, H=hidden, n_layers=n_layers, activation=activation)
    )


def zero_last_linear(net: nn.Module) -> None:
    """Zero the last Linear layer (Stabilized init)."""
    last: Optional[nn.Linear] = None
    for module in net.modules():
        if isinstance(module, nn.Linear):
            last = module
    if last is None:
        raise ValueError("network has no Linear layers")
    with torch.no_grad():
        last.weight.zero_()
        if last.bias is not None:
            last.bias.zero_()


def apply_init_kind(net: nn.Module, init_kind: str) -> None:
    kind = init_kind.lower()
    if kind == "standard":
        return
    if kind == "stabilized":
        zero_last_linear(net)
        return
    raise ValueError(f"unknown init_kind={init_kind!r}; expected 'standard' or 'stabilized'")


@torch.no_grad()
def measure_rms_delta_r(
    net: nn.Module,
    states: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
) -> float:
    """rms|ΔR| of preference pairs under ``net`` (i/j shaped [K,P] or flat)."""
    R = segment_returns(net, states)
    dR = R[i_idx.reshape(-1)] - R[j_idx.reshape(-1)]
    return float(torch.sqrt((dR**2).mean()).item())


@torch.no_grad()
def calibrate_states_to_target_rms(
    net: nn.Module,
    states: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    target_rms: float,
    *,
    n_iters: int = 24,
) -> Tuple[torch.Tensor, float]:
    """
    Scale trajectory states so init rms|ΔR| under a *fixed* learner ≈ ``target_rms``.

    PyTorch initialization of ``net`` is left unchanged. Searches a global
    multiplier ``s`` with states' = s · states (``target_rms <= 0`` ⇒ s = 0).

    Returns
    -------
    scaled_states : same shape as ``states``
    achieved_rms : float
    """
    if target_rms <= 0.0:
        return torch.zeros_like(states), 0.0

    base = measure_rms_delta_r(net, states, i_idx, j_idx)
    if base < 1e-10:
        # Degenerate pairs / flat head on these states — cannot hit a nonzero target.
        return states.clone(), 0.0

    lo, hi = 1e-4, 1e4
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        rms = measure_rms_delta_r(net, states * mid, i_idx, j_idx)
        if rms < target_rms:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    scaled = states * scale
    return scaled, measure_rms_delta_r(net, scaled, i_idx, j_idx)


def segment_returns(net: nn.Module, states: torch.Tensor) -> torch.Tensor:
    """
    states: [N, T, D] -> segment returns [N]
    """
    n, t, d = states.shape
    out = net(states.reshape(n * t, d))
    if out.dim() == 2 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    return out.reshape(n, t).sum(dim=1)


@torch.no_grad()
def teacher_segment_returns(
    states: np.ndarray,
    *,
    d: int,
    torch_seed: int,
    device: torch.device,
    hidden: int = DEFAULT_HIDDEN,
    n_layers: int = DEFAULT_N_LAYERS,
) -> np.ndarray:
    """Frozen independently initialized gen_net teacher; returns unnormalized R* [N]."""
    torch.manual_seed(torch_seed)
    teacher = build_reward_mlp(d, hidden=hidden, n_layers=n_layers).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    st = torch.as_tensor(states, dtype=torch.float32, device=device)
    return segment_returns(teacher, st).cpu().numpy()


def sample_expert_pairs(
    rng: np.random.Generator,
    *,
    seeds: int,
    k: int,
    n_seg: int,
    pairs: int,
    q: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample (i, j) indices per expert with shared-pair fraction q.

    Returns
    -------
    i_all, j_all : (seeds, k, pairs)
    """
    q = float(np.clip(q, 0.0, 1.0))
    n_shared = int(round(q * pairs))
    n_priv = pairs - n_shared
    bs = n_seg // k
    if bs < 2:
        raise ValueError(f"n_seg={n_seg} too small for k={k} disjoint blocks")
    blocks = [np.arange(b * bs, (b + 1) * bs) for b in range(k)]

    i_all = np.zeros((seeds, k, pairs), dtype=np.int64)
    j_all = np.zeros((seeds, k, pairs), dtype=np.int64)

    if n_shared > 0:
        i_s = rng.integers(0, n_seg, (seeds, n_shared))
        j_s = rng.integers(0, n_seg, (seeds, n_shared))
        same = i_s == j_s
        while same.any():
            j_s[same] = rng.integers(0, n_seg, int(same.sum()))
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

    return i_all, j_all


def _trust_coef_w(
    alpha: torch.Tensor,
    variant: SharedVariant,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (trust, coef, w) for a single-seed alpha of shape [K]."""
    if variant.use_alpha_tanh:
        trust = torch.tanh(alpha)
    else:
        trust = alpha
    abs_t = trust.abs()
    if variant.use_maxnorm:
        denom = torch.max(abs_t).clamp_min(1e-12)
        if variant.detach_maxnorm:
            denom = denom.detach()
        coef = trust / denom
    else:
        coef = trust
    if variant.use_confidence_weights:
        w = k * abs_t / abs_t.sum().clamp_min(1e-12)
        if variant.detach_weights:
            w = w.detach()
    else:
        w = torch.ones(k, device=alpha.device)
    return trust, coef, w


def train_one_seed_ttp(
    states: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    y: torch.Tensor,
    y_bar: torch.Tensor,
    variant: SharedVariant,
    *,
    steps: int,
    lr_theta: float,
    lr_alpha: float,
    alpha_init: float,
    torch_seed: int,
    hidden: int = DEFAULT_HIDDEN,
    n_layers: int = DEFAULT_N_LAYERS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Train one seed. states [N,T,D], i/j [K,P], y [K,P], y_bar [P].

    Returns R_hat [N], abar [K], init_rms (scalar for this seed).
    """
    device = states.device
    n, t, d = states.shape
    k, pairs = y.shape

    torch.manual_seed(torch_seed)
    net = build_reward_mlp(d, hidden=hidden, n_layers=n_layers).to(device)
    apply_init_kind(net, variant.init_kind)
    init_rms = measure_rms_delta_r(net, states, i_idx, j_idx)

    alpha = torch.nn.Parameter(torch.full((k,), float(alpha_init), device=device))

    opt = torch.optim.SGD(
        [
            {"params": net.parameters(), "lr": lr_theta},
            {"params": [alpha], "lr": lr_alpha},
        ]
    )

    for _ in range(steps):
        opt.zero_grad()
        R = segment_returns(net, states)
        delta = R[i_idx] - R[j_idx]  # [K, P]

        _, coef, w = _trust_coef_w(alpha, variant, k)

        logits_R = coef.detach().unsqueeze(1) * delta
        bce_R = F.binary_cross_entropy_with_logits(logits_R, y, reduction="none")
        loss_R = (w.unsqueeze(1) * bce_R).mean()

        if variant.consensus_coef > 0.0:
            loss_R = loss_R + variant.consensus_coef * F.binary_cross_entropy_with_logits(
                delta.mean(dim=0), y_bar, reduction="none"
            ).mean()

        logits_A = coef.unsqueeze(1) * delta.detach()
        if variant.detach_weights or not variant.use_confidence_weights:
            loss_A = F.binary_cross_entropy_with_logits(logits_A, y, reduction="none").mean()
        else:
            bce_A = F.binary_cross_entropy_with_logits(logits_A, y, reduction="none")
            loss_A = (w.unsqueeze(1) * bce_A).mean()

        (loss_R + loss_A).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()

    with torch.no_grad():
        R_hat = segment_returns(net, states).cpu().numpy()
        trust, coef, _ = _trust_coef_w(alpha, variant, k)
        if variant.use_maxnorm:
            abar = coef.cpu().numpy()
        else:
            abar = trust.cpu().numpy()

    return R_hat, abar, init_rms


def run_shared_variant(
    betas: Tuple[float, ...],
    variant: SharedVariant,
    *,
    seeds: int,
    n_seg: int = 500,
    T: int = 50,
    d: int = 16,
    pairs: int = 256,
    q: float = 0.0,
    steps: int = 400,
    lr_theta: float = 0.05,
    lr_alpha: float = 0.005,
    alpha_init: float = 0.01,
    seed: int = 0,
    device: Optional[torch.device] = None,
    hidden: int = DEFAULT_HIDDEN,
    n_layers: int = DEFAULT_N_LAYERS,
    progress: bool = True,
    progress_desc: Optional[str] = None,
    target_rms: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns
    -------
    rho : (seeds,) corr(R_hat, R*)
    alpha_bar : (seeds, K)
    init_rms : float mean empirical init rms|Delta R| across seeds

    Preferences are always generated from a frozen teacher gen_net (true reward).
    If ``target_rms`` is set, trajectory states are globally rescaled (learner
    init untouched) so preference-pair rms|ΔR| matches the target under the
    PyTorch-default learner; teacher labels are then built on those scaled states.
    """
    device = get_device(device)
    rng = np.random.default_rng(seed)
    k = len(betas)
    b = np.asarray(betas, dtype=np.float64)

    label = progress_desc or f"{variant.name} K={k}"
    rms_msg = f"target_rms={target_rms:g}" if target_rms is not None else f"init={variant.init_kind}"
    status_print(
        f"{label} | device={device} seeds={seeds} steps={steps} n={n_seg} T={T} d={d} "
        f"pairs={pairs} q={q:g} {rms_msg}"
    )

    states_np = rng.normal(size=(seeds, n_seg, T, d)).astype(np.float32)
    i_np, j_np = sample_expert_pairs(
        rng, seeds=seeds, k=k, n_seg=n_seg, pairs=pairs, q=q
    )

    # Optionally rescale trajectories so a fixed default-init learner has target rms|ΔR|.
    if target_rms is not None:
        for s in progress_range(
            seeds, desc=f"{label} scale-states", leave=False, disable=not progress
        ):
            torch_seed = seed + 1_000 + 31 * s
            torch.manual_seed(torch_seed)
            probe = build_reward_mlp(d, hidden=hidden, n_layers=n_layers).to(device)
            # Keep PyTorch default init — do not apply_init_kind / weight scaling.
            st = torch.as_tensor(states_np[s], dtype=torch.float32, device=device)
            i_idx = torch.as_tensor(i_np[s], dtype=torch.long, device=device)
            j_idx = torch.as_tensor(j_np[s], dtype=torch.long, device=device)
            scaled, _ = calibrate_states_to_target_rms(
                probe, st, i_idx, j_idx, float(target_rms)
            )
            states_np[s] = scaled.detach().cpu().numpy().astype(np.float32)

    r_star = np.zeros((seeds, n_seg), dtype=np.float64)
    for s in progress_range(
        seeds, desc=f"{label} teacher", leave=False, disable=not progress
    ):
        r = teacher_segment_returns(
            states_np[s],
            d=d,
            torch_seed=seed + 10_000 + 97 * s,
            device=device,
            hidden=hidden,
            n_layers=n_layers,
        )
        r_star[s] = (r - r.mean()) / (r.std() + 1e-12)

    y_np = np.zeros((seeds, k, pairs), dtype=np.float64)
    for e in range(k):
        d_star = np.take_along_axis(r_star, i_np[:, e], 1) - np.take_along_axis(
            r_star, j_np[:, e], 1
        )
        y_np[:, e] = (rng.random((seeds, pairs)) < sigmoid_np(b[e] * d_star)).astype(np.float64)
    consensus_np = y_np.mean(1)

    rhos = np.zeros(seeds, dtype=np.float64)
    abars = np.zeros((seeds, k), dtype=np.float64)
    init_rms_list: List[float] = []

    for s in progress_range(
        seeds, desc=f"{label} train", leave=True, disable=not progress
    ):
        states = torch.as_tensor(states_np[s], dtype=torch.float32, device=device)
        i_idx = torch.as_tensor(i_np[s], dtype=torch.long, device=device)
        j_idx = torch.as_tensor(j_np[s], dtype=torch.long, device=device)
        y = torch.as_tensor(y_np[s], dtype=torch.float32, device=device)
        y_bar = torch.as_tensor(consensus_np[s], dtype=torch.float32, device=device)

        # Same torch_seed as the probe above so default init matches the scaled ΔR.
        R_hat, abar, init_rms_s = train_one_seed_ttp(
            states,
            i_idx,
            j_idx,
            y,
            y_bar,
            variant,
            steps=steps,
            lr_theta=lr_theta,
            lr_alpha=lr_alpha,
            alpha_init=alpha_init,
            torch_seed=seed + 1_000 + 31 * s,
            hidden=hidden,
            n_layers=n_layers,
        )
        rhos[s] = float(rowwise_corr(R_hat[None, :], r_star[s : s + 1])[0])
        abars[s] = abar
        init_rms_list.append(init_rms_s)

    init_rms = float(np.mean(init_rms_list))
    status_print(
        f"{label} done | mean_rho={rhos.mean():+.3f} "
        f"correct={(rhos > 0.05).mean():.3f} init_rms={init_rms:.4g}"
    )
    return rhos, abars, init_rms


def build_k4_configs() -> Dict[str, Tuple[float, ...]]:
    return {
        "3R1N": (1.0, 1.0, 1.0, 0.0),
        "3R1A": (1.0, 1.0, 1.0, -1.0),
        "2R1A1N": (1.0, 1.0, -1.0, 0.0),
        "1R3A": (1.0, -1.0, -1.0, -1.0),
    }
