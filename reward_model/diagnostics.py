"""Diagnostics for reward-model preference pairs and trajectory buffers.

Logged quantities
-----------------
rms_delta_r
    rms|ΔR|_0 = sqrt( mean_{(i,j) in D} (R(τ_i) - R(τ_j))^2 )
    where D is the labeled preference-pair buffer and R(τ) is the
    ensemble-mean predicted segment return (sum over timesteps).

mean_sa_var / mean_sa_std / mean_sa_second_moment
    Moments of concatenated [obs, action] rows in the reward-model
    trajectory buffer (``inputs``).

mean_state_* / mean_action_*
    Same moments restricted to the state or action slice.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch


def _as_traj_array(traj) -> Optional[np.ndarray]:
    if traj is None or len(traj) == 0:
        return None
    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    return arr


def collect_sa_from_inputs(inputs: Sequence, ds: int, da: int) -> Optional[np.ndarray]:
    """Flatten reward-model ``inputs`` trajectories into [N, ds+da]."""
    chunks: List[np.ndarray] = []
    expected = ds + da
    for traj in inputs:
        arr = _as_traj_array(traj)
        if arr is None:
            continue
        if arr.shape[-1] != expected:
            # Skip malformed / empty placeholder rows.
            continue
        chunks.append(arr)
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def sa_moment_stats(x: np.ndarray, ds: int, da: int) -> Dict[str, float]:
    """Compute mean variance / std / second-moment of state, action, and SA."""
    assert x.ndim == 2 and x.shape[1] == ds + da
    obs = x[:, :ds]
    actions = x[:, ds : ds + da]

    def _moments(arr: np.ndarray) -> Dict[str, float]:
        var_per_dim = np.var(arr, axis=0)
        second_per_dim = np.mean(arr ** 2, axis=0)
        std_per_dim = np.sqrt(np.maximum(var_per_dim, 0.0))
        return {
            "mean_var": float(var_per_dim.mean()),
            "mean_std": float(std_per_dim.mean()),
            "mean_second_moment": float(second_per_dim.mean()),
        }

    state_m = _moments(obs)
    action_m = _moments(actions)
    sa_m = _moments(x)
    return {
        "n_transitions": float(x.shape[0]),
        "mean_state_var": state_m["mean_var"],
        "mean_state_std": state_m["mean_std"],
        "mean_state_second_moment": state_m["mean_second_moment"],
        "mean_action_var": action_m["mean_var"],
        "mean_action_std": action_m["mean_std"],
        "mean_action_second_moment": action_m["mean_second_moment"],
        "mean_sa_var": sa_m["mean_var"],
        "mean_sa_std": sa_m["mean_std"],
        "mean_sa_second_moment": sa_m["mean_second_moment"],
    }


@torch.no_grad()
def rms_delta_r_from_pairs(
    ensemble: Sequence[torch.nn.Module],
    seg1: np.ndarray,
    seg2: np.ndarray,
    device: Union[str, torch.device] = "cuda",
    batch_size: int = 256,
) -> Dict[str, float]:
    """rms|ΔR|_0 and companion stats over preference pairs (seg1, seg2).

    R(τ) = mean_ensemble sum_t r_θ(s_t, a_t). Alpha / trust scaling is NOT applied.
    """
    n = int(seg1.shape[0])
    if n == 0:
        return {
            "n_pairs": 0.0,
            "rms_delta_r": 0.0,
            "std_delta_r": 0.0,
            "var_delta_r": 0.0,
            "mean_abs_delta_r": 0.0,
        }

    device = torch.device(device) if not isinstance(device, torch.device) else device
    deltas: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s1 = torch.as_tensor(seg1[start:end], dtype=torch.float32, device=device)
        s2 = torch.as_tensor(seg2[start:end], dtype=torch.float32, device=device)

        r1_members = []
        r2_members = []
        for member in ensemble:
            # member output: [B, T, 1] or [B, T] depending on Sequential
            out1 = member(s1)
            out2 = member(s2)
            if out1.dim() == 3:
                out1 = out1.squeeze(-1)
                out2 = out2.squeeze(-1)
            r1_members.append(out1.sum(dim=1))
            r2_members.append(out2.sum(dim=1))

        r1 = torch.stack(r1_members, dim=0).mean(dim=0)
        r2 = torch.stack(r2_members, dim=0).mean(dim=0)
        deltas.append((r1 - r2).detach().cpu().numpy())

    delta = np.concatenate(deltas, axis=0)
    sq = delta ** 2
    return {
        "n_pairs": float(n),
        "rms_delta_r": float(np.sqrt(sq.mean())),
        "std_delta_r": float(delta.std()),
        "var_delta_r": float(delta.var()),
        "mean_abs_delta_r": float(np.abs(delta).mean()),
    }


def gather_preference_pairs(reward_models: Sequence) -> tuple:
    """Stack labeled preference segments from one or more RewardModel buffers."""
    seg1_list, seg2_list = [], []
    for rm in reward_models:
        n = rm.capacity if rm.buffer_full else rm.buffer_index
        if n <= 0:
            continue
        seg1_list.append(rm.buffer_seg1[:n])
        seg2_list.append(rm.buffer_seg2[:n])
    if not seg1_list:
        return (
            np.zeros((0, 1, 1), dtype=np.float32),
            np.zeros((0, 1, 1), dtype=np.float32),
        )
    return np.concatenate(seg1_list, axis=0), np.concatenate(seg2_list, axis=0)


def compute_reward_buffer_diagnostics(
    ensemble: Sequence[torch.nn.Module],
    reward_models: Sequence,
    ds: int,
    da: int,
    device: Union[str, torch.device] = "cuda",
    batch_size: int = 256,
) -> Dict[str, float]:
    """Full diagnostic dict for mixture / multi-buffer reward models."""
    seg1, seg2 = gather_preference_pairs(reward_models)
    stats = rms_delta_r_from_pairs(
        ensemble, seg1, seg2, device=device, batch_size=batch_size
    )

    # Trajectory buffers are duplicated across experts; use the first non-empty.
    x = None
    for rm in reward_models:
        x = collect_sa_from_inputs(rm.inputs, ds, da)
        if x is not None:
            break
    if x is None:
        stats.update(
            {
                "n_transitions": 0.0,
                "mean_state_var": 0.0,
                "mean_state_std": 0.0,
                "mean_state_second_moment": 0.0,
                "mean_action_var": 0.0,
                "mean_action_std": 0.0,
                "mean_action_second_moment": 0.0,
                "mean_sa_var": 0.0,
                "mean_sa_std": 0.0,
                "mean_sa_second_moment": 0.0,
            }
        )
    else:
        stats.update(sa_moment_stats(x, ds, da))
    return stats


def log_reward_buffer_diagnostics(logger, stats: Dict[str, float], step: int) -> None:
    if logger is None:
        return
    for key, value in stats.items():
        logger.log(f"reward/{key}", float(value), step)


def write_reward_buffer_diagnostics_csv(
    out_dir: str,
    stats: Dict[str, float],
    step: int,
    filename: str = "buffer_diagnostics.csv",
) -> str:
    """Write diagnostics to a dedicated CSV (avoids reward.csv fieldname lock-in)."""
    import csv
    import os

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    row = {"step": int(step), **{k: float(v) for k, v in stats.items()}}
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path
