#!/usr/bin/env python3
"""Roll out saved SAC actor checkpoint(s) and write mp4 video(s).

Works for DM Control (cheetah_run, walker_walk, ...) and MetaWorld
(metaworld_door-open-v2, metaworld_sweep-into-v2, ...) using the run's
``.hydra/config.yaml``.

Examples (inside ``mixture_pbrl_container``, from repo root):

  # Actor path (step inferred from filename)
  python scripts/record_agent_video.py \\
    --actor exp_.../cheetah_run/.../seed67890/actor_1000000.pt

  # Multiple actors / envs / steps in one call
  python scripts/record_agent_video.py --actor \\
    exp_.../metaworld_door-open-v2/.../actor_1000000.pt \\
    exp_.../metaworld_sweep-into-v2/.../actor_1000000.pt \\
    exp_.../walker_walk/.../actor_500000.pt

  # Classic ckpt_dir + step
  python scripts/record_agent_video.py \\
    --ckpt_dir exp_.../walker_walk/.../seed34512_... \\
    --step 500000
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Optional, Tuple

import imageio
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

# Allow running as scripts/record_agent_video.py from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import utils

ACTOR_RE = re.compile(r"actor_(\d+)\.pt$")


def parse_args():
    p = argparse.ArgumentParser(
        description="Record policy rollout video(s) from actor_*.pt checkpoints"
    )
    p.add_argument(
        "--actor",
        nargs="+",
        default=None,
        help="One or more paths to actor_{step}.pt (ckpt dir = parent; step from name)",
    )
    p.add_argument(
        "--ckpt_dir",
        default=None,
        help="Run directory containing actor_{step}.pt and .hydra/config.yaml",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step (required with --ckpt_dir; optional with --actor)",
    )
    p.add_argument("--episodes", type=int, default=1, help="Eval episodes per checkpoint")
    p.add_argument("--height", type=int, default=256, help="DMC render height")
    p.add_argument("--width", type=int, default=256, help="DMC render width")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="DMC camera id (ignored for MetaWorld)",
    )
    p.add_argument(
        "--mw_camera",
        default="corner",
        help="MetaWorld camera name: corner, corner2, corner3, topview, "
        "behindGripper, gripperPOV (default: corner)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output mp4 (single job only). Default: {ckpt_dir}/video_actor_{step}.mp4",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda / cpu (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--actor_only",
        action="store_true",
        help="Load only actor_*.pt (skip critic weights)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override env seed from config (default: keep config seed)",
    )
    return p.parse_args()


def resolve_jobs(args) -> List[Tuple[str, int, Optional[str]]]:
    """Return list of (ckpt_dir, step, optional_out_path)."""
    jobs: List[Tuple[str, int, Optional[str]]] = []

    if args.actor:
        for actor_path in args.actor:
            actor_path = os.path.abspath(actor_path)
            if not os.path.isfile(actor_path):
                raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")
            m = ACTOR_RE.search(os.path.basename(actor_path))
            if not m:
                raise ValueError(
                    f"Expected filename actor_<step>.pt, got: {os.path.basename(actor_path)}"
                )
            step = args.step if args.step is not None else int(m.group(1))
            ckpt_dir = os.path.dirname(actor_path)
            jobs.append((ckpt_dir, step, None))
        if args.out is not None:
            if len(jobs) != 1:
                raise ValueError("--out can only be used with a single --actor")
            jobs[0] = (jobs[0][0], jobs[0][1], args.out)
        return jobs

    if args.ckpt_dir:
        if args.step is None:
            raise ValueError("--step is required when using --ckpt_dir")
        ckpt_dir = os.path.abspath(args.ckpt_dir)
        jobs.append((ckpt_dir, args.step, args.out))
        return jobs

    raise ValueError("Provide --actor PATH [PATH ...] or --ckpt_dir DIR --step N")


def load_cfg(ckpt_dir: str, device: str, seed_override: Optional[int]):
    cfg_path = os.path.join(ckpt_dir, ".hydra", "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing Hydra config: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    cfg.device = device
    if seed_override is not None:
        cfg.seed = seed_override
    return cfg


def build_agent(cfg, env):
    # Saved configs keep Hydra placeholders (???) for dims filled at train time.
    cfg.agent.params.obs_dim = int(env.observation_space.shape[0])
    cfg.agent.params.action_dim = int(env.action_space.shape[0])
    cfg.agent.params.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]
    cfg.agent.params.device = cfg.device
    OmegaConf.resolve(cfg)
    return hydra.utils.instantiate(cfg.agent)


def load_weights(agent, ckpt_dir: str, step: int, device: str, actor_only: bool):
    actor_path = os.path.join(ckpt_dir, f"actor_{step}.pt")
    if not os.path.isfile(actor_path):
        raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")

    critic_path = os.path.join(ckpt_dir, f"critic_{step}.pt")
    critic_target_path = os.path.join(ckpt_dir, f"critic_target_{step}.pt")
    have_critics = os.path.isfile(critic_path) and os.path.isfile(critic_target_path)

    if actor_only or not have_critics:
        if not have_critics and not actor_only:
            print(f"warning: critics missing under {ckpt_dir}; loading actor only")
        agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
        return

    # agent.load uses bare torch.load; load with map_location manually for CPU/GPU flexibility.
    agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
    agent.critic_target.load_state_dict(
        torch.load(critic_target_path, map_location=device)
    )


def _metaworld_base_env(env):
    """Unwrap TimeLimit / NormalizedBoxEnv to the Sawyer env with ``sim``."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    if hasattr(base, "_wrapped_env"):
        base = base._wrapped_env
    while hasattr(base, "env"):
        base = base.env
    return base


# Sawyer body names that make up the robot (shared across MetaWorld tasks).
_SAWYER_BODY_PREFIXES = (
    "base",
    "controller_box",
    "pedestal",
    "torso",
    "right_",
    "left_",
    "hand",
    "head",
    "screen",
    "mocap",
)


def _is_sawyer_body(name: str) -> bool:
    if name in ("world", "tablelink", "RetainingWall"):
        return False
    return any(name == p or name.startswith(p) for p in _SAWYER_BODY_PREFIXES)


def color_metaworld_arm_red(env, rgba=(0.85, 0.05, 0.05, 1.0)) -> int:
    """Set visible Sawyer robot geoms to red. Returns number of geoms recolored."""
    base = _metaworld_base_env(env)
    model = base.sim.model
    color = np.asarray(rgba, dtype=np.float32)
    n = 0
    for i in range(model.ngeom):
        body = model.body_names[int(model.geom_bodyid[i])]
        if not _is_sawyer_body(body):
            continue
        alpha = float(model.geom_rgba[i][3])
        if alpha < 1e-6:
            continue  # keep invisible collision/helper geoms hidden
        model.geom_rgba[i][:3] = color[:3]
        model.geom_rgba[i][3] = alpha
        n += 1
    return n


def render_frame(
    env,
    is_metaworld: bool,
    height: int,
    width: int,
    camera_id: int,
    mw_camera: str = "corner",
):
    if is_metaworld:
        # Headless MetaWorld: gym render often breaks under GLEW; use mujoco-py offscreen.
        # sim.render already returns RGB (do NOT BGR-swap — that turns the red arm blue).
        base = _metaworld_base_env(env)
        frame = base.sim.render(
            width, height, mode="offscreen", camera_name=mw_camera
        )
        return np.asarray(frame)

    frame = env.render(
        mode="rgb_array",
        height=height,
        width=width,
        camera_id=camera_id,
    )
    return np.asarray(frame)


def record_one(
    ckpt_dir: str,
    step: int,
    *,
    device: str,
    episodes: int,
    height: int,
    width: int,
    fps: int,
    camera_id: int,
    mw_camera: str,
    actor_only: bool,
    seed_override: Optional[int],
    out_path: Optional[str],
) -> str:
    if out_path is None:
        cam_tag = f"_{mw_camera}" if mw_camera != "corner" else ""
        out_path = os.path.join(ckpt_dir, f"video_actor_{step}{cam_tag}.mp4")
    cfg = load_cfg(ckpt_dir, device, seed_override)
    is_metaworld = "metaworld" in str(cfg.env)

    if is_metaworld:
        # Dockerfile sets LD_PRELOAD for GLEW/GL; that breaks nvidia EGL offscreen.
        os.environ.pop("LD_PRELOAD", None)
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        env = utils.make_metaworld_env(cfg)
        n_red = color_metaworld_arm_red(env)
        print(f"colored {n_red} Sawyer geoms red; camera={mw_camera}")
    else:
        env = utils.make_env(cfg)

    agent = build_agent(cfg, env)
    load_weights(agent, ckpt_dir, step, device, actor_only)

    frames = []
    returns = []
    successes = []
    print(f"\n=== env={cfg.env}  step={step}  device={device}  ckpt={ckpt_dir} ===")

    for ep in range(episodes):
        obs = env.reset()
        agent.reset()
        done = False
        ep_ret = 0.0
        ep_success = 0.0
        while not done:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            frames.append(
                render_frame(
                    env, is_metaworld, height, width, camera_id, mw_camera=mw_camera
                )
            )
            obs, reward, done, extra = env.step(action)
            ep_ret += float(reward)
            if is_metaworld and isinstance(extra, dict) and "success" in extra:
                ep_success = max(ep_success, float(extra["success"]))
        returns.append(ep_ret)
        if is_metaworld:
            successes.append(ep_success)
            print(f"episode {ep}: return = {ep_ret:.1f}  success = {ep_success:.0f}")
        else:
            print(f"episode {ep}: true return = {ep_ret:.1f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    mean_ret = float(np.mean(returns)) if returns else 0.0
    msg = (
        f"saved {out_path} "
        f"({len(frames)} frames, {episodes} ep, mean return {mean_ret:.1f}"
    )
    if successes:
        msg += f", success rate {100.0 * float(np.mean(successes)):.0f}%"
    msg += ")"
    print(msg)
    return out_path


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    jobs = resolve_jobs(args)

    saved = []
    for ckpt_dir, step, out in jobs:
        saved.append(
            record_one(
                ckpt_dir,
                step,
                device=device,
                episodes=args.episodes,
                height=args.height,
                width=args.width,
                fps=args.fps,
                camera_id=args.camera_id,
                mw_camera=args.mw_camera,
                actor_only=args.actor_only,
                seed_override=args.seed,
                out_path=out,
            )
        )

    if len(saved) > 1:
        print("\nAll videos:")
        for path in saved:
            print(f"  {path}")


if __name__ == "__main__":
    main()
