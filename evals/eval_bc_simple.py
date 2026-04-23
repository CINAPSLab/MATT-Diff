"""
Evaluate BC policy with simple point-mass dynamics.

Differences from eval_bc.py:
  - env.simple_dynamics = True  (dx/dy actions instead of v/w)
  - action bounds derived from --step_size
  - robot trajectory saved to npz
  - multi-seed loop matching eval_dp_simple.py convention

Usage:
  python -u -m evals.eval_bc_simple
"""

from __future__ import annotations

import numpy as np
import torch
import argparse
from pathlib import Path
from typing import Any, Dict, List

from evals.eval_core import evaluate_episode, pack_from_eval_states
from envs.env_gym import SimpleGymEnv
from src.models.bc_policy import BCPolicy
from utilities.utils import load_houseexpo_image_as_grid

MAX_STEPS  = 1000
USE_ENV_KF = True


def run_bc_episode(env: SimpleGymEnv, policy: BCPolicy,
                   max_steps: int, seed: int) -> Dict[str, np.ndarray]:
    obs, info0 = env.reset(seed=seed)
    # Capture initial state at spawn (before first action) as timestep 0
    eval_states_seq: List[List[Dict[str, Any]]] = [info0["eval_states"]]

    for _ in range(int(max_steps)):
        action = policy.predict(obs)           # (2,) — dx, dy
        obs, _, terminated, truncated, info = env.step(action)
        eval_states_seq.append(info["eval_states"])
        if terminated or truncated:
            break

    mu, S, x_true, did, exists = pack_from_eval_states(eval_states_seq)

    robot_traj = None
    try:
        traj = env.get_trajectory()
        if isinstance(traj, np.ndarray) and traj.ndim == 2 and traj.shape[1] >= 3:
            robot_traj = traj[:, :3].astype(np.float32)
    except Exception:
        pass

    return {
        "mu": mu, "Sigma": S, "x_true": x_true,
        "did_update": did, "exist_mask": exists, "robot": robot_traj,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BC policy with simple point-mass dynamics")
    parser.add_argument("--ckpt",      type=str, default="output/model/bc/best.pt")
    parser.add_argument("--ts_dir",    type=str, default="output/results/bc_simple/timeseries")
    parser.add_argument("--map_path",  type=str, default="map/evaluation_map.png")
    parser.add_argument("--seed",      type=int, default=57)
    parser.add_argument("--episodes",  type=int, default=20)
    parser.add_argument("--step_size", type=float, default=10.0,
                        help="Action bound ±step_size for (dx, dy)")
    parser.add_argument("--momentum",     type=float, default=0.5)
    parser.add_argument("--theta_alpha",  type=float, default=0.3)
    parser.add_argument("--collision_freeze", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze robot on wall collision (default: True)")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_dir  = Path(args.ts_dir)
    ts_dir.mkdir(parents=True, exist_ok=True)

    # load BC policy
    policy = BCPolicy(args.ckpt, device=device, use_robot=False)

    # build env with simple dynamics
    grid = load_houseexpo_image_as_grid(args.map_path)
    env  = SimpleGymEnv(grid=grid, fov_rad=1.0, render_mode=None, use_env_kf=USE_ENV_KF)
    env.simple_dynamics   = True
    env.simple_momentum   = args.momentum
    env.simple_theta_alpha = args.theta_alpha
    env.collision_freeze  = args.collision_freeze

    # set action bounds to step_size range
    low  = np.array([-args.step_size, -args.step_size], dtype=np.float32)
    high = np.array([ args.step_size,  args.step_size], dtype=np.float32)
    policy.set_action_bounds(low, high)

    ep_metrics: List[Dict[str, float]] = []

    for ep in range(args.episodes):
        seed_ep = int(args.seed) + ep
        print(f"[bc-simple] ep={ep}/{args.episodes} seed={seed_ep}")

        data = run_bc_episode(env, policy, MAX_STEPS, seed=seed_ep)

        sigma_cap = float(getattr(env, "_sigma_cap_px", 250.0))
        m = evaluate_episode(
            mu=data["mu"], Sigma=data["Sigma"],
            x_true=data["x_true"], did_update=data["did_update"],
            exist_mask=data["exist_mask"],
            sigma_cap=sigma_cap, rmse_all_censored=False,
        )
        ep_metrics.append({
            "rmse_exist": float(m["rmse_exist"]),
            "nll":        float(m["nll"]),
            "entropy":    float(m["entropy"]),
            "episode": ep, "seed": seed_ep,
        })
        print(f"  rmse={m['rmse_exist']:.3f}  nll={m['nll']:.2f}  H={m['entropy']:.2f}")

        # save timeseries npz (including robot for trajectory plots)
        payload = {
            "mu": data["mu"], "Sigma": data["Sigma"],
            "x_true": data["x_true"], "did_update": data["did_update"],
            "exist_mask": data["exist_mask"],
        }
        if data.get("robot") is not None:
            payload["robot"] = data["robot"]
        np.savez_compressed(ts_dir / f"bc_ep{ep:03d}.npz", **payload)

    keys     = ["rmse_exist", "nll", "entropy"]
    task_avg = {k: float(np.mean([e[k] for e in ep_metrics])) for k in keys}
    task_std = {f"{k}_std": float(np.std([e[k] for e in ep_metrics])) for k in keys}

    print(f"\n[bc-simple] map={args.map_path}  seed={args.seed}  episodes={args.episodes}")
    print(f"  avg rmse={task_avg['rmse_exist']:.4f}  "
          f"nll={task_avg['nll']:.4f}  H={task_avg['entropy']:.4f}")


if __name__ == "__main__":
    main()
