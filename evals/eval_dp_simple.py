"""
Evaluate DP (MATT-Diff) with simple point-mass dynamics.

Usage:
  python -u -m evals.eval_dp_simple
"""

import numpy as np
import torch
import argparse
from pathlib import Path

from evals.eval_core import evaluate_episode, pack_from_eval_states
from envs.env_gym import SimpleGymEnv
from src.dp_agent import DPAgent
from src.models.dp_policy import DiffusionPolicyNetwork
from utilities.utils import load_houseexpo_image_as_grid, compute_s_lost_norm

from typing import Any, Dict, List

MAX_STEPS = 1000
USE_ENV_KF = True


def run_dp_episode(env: SimpleGymEnv, agent: DPAgent, max_steps: int,
                   seed: int) -> Dict[str, np.ndarray]:
    obs, info0 = env.reset(seed=seed)
    agent.reset()
    # Capture initial state at spawn (before first action) as timestep 0
    eval_states_seq: List[List[Dict[str, Any]]] = [info0["eval_states"]]

    for _ in range(int(max_steps)):
        a = agent.get_action(obs)
        # simple dynamics: action is (dx, dy) directly, no gain/smoothing needed
        action = np.asarray(a, dtype=np.float32).ravel()[:2]

        obs, reward, terminated, truncated, info = env.step(action)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="output/model/dp/best.pt")
    parser.add_argument("--ts_dir", type=str, default="output/results/dp_simple/timeseries")
    parser.add_argument("--map_path", type=str, default="map/evaluation_map.png")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--inf_steps", type=int, default=20)
    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--theta_alpha", type=float, default=0.3)
    parser.add_argument("--step_size", type=float, default=10.0)
    parser.add_argument("--collision_freeze", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze robot on wall collision (default: True)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    policy = DiffusionPolicyNetwork(
        action_dim=2,
        obs_horizon=cfg["obs_h"],
        pred_horizon=cfg["pred_h"],
        map_emb_dim=cfg["map_emb_dim"],
        unet_down_dims=(256, 512, 1024),
        unet_kernel_size=5,
        num_diffusion_iters=cfg["num_diffusion_iters"],
        use_age=False,
    ).to(device)
    policy.load_state_dict(ckpt["policy"])

    grid = load_houseexpo_image_as_grid(args.map_path)
    env = SimpleGymEnv(grid=grid, fov_rad=1.0, render_mode=None, use_env_kf=USE_ENV_KF)
    env.simple_dynamics = True
    env.simple_momentum = args.momentum
    env.simple_theta_alpha = args.theta_alpha
    env.collision_freeze = args.collision_freeze

    s_lost = compute_s_lost_norm(env.width, env.height)

    # action stats for denormalization: (dx, dy) in [-step_size, step_size]
    action_stats = {
        "low": np.array([-args.step_size, -args.step_size], dtype=np.float32),
        "high": np.array([args.step_size, args.step_size], dtype=np.float32),
    }

    agent = DPAgent(
        policy_model=policy,
        action_stats=action_stats,
        action_space_low=action_stats["low"],
        action_space_high=action_stats["high"],
        obs_horizon=cfg["obs_h"],
        pred_horizon=cfg["pred_h"],
        action_horizon=args.action_horizon,
        num_inference_steps=args.inf_steps,
        map_w=env.width,
        map_h=env.height,
        s_lost_norm=s_lost,
    )

    ts_dir = Path(args.ts_dir)
    ts_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[Dict[str, float]] = []

    for ep in range(args.episodes):
        seed_ep = int(args.seed) + ep
        print(f"[DP-simple] ep={ep}/{args.episodes} seed={seed_ep}")

        data = run_dp_episode(env, agent, MAX_STEPS, seed=seed_ep)
        sigma_cap_eval = float(getattr(env, "_sigma_cap_px", 250.0))
        m = evaluate_episode(
            mu=data["mu"], Sigma=data["Sigma"],
            x_true=data["x_true"], did_update=data["did_update"],
            exist_mask=data["exist_mask"],
            sigma_cap=sigma_cap_eval, rmse_all_censored=False,
        )

        metrics.append({
            "rmse_exist": float(m["rmse_exist"]),
            "nll": float(m["nll"]),
            "entropy": float(m["entropy"]),
            "episode": ep,
        })
        print(f"  rmse={m['rmse_exist']:.3f}  nll={m['nll']:.2f}  H={m['entropy']:.2f}")

        payload = {
            "mu": data["mu"], "Sigma": data["Sigma"],
            "x_true": data["x_true"], "did_update": data["did_update"],
            "exist_mask": data["exist_mask"],
        }
        if data.get("robot") is not None:
            payload["robot"] = data["robot"]
        np.savez_compressed(ts_dir / f"dp_ep{ep:03d}.npz", **payload)

    keys = ["rmse_exist", "nll", "entropy"]
    task_avg = {k: float(np.mean([e[k] for e in metrics])) for k in keys}
    task_std = {f"{k}_std": float(np.std([e[k] for e in metrics])) for k in keys}

    print(f"\n[dp-simple] map={args.map_path}  seed={args.seed}  episodes={args.episodes}")
    print(f"  avg rmse={task_avg['rmse_exist']:.4f}  "
          f"nll={task_avg['nll']:.4f}  H={task_avg['entropy']:.4f}")


if __name__ == "__main__":
    main()
