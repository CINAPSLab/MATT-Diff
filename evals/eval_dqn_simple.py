"""
python -m evals.eval_dqn_simple \
  --ckpt "output/model/dqn_simple_100k.zip" \
  --out "output/results/dqn_simple/map0/dqn_simple_map0_seed27.json" \
  --ts_dir "output/results/dqn_simple/map0/timeseries/map0_seed27" \
  --map_path "map/0.png" \
  --seed 27 \
  --episodes 10
"""

import numpy as np
import torch
import torch.nn as nn
import json
import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from evals.eval_core import evaluate_episode, pack_from_eval_states
from envs.env_gym import SimpleGymEnv
from utilities.utils import load_houseexpo_image_as_grid

from typing import Any, Dict, List

MAX_STEPS = 1000
USE_ENV_KF = True


# ==========================================
# 1. Action Wrapper — simple dynamics (dx, dy)
# ==========================================
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.vx_options = [-10.0, -5.0, 5.0, 10.0]
        self.vy_options = [-10.0, -5.0, 5.0, 10.0]

        self.num_actions = len(self.vx_options) * len(self.vy_options)  # 16
        self.action_space = spaces.Discrete(self.num_actions)

        self.action_mapping = []
        for vx in self.vx_options:
            for vy in self.vy_options:
                self.action_mapping.append(np.array([vx, vy], dtype=np.float32))

    def action(self, act):
        return self.action_mapping[act]


# ==========================================
# 2. ATTN Feature Extractor (must match training)
# ==========================================
class ATTNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        ego_map_shape = observation_space.spaces["ego_map"].shape
        n_input_channels = ego_map_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 20, kernel_size=4, stride=3),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_map = torch.as_tensor(observation_space.spaces["ego_map"].sample()[None]).float()
            cnn_out_dim = self.cnn(dummy_map).shape[1]

        robot_dim = observation_space.spaces["robot"].shape[0]
        slots_dim = np.prod(observation_space.spaces["slots"].shape)
        vector_dim = robot_dim + slots_dim

        total_concat_dim = cnn_out_dim + vector_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_concat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        ego_map = observations["ego_map"].float() / 255.0
        cnn_features = self.cnn(ego_map)
        robot = observations["robot"].float()
        slots = observations["slots"].float().view(robot.shape[0], -1)

        concat_features = torch.cat([cnn_features, robot, slots], dim=1)
        return self.mlp(concat_features)


# ==========================================
# 3. Evaluation Core
# ==========================================
def run_dqn_episode(env: gym.Env, model: DQN, max_steps: int,
                    seed: int) -> Dict[str, np.ndarray]:

    obs, info0 = env.reset(seed=seed)
    # Capture initial state at spawn (before first action) as timestep 0
    eval_states_seq: List[List[Dict[str, Any]]] = [info0["eval_states"]]

    for _ in range(int(max_steps)):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        eval_states_seq.append(info["eval_states"])

        if terminated or truncated:
            break

    mu, S, x_true, did, exists = pack_from_eval_states(eval_states_seq)

    robot_traj = None
    try:
        traj = env.unwrapped.get_trajectory()
        if isinstance(traj, np.ndarray) and traj.ndim == 2:
            if traj.shape[1] >= 3:
                robot_traj = traj[:, :3].astype(np.float32)
            elif traj.shape[1] == 2:
                pad = np.zeros((traj.shape[0], 1), dtype=np.float32)
                robot_traj = np.concatenate([traj[:, :2], pad], axis=1)
    except Exception:
        pass

    return {
        "mu": mu,
        "Sigma": S,
        "x_true": x_true,
        "did_update": did,
        "exist_mask": exists,
        "robot": robot_traj,
    }


def dump_timeseries_npz(save_dir: Path, ep_idx: int, data: Dict[str, np.ndarray]):
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"dqn_ep{ep_idx:03d}.npz"
    payload = {
        "mu": data["mu"],
        "Sigma": data["Sigma"],
        "x_true": data["x_true"],
        "exist_mask": data["exist_mask"],
        "did_update": data["did_update"],
    }

    if data.get("robot") is not None:
        payload["robot"] = data["robot"]

    np.savez_compressed(out_path, **payload)


def main(args: argparse.Namespace):
    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    timeseries_dir = Path(args.ts_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    timeseries_dir.mkdir(parents=True, exist_ok=True)

    grid = load_houseexpo_image_as_grid(args.map_path)
    raw_env = SimpleGymEnv(grid=grid, fov_rad=1.0, render_mode=None, use_env_kf=USE_ENV_KF)
    raw_env.simple_dynamics = True
    raw_env.collision_freeze = args.collision_freeze
    env = DiscreteActionWrapper(raw_env)

    print(f"Loading DQN model from {ckpt_path}...")
    model = DQN.load(ckpt_path)

    metrics: List[Dict[str, float]] = []

    print(f"Starting evaluation for {args.episodes} episodes...")
    for ep in range(int(args.episodes)):
        seed_ep = int(args.seed) + ep
        data = run_dqn_episode(env, model, MAX_STEPS, seed=seed_ep)

        sigma_cap_eval = float(getattr(raw_env, "_sigma_cap_px", 250.0))
        m = evaluate_episode(
            mu=data["mu"],
            Sigma=data["Sigma"],
            x_true=data["x_true"],
            did_update=data["did_update"],
            exist_mask=data["exist_mask"],
            sigma_cap=sigma_cap_eval,
            rmse_all_censored=False,
        )

        metrics.append(
            {
                "rmse_exist": float(m["rmse_exist"]),
                "nll": float(m["nll"]),
                "entropy": float(m["entropy"]),
                "episode": ep,
            }
        )

        print(f"Episode {ep} | RMSE: {m['rmse_exist']:.2f} | NLL: {m['nll']:.2f}")

        dump_timeseries_npz(timeseries_dir, ep, data)

    out_obj = {
        "args": vars(args),
        "metrics": metrics,
    }

    with open(out_path, "w") as f:
        json.dump(out_obj, f, indent=2)

    print(f"Evaluation complete. Metrics saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN Simple Baseline Model")

    parser.add_argument("--ckpt", type=str, default="output/model/dqn_simple_100k.zip")
    parser.add_argument("--out", type=str, default="output/results/dqn_simple/dqn_simple_eval.json")
    parser.add_argument("--ts_dir", type=str, default="output/results/dqn_simple/timeseries")
    parser.add_argument("--map_path", type=str, default="map/0.png")
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--collision_freeze", action="store_true",
                        help="Freeze robot on first wall collision")

    args = parser.parse_args()
    main(args)
