from __future__ import annotations

import numpy as np
import torch
import json
from pathlib import Path

from evals.eval_core import evaluate_episode, pack_from_eval_states
from envs.env_gym import SimpleGymEnv
from src.models.bc_policy import BCPolicy
from utilities.utils import load_houseexpo_image_as_grid

from typing import Any, Dict, List


CKPT_PATH = Path("src/run/bc/best.pt")
OUT_PATH = Path("results/bc/bc.json")
SAVE_TIMESERIES_DIR = Path("results/bc/bc")

SEED = 29
EPISODES = 5
MAX_STEPS = 1000

MAP_FILENAME = "map/0a1a5807d65749c1194ce1840354be39.png"
USE_ENV_KF = True



def run_bc_episode(env: SimpleGymEnv, policy: BCPolicy, max_steps: int
                   ,seed: int,) -> Dict[str, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    eval_states_seq: List[List[Dict[str, Any]]] = []

    for _ in range(int(max_steps)):
        action = policy.predict(obs)  # shape (2,)
        obs, reward, terminated, truncated, info = env.step(action)
        eval_states_seq.append(info["eval_states"])

        if terminated or truncated:
            break

    mu, S, x_true, did, exists = pack_from_eval_states(eval_states_seq)

    return {
        "mu": mu,
        "Sigma": S,
        "x_true": x_true,
        "did_update": did,
        "exist_mask": exists,
    }


def dump_timeseries_npz(save_dir: Path, ep_idx: int, data: Dict[str, np.ndarray]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"bc_ep{ep_idx:03d}.npz"
    payload = {
        "mu": data["mu"],
        "Sigma": data["Sigma"],
        "x_true": data["x_true"],
        "exist_mask": data["exist_mask"],
        "did_update": data["did_update"],
    }
    np.savez_compressed(out_path, **payload)



def main() -> None:
    device = torch.device("cuda")

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    policy = BCPolicy(str(CKPT_PATH), device=device, use_robot=False)

    grid = load_houseexpo_image_as_grid(MAP_FILENAME)
    env = SimpleGymEnv(grid=grid, fov_rad=1.0, render_mode=None, use_env_kf=USE_ENV_KF)
    policy.set_action_bounds(env.action_space.low, env.action_space.high)

    metrics: List[Dict[str, float]] = []

    for ep in range(int(EPISODES)):
        seed_ep = int(SEED) + ep
        data = run_bc_episode(env, policy, MAX_STEPS, seed=seed_ep)

        sigma_cap_eval = float(getattr(env, "_sigma_cap_px", 250.0))
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

        dump_timeseries_npz(SAVE_TIMESERIES_DIR, ep, data)

    out_obj = {
        "args": {
            "ckpt": str(CKPT_PATH),
            "seed": int(SEED),
            "save_timeseries_dir": str(SAVE_TIMESERIES_DIR),
        },
        "metrics": metrics,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out_obj, f, indent=2)


if __name__ == "__main__":
    main()