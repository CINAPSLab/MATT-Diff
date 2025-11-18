import numpy as np
import torch
import json
from pathlib import Path

from evals.eval_core import evaluate_episode, pack_from_eval_states
from envs.env_gym import SimpleGymEnv
from src.dp_agent import DPAgent
from src.models.dp_policy import DiffusionPolicyNetwork
from utilities.utils import load_houseexpo_image_as_grid

from typing import Any, Dict, List

CKPT_PATH = Path("src/run/dp/best.pt")
OUT_PATH = Path("results/dp/dp.json")
SAVE_TIMESERIES_DIR = Path("results/dp/")

SEED = 29
EPISODES = 5
MAX_STEPS = 1000

MAP_FILENAME = "map/0a1a5807d65749c1194ce1840354be39.png"
USE_ENV_KF = True

INF_STEPS = 20
ACTION_HORIZON = 4

TURN_THR = 0.08
W_GAIN = 1.5
V_GAIN = 0.8
CURV_CAP = 0.6
MIN_V_TURN = 0.30
W_DEADBAND = 0.02
W_SMOOTH_ALPHA = 0.6


def run_dp_episode(env: SimpleGymEnv, agent: DPAgent, max_steps: int, 
                   seed: int,) -> Dict[str, np.ndarray]:

    obs, _ = env.reset(seed=seed)
    agent.reset()
    eval_states_seq: List[List[Dict[str, Any]]] = []

    lo = env.action_space.low.astype(np.float32)
    hi = env.action_space.high.astype(np.float32)

    prev_w = 0.0

    for _ in range(int(max_steps)):
        a = agent.get_action(obs)
        v_raw, w_raw = float(a[0]), float(a[1])

        v = v_raw * V_GAIN
        w = w_raw * W_GAIN

        alpha = W_SMOOTH_ALPHA
        w = alpha * prev_w + (1.0 - alpha) * w

        if abs(w) < W_DEADBAND:
            w = 0.0
            prev_w = 0.0
        else:
            prev_w = w

        if abs(w) > TURN_THR and abs(v) < MIN_V_TURN:
            v = (1.0 if v == 0.0 else np.sign(v)) * MIN_V_TURN

        if CURV_CAP > 0.0:
            v_eff = max(abs(v), 1e-3)
            if abs(w) > CURV_CAP * v_eff:
                w = np.sign(w) * CURV_CAP * v_eff

        a_env = np.array([v, w], dtype=np.float32)
        action = np.clip(a_env, lo, hi)

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


def dump_timeseries_npz(save_dir: Path, ep_idx: int, data: Dict[str, np.ndarray]):
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"dp_ep{ep_idx:03d}.npz"
    payload = {
        "mu": data["mu"],
        "Sigma": data["Sigma"],
        "x_true": data["x_true"],
        "exist_mask": data["exist_mask"],
        "did_update": data["did_update"],
    }
    np.savez_compressed(out_path, **payload)


def main():
    device = torch.device("cuda")

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
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
    policy.s_lost_norm = 1.01

    grid = load_houseexpo_image_as_grid(MAP_FILENAME)
    env = SimpleGymEnv(grid=grid, fov_rad=1.0, render_mode=None, use_env_kf=USE_ENV_KF)

    agent = DPAgent(
        policy_model=policy,
        robot_stats=None,
        action_stats=None,
        action_space_low=env.action_space.low,
        action_space_high=env.action_space.high,
        obs_horizon=cfg["obs_h"],
        pred_horizon=cfg["pred_h"],
        action_horizon=ACTION_HORIZON,
        num_inference_steps=INF_STEPS,
    )

    metrics: List[Dict[str, float]] = []

    for ep in range(int(EPISODES)):
        seed_ep = int(SEED) + ep
        data = run_dp_episode(env, agent, MAX_STEPS, seed=seed_ep)

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
            "ckpt_path": str(CKPT_PATH),
            "seed": int(SEED),
            "save_timeseries_dir": str(SAVE_TIMESERIES_DIR),
        },
        "metrics": metrics,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out_obj, f, indent=2)


if __name__ == "__main__":
    main()