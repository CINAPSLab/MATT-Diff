from __future__ import annotations

import numpy as np
import json
import os

from typing import Any, Dict, List, Optional

from .eval_core import evaluate_episode, pack_from_eval_states
from envs.env_gym import SimpleGymEnv
from pursuer.mm_planner import Coordinator as MMPlanner
from utilities.utils import load_houseexpo_image_as_grid


SEED = 29
EPISODES = 5
MAX_STEPS = 1000

MAP_FILENAME = "map/0a1a5807d65749c1194ce1840354be39.png"
USE_ENV_KF = True

SIGMA_CAP = 250.0

BASE_DIR = os.path.join("results", "reacq")
OUT_JSON = os.path.join(BASE_DIR, "reacq_results.json")
SAVE_TIMESERIES_DIR: Optional[str] = BASE_DIR


REACQ_STRONG = {
    "FORCE_EXPLORE_ONLY": False,
    "ENABLE_REACQUIRE": True,
    "SWITCH_ON_NEW": True,
    "plan_safety_margin_px": 8,
    "sigma_reacq_lo": 150.0,
    "sigma_abort": 350.0,
    "stale_S": 60,
    "T_track": 70,
    "cooldown": 60,
    "goal_hold_steps": 20,
}

GOAL_PARAMS_BASE = {
    "use_simple_nearest": True,
    "frontier_gain_weight": 1.2,
    "frontier_area_weight": 1.2,
    "frontier_dist_weight": 10.0,
    "frontier_known_weight": 0.4,
    "frontier_min_area": 300,
    "frontier_excl_radius": 80,
    "frontier_gain_radius": 60,
    "softmax_temp": 0.5,
}


def _ensure_action(action: Optional[np.ndarray]) -> np.ndarray:
    if action is None:
        return np.zeros(2, dtype=np.float32)
    a = np.asarray(action, dtype=np.float32).ravel()
    if a.size < 2:
        out = np.zeros(2, dtype=np.float32)
        out[:a.size] = a
        return out
    return a[:2]


def run_episode(env: SimpleGymEnv, planner: MMPlanner,
                max_steps: int) -> Dict[str, np.ndarray]:
    eval_states_seq: List[List[Dict[str, Any]]] = []

    if hasattr(planner, "reset"):
        try:
            planner.reset(env._get_obs())
        except Exception:
            pass

    for _ in range(int(max_steps)):
        try:
            result = planner.step(env._rbt, None, env.get_visible_region(), tmask=env._tmask)
            action = _ensure_action(result.get("action") if isinstance(result, dict) else None)
        except Exception:
            action = np.zeros(2, dtype=np.float32)

        _, _, terminated, truncated, info = env.step(action)
        states = info.get("eval_states", None)
        if states is None:
            raise RuntimeError("env did not provide info['eval_states'].")

        eval_states_seq.append(states)

        if terminated or truncated:
            break

    mu, S, x_true, did, exists = pack_from_eval_states(eval_states_seq)

    robot_obs = None
    try:
        traj = env.get_trajectory()
        if traj is not None:
            robot_obs = traj
    except Exception:
        pass

    return {
        "mu": mu,
        "Sigma": S,
        "x_true": x_true,
        "did_update": did,
        "exist_mask": exists,
        "robot": robot_obs,
    }


def main() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)

    grid = load_houseexpo_image_as_grid(MAP_FILENAME)
    env = SimpleGymEnv(grid=grid, fov_rad=1.0, render_mode=None, use_env_kf=USE_ENV_KF)

    if SAVE_TIMESERIES_DIR is not None:
        os.makedirs(SAVE_TIMESERIES_DIR, exist_ok=True)

    metrics_for_json: List[Dict[str, float]] = []

    for ep in range(int(EPISODES)):
        seed = None if SEED is None else int(SEED + ep)
        env.reset(seed=seed)

        goal_params = dict(GOAL_PARAMS_BASE)
        if getattr(env, "rng_spawn", None):
            goal_params["random_seed"] = int(env.rng_spawn.integers(1, 1_000_000))

        planner = MMPlanner(
            grid=env._global_map,
            kf_targets=env._kf_targets,
            plan_grid=env._planning_grid,
            params=dict(REACQ_STRONG),
            goal_params=goal_params,
        )

        print(f"[Reacq] Episode {ep}/{EPISODES} (seed={seed})")

        mdata = run_episode(env, planner, MAX_STEPS)
        sigma_cap_eval = float(getattr(env, "_sigma_cap_px", SIGMA_CAP))

        m = evaluate_episode(
            mu=mdata["mu"],
            Sigma=mdata["Sigma"],
            x_true=mdata["x_true"],
            did_update=mdata["did_update"],
            exist_mask=mdata["exist_mask"],
            sigma_cap=sigma_cap_eval,
            rmse_all_censored=False,
        )
        m["episode"] = int(ep)

        metrics_for_json.append({
            "rmse_exist": float(m["rmse_exist"]),
            "nll": float(m["nll"]),
            "entropy": float(m["entropy"]),
            "episode": int(m["episode"]),
        })

        print(
            f"[Reacq] ep={ep} "
            f"rmse_exist={m['rmse_exist']:.3f} "
            f"nll={m['nll']:.2f} H={m['entropy']:.2f}"
        )

        if SAVE_TIMESERIES_DIR is not None:
            dump = {
                "mu": mdata["mu"],
                "Sigma": mdata["Sigma"],
                "x_true": mdata["x_true"],
                "did_update": mdata["did_update"],
                "exist_mask": mdata["exist_mask"],
            }
            if mdata.get("robot", None) is not None:
                dump["robot"] = mdata["robot"]

            out_npz = os.path.join(SAVE_TIMESERIES_DIR, f"reacq_ep{ep:03d}.npz")
            np.savez_compressed(out_npz, **dump)
            print(f"[Reacq] saved timeseries -> {out_npz}")

    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "args": {
                    "SEED": SEED,
                    "EPISODES": EPISODES,
                },
                "metrics": metrics_for_json,
            },
            f,
            indent=2,
        )
    print(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()