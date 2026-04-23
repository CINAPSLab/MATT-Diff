"""
Test expert planners with simplified point-mass dynamics.

Purpose: isolate whether wall collisions come from RRT* planning
or from unicycle + Pure Pursuit tracking.

Dynamics:
    x_{k+1} = x_k + a1_k
    y_{k+1} = y_k + a2_k
    theta_{k+1} = arctan2(a2_k, a1_k)   (unchanged if stationary)

Usage:
    python -m evals.eval_simple_dynamics \
        --map_path map/5.png --seed 47 --episodes 10 \
        --out_dir output/simple_dynamics_test
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
from typing import Any, Dict, List, Optional

from .eval_core import evaluate_episode, pack_from_eval_states
from envs.env_gym import SimpleGymEnv
from pursuer.mm_planner import Coordinator as MMPlanner
from pursuer.explore_only import Coordinator as FrontierPlanner
from pursuer.controller.simple_follow import SimpleFollowController
from utilities.utils import load_houseexpo_image_as_grid

# ── CLI ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--map_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=47)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--step_size", type=float, default=10.0,
                    help="Max displacement per step (px)")
parser.add_argument("--out_dir", type=str,
                    default="output/simple_dynamics_test")
parser.add_argument("--method", type=str, default=None,
                    choices=["track", "reacq", "frontier"],
                    help="Run only this planner (default: all three)")
parser.add_argument("--collision_freeze", action="store_true",
                    help="Freeze robot on first wall collision (no more movement)")
args = parser.parse_args()

SIGMA_CAP = 250.0

# ── Planner configs (same as originals) ──────────────────────────
TRACK_PARAMS = {
    "FORCE_EXPLORE_ONLY": False,
    "ENABLE_REACQUIRE": True,
    "SWITCH_ON_NEW": True,
    "plan_safety_margin_px": 8,
    "sigma_reacq_lo": 350.0,
    "sigma_abort": 350.0,
    "stale_S": 200,
    "T_track": 150,
    "cooldown": 200,
    "goal_hold_steps": 20,
}

REACQ_PARAMS = {
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

GOAL_COMMON = {
    "frontier_gain_weight": 1.0,
    "frontier_area_weight": 1.0,
    "frontier_dist_weight": 8.0,
    "frontier_known_weight": 0.5,
    "frontier_min_area": 8,
    "frontier_excl_radius": 10,
    "frontier_gain_radius": 60,
    "use_simple_nearest": True,
    "softmax_temp": 3.4,
}

REACQ_GOAL = {
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


# ── Helpers ───────────────────────────────────────────────────────
def _ensure_action(action) -> np.ndarray:
    if action is None:
        return np.zeros(2, dtype=np.float32)
    a = np.asarray(action, dtype=np.float32).ravel()
    if a.size < 2:
        out = np.zeros(2, dtype=np.float32)
        out[: a.size] = a
        return out
    return a[:2]


def run_episode(env: SimpleGymEnv, planner, max_steps: int) -> Dict[str, np.ndarray]:
    # Capture initial state at spawn (before first action) as timestep 0
    eval_states_seq: List[List[Dict[str, Any]]] = [env._get_info()["eval_states"]]

    if hasattr(planner, "reset"):
        try:
            planner.reset(env._get_obs())
        except Exception:
            pass

    for _ in range(max_steps):
        try:
            result = planner.step(
                env._rbt, None, env.get_visible_region(), tmask=env._tmask
            )
            action = _ensure_action(
                result.get("action") if isinstance(result, dict) else None
            )
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
        "mu": mu, "Sigma": S, "x_true": x_true,
        "did_update": did, "exist_mask": exists, "robot": robot_obs,
    }


def make_planner(tag: str, env: SimpleGymEnv, step_size: float):
    """Create planner with SimpleFollowController swapped in."""
    if tag == "frontier":
        coord_params = {
            "FORCE_EXPLORE_ONLY": True,
            "ENABLE_REACQUIRE": False,
            "SWITCH_ON_NEW": False,
            "GOAL_HOLD_STEPS": 10,
            "plan_safety_margin_px": 8,
        }
        goal_params = {
            "frontier_gain_weight": 1.8,
            "frontier_area_weight": 0.8,
            "frontier_dist_weight": 10.0,
            "frontier_known_weight": 0.25,
            "frontier_min_area": 300,
            "frontier_excl_radius": 80,
            "frontier_gain_radius": int(0.7 * env._radius),
            "use_simple_nearest": True,
            "softmax_temp": 0.55,
        }
        if getattr(env, "rng_spawn", None):
            goal_params["random_seed"] = int(env.rng_spawn.integers(1, 1_000_000))
        planner = FrontierPlanner(
            grid=env._global_map,
            kf_targets=env._kf_targets,
            plan_grid=env._planning_grid,
            params=coord_params,
            goal_params=goal_params,
        )
    else:
        params = dict(TRACK_PARAMS if tag == "track" else REACQ_PARAMS)
        goal_params = dict(GOAL_COMMON if tag == "track" else REACQ_GOAL)
        if tag == "reacq" and getattr(env, "rng_spawn", None):
            goal_params["random_seed"] = int(env.rng_spawn.integers(1, 1_000_000))
        planner = MMPlanner(
            grid=env._global_map,
            kf_targets=env._kf_targets,
            plan_grid=env._planning_grid,
            params=params,
            goal_params=goal_params,
        )

    # Swap in simple controller
    planner.controller = SimpleFollowController(step_size=step_size)
    return planner


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    grid = load_houseexpo_image_as_grid(args.map_path)

    methods = (args.method,) if args.method else ("track", "reacq", "frontier")
    for tag in methods:
        ts_dir = os.path.join(args.out_dir, f"timeseries_{tag}")
        os.makedirs(ts_dir, exist_ok=True)

        metrics_list: List[Dict[str, float]] = []

        for ep in range(args.episodes):
            seed = int(args.seed + ep)
            env = SimpleGymEnv(
                grid=grid, fov_rad=1.0, render_mode=None, use_env_kf=True,
            )
            env.simple_dynamics = True       # enable point-mass dynamics
            env.simple_theta_alpha = 0.3     # smooth FoV direction
            env.simple_momentum = 0.5        # velocity momentum
            env.collision_freeze = args.collision_freeze
            env.reset(seed=seed)

            planner = make_planner(tag, env, step_size=args.step_size)

            print(f"[{tag}] ep={ep}/{args.episodes} seed={seed}")
            mdata = run_episode(env, planner, args.max_steps)

            sigma_cap_eval = float(getattr(env, "_sigma_cap_px", SIGMA_CAP))
            m = evaluate_episode(
                mu=mdata["mu"], Sigma=mdata["Sigma"],
                x_true=mdata["x_true"], did_update=mdata["did_update"],
                exist_mask=mdata["exist_mask"],
                sigma_cap=sigma_cap_eval, rmse_all_censored=False,
            )
            m["episode"] = ep
            metrics_list.append({
                "rmse_exist": float(m["rmse_exist"]),
                "nll": float(m["nll"]),
                "entropy": float(m["entropy"]),
                "episode": ep,
            })
            print(f"  rmse={m['rmse_exist']:.3f}  nll={m['nll']:.2f}  H={m['entropy']:.2f}")

            dump = {
                "mu": mdata["mu"], "Sigma": mdata["Sigma"],
                "x_true": mdata["x_true"], "did_update": mdata["did_update"],
                "exist_mask": mdata["exist_mask"],
            }
            if mdata.get("robot") is not None:
                dump["robot"] = mdata["robot"]
            npz_path = os.path.join(ts_dir, f"{tag}_ep{ep:03d}.npz")
            np.savez_compressed(npz_path, **dump)

        json_path = os.path.join(args.out_dir, f"{tag}_simple.json")
        with open(json_path, "w") as f:
            json.dump({"args": vars(args), "metrics": metrics_list}, f, indent=2)
        print(f"[{tag}] Done. Saved -> {json_path}")

    print(f"\n[ALL DONE] Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
