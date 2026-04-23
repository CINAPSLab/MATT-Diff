"""
Data collection for MATT-Diff training with simple point-mass dynamics.

Saves per-step observations (robot, slots, ego_map) and actions (dx, dy)
in the same npz format as explore_logger.py.

Usage:
  python logger/collect_simple_dynamics.py \
    --map map/5.png --out train_data_simple --episodes 500 --seed 1000
"""

import numpy as np
import os
import random
import json
import time
import argparse
from pathlib import Path
from collections import deque
from tqdm import tqdm

from envs.env_gym import SimpleGymEnv
from pursuer.mm_planner import Coordinator as MMPlanner
from pursuer.explore_only import Coordinator as FrontierPlanner
from pursuer.controller.simple_follow import SimpleFollowController
from utilities.utils import load_houseexpo_image_as_grid, compute_s_lost_norm

# ── CLI ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--map", type=str, required=True)
parser.add_argument("--out", type=str, default="train_data_simple")
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--max_steps", type=int, default=1500)
parser.add_argument("--step_size", type=float, default=10.0)
parser.add_argument("--momentum", type=float, default=0.5)
parser.add_argument("--theta_alpha", type=float, default=0.3)
parser.add_argument("--planner", type=str, default="frontier",
                    choices=["track", "reacq", "frontier"])
args = parser.parse_args()

# ── Planner configs ───────────────────────────────────────────────
TRACK_PARAMS = {
    "FORCE_EXPLORE_ONLY": False, "ENABLE_REACQUIRE": True,
    "SWITCH_ON_NEW": True, "plan_safety_margin_px": 8,
    "sigma_reacq_lo": 350.0, "sigma_abort": 350.0,
    "stale_S": 200, "T_track": 150, "cooldown": 200, "goal_hold_steps": 20,
}
TRACK_GOAL = {
    "frontier_gain_weight": 1.0, "frontier_area_weight": 1.0,
    "frontier_dist_weight": 8.0, "frontier_known_weight": 0.5,
    "frontier_min_area": 8, "frontier_excl_radius": 10,
    "frontier_gain_radius": 60, "use_simple_nearest": True, "softmax_temp": 3.4,
}
REACQ_PARAMS = {
    "FORCE_EXPLORE_ONLY": False, "ENABLE_REACQUIRE": True,
    "SWITCH_ON_NEW": True, "plan_safety_margin_px": 8,
    "sigma_reacq_lo": 150.0, "sigma_abort": 350.0,
    "stale_S": 60, "T_track": 70, "cooldown": 60, "goal_hold_steps": 20,
}
REACQ_GOAL = {
    "use_simple_nearest": True, "frontier_gain_weight": 1.2,
    "frontier_area_weight": 1.2, "frontier_dist_weight": 10.0,
    "frontier_known_weight": 0.4, "frontier_min_area": 300,
    "frontier_excl_radius": 80, "frontier_gain_radius": 60, "softmax_temp": 0.5,
}

STUCK_WINDOW = 200
STUCK_DIST_EPS = 50.0        # cumulative distance over window must exceed this


def make_planner(tag, env, step_size):
    if tag == "frontier":
        params = {
            "FORCE_EXPLORE_ONLY": True, "ENABLE_REACQUIRE": False,
            "SWITCH_ON_NEW": False, "GOAL_HOLD_STEPS": 20,
            "plan_safety_margin_px": 8,
        }
        goal_params = {
            "frontier_gain_weight": 1.8, "frontier_area_weight": 0.8,
            "frontier_dist_weight": 10.0, "frontier_known_weight": 0.25,
            "frontier_min_area": 300, "frontier_excl_radius": 80,
            "frontier_gain_radius": int(0.7 * env._radius),
            "use_simple_nearest": True, "softmax_temp": 0.55,
        }
        if getattr(env, "rng_spawn", None):
            goal_params["random_seed"] = int(env.rng_spawn.integers(1, 1_000_000))
        planner = FrontierPlanner(
            grid=env._global_map, kf_targets=env._kf_targets,
            plan_grid=env._planning_grid, params=params, goal_params=goal_params,
        )
    elif tag == "track":
        planner = MMPlanner(
            grid=env._global_map, kf_targets=env._kf_targets,
            plan_grid=env._planning_grid,
            params=dict(TRACK_PARAMS), goal_params=dict(TRACK_GOAL),
        )
    else:  # reacq
        goal_params = dict(REACQ_GOAL)
        if getattr(env, "rng_spawn", None):
            goal_params["random_seed"] = int(env.rng_spawn.integers(1, 1_000_000))
        planner = MMPlanner(
            grid=env._global_map, kf_targets=env._kf_targets,
            plan_grid=env._planning_grid,
            params=dict(REACQ_PARAMS), goal_params=goal_params,
        )
    planner.controller = SimpleFollowController(step_size=step_size, rdp_epsilon=5.0)
    return planner


def save_episode(output_dir, episode_idx, episode_data, episode_meta,
                 map_w, map_h, map_id, s_lost_norm):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"episode_{episode_idx:04d}"
    npz_path = f"{base}.npz"
    json_path = f"{base}_stats.json"

    save_data = {
        "ego_map": np.array(episode_data["ego_map"], dtype=np.uint8),
        "robot": np.array(episode_data["robot"], dtype=np.float32),
        "slots": np.array(episode_data["slots"], dtype=np.float32),
        "action": np.array(episode_data["action"], dtype=np.float32),
        "map_w": np.array(map_w, dtype=np.int32),
        "map_h": np.array(map_h, dtype=np.int32),
        "action_low": np.array([-args.step_size, -args.step_size], dtype=np.float32),
        "action_high": np.array([args.step_size, args.step_size], dtype=np.float32),
    }

    # slot_mask
    slots_arr = save_data["slots"]
    maxsig = np.maximum(slots_arr[..., 2], slots_arr[..., 3])
    save_data["slot_mask"] = (maxsig < s_lost_norm).astype(np.float32)

    np.savez_compressed(npz_path, **save_data)
    with open(json_path, "w") as f:
        json.dump(episode_meta, f, indent=2)

    print(f"[Save] ep={episode_idx:04d}  steps={episode_meta.get('steps_done', -1)}  "
          f"end={episode_meta.get('end_reason', 'n/a')} -> {npz_path}")


def main():
    output_path = Path(args.out)
    grid = load_houseexpo_image_as_grid(args.map)
    map_h, map_w = grid.shape
    map_id = Path(args.map).stem
    s_lost_norm = compute_s_lost_norm(map_w, map_h)

    expected_shapes = {
        "ego_map": (4, 128, 128),
        "robot": (3,),
        "slots": (6, 4),  # KMAX=6
        "action": (2,),
    }

    t0_all = time.time()
    for episode_idx in range(args.episodes):
        ep_seed = int(args.seed + episode_idx)

        os.environ["PYTHONHASHSEED"] = str(ep_seed)
        random.seed(ep_seed)
        np.random.seed(ep_seed)

        print(f"\n--- Episode {episode_idx+1}/{args.episodes} | seed={ep_seed} | planner={args.planner} ---")

        episode_data = {key: [] for key in expected_shapes}
        episode_meta = {
            "episode_id": int(episode_idx), "seed": ep_seed,
            "steps_done": 0, "end_reason": "timeout",
            "planner": args.planner, "dynamics": "simple",
            "momentum": args.momentum, "theta_alpha": args.theta_alpha,
            "step_size": args.step_size,
        }

        env = SimpleGymEnv(grid=grid, fov_rad=1.0, render_mode=None, use_env_kf=True)
        env.simple_dynamics = True
        env.simple_momentum = args.momentum
        env.simple_theta_alpha = args.theta_alpha
        obs, info = env.reset(seed=ep_seed)

        expert = make_planner(args.planner, env, step_size=args.step_size)
        pose_hist = deque(maxlen=STUCK_WINDOW + 1)
        pose_hist.append(env._rbt.copy())
        cumul_dist = 0.0

        t0_ep = time.time()
        for step in tqdm(range(args.max_steps), desc=f"Episode {episode_idx+1}", leave=False):
            current_obs = obs

            result = expert.step(env._rbt, None, env.get_visible_region(), tmask=env._tmask)
            action = result.get("action") if isinstance(result, dict) else None
            if action is None:
                action = np.zeros(2, dtype=np.float32)
            else:
                action = np.asarray(action, dtype=np.float32).ravel()[:2]

            # save observation + action
            for key, shape in expected_shapes.items():
                if key == "action":
                    continue
                arr = current_obs.get(key, None)
                if arr is None or arr.shape != shape:
                    arr = np.zeros(shape, dtype=np.float32 if key != "ego_map" else np.uint8)
                episode_data[key].append(arr)
            episode_data["action"].append(action)

            obs, _, terminated, truncated, info = env.step(action)

            prev_pos = pose_hist[-1][:2]
            pose_hist.append(env._rbt.copy())
            step_dist = np.linalg.norm(env._rbt[:2] - prev_pos)
            cumul_dist += step_dist
            if len(pose_hist) >= (STUCK_WINDOW + 1):
                # subtract the oldest leg that just fell out of the window
                oldest = pose_hist[0][:2]
                second = pose_hist[1][:2]
                cumul_dist -= np.linalg.norm(second - oldest)
                if cumul_dist < STUCK_DIST_EPS:
                    episode_meta["end_reason"] = "stuck"
                    episode_meta["steps_done"] = step + 1
                    break

            if terminated or truncated:
                episode_meta["end_reason"] = "env_done"
                episode_meta["steps_done"] = step + 1
                break

        if episode_meta["steps_done"] == 0:
            episode_meta["steps_done"] = min(args.max_steps, len(episode_data["action"]))
        episode_meta["wall_time_sec"] = float(time.time() - t0_ep)

        save_episode(output_path, episode_idx, episode_data, episode_meta,
                     map_w=map_w, map_h=map_h, map_id=map_id,
                     s_lost_norm=s_lost_norm)

    print(f"\n[ALL DONE] {args.episodes} episodes saved to {output_path}/")
    print(f"Total wall time: {time.time() - t0_all:.1f}s")
    print(f"\nTo train: python src/train_dp.py --data_dir {output_path} "
          f"--act_low {-args.step_size} {-args.step_size} "
          f"--act_high {args.step_size} {args.step_size}")


if __name__ == "__main__":
    main()
