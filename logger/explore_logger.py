import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import time

from envs.env_gym import SimpleGymEnv
from pursuer.explore_only import Coordinator
from utilities.utils import load_houseexpo_image_as_grid

MAP_FILE = "map/0a1a5807d65749c1194ce1840354be39.png"
OUTPUT_DIR = "train_data"
NUM_EPISODES = 300
MAX_STEPS_PER_EPISODE = 1500

USE_RANDOM_SEED_PER_EPISODE = False

STUCK_WINDOW = 80



def build_local_profile(env: SimpleGymEnv) -> tuple[dict, dict]:
    coord_params = {
        "FORCE_EXPLORE_ONLY": True,
        "ENABLE_REACQUIRE": False,
        "SWITCH_ON_NEW": False,
        "GOAL_HOLD_STEPS": 20,
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
        "random_seed": int(env.rng_spawn.integers(1, 1_000_000)) if getattr(env, 'rng_spawn', None) else None,
    }
    return coord_params, goal_params

def save_episode(output_dir: Path, episode_idx: int, episode_data: dict, episode_meta: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"explore_episode_{episode_idx:03d}"
    npz_path = f"{base}.npz"
    json_path = f"{base}_stats.json"

    dtype_map = {
        "ego_map": np.uint8,
        "robot": np.float32,
        "slots": np.float32,
        "action": np.float32,
        "step_idx": np.int32,
        "episode_id": np.int32,
        "seed": np.int32,
    }
    save_data = {k: np.array(v, dtype=dtype_map.get(k)) for k,v in episode_data.items() if k in dtype_map}

    np.savez_compressed(npz_path, **save_data)
    with open(json_path, "w") as f:
        json.dump(episode_meta, f, indent=2)

    print(f"[Save] ep={episode_idx:03d}  steps={episode_meta.get('steps_done', -1)}  end={episode_meta.get('end_reason','n/a')} -> {npz_path}")

def collect_expert_data():
    output_path = Path(OUTPUT_DIR)

    grid = load_houseexpo_image_as_grid(MAP_FILE)
    env = SimpleGymEnv(grid=grid, fov_rad=1.0, render_mode=None)

    expected_shapes = {
        "ego_map": (4, 128, 128),
        "robot": (3,),
        "slots": (env.KMAX, 4),
        "action": (2,),
    }

    t0_all = time.time()
    for episode_idx in range(NUM_EPISODES):
        print(f"\n--- Episode {episode_idx+1}/{NUM_EPISODES} ---")

        episode_data = {key: [] for key in list(expected_shapes.keys()) + ["step_idx", "episode_id", "seed"]}
        episode_meta = {
            "episode_id": int(episode_idx), "seed": None, "steps_done": 0,
            "end_reason": "timeout", "stuck_window": STUCK_WINDOW,
            "wall_time_sec": 0.0,
        }

        if USE_RANDOM_SEED_PER_EPISODE:
            episode_seed = int(np.random.randint(0, 2**31 - 1))
        else:
            episode_seed = None
        obs, info = env.reset(seed=episode_seed)
        episode_meta["seed"] = (-1 if episode_seed is None else int(episode_seed))

        coord_params, goal_params = build_local_profile(env)
        print("Using Profile: 'local' (explore-only)")

        expert = Coordinator(
            grid=env._global_map, kf_targets=env._kf_targets, plan_grid=env._planning_grid,
            params=coord_params, goal_params=goal_params,
        )

        prev_known_count = int(np.count_nonzero(env._known == 1))
        no_new_counter = 0

        t0_ep = time.time()
        for step in tqdm(range(MAX_STEPS_PER_EPISODE), desc=f"Episode {episode_idx+1}"):
            current_obs = obs

            detections = []
            for tid in range(env.KMAX):
                tgt = env._tgts[tid]
                visible = (tgt is not None) and env.is_target_visible(tgt)
                detections.append((tid, tgt.get_pose()[:2] if visible else None))

            expert_result = expert.step(env._rbt, detections, env.get_visible_region(), tmask=env._tmask)
            
            action = expert_result.get("action")
            if action is None:
                action = np.zeros(expected_shapes["action"], dtype=np.float32)
            
            for key, expected_shape in expected_shapes.items():
                if key == 'action':
                    continue
                arr = current_obs.get(key, None)
                if arr is None:
                    arr = np.zeros(expected_shape, dtype=np.float32 if key != 'ego_map' else np.uint8)
                if arr.shape != expected_shape:
                    print(f"\nWarning: Shape mismatch for '{key}' at step {step}. Got {arr.shape}, expected {expected_shape}. Padding with zeros.")
                    arr = np.zeros(expected_shape, dtype=arr.dtype)
                episode_data[key].append(arr)

            episode_data["action"].append(action)
            episode_data["step_idx"].append(step)
            episode_data["episode_id"].append(episode_idx)
            episode_data["seed"].append(-1 if episode_seed is None else int(episode_seed))

            obs, reward, terminated, truncated, info = env.step(action)


            cur_known_count = int(np.count_nonzero(env._known == 1))
            new_known = cur_known_count - prev_known_count
            prev_known_count = cur_known_count
            no_new_counter = 0 if new_known > 0 else no_new_counter + 1
            if no_new_counter >= STUCK_WINDOW:
                episode_meta["end_reason"] = "stuck_no_new_cells"
                episode_meta["steps_done"] = step + 1
                break

            if terminated or truncated:
                episode_meta["end_reason"] = "env_done"
                episode_meta["steps_done"] = step + 1
                break

        if episode_meta["steps_done"] == 0:
            episode_meta["steps_done"] = min(MAX_STEPS_PER_EPISODE, len(episode_data["step_idx"]))
        episode_meta["wall_time_sec"] = float(time.time() - t0_ep)

        try:
            episode_meta["frontier_stats"] = getattr(expert.goal_selector, "last_frontier_stats", {})
        except Exception:
            episode_meta["frontier_stats"] = {}

        save_episode(output_path, episode_idx, episode_data, episode_meta)

    print(f"\n[Main] Total wall time: {time.time() - t0_all:.1f}s")

if __name__ == "__main__":
    collect_expert_data()