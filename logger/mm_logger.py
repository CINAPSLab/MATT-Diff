from __future__ import annotations

import os
import time
import json
import random
from pathlib import Path
from collections import deque

import numpy as np
import cv2
from tqdm import tqdm

from envs.env_gym import SimpleGymEnv
from pursuer.mm_planner import Coordinator as MMCoordinator
from utilities.utils import load_houseexpo_image_as_grid


MAP_FILE = "map/0a1a5807d65749c1194ce1840354be39.png"
OUTPUT_DIR = "train_data"
DEBUG_DIR = "debug_vis_mm"

NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 2000

BASE_SEED = 100000

SAVE_VIS = False
VIS_EVERY = 5000

STUCK_WINDOW = 200
POS_EPS = 10.0
YAW_EPS_DEG = 10.0

ZERO_BURST_ABORT = 150
NO_PATH_BURST_ABORT = 200

PROFILES = [
    ("track-strong", 0),
    ("reacq-strong", 1),
]


def profile_params(name: str) -> dict:
    if name == "track-strong":
        return {
            "FORCE_EXPLORE_ONLY": False,
            "ENABLE_REACQUIRE": True,
            "SWITCH_ON_NEW": True,
            "plan_safety_margin_px": 8,
            "sigma_reacq_lo": 150.0,
            "sigma_abort": 240.0,
            "stale_S": 500,
            "T_track": 180,
            "cooldown": 600,
            "goal_hold_steps": 20,
        }
    if name == "reacq-strong":
        return {
            "FORCE_EXPLORE_ONLY": False,
            "ENABLE_REACQUIRE": True,
            "SWITCH_ON_NEW": True,
            "plan_safety_margin_px": 8,
            "sigma_reacq_lo": 80.0,
            "sigma_abort": 230.0,
            "stale_S": 75,
            "T_track": 10,
            "cooldown": 60,
            "goal_hold_steps": 20,
        }
    raise ValueError(f"unknown profile: {name}")


def build_goal_params(env: SimpleGymEnv) -> dict:
    return {
        "frontier_gain_weight": 1.2,
        "frontier_area_weight": 1.2,
        "frontier_dist_weight": 10.0,
        "frontier_known_weight": 0.4,
        "frontier_min_area": 300,
        "frontier_excl_radius": 80,
        "frontier_gain_radius": int(0.7 * getattr(env, "_radius", 120)),
        "use_simple_nearest": True,
        "sample_frontier": True,
        "softmax_temp": 0.55,
        "random_seed": int(env.rng_spawn.integers(1, 1_000_000))
        if getattr(env, "rng_spawn", None)
        else None,
    }


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _pose_progress_stuck(p_start: np.ndarray, p_now: np.ndarray, pos_eps: float = POS_EPS,
    yaw_eps_deg: float = YAW_EPS_DEG,) -> bool:
    dx = float(p_now[0] - p_start[0])
    dy = float(p_now[1] - p_start[1])
    dist = (dx * dx + dy * dy) ** 0.5
    dth_deg = abs(_wrap_deg(np.degrees(float(p_now[2] - p_start[2]))))
    return (dist < pos_eps) and (dth_deg < yaw_eps_deg)


def quick_snapshot(env: SimpleGymEnv) -> np.ndarray | None:
    canvas = cv2.cvtColor(
        (env._global_map * 225).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )
    poly = env.get_visible_region().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [poly], True, (0, 0, 255), 2)
    corners = env.get_robot_corners()
    cv2.fillPoly(canvas, [corners], color=(0, 255, 255))
    cv2.polylines(canvas, [corners], True, (0, 0, 0), 2)
    return canvas


def save_episode(
    output_dir: Path, episode_idx: int, episode_data: dict, episode_meta: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"mm_episode_{episode_idx:03d}"
    npz_path = f"{base}.npz"
    json_path = f"{base}_stats.json"

    dtype_map = {
        "ego_map": np.uint8,
        "robot": np.float32,
        "slots": np.float32,
        "action": np.float32,
        "step_idx": np.int32,
        "episode_id": np.int32,
        "profile_id": np.int32,
    }
    save_data = {
        k: np.array(v, dtype=dtype_map.get(k))
        for k, v in episode_data.items()
        if k in dtype_map
    }

    np.savez_compressed(npz_path, **save_data)
    with open(json_path, "w") as f:
        json.dump(episode_meta, f, indent=2)
    print(
        f"[Save] ep_idx={episode_idx:03d} steps={episode_meta.get('steps_done', -1)} "
        f"end={episode_meta.get('end_reason','n/a')} seed={episode_meta.get('episode_seed','-')} "
        f"profile={episode_meta.get('profile','-')} -> {npz_path}"
    )


def save_dropped_meta(output_dir: Path, episode_idx: int, meta: dict):
    d = Path(output_dir) / "dropped"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"mm_drop_ep_{episode_idx:03d}.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Drop] ep_idx={episode_idx:03d} reason={meta.get('end_reason')} -> {path}")


def collect_mm_expert():
    grid = load_houseexpo_image_as_grid(MAP_FILE)
    env = SimpleGymEnv(grid=grid, fov_rad=1.0, render_mode=None, use_env_kf=True)

    expected_shapes = {
        "ego_map": (4, 128, 128),
        "robot": (3,),
        "slots": (env.KMAX, 4),
        "action": (2,),
    }

    out_dir = Path(OUTPUT_DIR)
    dbg_dir = out_dir / DEBUG_DIR
    if SAVE_VIS:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    t0_all = time.time()

    for ep in range(NUM_EPISODES):
        ep_seed = int(BASE_SEED + ep)

        for profile_name, profile_id in PROFILES:
            out_idx = ep * len(PROFILES) + profile_id

            os.environ["PYTHONHASHSEED"] = str(ep_seed)
            random.seed(ep_seed)
            np.random.seed(ep_seed)

            params = profile_params(profile_name)
            goal_params = build_goal_params(env)

            print(
                f"\n--- Episode {ep+1}/{NUM_EPISODES} | profile={profile_name} "
                f"(id={profile_id}) | seed={ep_seed} ---",
                flush=True,
            )

            obs, info = env.reset(seed=ep_seed)

            # DP / BC と同条件にしたい場合はここは呼ばない
            if hasattr(env, "sync_tick"):
                obs, info = env.sync_tick()

            expert = MMCoordinator(
                grid=env._global_map,
                kf_targets=env._kf_targets,
                plan_grid=getattr(env, "_planning_grid", None),
                params=params,
                goal_params=goal_params,
            )

            epi = {
                k: []
                for k in [
                    "ego_map",
                    "robot",
                    "slots",
                    "action",
                    "step_idx",
                    "episode_id",
                    "profile_id",
                ]
            }
            meta = {
                "episode_seed": int(ep_seed),
                "episode_id": int(ep),
                "profile": profile_name,
                "profile_id": int(profile_id),
                "params": params,
                "goal_params": goal_params,
                "steps_done": 0,
                "end_reason": "timeout",
                "dropped": False,
            }

            pose_hist = deque(maxlen=STUCK_WINDOW + 1)

            def _push_pose():
                pose_hist.append(
                    np.array(
                        [env._rbt[0], env._rbt[1], env._rbt[2]], dtype=np.float32
                    )
                )

            _push_pose()

            zero_burst = 0
            no_path_burst = 0

            dropped = False

            for step in tqdm(
                range(MAX_STEPS_PER_EPISODE),
                desc=f"Ep {ep+1}/{NUM_EPISODES} | {profile_name}",
                leave=False,
            ):
                detections = []
                for tid in range(env.KMAX):
                    tgt = env._tgts[tid]
                    if tgt is None:
                        detections.append((tid, None))
                        continue
                    vis = env.is_target_visible(tgt)
                    if vis and env._tmask[tid] == 0:
                        env._tmask[tid] = 1
                    detections.append(
                        (tid, tgt.get_pose()[:2] if vis else None)
                    )

                expert_out = expert.step(
                    env._rbt, detections, env.get_visible_region(), tmask=env._tmask
                )

                action = expert_out.get("action")
                if action is None:
                    action = np.zeros((2,), dtype=np.float32)
                a = np.asarray(action, dtype=np.float32).ravel()[:2]

                if (abs(a[0]) < 1e-3) and (abs(a[1]) < 1e-3):
                    zero_burst += 1
                else:
                    zero_burst = 0

                path_present = expert_out.get("path") is not None
                if not path_present:
                    no_path_burst += 1
                else:
                    no_path_burst = 0

                if zero_burst >= ZERO_BURST_ABORT:
                    meta["end_reason"] = "zero_burst_abort"
                    meta["steps_done"] = step + 1
                    meta["dropped"] = True
                    dropped = True
                    break
                if no_path_burst >= NO_PATH_BURST_ABORT:
                    meta["end_reason"] = "no_path_burst_abort"
                    meta["steps_done"] = step + 1
                    meta["dropped"] = True
                    dropped = True
                    break

                for k, shp in expected_shapes.items():
                    if k == "action":
                        continue
                    arr = obs.get(k, None)
                    if arr is None or tuple(arr.shape) != shp:
                        arr = np.zeros(
                            shp,
                            dtype=np.float32 if k != "ego_map" else np.uint8,
                        )
                    epi[k].append(arr)
                epi["action"].append(a.astype(np.float32))
                epi["step_idx"].append(step)
                epi["episode_id"].append(ep)
                epi["profile_id"].append(profile_id)

                obs, reward, terminated, truncated, info = env.step(a)

                if SAVE_VIS and step > 0 and (step % VIS_EVERY == 0):
                    snap = quick_snapshot(env)
                    if snap is not None:
                        out_img = (
                            dbg_dir
                            / f"ep{out_idx:03d}_p{profile_id}_step{step:05d}.png"
                        )
                        cv2.imwrite(str(out_img), snap)

                _push_pose()
                if len(pose_hist) >= (STUCK_WINDOW + 1):
                    p0, pN = pose_hist[0], pose_hist[-1]
                    if _pose_progress_stuck(p0, pN):
                        meta["end_reason"] = "stuck_pose"
                        meta["steps_done"] = step + 1
                        break

                if terminated or truncated:
                    meta["end_reason"] = "env_done"
                    meta["steps_done"] = step + 1
                    break

            if meta["steps_done"] == 0:
                meta["steps_done"] = min(
                    MAX_STEPS_PER_EPISODE, len(epi["step_idx"])
                )

            if dropped:
                save_dropped_meta(OUTPUT_DIR, out_idx, meta)
            else:
                save_episode(Path(OUTPUT_DIR), out_idx, epi, meta)

    print(f"\n[mm_logger_strong] Total wall time: {time.time() - t0_all:.1f}s")


if __name__ == "__main__":
    collect_mm_expert()