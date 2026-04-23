"""
Calibration study: DP WITHOUT map encoder, simple point-mass dynamics.

Loads a trained checkpoint but zeroes out the map embedding in
`_build_global_cond`, so the global condition only contains
target-encoder information.  Everything else (target encoder,
U-Net, layer norms) uses the original trained weights.

Usage:
  python -m evals.eval_dp_no_map_simple \
    --ckpt output/model/dp_apr9/best.pt \
    --map_path map/0.png --seed 27 --episodes 10 \
    --out output/results/dp_no_map_simple/map0/map0_seed27_ep10.json \
    --ts_dir output/results/dp_no_map_simple/map0/timeseries/map0_seed27_ep10
"""

import types
import numpy as np
import torch
import json
import argparse
from pathlib import Path

from evals.eval_core import evaluate_episode, pack_from_eval_states
from evals.eval_dp_simple import run_dp_episode, MAX_STEPS, USE_ENV_KF
from envs.env_gym import SimpleGymEnv
from src.dp_agent import DPAgent
from src.models.dp_policy import DiffusionPolicyNetwork
from utilities.utils import load_houseexpo_image_as_grid, compute_s_lost_norm

from typing import Dict, List


# ---- patched forward: pass blank map to preserve LayerNorm ---- #
def _build_global_cond_no_map(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Pass a completely blank map to preserve LayerNorm statistics."""
    dummy_ego_map = torch.zeros_like(batch["ego_map"])
    
    z_map = self.ln_map(self._encode_map(dummy_ego_map))

    slot_features = batch["slot_features"][:, :, :self.k_slots, :]
    slot_mask = batch["slot_mask"][:, :, :self.k_slots]

    slot_feat_last = slot_features[:, -1]
    mask_last = slot_mask[:, -1]
    z_target = self.ln_target(self.tse(slot_feat_last, mask_last))

    return self.ln_gc(torch.cat([z_map, z_target], dim=-1))


def main():
    parser = argparse.ArgumentParser(
        description="Calibration study: DP WITHOUT map encoder, simple dynamics"
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="output/results/dp_no_map_simple_eval.json")
    parser.add_argument("--ts_dir", type=str, default="output/results/dp_no_map_simple/timeseries")
    parser.add_argument("--map_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--inf_steps", type=int, default=20)
    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--theta_alpha", type=float, default=0.3)
    parser.add_argument("--step_size", type=float, default=10.0)
    parser.add_argument("--collision_freeze", action="store_true",
                        help="Freeze robot on first wall collision")
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

    # ---- patch: disable map encoder ---- #
    policy._build_global_cond = types.MethodType(_build_global_cond_no_map, policy)

    grid = load_houseexpo_image_as_grid(args.map_path)
    env = SimpleGymEnv(grid=grid, fov_rad=1.0, render_mode=None, use_env_kf=USE_ENV_KF)
    env.simple_dynamics = True
    env.simple_momentum = args.momentum
    env.simple_theta_alpha = args.theta_alpha
    env.collision_freeze = args.collision_freeze

    s_lost = compute_s_lost_norm(env.width, env.height)

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
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    metrics: List[Dict[str, float]] = []

    for ep in range(args.episodes):
        seed_ep = int(args.seed) + ep
        print(f"[DP-no-map-simple] ep={ep}/{args.episodes} seed={seed_ep}")

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
        np.savez_compressed(ts_dir / f"dp_no_map_ep{ep:03d}.npz", **payload)

    with open(args.out, "w") as f:
        json.dump({"ablation": "no_map_encoder", "args": vars(args), "metrics": metrics}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
