import numpy as np
import torch

from src.models.dp_policy import DiffusionPolicyNetwork
from utilities.utils import compute_s_lost_norm, compute_slot_features_normalized

import collections
from typing import Dict, Any, Optional
from numpy.typing import NDArray

class DPAgent:
    def __init__(
        self,
        policy_model: DiffusionPolicyNetwork,
        action_space_low: NDArray[np.float_],
        action_space_high: NDArray[np.float_],
        obs_horizon: int,
        pred_horizon: int,
        action_horizon: int = 8,
        num_inference_steps: int = 50,
        action_stats: Optional[Dict[str, NDArray[np.float_]]] = None,
        map_w: int = 1909,
        map_h: int = 955,
        s_lost_norm: Optional[float] = None,        

    ):
        self.policy = policy_model.eval()
        self.device = next(policy_model.parameters()).device

        self.action_stats = action_stats
        self.action_low = action_space_low.astype(np.float32)
        self.action_high = action_space_high.astype(np.float32)

        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)

        self.map_w = int(map_w)
        self.map_h = int(map_h)
        self.s_lost_norm = float(s_lost_norm if s_lost_norm is not None
                                  else compute_s_lost_norm(map_w, map_h))
        

        self.obs_history = collections.deque(maxlen=self.obs_horizon)
        self.action_plan = collections.deque(maxlen=self.pred_horizon)
        self._steps_since_plan = 0

    def reset(self):
        self.obs_history.clear()
        self.action_plan.clear()
        self._steps_since_plan = 0


    def _denorm_action(self, action_norm: np.ndarray) -> np.ndarray:
        if self.action_stats is not None:
            low  = np.asarray(self.action_stats["low"],  dtype=np.float32).reshape(-1)
            high = np.asarray(self.action_stats["high"], dtype=np.float32).reshape(-1)
        else:
            low  = self.action_low
            high = self.action_high
        return (0.5 * (action_norm + 1.0) * (high - low) + low).astype(np.float32)

    def _prepare_batch(self) -> Dict[str, torch.Tensor]:
        hist = list(self.obs_history)
        robot_states = np.stack([o["robot"] for o in hist]).astype(np.float32)
        slots = np.stack([o["slots"] for o in hist]).astype(np.float32)
        ego_map_np = np.stack([o["ego_map"] for o in hist])

        slots[..., :2] = np.clip(slots[..., :2], -1.0, 1.0)
        slots[..., 2:4] = np.clip(slots[..., 2:4], 0.0, 1.0)

        # Compute slot features [H, K, 7]
        slot_features = compute_slot_features_normalized(
            slots, robot_states, self.map_w, self.map_h)

        # Compute slot mask [H, K]
        maxsig = np.maximum(slots[..., 2], slots[..., 3])
        slot_mask = (maxsig < self.s_lost_norm).astype(np.float32)

        return {
            "ego_map": torch.from_numpy(ego_map_np).unsqueeze(0).to(self.device),
            "slot_features": torch.from_numpy(slot_features).unsqueeze(0).to(self.device),
            "slot_mask": torch.from_numpy(slot_mask).unsqueeze(0).to(self.device),
        }

    def get_action(self, current_obs: Dict[str, np.ndarray]) -> np.ndarray:
        self.obs_history.append(current_obs)
        while len(self.obs_history) < self.obs_horizon:
            self.obs_history.append(current_obs)

        if (not self.action_plan) or (self._steps_since_plan >= self.action_horizon):
            batch = self._prepare_batch()
            with torch.inference_mode():
                a_seq = self.policy.predict(batch, num_inference_steps=self.num_inference_steps)
            a_seq_np = a_seq.squeeze(0).cpu().numpy()
            unnorm = self._denorm_action(a_seq_np)
            clipped = np.clip(unnorm, self.action_low, self.action_high)
            self.action_plan.clear()
            self.action_plan.extend(clipped)
            self._steps_since_plan = 0

        act = np.asarray(self.action_plan.popleft(), dtype=np.float32)
        self._steps_since_plan += 1
        return act