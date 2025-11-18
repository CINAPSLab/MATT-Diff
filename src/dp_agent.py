import numpy as np
import torch

from src.models.dp_policy import DiffusionPolicyNetwork

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
        robot_stats: Optional[Dict[str, NDArray[np.float_]]] = None,
        action_stats: Optional[Dict[str, NDArray[np.float_]]] = None,
    ):
        self.policy = policy_model.eval()
        self.device = next(policy_model.parameters()).device

        self.robot_stats = robot_stats
        self.action_stats = action_stats
        self.action_low = action_space_low.astype(np.float32)
        self.action_high = action_space_high.astype(np.float32)

        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)

        self.obs_history = collections.deque(maxlen=self.obs_horizon)
        self.action_plan = collections.deque(maxlen=self.pred_horizon)
        self._steps_since_plan = 0

    def reset(self):
        self.obs_history.clear()
        self.action_plan.clear()
        self._steps_since_plan = 0

    def _normalize_robot_state(self, robot_state: np.ndarray) -> np.ndarray:
        if self.robot_stats is None:
            return robot_state.astype(np.float32)
        stats = self.robot_stats
        scale = (stats["max"] - stats["min"]).astype(np.float32)
        scale[scale == 0.0] = 1e-6
        norm_state = (robot_state - stats["min"]) / scale
        return (norm_state * 2.0 - 1.0).astype(np.float32)

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
        robot_states = np.stack([o["robot"] for o in hist])
        slots        = np.stack([o["slots"] for o in hist])
        ego_map_np   = np.stack([o["ego_map"] for o in hist])

        slots = slots.astype(np.float32, copy=False)
        slots[..., :2] = np.clip(slots[..., :2], -1.0, 1.0)
        slots[..., 2:4] = np.clip(slots[..., 2:4],  0.0, 1.0)

        return {
            "robot":   torch.from_numpy(self._normalize_robot_state(robot_states)).unsqueeze(0).to(self.device),
            "slots":   torch.from_numpy(slots).unsqueeze(0).to(self.device),
            "ego_map": torch.from_numpy(ego_map_np).unsqueeze(0).to(self.device),
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