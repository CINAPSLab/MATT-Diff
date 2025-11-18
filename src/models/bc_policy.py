from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Optional


class MapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return h.flatten(1)


class SlotEncoderLight(nn.Module):
    def __init__(self, d_hidden: int = 64, d_out: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(4, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.proj = nn.Linear(d_hidden * 2, d_out)
        self.ln = nn.LayerNorm(d_out)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(slots))
        x = F.relu(self.fc2(x))
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        h = torch.cat([mean_pool, max_pool], dim=1)
        h = self.proj(h)
        return self.ln(h)


class BCSlotsBaseline(nn.Module):
    def __init__(self, use_robot: bool = False):
        super().__init__()
        self.use_robot = bool(use_robot)

        self.map_enc = MapEncoder()
        self.slot_enc = SlotEncoderLight()

        trunk_in = 256 + 128
        if self.use_robot:
            self.rbt = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            trunk_in += 64

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head = nn.Linear(256, 2)

    def forward(self, ego_map_u8: torch.Tensor, slots: torch.Tensor, 
                robot: Optional[torch.Tensor] = None,) -> torch.Tensor:
        x_map = ego_map_u8.float() / 255.0 if ego_map_u8.dtype == torch.uint8 else ego_map_u8
        h_map = self.map_enc(x_map)
        h_slt = self.slot_enc(slots)
        feats = [h_map, h_slt]

        if self.use_robot and (robot is not None):
            feats.append(self.rbt(robot))

        h = torch.cat(feats, dim=1)
        h = self.trunk(h)
        return self.head(h)


class BCPolicy:
    def __init__(self, ckpt_path: str, device: torch.device, use_robot: bool = False):
        self.device = device
        self.use_robot = bool(use_robot)

        self.model = BCSlotsBaseline(use_robot=self.use_robot).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self._low: Optional[np.ndarray] = None
        self._high: Optional[np.ndarray] = None

    def set_action_bounds(self, low: np.ndarray, high: np.ndarray) -> None:
        self._low = np.asarray(low, dtype=np.float32)
        self._high = np.asarray(high, dtype=np.float32)

    @torch.no_grad()
    def predict(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        ego_map = torch.from_numpy(obs["ego_map"]).unsqueeze(0).to(self.device)
        slots = torch.from_numpy(obs["slots"]).unsqueeze(0).to(self.device)
        robot = None
        if self.use_robot and "robot" in obs:
            robot = torch.from_numpy(obs["robot"]).unsqueeze(0).to(self.device)

        act = self.model(ego_map, slots, robot).squeeze(0).cpu().numpy().astype(np.float32)

        if self._low is not None and self._high is not None:
            act = np.clip(act, self._low, self._high)

        return act


__all__ = [
    "MapEncoder",
    "SlotEncoderLight",
    "BCSlotsBaseline",
    "BCPolicy",
]