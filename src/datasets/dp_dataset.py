from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader


class ActionMinMax:
    def __init__(self, low: np.ndarray, high: np.ndarray, eps: float = 1e-6):
        low = np.asarray(low, dtype=np.float32).ravel()
        high = np.asarray(high, dtype=np.float32).ravel()
        self.low = low
        self.high = high
        self.center = 0.5 * (high + low)
        self.scale = np.maximum(0.5 * (high - low), eps)

    def normalize(self, a: np.ndarray, clip: bool = True) -> np.ndarray:
        z = (a - self.center) / self.scale
        return np.clip(z, -1.0, 1.0) if clip else z

    def denormalize(self, z: np.ndarray, clip: bool = True) -> np.ndarray:
        a = z * self.scale + self.center
        if clip:
            a = np.clip(a, self.low, self.high)
        return a

    def to_json(self) -> dict:
        return {
            "type": "minmax",
            "low": self.low.tolist(),
            "high": self.high.tolist(),
        }


class DPSequenceDataset(Dataset):
    def __init__(self, episode_paths: List[Path], obs_h: int, pred_h: int,scaler: ActionMinMax, 
        clip_actions: bool = True,):
        super().__init__()
        self.obs_h = int(obs_h)
        self.pred_h = int(pred_h)
        self.scaler = scaler
        self.clip_actions = bool(clip_actions)

        self.index: List[Tuple[Path, int]] = []
        for p in episode_paths:
            with np.load(p, allow_pickle=False) as d:
                T_ep = int(d["action"].shape[0])
            max_t = T_ep - (self.obs_h + self.pred_h)
            for t0 in range(max_t + 1):
                self.index.append((p, t0))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        p, t0 = self.index[i]
        H, T = self.obs_h, self.pred_h
        with np.load(p, allow_pickle=False) as d:
            robot = d["robot"][t0 : t0 + H].astype(np.float32)
            slots = d["slots"][t0 : t0 + H].astype(np.float32)
            ego = d["ego_map"][t0 : t0 + H]
            act = d["action"][t0 + H : t0 + H + T].astype(np.float32)

        if self.clip_actions:
            act = np.clip(act, self.scaler.low, self.scaler.high)
        act_norm = self.scaler.normalize(act).astype(np.float32)

        return {
            "robot": torch.from_numpy(robot),
            "slots": torch.from_numpy(slots),
            "ego_map": torch.from_numpy(ego),
            "actions": torch.from_numpy(act_norm),
        }


def split_episodes(data_dir: Path,val_ratio: float, test_ratio: float, seed: int,
      ) -> Tuple[List[Path], List[Path], List[Path]]:
    all_paths = sorted(list(Path(data_dir).glob("**/*.npz")))
    n = len(all_paths)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    rng = random.Random(seed)
    rng.shuffle(all_paths)

    test_eps = all_paths[:n_test]
    val_eps = all_paths[n_test : n_test + n_val]
    train_eps = all_paths[n_test + n_val :]
    return train_eps, val_eps, test_eps


def maybe_read_low_high_from_episode(ep_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(ep_path, allow_pickle=False) as d:
        low = np.asarray(d["action_low"], dtype=np.float32).ravel()
        high = np.asarray(d["action_high"], dtype=np.float32).ravel()
    return low, high


# dp_dataset.py 内

def make_loaders(
    data_dir: Path,
    batch_size: int,
    obs_h: int,
    pred_h: int,
    num_workers: int,
    prefetch_factor: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    action_low: Tuple[float, float],
    action_high: Tuple[float, float],
):

    train_eps, val_eps, _ = split_episodes(data_dir, val_ratio, test_ratio, seed)

    low, high = maybe_read_low_high_from_episode(train_eps[0])
    scaler = ActionMinMax(low, high)

    ds_tr = DPSequenceDataset(train_eps, obs_h, pred_h, scaler)
    ds_va = DPSequenceDataset(val_eps, obs_h, pred_h, scaler) if val_eps else None

    persistent = num_workers > 0
    pf = prefetch_factor if persistent else None

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=pf,
    )
    dl_va = None
    if ds_va and len(ds_va) > 0:
        dl_va = DataLoader(
            ds_va,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent,
            prefetch_factor=pf,
        )
    return dl_tr, dl_va, scaler