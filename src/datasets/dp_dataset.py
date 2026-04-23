from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import random
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from utilities.utils import compute_s_lost_norm, compute_slot_features_normalized

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

class DPHDF5Dataset(Dataset):
    def __init__(self, h5_path: Path, episode_keys: List[str], obs_h: int, pred_h: int, scaler: ActionMinMax, clip_actions: bool = True):
        super().__init__()
        self.h5_path = h5_path
        self.obs_h = int(obs_h)
        self.pred_h = int(pred_h)
        self.scaler = scaler
        self.clip_actions = bool(clip_actions)
        self.episode_keys = list(episode_keys)

        self.index: List[Tuple[str, int]] = []
        self._h5_file = None

        # Build sequence index from the HDF5 structure
        print(f"Building index for {len(self.episode_keys)} episodes...")
        with h5py.File(self.h5_path, 'r') as f:
            for ep_key in self.episode_keys:
                T_ep = int(f[ep_key]["action"].shape[0])
                max_t = T_ep - (self.obs_h + self.pred_h)
                for t0 in range(max_t + 1):
                    self.index.append((ep_key, t0))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        # Open the HDF5 file inside the worker process to avoid multiprocessing errors
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r', swmr=True)

        ep_key, t0 = self.index[i]
        H, T = self.obs_h, self.pred_h

        grp = self._h5_file[ep_key]

        # Read slices directly from the HDF5 file into numpy arrays
        robot = grp["robot"][t0 : t0 + H].astype(np.float32)
        slots = grp["slots"][t0 : t0 + H].astype(np.float32)
        ego = grp["ego_map"][t0 : t0 + H]
        act = grp["action"][t0 + H : t0 + H + T].astype(np.float32)

        if "slot_mask" in grp:
            slot_mask = grp["slot_mask"][t0 : t0 + H].astype(np.float32)
        else:
            mw = int(grp["map_w"][()]) if "map_w" in grp else 1909
            mh = int(grp["map_h"][()]) if "map_h" in grp else 955
            s_lost = compute_s_lost_norm(mw, mh)
            maxsig = np.maximum(slots[..., 2], slots[..., 3])
            slot_mask = (maxsig < s_lost).astype(np.float32)

        mw = int(grp["map_w"][()]) if "map_w" in grp else 1909
        mh = int(grp["map_h"][()]) if "map_h" in grp else 955

        slot_features = compute_slot_features_normalized(slots, robot, mw, mh)

        if self.clip_actions:
            act = np.clip(act, self.scaler.low, self.scaler.high)
        act_norm = self.scaler.normalize(act).astype(np.float32)

        return {
            "ego_map": torch.from_numpy(ego),
            "actions": torch.from_numpy(act_norm),
            "slot_features": torch.from_numpy(slot_features),
            "slot_mask": torch.from_numpy(slot_mask),
        }

def split_episodes_h5(h5_path: Path, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    with h5py.File(h5_path, 'r') as f:
        all_keys = sorted(list(f.keys()))
        
    n = len(all_keys)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    rng = random.Random(seed)
    rng.shuffle(all_keys)

    test_keys = all_keys[:n_test]
    val_keys = all_keys[n_test : n_test + n_val]
    train_keys = all_keys[n_test + n_val :]
    return train_keys, val_keys, test_keys

def maybe_read_low_high_from_h5(h5_path: Path, keys: List[str]):
    if not keys:
        return None, None
    with h5py.File(h5_path, 'r') as f:
        grp = f[keys[0]]
        if "action_low" in grp and "action_high" in grp:
            low = np.asarray(grp["action_low"][()], dtype=np.float32).ravel()
            high = np.asarray(grp["action_high"][()], dtype=np.float32).ravel()
            return low, high
    return None, None

class DPNpzDataset(Dataset):
    """In-memory dataset built from a directory tree of .npz episode files.

    All episodes are loaded into RAM at construction time, which is fast and
    avoids multiprocessing / file-handle issues.  Suitable for ~300 episodes.
    """

    def __init__(self, episodes: List[Dict], obs_h: int, pred_h: int,
                 scaler: ActionMinMax, clip_actions: bool = True):
        super().__init__()
        self.obs_h = int(obs_h)
        self.pred_h = int(pred_h)
        self.scaler = scaler
        self.clip_actions = bool(clip_actions)
        self.episodes = episodes  # list of dicts with numpy arrays already loaded

        self.index: List[Tuple[int, int]] = []
        for ep_i, ep in enumerate(episodes):
            T_ep = ep["action"].shape[0]
            max_t = T_ep - (self.obs_h + self.pred_h)
            for t0 in range(max_t + 1):
                self.index.append((ep_i, t0))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ep_i, t0 = self.index[i]
        ep = self.episodes[ep_i]
        H, T = self.obs_h, self.pred_h

        robot = ep["robot"][t0 : t0 + H].astype(np.float32)
        slots = ep["slots"][t0 : t0 + H].astype(np.float32)
        ego   = ep["ego_map"][t0 : t0 + H]
        act   = ep["action"][t0 + H : t0 + H + T].astype(np.float32)
        slot_mask = ep["slot_mask"][t0 : t0 + H].astype(np.float32)
        mw = int(ep["map_w"])
        mh = int(ep["map_h"])

        slot_features = compute_slot_features_normalized(slots, robot, mw, mh)

        if self.clip_actions:
            act = np.clip(act, self.scaler.low, self.scaler.high)
        act_norm = self.scaler.normalize(act).astype(np.float32)

        return {
            "ego_map":      torch.from_numpy(ego),
            "actions":      torch.from_numpy(act_norm),
            "slot_features": torch.from_numpy(slot_features),
            "slot_mask":    torch.from_numpy(slot_mask),
        }


def load_npz_episodes(data_dir: Path) -> List[Dict]:
    """Glob all .npz files under data_dir recursively and load into memory."""
    paths = sorted(data_dir.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")
    print(f"[DPNpzDataset] Loading {len(paths)} npz files into memory ...")
    episodes = []
    for p in paths:
        d = dict(np.load(p, allow_pickle=False))
        # compute slot_mask if missing
        if "slot_mask" not in d:
            from utilities.utils import compute_s_lost_norm
            mw = int(d["map_w"])
            mh = int(d["map_h"])
            s_lost = compute_s_lost_norm(mw, mh)
            maxsig = np.maximum(d["slots"][..., 2], d["slots"][..., 3])
            d["slot_mask"] = (maxsig < s_lost).astype(np.float32)
        episodes.append(d)
    print(f"[DPNpzDataset] Loaded {len(episodes)} episodes.")
    return episodes


def split_episodes_npz(episodes: List[Dict], val_ratio: float, test_ratio: float,
                       seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    n = len(episodes)
    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    test_ep  = [episodes[i] for i in idx[:n_test]]
    val_ep   = [episodes[i] for i in idx[n_test : n_test + n_val]]
    train_ep = [episodes[i] for i in idx[n_test + n_val :]]
    return train_ep, val_ep, test_ep


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
    data_dir = Path(data_dir)

    # ── .h5 file path → HDF5 dataset (original behaviour) ──────────
    if data_dir.is_file():
        h5_path = data_dir
        train_keys, val_keys, _ = split_episodes_h5(h5_path, val_ratio, test_ratio, seed)

        low, high = maybe_read_low_high_from_h5(h5_path, train_keys)
        if low is None or high is None:
            low, high = action_low, action_high

        scaler = ActionMinMax(low, high)
        ds_tr = DPHDF5Dataset(h5_path, train_keys, obs_h, pred_h, scaler)
        ds_va = DPHDF5Dataset(h5_path, val_keys,   obs_h, pred_h, scaler) if val_keys else None

    # ── directory → in-memory .npz dataset ─────────────────────────
    elif data_dir.is_dir():
        all_eps = load_npz_episodes(data_dir)

        # infer action bounds from first episode
        low  = np.asarray(all_eps[0]["action_low"],  dtype=np.float32).ravel() if "action_low"  in all_eps[0] else np.asarray(action_low,  dtype=np.float32)
        high = np.asarray(all_eps[0]["action_high"], dtype=np.float32).ravel() if "action_high" in all_eps[0] else np.asarray(action_high, dtype=np.float32)

        scaler = ActionMinMax(low, high)
        train_ep, val_ep, _ = split_episodes_npz(all_eps, val_ratio, test_ratio, seed)

        ds_tr = DPNpzDataset(train_ep, obs_h, pred_h, scaler)
        ds_va = DPNpzDataset(val_ep,   obs_h, pred_h, scaler) if val_ep else None

    else:
        raise ValueError(f"data_dir must be an .h5 file or a directory: {data_dir}")

    persistent = False
    pf = prefetch_factor if num_workers > 0 else None

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
    if ds_va is not None:
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