from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

import os
import glob
import csv
import random
import argparse
from dataclasses import dataclass
import time
from datetime import datetime

from src.models.bc_policy import BCSlotsBaseline

from typing import List, Dict, Any, Optional, Tuple



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def sizeof_fmt(num: float) -> str:
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}B"
        num /= 1024.0
    return f"{num:.1f}EB"



class NpzMultiSampleReader:
    def __init__(self, path: str):
        self.path = path
        with np.load(self.path, allow_pickle=False, mmap_mode=None) as z:
            ego = z["ego_map"]
            self.length = int(ego.shape[0]) if ego.ndim == 4 else 1

    def __len__(self) -> int:
        return self.length

    def get(self, idx: int) -> Dict[str, np.ndarray]:
        with np.load(self.path, allow_pickle=False, mmap_mode=None) as z:
            if self.length == 1:
                item = {
                    "ego_map": z["ego_map"],
                    "slots": z["slots"],
                    "action": z["action"],
                }
                if "robot" in z.files:
                    item["robot"] = z["robot"]
            else:
                item = {
                    "ego_map": z["ego_map"][idx],
                    "slots": z["slots"][idx],
                    "action": z["action"][idx],
                }
                if "robot" in z.files:
                    item["robot"] = z["robot"][idx]
        return item


class FolderNPZDataset(Dataset):
    def __init__(self, data_dir: str, require_robot: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.require_robot = require_robot

        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        self.readers: List[NpzMultiSampleReader] = []
        self.index: List[Tuple[int, int]] = []
        total = 0
        print("[dataset] scanning .npz files...")
        for f in files:
            r = NpzMultiSampleReader(f)
            n = len(r)
            if n == 0:
                continue
            self.readers.append(r)
            rid = len(self.readers) - 1
            self.index.extend([(rid, i) for i in range(n)])

            try:
                size_bytes = os.path.getsize(f)
            except Exception:
                size_bytes = 0
            total += n
            print(f"  - {os.path.basename(f)}: {n} samples ({sizeof_fmt(size_bytes)})")
        print(f"[dataset] total samples: {total}")

        # light validation
        x0 = self[0]
        ego = x0["ego_map"]
        if ego.ndim == 3:
            assert ego.shape[0] == 4, "ego_map must be (4,128,128) or (N,4,128,128)"
        else:
            assert ego.shape[1] == 4, "ego_map batch first unexpected"
        assert ego.dtype == np.uint8, "ego_map must be uint8"
        assert x0["slots"].ndim == 2 and x0["slots"].shape[1] == 4, "slots shape unexpected"
        assert x0["action"].shape[-1] == 2, "action shape unexpected"
        if require_robot:
            assert "robot" in x0 and x0["robot"].shape[-1] == 3, "robot is required but missing"

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        rid, li = self.index[idx]
        try:
            return self.readers[rid].get(li)
        except Exception as e:
            print(f"[warn] skip sample: file={self.readers[rid].path}, idx={li}, err={e}")
            j = (idx + 1) % len(self.index)
            rid2, li2 = self.index[j]
            return self.readers[rid2].get(li2)


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    ego = np.stack([b["ego_map"] for b in batch], axis=0)
    if ego.ndim == 3:
        ego = np.expand_dims(ego, 0)
    ego_t = torch.from_numpy(ego)  # uint8

    slt = torch.from_numpy(np.stack([b["slots"] for b in batch], axis=0)).float()
    act = torch.from_numpy(np.stack([b["action"] for b in batch], axis=0)).float()

    if "robot" in batch[0]:
        rbt = torch.from_numpy(np.stack([b["robot"] for b in batch], axis=0)).float()
    else:
        rbt = None

    return {"ego_map": ego_t, "slots": slt, "robot": rbt, "action": act}



@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-4
    wd: float = 1e-6
    workers: int = 8
    val_ratio: float = 0.1
    seed: int = 42
    amp: bool = True
    use_robot: bool = False
    resume: Optional[str] = None 
    reset_optim: bool = False
    terse: bool = False


@torch.no_grad()
def _peak_mem_log(tag: str = "") -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / 1e9
        reserv = torch.cuda.max_memory_reserved() / 1e9
        print(f"[mem{(':'+tag) if tag else ''}] alloc_peak={alloc:.2f}GB reserved_peak={reserv:.2f}GB")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True,) -> Dict[str, float]:
    model.eval()
    mse_loss = 0.0
    n = 0
    abs_err_v = 0.0
    abs_err_w = 0.0

    with torch.no_grad():
        for batch in loader:
            ego = batch["ego_map"].to(device, non_blocking=True)
            slt = batch["slots"].to(device, non_blocking=True)
            act = batch["action"].to(device, non_blocking=True)
            rbt = batch["robot"].to(device, non_blocking=True) if (batch["robot"] is not None) else None

            if amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = model(ego, slt, rbt)
                    loss = F.mse_loss(pred, act, reduction="sum")
            else:
                pred = model(ego, slt, rbt)
                loss = F.mse_loss(pred, act, reduction="sum")

            mse_loss += loss.item()
            n += act.shape[0]
            ae = (pred - act).abs()
            abs_err_v += ae[:, 0].sum().item()
            abs_err_w += ae[:, 1].sum().item()

    return {
        "mse": mse_loss / max(n, 1),
        "mae_v": abs_err_v / max(n, 1),
        "mae_w": abs_err_w / max(n, 1),
        "n": n,
    }


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device("cuda:0")
    print(f"[device] {device}, name={torch.cuda.get_device_name(0)}")

    ds = FolderNPZDataset(cfg.data_dir, require_robot=cfg.use_robot)
    val_len = max(1, int(len(ds) * cfg.val_ratio))
    train_len = len(ds) - val_len
    ds_train, ds_val = random_split(
        ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    print(f"[split] train={train_len}, val={val_len}")

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(cfg.workers > 0),
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(1, cfg.workers // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=(cfg.workers > 0),
        prefetch_factor=2,
    )

    model = BCSlotsBaseline(use_robot=cfg.use_robot).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    os.makedirs(cfg.out_dir, exist_ok=True)

    log_path = os.path.join(cfg.out_dir, "log.csv")
    write_header = not os.path.exists(log_path)
    if write_header:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_mse", "val_mse", "val_mae_v", "val_mae_w", "n_val"])

    start_epoch = 0
    best_val = float("inf")
    if cfg.resume is not None and os.path.isfile(cfg.resume):
        print(f"[resume] loading checkpoint from: {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
            if not cfg.reset_optim and "optim" in ckpt:
                try:
                    optim.load_state_dict(ckpt["optim"])
                except Exception as e:
                    print(f"[resume] optimizer load failed ({e}); continue with fresh optimizer.")
            if cfg.amp and (not cfg.reset_optim) and "scaler" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception as e:
                    print(f"[resume] scaler load failed ({e}); continue with fresh scaler.")
            start_epoch = int(ckpt.get("epoch", 0))
            best_val = float(ckpt.get("val_mse", float("inf")))
            print(f"[resume] start_epoch={start_epoch}, best_val={best_val:.6f}")
        else:
            print("[resume] checkpoint format unexpected; training from scratch.")
        model.to(device)
    else:
        print("[resume] no checkpoint loaded; training from scratch.")

    if cfg.epochs <= start_epoch:
        print(f"[warn] target epochs ({cfg.epochs}) <= start_epoch ({start_epoch}). nothing to do.")
        print(f"CSV log: {log_path}")
        return

    # optional warmup batch
    if cfg.terse:
        t0 = time.time()
        try:
            b0 = next(iter(train_loader))
            _ego = b0["ego_map"].to(device, non_blocking=True)
            _slt = b0["slots"].to(device, non_blocking=True)
            _rbt = b0["robot"].to(device, non_blocking=True) if (b0["robot"] is not None) else None
            model.eval()
            with torch.no_grad():
                if cfg.amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        _ = model(_ego, _slt, _rbt)
                else:
                    _ = model(_ego, _slt, _rbt)
            torch.cuda.synchronize()
        except StopIteration:
            pass
        print(f"SingleProcess AUTOTUNE benchmarking takes {time.time() - t0:.4f} seconds")

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss, n = 0.0, 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if cfg.terse:
            iterator = train_loader
            for batch in iterator:
                ego = batch["ego_map"].to(device, non_blocking=True)
                slt = batch["slots"].to(device, non_blocking=True)
                act = batch["action"].to(device, non_blocking=True)
                rbt = batch["robot"].to(device, non_blocking=True) if (batch["robot"] is not None) else None

                optim.zero_grad(set_to_none=True)
                if cfg.amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        pred = model(ego, slt, rbt)
                        loss = F.mse_loss(pred, act)
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    pred = model(ego, slt, rbt)
                    loss = F.mse_loss(pred, act)
                    loss.backward()
                    optim.step()

                bs = act.size(0)
                total_loss += loss.item() * bs
                n += bs
        else:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
            for step, batch in enumerate(pbar):
                ego = batch["ego_map"].to(device, non_blocking=True)
                slt = batch["slots"].to(device, non_blocking=True)
                act = batch["action"].to(device, non_blocking=True)
                rbt = batch["robot"].to(device, non_blocking=True) if (batch["robot"] is not None) else None

                optim.zero_grad(set_to_none=True)
                if cfg.amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        pred = model(ego, slt, rbt)
                        loss = F.mse_loss(pred, act)
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    pred = model(ego, slt, rbt)
                    loss = F.mse_loss(pred, act)
                    loss.backward()
                    optim.step()

                bs = act.size(0)
                total_loss += loss.item() * bs
                n += bs
                pbar.set_postfix({"train_mse(avg)": f"{(total_loss / max(n, 1)):.6f}"})

                if (step % 200) == 0 and torch.cuda.is_available():
                    _peak_mem_log(tag=f"ep{epoch}-step{step}")

        train_mse = total_loss / max(n, 1)

        val_metrics = evaluate(model, val_loader, device, amp=cfg.amp)
        val_mse = val_metrics["mse"]
        val_mae_v = val_metrics["mae_v"]
        val_mae_w = val_metrics["mae_w"]
        n_val = val_metrics["n"]

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_mse, val_mse, val_mae_v, val_mae_w, n_val])

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": vars(cfg),
            "val_mse": val_mse,
        }
        torch.save(ckpt, os.path.join(cfg.out_dir, "ckpt_last.pt"))
        if val_mse < best_val:
            best_val = val_mse
            torch.save(ckpt, os.path.join(cfg.out_dir, "ckpt_best.pt"))

        if cfg.terse:
            sec = time.time() - epoch_start
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] epoch {epoch:03d}/{cfg.epochs}  train={train_mse:.6f}  val={val_mse:.6f}  time={sec:.1f}s")
        else:
            print(f"[val] mse={val_mse:.6f}, mae_v={val_mae_v:.6f}, mae_w={val_mae_w:.6f}, n={n_val}")
            _peak_mem_log(tag=f"ep{epoch}")

    print("[done] training finished.")
    print(f"CSV log: {log_path}")



def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory containing .npz files")
    ap.add_argument("--out_dir", type=str, default="./run/bc/")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    ap.add_argument("--use_robot", action="store_true", help="Use obs['robot'] as input")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint (ckpt_last.pt or ckpt_best.pt)")
    ap.add_argument("--reset_optim", action="store_true", help="Ignore optimizer/scaler states when resuming")
    ap.add_argument("--terse", action="store_true", help="Minimal console log per epoch")
    args = ap.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        workers=args.workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        amp=not args.no_amp,
        use_robot=args.use_robot,
        resume=args.resume,
        reset_optim=args.reset_optim,
        terse=args.terse,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)