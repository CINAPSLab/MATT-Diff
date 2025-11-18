from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os, math, time, json, argparse, random
from pathlib import Path
from dataclasses import dataclass

from src.datasets.dp_dataset import make_loaders, ActionMinMax
from src.models.dp_policy import DiffusionPolicyNetwork

from typing import Dict, List, Tuple, Optional

@dataclass
class Config:
    data_dir: Path
    out_dir: Path
    epochs: int = 30
    n_freeze: int = 10
    batch_size: int = 32
    grad_clip: float = 1.0
    lr_unet: float = 3e-4
    lr_map: float = 3e-5
    weight_decay_unet: float = 1e-4
    weight_decay_map: float = 1e-5
    warmup_steps: int = 800
    obs_h: int = 6
    pred_h: int = 16
    num_diffusion_iters: int = 1000
    num_workers: int = 4
    prefetch_factor: int = 2
    seed: int = 42
    amp_dtype: torch.dtype = torch.bfloat16
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    map_emb_dim: int = 256
    act_low: Tuple[float, float] = (0.0, -1.2)
    act_high: Tuple[float, float] = (13.0, 1.2)



def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / warmup
    prog = (step - warmup) / (total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def write_csv_header(path: Path):
    with open(path, "w") as f:
        f.write("step,epoch,phase,loss,lr_unet,lr_map\n")


def append_csv(path: Path, step: int, epoch: int, phase: str,
               loss: float, lr_unet: float, lr_map: float):
    with open(path, "a") as f:
        f.write(f"{step},{epoch},{phase},{loss:.6f},{lr_unet:.6e},{lr_map:.6e}\n")



def make_optim(cfg: Config, policy: DiffusionPolicyNetwork):
    return torch.optim.AdamW(
        [
            {"params": policy.noise_pred_net.parameters(),
             "lr": cfg.lr_unet, "weight_decay": cfg.weight_decay_unet},

            {"params": policy.tse.parameters(),
             "lr": cfg.lr_unet, "weight_decay": cfg.weight_decay_unet},

            {"params": policy.map_encoder.parameters(),
             "lr": cfg.lr_map, "weight_decay": cfg.weight_decay_map},
        ]
    )


def make_sched(cfg: Config, opt: torch.optim.Optimizer, steps_per_epoch: int):

    total = cfg.epochs * steps_per_epoch
    unfreeze_step = cfg.n_freeze * steps_per_epoch

    def lr_unet(s):
        return cosine_with_warmup(s, cfg.warmup_steps, total)

    def lr_map(s):
        if s < unfreeze_step:
            return 0.0
        t = s - unfreeze_step
        rest = total - unfreeze_step
        if t < cfg.warmup_steps // 2:
            return t / (cfg.warmup_steps // 2)
        prog = (t - cfg.warmup_steps // 2) / (rest - cfg.warmup_steps // 2)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=[lr_unet, lr_unet, lr_map]
    )



def train_loop(cfg: Config, device: torch.device, policy: DiffusionPolicyNetwork,
    opt: torch.optim.Optimizer, sched: torch.optim.lr_scheduler.LambdaLR,
    dl_tr: DataLoader, dl_va: Optional[DataLoader], log_path: Path,):

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    step = 0
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):

        train_map = epoch > cfg.n_freeze
        for p in policy.map_encoder.parameters():
            p.requires_grad = train_map
        policy.map_encoder.train(train_map)

        policy.train()
        for batch in dl_tr:
            step += 1

            batch = {k: v.to(device, non_blocking=True)
                     for k, v in batch.items()}

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=cfg.amp_dtype):
                out = policy(batch)
                loss = F.mse_loss(out["pred_noise"], out["target_noise"])

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            append_csv(
                log_path,
                step, epoch, "train",
                loss.item(),
                opt.param_groups[0]["lr"],
                opt.param_groups[2]["lr"],
            )

        if dl_va is not None:
            policy.eval()
            vsum = 0.0
            cnt = 0
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=cfg.amp_dtype):
                for batch in dl_va:
                    batch = {k: v.to(device, non_blocking=True)
                             for k, v in batch.items()}
                    out = policy(batch)
                    vsum += F.mse_loss(out["pred_noise"], out["target_noise"]).item()
                    cnt += 1
            val_loss = vsum / cnt
            append_csv(
                log_path,
                step, epoch, "val",
                val_loss,
                opt.param_groups[0]["lr"],
                opt.param_groups[2]["lr"],
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "cfg": cfg.__dict__,
                        "epoch": epoch,
                        "step": step,
                        "best_val": best_val,
                    },
                    cfg.out_dir / "best.pt",
                )

        torch.save(
            {
                "policy": policy.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "step": step,
                "best_val": best_val,
            },
            cfg.out_dir / "last.pt",
        )



def main(cfg: Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    device = torch.device("cuda")

    dl_tr, dl_va, scaler = make_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        obs_h=cfg.obs_h,
        pred_h=cfg.pred_h,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
        action_low=cfg.act_low,
        action_high=cfg.act_high,
    )

    policy = DiffusionPolicyNetwork(
        action_dim=2,
        obs_horizon=cfg.obs_h,
        pred_horizon=cfg.pred_h,
        map_emb_dim=cfg.map_emb_dim,
        unet_down_dims=(256, 512, 1024),
        unet_kernel_size=5,
        num_diffusion_iters=cfg.num_diffusion_iters,
        use_age=False,
    ).to(device)

    opt = make_optim(cfg, policy)
    sched = make_sched(cfg, opt, steps_per_epoch=len(dl_tr))

    log_path = cfg.out_dir / "train_log.csv"
    write_csv_header(log_path)

    train_loop(
        cfg=cfg,
        device=device,
        policy=policy,
        opt=opt,
        sched=sched,
        dl_tr=dl_tr,
        dl_va=dl_va,
        log_path=log_path,
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("src/run/dp"))

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n_freeze", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lr_unet", type=float, default=3e-4)
    ap.add_argument("--lr_map", type=float, default=3e-5)
    ap.add_argument("--weight_decay_unet", type=float, default=1e-4)
    ap.add_argument("--weight_decay_map", type=float, default=1e-5)
    ap.add_argument("--warmup_steps", type=int, default=800)

    ap.add_argument("--obs_h", type=int, default=6)
    ap.add_argument("--pred_h", type=int, default=16)
    ap.add_argument("--num_diffusion_iters", type=int, default=1000)

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--map_emb_dim", type=int, default=256)

    ap.add_argument("--act_low", nargs="+", type=float, default=(0.0, -1.2))
    ap.add_argument("--act_high", nargs="+", type=float, default=(13.0, 1.2))

    args = ap.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        n_freeze=args.n_freeze,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        lr_unet=args.lr_unet,
        lr_map=args.lr_map,
        weight_decay_unet=args.weight_decay_unet,
        weight_decay_map=args.weight_decay_map,
        warmup_steps=args.warmup_steps,
        obs_h=args.obs_h,
        pred_h=args.pred_h,
        num_diffusion_iters=args.num_diffusion_iters,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        map_emb_dim=args.map_emb_dim,
        act_low=tuple(args.act_low),
        act_high=tuple(args.act_high),
    )

    main(cfg)