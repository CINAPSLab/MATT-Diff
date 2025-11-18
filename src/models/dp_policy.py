from __future__ import annotations

import math
import torch
import torch.nn as nn
from diffusers import DDPMScheduler

from .unet1d import ConditionalUnet1D
from .map_encoder import OccPerformerEncoder_v2
from .target_encoder import TargetSetEncoder
from src.common.constants import MAP_WH, K_OUT, S_LOST_NORM
from src.common.utils import ego_to_float01

from typing import Dict, Optional, Tuple


class DiffusionPolicyNetwork(nn.Module):
    """
    ## add comment (English, obey to my paper)
    """

    def __init__(self, action_dim: int, obs_horizon: int, pred_horizon: int,
        *,
        map_emb_dim: int = 256, unet_down_dims: Tuple[int, ...] = (256, 512, 1024),
        unet_kernel_size: int = 5, num_diffusion_iters: int = 1000, use_age: bool = False,):
        super().__init__()
        self.action_dim = int(action_dim)
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)

        self.k_slots = int(K_OUT)
        self.s_lost_norm = float(S_LOST_NORM)
        W, H = MAP_WH
        self.map_w, self.map_h = int(W), int(H)

        self.use_age = bool(use_age)
        self.slot_feat_dim = 8 if self.use_age else 7 


        self.map_encoder = OccPerformerEncoder_v2(
            in_ch=4, in_hw=128, emb_dim=map_emb_dim, heads=8, layers=6,
            dropout=0.0, drop_path=0.1, attn_kernel="rff", rff_dim=128,
            patch32=4, patch16=2, use_rope=True, use_timestep_film=False, return_tokens=False
        )

        # Target Set encoder
        self.tse = TargetSetEncoder(slot_dim=self.slot_feat_dim, d_model=256, nhead=4, num_layers=2, dropout=0.1)

        # normalization
        self.ln_map = nn.LayerNorm(map_emb_dim)
        self.ln_target = nn.LayerNorm(256)

        # Global condition 
        gdim = map_emb_dim + 256
        self.ln_gc = nn.LayerNorm(gdim)

        # Conditional U-net
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=gdim,
            diffusion_step_embed_dim=256,
            down_dims=unet_down_dims,
            kernel_size=unet_kernel_size,
            n_groups=8,
        )

        # DDPM
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )


    @torch.no_grad()
    def _build_slot_mask(self, slots: torch.Tensor) -> torch.Tensor:

        maxsig = slots[..., 2:4].max(dim=-1).values
        return (maxsig < self.s_lost_norm).float()

    @torch.no_grad()
    def _featurize_slots(self, slots_last: torch.Tensor, robot_last: torch.Tensor,
        age_last: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K, _ = slots_last.shape
        W = float(self.map_w)
        HH = float(self.map_h)

        x_px = ((slots_last[..., 0] + 1.0) * 0.5 * (W - 1.0)).clamp(0.0, W - 1.0)
        y_px = ((slots_last[..., 1] + 1.0) * 0.5 * (HH - 1.0)).clamp(0.0, HH - 1.0)

        rx = robot_last[..., 0].unsqueeze(-1)
        ry = robot_last[..., 1].unsqueeze(-1)
        th = robot_last[..., 2].unsqueeze(-1)
        dx = x_px - rx
        dy = y_px - ry
        c = torch.cos(th)
        s = torch.sin(th)
        dx_r =  c * dx + s * dy
        dy_r = -s * dx + c * dy

        phi = torch.atan2(dy_r, dx_r)
        r = torch.sqrt(dx * dx + dy * dy)

        dxn = torch.clamp(dx_r / max(W / 2.0, 1.0), -1.0, 1.0)
        dyn = torch.clamp(dy_r / max(HH / 2.0, 1.0), -1.0, 1.0)
        r_cap = math.sqrt((W / 2.0) ** 2 + (HH / 2.0) ** 2)
        rn = torch.clamp(r / max(r_cap, 1.0), 0.0, 1.0)
        sinp = torch.sin(phi)
        cosp = torch.cos(phi)

        sx = slots_last[..., 2].clamp(0.0, 1.0)
        sy = slots_last[..., 3].clamp(0.0, 1.0)

        feats = [dxn, dyn, rn, sinp, cosp, sx, sy]
        if self.use_age:
            age = torch.zeros_like(dxn) if (age_last is None) else age_last.clamp(0.0, 1.0)
            feats.append(age)

        return torch.stack(feats, dim=-1).to(slots_last.dtype)

    def _encode_map(self, ego_map: torch.Tensor) -> torch.Tensor:
        B, H = ego_map.shape[:2]
        x = ego_map.reshape(B * H, 4, 128, 128)
        x = ego_to_float01(x)
        z_map, _ = self.map_encoder(x)
        return z_map.reshape(B, H, -1).mean(dim=1)

    def _build_global_cond(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ego_map = batch["ego_map"]
        robot   = batch["robot"]
        slots   = batch["slots"][:, :, :self.k_slots, :]

        # map
        z_map = self.ln_map(self._encode_map(ego_map))  

        slots_last = slots[:, -1]
        robot_last = robot[:, -1]
        mask_last  = self._build_slot_mask(slots)[:, -1]
        age_last   = batch.get("age", None)
        if self.use_age and (age_last is not None):
            age_last = age_last[:, -1]
        else:
            age_last = None

        slot_feat = self._featurize_slots(slots_last, robot_last, age_last)
        z_target  = self.ln_target(self.tse(slot_feat, mask_last))

        return self.ln_gc(torch.cat([z_map, z_target], dim=-1))


    ## ------- train forward pass ------- ##
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        actions = batch["actions"]
        B, device = actions.size(0), actions.device

        global_cond = self._build_global_cond(batch)

        noise = torch.randn_like(actions)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
        noisy = self.noise_scheduler.add_noise(actions, noise, t)
        pred = self.noise_pred_net(sample=noisy, timestep=t, global_cond=global_cond)
        return {"pred_noise": pred, "target_noise": noise}


    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor], num_inference_steps: Optional[int] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        B = batch["ego_map"].size(0)
        T = self.pred_horizon

        global_cond = self._build_global_cond(batch)

        actions = torch.randn((B, T, self.action_dim), device=device, generator=getattr(self, "_generator", None))
        if hasattr(self.noise_scheduler, "init_noise_sigma"):
            actions = actions * self.noise_scheduler.init_noise_sigma

        steps = num_inference_steps or self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(steps, device=device)

        for t in self.noise_scheduler.timesteps:
            tt = t if torch.is_tensor(t) else torch.tensor([t], device=device, dtype=torch.long)
            tt = tt.expand(B)
            pred = self.noise_pred_net(sample=actions, timestep=tt, global_cond=global_cond)
            actions = self.noise_scheduler.step(pred, t, actions).prev_sample

        return actions


    def set_generator(self, gen: Optional[torch.Generator]) -> None:
        self._generator = gen