from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import DDPMScheduler

from .unet1d import ConditionalUnet1D
from .map_encoder import OccPerformerEncoder_v2
from .target_encoder import TargetSetEncoder
from src.common.constants import K_OUT
from src.common.utils import ego_to_float01

from typing import Dict, Optional, Tuple


class DiffusionPolicyNetwork(nn.Module):

    def __init__(self, action_dim: int, obs_horizon: int, pred_horizon: int,
        *,
        map_emb_dim: int = 256, unet_down_dims: Tuple[int, ...] = (256, 512, 1024),
        unet_kernel_size: int = 5, num_diffusion_iters: int = 1000, use_age: bool = False,):
        super().__init__()
        self.action_dim = int(action_dim)
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)

        self.k_slots = int(K_OUT)
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

        # Temporal projection for map embeddings
        self.map_temporal_proj = nn.Linear(obs_horizon * map_emb_dim, map_emb_dim)

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



    def _encode_map(self, ego_map: torch.Tensor) -> torch.Tensor:
        B, H = ego_map.shape[:2]
        x = ego_map.reshape(B * H, 4, 128, 128)
        x = ego_to_float01(x)
        z_map, _ = self.map_encoder(x)
        return self.map_temporal_proj(z_map.reshape(B, -1))

    def _build_global_cond(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ego_map = batch["ego_map"]
        slot_features = batch["slot_features"][:, :, :self.k_slots, :]
        slot_mask = batch["slot_mask"][:, :, :self.k_slots]

        # map
        z_map = self.ln_map(self._encode_map(ego_map))  

        slot_feat_last = slot_features[:, -1]   # [B, K, 7]
        mask_last = slot_mask[:, -1]             # [B, K]

        z_target = self.ln_target(self.tse(slot_feat_last, mask_last))
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