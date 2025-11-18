import math
from typing import Union, Optional
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        g = n_groups
        for cand in [n_groups, 4, 2, 1]:
            if out_ch % cand == 0:
                g = cand
                break
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(g, out_ch),
            nn.Mish(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3, n_groups: int = 8):
        
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : [B, in_channels,  T]
        cond : [B, cond_dim]
        out  : [B, out_channels, T]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias  = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim: int, global_cond_dim: int, diffusion_step_embed_dim: int = 256,
                 down_dims=(256, 512, 1024), kernel_size: int = 5,n_groups: int = 8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,sample: torch.Tensor, timestep: Union[torch.Tensor, float, int],
                global_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        x = sample.moveaxis(-1, -2)

        t = timestep
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        elif t.ndim == 0:
            t = t[None].to(x.device)
        t = t.expand(x.shape[0])

        gfeat = self.diffusion_step_encoder(t)
        if global_cond is not None:
            gfeat = torch.cat([gfeat, global_cond], dim=-1)

        h = []
        for res1, res2, down in self.down_modules:
            x = res1(x, gfeat)
            x = res2(x, gfeat)
            h.append(x)
            x = down(x)

        for mid in self.mid_modules:
            x = mid(x, gfeat)

        for res1, res2, up in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = res1(x, gfeat)
            x = res2(x, gfeat)
            x = up(x)

        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)
        return x
