import torch
import torch.nn as nn

from typing import Optional

class TargetSetEncoder(nn.Module):
    """
    """
    def __init__(self, slot_dim: int, d_model: int = 256, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, use_key_padding_mask: bool = True,
                 use_null_replace: bool = True, use_global_fallback: bool = True):
        super().__init__()
        self.use_kpm = use_key_padding_mask
        self.use_null_replace = use_null_replace
        self.use_global_fallback = use_global_fallback
        self.slot_mlp = nn.Sequential(
            nn.Linear(slot_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU()
        )
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_ln = nn.LayerNorm(d_model)
        self.global_null = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, slots_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = slots_feat.size(0)
        tok = self.slot_mlp(slots_feat)
        tok = torch.where(mask.to(torch.bool).unsqueeze(-1), tok, self.null_token.expand(B, 1, -1))
        tok = self.encoder(tok, mask=None, src_key_padding_mask=(mask == 0))
        w = mask.unsqueeze(-1).type_as(tok)
        z = (tok * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        if self.use_global_fallback:
            z = torch.where((mask.sum(dim=1) == 0).unsqueeze(-1), self.global_null.expand(B, -1), z)
        return self.out_ln(z)