from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class EfficientProbing(nn.Module):
    """Probe head kept compatible with training checkpoints."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 4,
        d_out: int = 8,
        use_layernorm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.d_out = d_out

        head_dim = feat_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.norm = nn.LayerNorm(feat_dim // d_out) if use_layernorm else None
        self.v = nn.Linear(feat_dim, feat_dim // d_out, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, feat_dim) * 0.02)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(feat_dim // d_out, num_classes, bias=True)

        self.attention_map: Optional[torch.Tensor] = None

    def forward(self, feats: torch.Tensor, cls: Optional[torch.Tensor] = None) -> torch.Tensor:
        if feats.dim() == 2:
            feats = feats.view(feats.size(0), feats.size(1) // self.feat_dim, self.feat_dim)

        bsz, n_tokens, channels = feats.shape
        cls_dim = channels // self.d_out

        if cls is not None:
            cls_token = cls
        else:
            cls_token = self.cls_token.expand(bsz, -1, -1)

        q = cls_token.reshape(
            bsz, self.num_queries, self.num_heads, channels // self.num_heads
        ).permute(0, 2, 1, 3)
        k = feats.reshape(
            bsz, n_tokens, self.num_heads, channels // self.num_heads
        ).permute(0, 2, 1, 3)
        q = q * self.scale

        v = self.v(feats).reshape(
            bsz,
            n_tokens,
            self.num_queries,
            channels // (self.d_out * self.num_queries),
        ).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attention_map = attn.squeeze(1)

        pooled = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        pooled = pooled.view(bsz, cls_dim)

        if self.norm is not None:
            pooled = self.norm(pooled)
        pooled = self.proj_drop(pooled)
        return self.classifier(pooled)

