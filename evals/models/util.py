import os
import ssl

import gdown
import timm.models as tm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wget
import numpy as np
from typing import Final
from time import time
from mmselfsup.models.backbones import ResNet

ckpt_dir = "pretrained_models"
os.makedirs(ckpt_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


class MMSelfSupResnet50(ResNet):
    def __init__(self):
        super().__init__(depth=50)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = super().forward(x)[0]
        return self.adaptive_pool(x).squeeze(-1).squeeze(-1)


def initialize_backbone(arch: str, **kwargs):
    if arch == "resnet50":
        model = torchvision.models.resnet50(weights=None, **kwargs)
    elif arch == "vits16":
        model = tm.vit_small_patch16_224(pretrained=False, **kwargs)
    elif arch == "vitb16":
        model = tm.vit_base_patch16_224(pretrained=False, **kwargs)
    elif arch == "vitl16":
        model = tm.vit_large_patch16_224(pretrained=False, **kwargs)
    elif arch == "vits8":
        model = tm.vision_transformer.VisionTransformer(
            patch_size=8, embed_dim=384, depth=12, num_heads=6, **kwargs
        )
    elif arch == "vitb8":
        model = tm.vision_transformer.VisionTransformer(
            patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs
        )
    else:
        raise NotImplementedError(f"Arch {arch} not implemented for now.")

    if arch.startswith("vit"):
        model.head = torch.nn.Identity()  # remove the original head
    else:
        model.fc = torch.nn.Identity()  # remove the final fc layer
    return model


def load_checkpoint(url: str, filename: str):
    path = os.path.join(ckpt_dir, filename)
    # print("checkpoint path", path)
    if not os.path.exists(path):
        print(f"Downloading checkpoint file: {url}")
        if url.startswith("https://drive.google.com"):
            gdown.download(url, path, quiet=False, fuzzy=True)
        else:
            ssl._create_default_https_context = ssl._create_unverified_context
            wget.download(url, path)
    ckpt = torch.load(path, map_location=device)
    return ckpt


def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def unfreeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True
    model.train()


def partially_freeze_model(model: torch.nn.Module):
    """Freeze first half of the model"""
    unfreeze_model(model)
    if model.__class__.__name__ == "VisionTransformer":
        for param in model.patch_embed.parameters():
            param.requires_grad = False
        for param in model.pos_embed.parameters():
            param.requires_grad = False
        for param in model.blocks[: len(model.blocks) // 2].parameters():
            param.requires_grad = False
    elif model.__class__.__name__ == "ResNet":
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.maxpool.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
    else:
        raise NotImplementedError(
            f"Model {model.__class__.__name__} not implemented for now."
        )


def prepare_state_dict(
    state_dict, remove_prefix=None, delete_prefixes=("head.", "fc.")
):
    for k in list(state_dict.keys()):
        if remove_prefix is not None:
            if k.startswith(remove_prefix):
                state_dict[k[len(remove_prefix) :]] = state_dict[k]
                del state_dict[k]

    if delete_prefixes is not None:
        for k in list(state_dict.keys()):
            for del_prefix in delete_prefixes:
                if k.startswith(del_prefix):
                    del state_dict[k]
    return state_dict

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f'{dim=} should be divisible by {num_heads=}'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False  # use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        # self.cls_bias = None

    def forward(self, x: torch.Tensor, temperature: float=1) -> torch.Tensor:
        s0 = time()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            assert False
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn / temperature
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn