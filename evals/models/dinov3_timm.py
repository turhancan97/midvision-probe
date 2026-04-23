from __future__ import annotations

import torch
import torch.nn as nn

from .utils import center_padding, tokens_to_output


SUPPORTED_DINOV3_TIMM_MODELS = {
    "vit_small_patch16_dinov3.lvd1689m",
    "vit_base_patch16_dinov3.lvd1689m",
    "vit_large_patch16_dinov3.lvd1689m",
    "vit_huge_plus_patch16_dinov3.lvd1689m",
    "vit_7b_patch16_dinov3.lvd1689m",
}


class DINOV3TIMM(nn.Module):
    """DINOv3 backbone loaded from timm Hugging Face model IDs.

    This wrapper mirrors the interface used across backbone wrappers in the repository
    and supports:
    - final/intermediate layer extraction,
    - cls / gap / dense / dense-cls outputs,
    - return_cls / mean_pool / efficient_probe shortcuts.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        output: str = "dense",
        layer: int = -1,
        return_multilayer: bool = False,
        add_norm: bool = False,
        return_cls: bool = False,
        mean_pool: bool = False,
        efficient_probe: bool = False,
        pretrained: bool = True,
        strict_model_name: bool = True,
    ):
        super().__init__()

        self.arch = "vit"
        self.model_name = model_name
        self.output = output
        self.return_cls = return_cls
        self.mean_pool = mean_pool
        self.efficient_probe = efficient_probe
        self.add_norm = add_norm

        if self.output not in ["cls", "gap", "dense", "dense-cls"]:
            raise ValueError("output must be one of ['cls', 'gap', 'dense', 'dense-cls']")

        if strict_model_name and model_name not in SUPPORTED_DINOV3_TIMM_MODELS:
            allowed = ", ".join(sorted(SUPPORTED_DINOV3_TIMM_MODELS))
            raise ValueError(
                f"Unsupported DINOv3 timm model_name '{model_name}'. "
                f"Allowed: {allowed}"
            )

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "DINOV3TIMM requires timm. Install it with `pip install timm`."
            ) from exc

        model_id = model_name if model_name.startswith("hf_hub:") else f"hf_hub:timm/{model_name}"

        try:
            vit = timm.create_model(model_id, pretrained=pretrained)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load timm model '{model_id}'. "
                "Check network access, model ID, and timm/huggingface compatibility."
            ) from exc

        self.checkpoint_name = f"dinov3_timm_{model_name}"
        self.vit = vit.eval().to(torch.float32)

        patch_size = getattr(self.vit.patch_embed, "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
                raise ValueError(f"Unsupported patch_size format: {patch_size}")
            self.patch_size = int(patch_size[0])
        else:
            self.patch_size = int(patch_size)

        feat_dim = int(getattr(self.vit, "num_features", getattr(self.vit, "embed_dim")))
        out_feat_dim = feat_dim * 2 if self.output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        if num_layers < 1:
            raise ValueError("Backbone has no transformer blocks.")

        default_multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        # Clamp to valid range in case extremely shallow models are used.
        default_multilayers = [max(0, min(num_layers - 1, x)) for x in default_multilayers]

        if return_multilayer:
            self.feat_dim = [out_feat_dim] * len(default_multilayers)
            self.multilayers = default_multilayers
        else:
            chosen = default_multilayers[-1] if layer == -1 else int(layer)
            if chosen < 0 or chosen >= num_layers:
                raise ValueError(f"Invalid layer index {chosen}. Valid range: [0, {num_layers - 1}]")
            self.feat_dim = out_feat_dim
            self.multilayers = [chosen]

        self.layer = "-".join(str(x) for x in self.multilayers)

        # Optional channel-wise normalization over token sequence.
        bn_dim = out_feat_dim // 2 if self.output == "dense-cls" else out_feat_dim
        self.batchnorms = nn.ModuleList([nn.BatchNorm1d(bn_dim) for _ in self.multilayers])

    def _prepare_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, object, object]:
        x = self.vit.patch_embed(images)
        rope = None
        attn_mask = None

        # Preferred timm path for modern ViTs (handles cls/register tokens + pos embedding).
        if hasattr(self.vit, "_pos_embed"):
            x = self.vit._pos_embed(x)
            if isinstance(x, tuple):
                if len(x) >= 1:
                    rope = x[1] if len(x) >= 2 else None
                    attn_mask = x[2] if len(x) >= 3 else None
                    x = x[0]
                else:
                    raise RuntimeError("Unexpected empty tuple from _pos_embed.")
        else:
            # Fallback for older/simple ViT implementations.
            if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
                cls = self.vit.cls_token.expand(images.shape[0], -1, -1)
                x = torch.cat((cls, x), dim=1)
            if hasattr(self.vit, "pos_embed") and self.vit.pos_embed is not None:
                x = x + self.vit.pos_embed

        # Some timm ViT variants expose optional modules that may be set to None.
        patch_drop = getattr(self.vit, "patch_drop", None)
        if patch_drop is not None:
            x = patch_drop(x)
        norm_pre = getattr(self.vit, "norm_pre", None)
        if norm_pre is not None:
            x = norm_pre(x)

        return x, rope, attn_mask

    def forward(self, images: torch.Tensor):
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size
        num_spatial = h * w

        x, rope, attn_mask = self._prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            # Some DINOv3 timm variants (EVA-style) require rope/attn_mask.
            if rope is not None or attn_mask is not None:
                try:
                    x = blk(x, rope=rope, attn_mask=attn_mask)
                except TypeError:
                    x = blk(x)
            else:
                x = blk(x)

            # Be robust to variants that return tuples from blocks.
            if isinstance(x, tuple):
                if len(x) == 0:
                    raise RuntimeError("Transformer block returned an empty tuple.")
                x = x[0]
            if i in self.multilayers:
                if self.add_norm:
                    x_batched = self.batchnorms[self.multilayers.index(i)](x.permute(0, 2, 1)).permute(0, 2, 1)
                    embeds.append(x_batched)
                else:
                    embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        probe_tokens = []
        for x_i in embeds:
            # Always take spatial tokens from the tail; this is robust to cls/register prefix tokens.
            spatial = x_i[:, -num_spatial:]
            cls_tok = x_i[:, 0]

            x_out = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_out)

            # Efficient probing expects [B, 1 + N, C] with cls first and patch tokens after.
            probe_tokens.append(torch.cat([x_i[:, :1], spatial], dim=1))

        if self.efficient_probe:
            return probe_tokens[0] if len(probe_tokens) == 1 else probe_tokens

        if len(outputs) == 1:
            tokens = probe_tokens[0]
            if self.mean_pool and not self.return_cls:
                return tokens[:, 1:].mean(dim=1)
            if self.return_cls and not self.mean_pool:
                return tokens[:, 0]
            if self.mean_pool and self.return_cls:
                return tokens.mean(dim=1)

        return outputs[0] if len(outputs) == 1 else outputs
