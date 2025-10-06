import torch

from .utils import center_padding, tokens_to_output
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_tensor
import torch.nn as nn

import sys


class VGGT1B(torch.nn.Module):
    def __init__(
        self,
        checkpoint="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        repo_dir="",
        output="dense",
        layer=-1,
        return_multilayer=False,
        add_norm=False,
        return_kqv=False,  # Flag to return K, Q, V
        fixed_size=480,
        mode_selected="k",
        return_layers=None,
        return_cls=False,
        mean_pool=False,
        efficient_probe=False,
    ):
        super().__init__()
        
        # get model
        self.return_cls = return_cls
        self.mean_pool = mean_pool
        self.repo_dir = repo_dir
        self.checkpoint_name = checkpoint
        sys.path.append(self.repo_dir)
        from vggt.models.vggt import VGGT
        model = VGGT()
        model.load_state_dict(torch.hub.load_state_dict_from_url(self.checkpoint_name))
        self.vit = model.eval().to(torch.float32)

        self.patch_size = self.vit.aggregator.patch_embed.patch_embed.proj.kernel_size[0]
        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output

        feat_dim = 1024
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.aggregator.patch_embed.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

        # Define BatchNorm1d layers for each selected layer
        if output == "dense-cls":
            feat_dim = feat_dim // 2
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )
        self.add_norm = add_norm  # Store the flag to control batch normalization
        self.return_kqv = return_kqv  # Store the flag to return K, Q, V
        self.fixed_size = fixed_size
        self.resize_transform = Resize((fixed_size, fixed_size))
        self.mode_selected = mode_selected
        self.efficient_probe = efficient_probe

    def extract_kqv(self, images):
        """Helper function to extract K, Q, V from the last attention layer"""
        feat_out = {}
        bs = images.shape[0]
        feat_h, feat_w = (
            images.shape[-2] // self.patch_size,
            images.shape[-1] // self.patch_size,
        )
        if images.ndim == 3:  # Check if the input is 3D (C, H, W)
            images = images.unsqueeze(0)  # Add batch dimension to make it (1, C, H, W)
        if images.ndim == 5:  # Check if the input is 3D (C, H, W)
            images = images.squeeze(0)  # Add batch dimension to make it (1, C, H, W)

        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output

        self.vit._modules["blocks"][-1]._modules["attn"]._modules[
            "qkv"
        ].register_forward_hook(hook_fn_forward_qkv)

        # Forward pass
        with torch.no_grad():
            x = self.vit.prepare_tokens(images)
            for i, blk in enumerate(self.vit.blocks):
                if i < len(self.vit.blocks) - 1:
                    x = blk(x)
                else:
                    x = blk(x, return_attention=True)

        bs, nb_head, nb_token = (
            x.shape[0],
            x.shape[1],
            x.shape[2],
        )

        qkv = (
            feat_out["qkv"].reshape(bs, nb_token, 3, nb_head, -1).permute(2, 0, 3, 1, 4)
        )
        # torch.Size([1, 1, 901, 3, 256])
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k.transpose(1, 2).reshape(bs, nb_token, -1)
        q = q.transpose(1, 2).reshape(bs, nb_token, -1)
        v = v.transpose(1, 2).reshape(bs, nb_token, -1)

        if self.mode_selected == "k":
            feats = k[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
        elif self.mode_selected == "q":
            feats = q[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
        elif self.mode_selected == "v":
            feats = v[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
        elif self.mode_selected == "kqv":
            k = k[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
            q = q[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
            v = v[:, 1:].transpose(1, 2).reshape(bs, self.feat_dim, feat_h * feat_w)
            feats = torch.cat([k, q, v], dim=1)
        return feats

    def preprocess_image(self, rgb_image):
        """
        Preprocess the RGB image tensor inside the DINO class.
        Args:
        - rgb_image: Tensor of shape (C, H, W) representing the RGB image.

        Returns:
        - tensor: Processed tensor ready to be fed into ViT (1, C, H, W).
        - feat_w, feat_h: Feature width and height after patch embedding.
        """
        # Resize the image to the fixed size (fixed_size x fixed_size)
        rgb_resized = self.resize_transform(rgb_image)

        # Calculate feature map dimensions after patch embedding
        feat_w, feat_h = (
            self.fixed_size // self.patch_size,
            self.fixed_size // self.patch_size,
        )

        # Unsqueeze to add batch dimension and return
        tensor = rgb_resized.unsqueeze(0)  # Shape: (1, C, H, W)

        return tensor, feat_w, feat_h

    def forward(self, images):
        if self.return_kqv:
            processed_tensor, feat_w, feat_h = self.preprocess_image(images)

            feats = self.extract_kqv(processed_tensor)
            return feats

        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        x = self.vit.aggregator.patch_embed.prepare_tokens_with_masks(images, None)

        embeds = []
        for i, blk in enumerate(self.vit.aggregator.patch_embed.blocks):
            x = blk(x)
            if i in self.multilayers:
                if self.add_norm:
                    x_batched = self.batchnorms[self.multilayers.index(i)](
                        x.permute(0, 2, 1)
                    ).permute(
                        0, 2, 1
                    )  # Exclude the class token
                    embeds.append(x_batched)
                else:
                    embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):

            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        embeds_patch = embeds[0][:, (-1 * num_spatial):]
        embeds_cls = embeds[0][:, :1]
        embeds = [torch.cat([embeds_cls, embeds_patch], dim=1)]
        if self.efficient_probe:
            return embeds[0]
        if len(outputs) == 1 and self.mean_pool and not self.return_cls:
            return embeds[0][:, 1:].mean(dim=1)
        elif len(outputs) == 1 and self.return_cls and not self.mean_pool:
            return embeds[0][:, 0]
        elif len(outputs) == 1 and self.mean_pool and self.return_cls:  
            return embeds[0].mean(dim=1)
        else:
            pass

        return outputs[0] if len(outputs) == 1 else outputs
