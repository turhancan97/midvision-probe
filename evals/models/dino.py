import torch
from .utils import center_padding, tokens_to_output
import torch.nn as nn
import torch.nn.functional as F


class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dino",
        model_name="vitb16",
        repo_dir="",
        weights="",
        output="dense",
        layer=-1,
        return_multilayer=False,
        add_norm=False,
        return_cls=False,
        mean_pool=False,
        efficient_probe=False,
    ):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitl14_reg": 1024,
            "vitg14": 1536,
        }

        # get model
        self.arch = "vit"
        self.return_cls = return_cls
        self.mean_pool = mean_pool
        self.efficient_probe = efficient_probe
        self.add_norm = add_norm
        self.repo_dir = repo_dir
        self.weights = weights
        self.dino_name = dino_name
        self.model_name = model_name

        if dino_name == "dinov3":
            REPO_DIR = self.repo_dir
            dino_vit = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local',
                               weights=self.weights)
            self.checkpoint_name = 'dinov3_vitb16'
        else:
            self.checkpoint_name = f"{dino_name}_{model_name}"
            dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)

        self.vit = dino_vit.eval().to(torch.float32)
        self.has_registers = "_reg" in model_name

        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]
        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
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

    def forward(self, images):
        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if self.dino_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        elif self.dino_name == "dinov3":
            x, (H, W) = self.vit.prepare_tokens_with_masks(images, None)
            rope_sincos = self.vit.rope_embed(H=H, W=W)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            if self.dino_name == "dinov3":
                x = blk(x, rope_sincos)
            else:
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
