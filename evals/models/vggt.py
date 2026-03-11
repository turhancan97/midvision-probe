import torch
from .utils import center_padding, tokens_to_output
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
        return_cls=False,
        mean_pool=False,
        efficient_probe=False,
    ):
        super().__init__()
        
        # get model
        self.arch = "vit"
        self.return_cls = return_cls
        self.mean_pool = mean_pool
        self.efficient_probe = efficient_probe
        self.add_norm = add_norm  # Store the flag to control batch normalization
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

    def forward(self, images):
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
