import torch
import torch.nn.functional as F
from mmcls.models import VisionTransformer as MMViT
import torch.nn as nn
from .util import load_checkpoint, prepare_state_dict

GLOBAL_POOL = False

checkpoints = {
    "vitb16": {
        "url": "https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221101-6dfc8bf3.pth",
        "filename": "maskfeat_vitb16.pth",
    },
}


class MMSelfSupMaskFeatViT(MMViT):
    def __init__(self, global_pool=GLOBAL_POOL):
        super().__init__(img_size=224, patch_size=16)
        self.global_pool = global_pool

    def forward(self, x):
        x = super().forward(x)
        if self.global_pool:
            x = x[0][0]  # Take the CLS token
            return x.reshape(x.shape[0], 768, -1).mean(dim=-1)
        else:
            return x[0][1]  # Return the spatial features


def load_model(arch: str, global_pool=GLOBAL_POOL, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = MMSelfSupMaskFeatViT(global_pool=global_pool)
    ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
    ckpt = prepare_state_dict(
        ckpt,
        remove_prefix="backbone.",
        delete_prefixes=["target_generator.", "neck.", "mask_token"],
    )
    model.load_state_dict(ckpt)

    return model


class MASKFEAT(torch.nn.Module):
    def __init__(
        self,
        model_name="maskfeat_vitb16",
        arch="vitb16",
        output="dense",
        layer=-1,
        global_pool=GLOBAL_POOL,
        return_multilayer=False,
        add_norm=False,
        return_cls=False,
        mean_pool=False,
        efficient_probe=False,
    ):
        super().__init__()
        self.arch = "vit"
        self.return_cls = return_cls
        self.mean_pool = mean_pool
        self.efficient_probe = efficient_probe

        self.model = load_model(arch, global_pool=global_pool)
        self.output = output
        feat_dim = 768  # ViT-B/16 has a dimension of 768
        self.patch_size = self.model.patch_embed.projection.kernel_size[0]
        self.add_norm = add_norm

        num_layers = len(self.model.layers)
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
        self.layer = "-".join(str(_x) for _x in self.multilayers)
        self.checkpoint_name = f"$maskfeat$_{model_name}_{output}_{self.layer}"

        # Define BatchNorm2d layers for each multilayer
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )

    def forward(self, images):

        # Forward pass through the patch embedding and position embedding
        x, patch_resolution = self.model.patch_embed(images)

        # Add class token if it exists
        if self.model.cls_token is not None:
            cls_token = self.model.cls_token.expand(images.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x = x + self.model.resize_pos_embed(
            self.model.pos_embed,
            self.model.patch_resolution,
            patch_resolution,
            mode=self.model.interpolate_mode,
            num_extra_tokens=self.model.num_extra_tokens,
        )
        x = self.model.drop_after_pos(x)

        embeds = []
        for i, blk in enumerate(self.model.layers):
            x = blk(x)
            if i in self.multilayers:
                if self.add_norm:
                    x_batched = self.batchnorms[self.multilayers.index(i)](
                        x.permute(0, 2, 1)
                    ).permute(
                        0, 2, 1
                    )
                    embeds.append(x_batched)
                else:
                    embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for x_i in embeds:
            x_i = x_i[:, 1:]
            b, n, c = x_i.shape
            h = w = int(n**0.5)  # Assuming square spatial dimensions (e.g., 14x14)
            x_i = x_i.permute(0, 2, 1).contiguous().view(b, c, h, w)
            outputs.append(x_i)
        
        if self.efficient_probe:
            return embeds[0]
        
        if len(self.multilayers) == 1 and self.mean_pool and not self.return_cls:
            return embeds[0][:, 1:].mean(dim=1)
        elif len(self.multilayers) == 1 and self.return_cls and not self.mean_pool:
            return embeds[0][:, 0]
        elif len(self.multilayers) == 1 and self.mean_pool and self.return_cls:
            return embeds[0].mean(dim=1)
        else:
            pass

        return outputs[0] if len(outputs) == 1 else outputs
