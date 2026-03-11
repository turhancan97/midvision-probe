import torch
from torch import nn
from .deit_utils import deit_base_patch16_LS, deit_large_patch16_LS
from .utils import resize_pos_embed, tokens_to_output


class DeIT(torch.nn.Module):
    def __init__(
        self,
        model_size="base",
        img_size=384,
        patch_size=16,
        output="dense",
        layer=-1,
        return_multilayer=False,
        add_norm=False,
        return_cls=False,
        mean_pool=False,
        efficient_probe=False,
    ):
        super().__init__()
        self.arch = "vit"
        assert output in ["cls", "gap", "dense"], "Options: [cls, gap, dense]"
        self.output = output
        self.return_cls = return_cls
        self.mean_pool = mean_pool
        self.checkpoint_name = f"deit3_{model_size}-{patch_size}_{img_size}"
        if model_size == "base":
            vit = deit_base_patch16_LS(True, img_size, True)
        elif model_size == "large":
            vit = deit_large_patch16_LS(True, img_size, True)

        self.vit = vit.eval()
        self.patch_size = patch_size
        self.embed_size = (img_size / self.patch_size, img_size, self.patch_size)
        # deactivate strict image size for positional embeding resizing
        self.vit.patch_embed.strict_img_size = False

        num_layers = len(self.vit.blocks)
        feat_dim = self.vit.num_features

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

        # Add BatchNorm1d layers for each transformer block output
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )
        self.add_norm = add_norm
        self.efficient_probe = efficient_probe

    def forward(self, images):
        B, _, h, w = images.shape
        h, w = h // self.patch_size, w // self.patch_size
        out_hw = (h, w)

        if (h, w) != self.embed_size:
            self.embed_size = (h, w)
            self.vit.pos_embed.data = resize_pos_embed(
                self.vit.pos_embed[0], self.embed_size, False
            )[None, :, :]

        x = self.vit.patch_embed(images)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = x + self.vit.pos_embed
        x = torch.cat((cls_tokens, x), dim=1)
        
        embeds = []
        for i, blk in enumerate(self.vit.blocks):
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

        h, w = out_hw
        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):

            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :] #TODO: check if this is correct
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

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
