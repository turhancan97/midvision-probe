import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import center_padding, tokens_to_output
import sys
from .util import load_checkpoint


checkpoints = {
    "vitb16": {
        "url": "",
        "filename": "spa-b.ckpt",
    },
    "vitl16": {
        "url": "",
        "filename": "spa-l.ckpt",
    }
}

class SPA(nn.Module):
    def __init__(
        self,
        model_name="vitb16",
        repo_dir="",
        size_image=672,
        pretrained=True,
        layer=-1,
        output="dense",
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
        self.add_norm = add_norm
        self.efficient_probe = efficient_probe
        self.size_image = size_image
        self.repo_dir = repo_dir
        sys.path.append(self.repo_dir)
        # Load the model within __init__
        self.model = self.load_model(model_name, pretrained)
        num_layers = len(self.model.blocks)
        self.output = output
        self.checkpoint_name = f"spa_{model_name}_{output}"
        self.patch_size = 16  # SPA typically uses a 16x16 patch size

        feat_dim = 768 if model_name == "vitb16" else 1024
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        # Set up batch normalization layers, adapted to CroCo's feature dimension (512)
        if return_multilayer:
            self.feat_dim = [feat_dim] * 4
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            self.multilayers = [multilayers[-1]]

        self.layer = "-".join(str(_x) for _x in self.multilayers)

        # Define BatchNorm1d layers for each selected layer (adjusted to 512 for CroCoNet)
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(feat_dim) for _ in self.multilayers]
        )

    def load_model(self, model_name: str, pretrained: bool):
        """Load the SPA model from checkpoint."""
        from spa.models import spa_vit_base_patch16, spa_vit_large_patch16
        assert model_name in checkpoints.keys(), f"Invalid model: {model_name}"
        if pretrained:
            if model_name == "vitb16":
                model = spa_vit_base_patch16(img_size=self.size_image, pretrained=True)
            else:
                model = spa_vit_large_patch16(img_size=self.size_image, pretrained=True)
        else:
            if model_name == "vitb16":
                model = spa_vit_base_patch16(img_size=self.size_image, pretrained=False) # TODO: currently only supports 224x224 images
            else:
                model = spa_vit_large_patch16(img_size=self.size_image, pretrained=False) # TODO: currently only supports 224x224 images
            ckpt = load_checkpoint(**checkpoints[model_name])
            # remove "model." prefix
            ckpt["state_dict"] = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
            # remove "img_backbone." prefix
            ckpt["state_dict"] = {k.replace("img_backbone.", ""): v for k, v in ckpt["state_dict"].items()}
            model.load_state_dict(ckpt["state_dict"], strict=False)
        return model.eval()

    def forward(self, images):
        images = F.interpolate(
            images, size=(self.size_image, self.size_image), mode="bilinear", align_corners=False
        )

        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        # Patch embedding
        x = self.model.patch_embed(images)
        
        # Add position embeddings (cls token position + patch positions)
        # pos_embed shape: (1, 1 + num_patches, embed_dim)
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        x = x + self.model.pos_embed[:, 1:, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        embeds = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.multilayers:
                # Apply final LayerNorm for the last layer (critical for good features!)
                if i == self.multilayers[-1]:
                    x_normed = self.model.norm(x)
                else:
                    x_normed = x
                    
                if self.add_norm:
                    x_batched = self.batchnorms[self.multilayers.index(i)](
                        x_normed.permute(0, 2, 1)
                    ).permute(0, 2, 1)
                    embeds.append(x_batched)
                else:
                    embeds.append(x_normed)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = [
            tokens_to_output(self.output, embed[:, 1:], embed[:, 0], (h, w)) for embed in embeds
        ]

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
