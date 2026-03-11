import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import center_padding, tokens_to_output
from .util import load_checkpoint
import sys

checkpoints = {
    "vitb16": {
        "url": "https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo.pth",  # Replace with actual URL
        "filename": "CroCo.pth",
    }
}


class CROCO(nn.Module):
    def __init__(
        self,
        model_name="vitb16",
        repo_dir="",
        size_image=224, # TODO: CROCO is currently only supported for 224x224 images
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
        self.repo_dir = repo_dir
        self.size_image = size_image
        self.patch_size = 16  # CroCoNet typically uses a 16x16 patch size
        self.add_norm = add_norm
        self.efficient_probe = efficient_probe
        sys.path.append(self.repo_dir)
        # Load the model within __init__
        self.model = self.load_model(model_name)
        num_layers = len(self.model.enc_blocks)
        self.output = output
        self.checkpoint_name = f"croco_{model_name}_{output}"

        feat_dim = 768
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

    def load_model(self, model_name: str):
        """Load the CroCo model from checkpoint."""
        from models.croco import CroCoNet
        assert model_name in checkpoints.keys(), f"Invalid model: {model_name}"
        ckpt = load_checkpoint(**checkpoints[model_name])
        model = CroCoNet(img_size=self.size_image,
            **ckpt.get("croco_kwargs", {})
        )  # Initialize CroCoNet with arguments
        model.load_state_dict(ckpt["model"], strict=True)
        return model.eval()

    def forward(self, images):
        images = F.interpolate(
            images, size=(self.size_image, self.size_image), mode="bilinear", align_corners=False
        )
        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        x, pos = self.model.patch_embed(images)

        if self.model.enc_pos_embed is not None:
            x = x + self.model.enc_pos_embed[None, ...]

        B, N, C = x.size()
        masks = torch.zeros((B, N), dtype=bool)
        posvis = pos
        posvis = pos[~masks].view(B, -1, 2)

        embeds = []  #* CroCo has no cls_token
        for i, blk in enumerate(self.model.enc_blocks):
            x = blk(x, posvis)
            if i in self.multilayers:
                if self.add_norm:
                    x_batched = self.batchnorms[self.multilayers.index(i)](
                        x.permute(0, 2, 1)
                    ).permute(0, 2, 1)
                    embeds.append(x_batched)
                else:
                    embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = [
            tokens_to_output(self.output, embed, None, (h, w)) for embed in embeds
        ]

        if self.efficient_probe:
            return embeds[0]

        # NO CLS TOKEN, so no below conditions are the same
        if len(outputs) == 1 and self.mean_pool and not self.return_cls:
            return embeds[0].mean(dim=1)
        elif len(outputs) == 1 and self.return_cls and not self.mean_pool:
            return embeds[0].mean(dim=1)
        elif len(outputs) == 1 and self.mean_pool and self.return_cls:
            return embeds[0].mean(dim=1)
        else:
            pass

        return outputs[0] if len(outputs) == 1 else outputs
