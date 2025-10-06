import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

from typing import Optional

from .util import get_2d_sincos_pos_embed, Attention

class BinaryHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="dpt",
        uncertainty_aware=False,
        hidden_dim=512,
        kernel_size=1,
        output_dim=2,
        pred_type="sigmoid",
    ):
        super().__init__()
        self.uncertainty_aware = uncertainty_aware
        self.kernel_size = kernel_size
        assert head_type in ["linear", "multiscale", "dpt"]
        name = f"snorm_{head_type}_k{kernel_size}"
        self.name = f"{name}_UA" if uncertainty_aware else name
        self.pred_type = pred_type
        if pred_type == "sigmoid":
            self.batch_norm = nn.BatchNorm2d(output_dim)
        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        feats = self.head(feats)
        if self.pred_type == "sigmoid":
            feats = self.batch_norm(feats)
            return torch.sigmoid(feats)
        elif self.pred_type == "tanh":
            return torch.tanh(feats)
        return feats


class TaskonomyHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="dpt",
        uncertainty_aware=False,
        hidden_dim=512,
        kernel_size=1,
        output_dim=1,
        pred_type="sigmoid",
    ):
        super().__init__()

        self.uncertainty_aware = uncertainty_aware
        self.kernel_size = kernel_size
        assert head_type in ["linear", "multiscale", "dpt"]
        name = f"snorm_{head_type}_k{kernel_size}"
        self.name = f"{name}_UA" if uncertainty_aware else name
        self.pred_type = pred_type
        if pred_type == "sigmoid":
            self.batch_norm = nn.BatchNorm2d(output_dim)
        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        feats = self.head(feats)
        if self.pred_type == "sigmoid":
            feats = self.batch_norm(feats)
            return torch.sigmoid(feats)
        elif self.pred_type == "tanh":
            return torch.tanh(feats)
        return feats


class SurfaceNormalHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        uncertainty_aware=False,
        hidden_dim=512,
        kernel_size=1,
    ):
        super().__init__()

        self.uncertainty_aware = uncertainty_aware
        output_dim = 4 if uncertainty_aware else 3

        self.kernel_size = kernel_size

        assert head_type in ["linear", "multiscale", "dpt"]
        name = f"snorm_{head_type}_k{kernel_size}"
        self.name = f"{name}_UA" if uncertainty_aware else name

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        return self.head(feats)


class DepthHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        min_depth=0.001,
        max_depth=10,
        prediction_type="sigdepth",
        hidden_dim=512,
        kernel_size=1,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.name = f"{prediction_type}_{head_type}_k{kernel_size}"

        if prediction_type == "bindepth":
            output_dim = 256
            self.predict = DepthBinPrediction(min_depth, max_depth, n_bins=output_dim)
        elif prediction_type == "sigdepth":
            output_dim = 1
            self.predict = DepthSigmoidPrediction(min_depth, max_depth)
        else:
            raise ValueError()

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        """Prediction each pixel."""
        feats = self.head(feats)
        depth = self.predict(feats)
        return depth


class DepthBinPrediction(nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=10,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
    ):
        super().__init__()
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm_strategy = norm_strategy
        self.bins_strategy = bins_strategy

    def forward(self, prob):
        if self.bins_strategy == "UD":
            bins = torch.linspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )
        elif self.bins_strategy == "SID":
            bins = torch.logspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )

        # following Adabins, default linear
        if self.norm_strategy == "linear":
            prob = torch.relu(prob)
            eps = 0.1
            prob = prob + eps
            prob = prob / prob.sum(dim=1, keepdim=True)
        elif self.norm_strategy == "softmax":
            prob = torch.softmax(prob, dim=1)
        elif self.norm_strategy == "sigmoid":
            prob = torch.sigmoid(prob)
            prob = prob / prob.sum(dim=1, keepdim=True)

        depth = torch.einsum("ikhw,k->ihw", [prob, bins])
        depth = depth.unsqueeze(dim=1)
        return depth


class DepthSigmoidPrediction(nn.Module):
    def __init__(self, min_depth=0.001, max_depth=10):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, pred):
        depth = pred.sigmoid()
        depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        return depth


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features,
        kernel_size=3,
        with_skip=True,
        upsample=False,
        is_transformer=False,
    ):
        """
        Init.

        Args:
            features (int): number of features.
            kernel_size (int): kernel size for the residual conv units (default 3).
            with_skip (bool): if True, includes a skip connection from the input to the output.
            upsample (bool): if True, upsamples the output by a factor of 2 using bilinear interpolation.
            is_transformer (bool): if True, uses the Transformer-based implementation. If False, uses the CNN-based implementation.
        """
        super().__init__()
        self.with_skip = with_skip
        self.upsample = upsample

        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit(
                features, kernel_size, is_transformer=is_transformer
            )

        self.resConfUnit2 = ResidualConvUnit(
            features, kernel_size, is_transformer=is_transformer
        )
        self.is_transformer = is_transformer

    def forward(self, x, skip_x=None):
        if skip_x is not None and self.with_skip:
            assert skip_x.shape == x.shape, "Shape of skip_x must match x"
            x = self.resConfUnit1(x) + skip_x

        x = self.resConfUnit2(x)

        if not self.is_transformer:
            x = nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            )

        return x


class ResidualConvUnit(nn.Module):
    def __init__(
        self, features, kernel_size=3, is_transformer=False, inplace_relu=True
    ):
        """
        Init.

        Args:
            features (int): number of features.
            kernel_size (int): kernel size for convolution layers (default 3).
            is_transformer (bool): if True, uses the Transformer-based implementation. If False, uses the CNN-based implementation.
            inplace_relu (bool): if True, uses in-place ReLU operations.
        """
        super().__init__()

        if is_transformer:
            assert (
                kernel_size % 2 == 1
            ), "Kernel size needs to be odd for transformer-based implementation"
            padding = kernel_size // 2
            self.conv = nn.Sequential(
                nn.Conv2d(features, features, kernel_size, padding=padding),
                nn.ReLU(inplace=inplace_relu),
                nn.Conv2d(features, features, kernel_size, padding=padding),
                nn.ReLU(inplace=inplace_relu),
            )
        else:
            self.conv1 = nn.Conv2d(
                features, features, kernel_size=3, stride=1, padding=1, bias=True
            )
            self.conv2 = nn.Conv2d(
                features, features, kernel_size=3, stride=1, padding=1, bias=True
            )
            self.relu = nn.ReLU(inplace=inplace_relu)

    def forward(self, x):
        if hasattr(self, "conv"):
            return self.conv(x) + x
        else:
            out = self.relu(x)
            out = self.conv1(out)
            out = self.relu(out)
            out = self.conv2(out)
            return out + x


class DPT(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=3):
        super().__init__()
        assert len(input_dims) == 4

        # Determine if we're using ResNet (CNN-based) or Transformer
        self.resnet = not isinstance(input_dims[0], int)

        # Initialize convolutional layers differently based on backbone type
        if self.resnet:
            self.conv_0 = nn.Conv2d(
                input_dims[0][0],
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.conv_1 = nn.Conv2d(
                input_dims[1][0],
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.conv_2 = nn.Conv2d(
                input_dims[2][0],
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.conv_3 = nn.Conv2d(
                input_dims[3][0],
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        else:
            self.conv_0 = nn.Conv2d(input_dims[0], hidden_dim, 1, padding=0)
            self.conv_1 = nn.Conv2d(input_dims[1], hidden_dim, 1, padding=0)
            self.conv_2 = nn.Conv2d(input_dims[2], hidden_dim, 1, padding=0)
            self.conv_3 = nn.Conv2d(input_dims[3], hidden_dim, 1, padding=0)

        # Initialize FeatureFusionBlock with resnet flag to determine behavior
        self.ref_0 = FeatureFusionBlock(
            hidden_dim, kernel_size, is_transformer=not self.resnet
        )
        self.ref_1 = FeatureFusionBlock(
            hidden_dim, kernel_size, is_transformer=not self.resnet
        )
        self.ref_2 = FeatureFusionBlock(
            hidden_dim, kernel_size, is_transformer=not self.resnet
        )
        self.ref_3 = FeatureFusionBlock(
            hidden_dim, kernel_size, is_transformer=not self.resnet, with_skip=False
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, feats):
        """Prediction each pixel."""
        assert len(feats) == 4
        feats[0] = self.conv_0(feats[0])
        feats[1] = self.conv_1(feats[1])
        feats[2] = self.conv_2(feats[2])
        feats[3] = self.conv_3(feats[3])

        # We no longer need to resize explicitly for ResNet features
        # Let the FeatureFusionBlocks handle the dimensions
        if not self.resnet:
            feats = [F.interpolate(x, scale_factor=2) for x in feats]

        out = self.ref_3(feats[3], None)
        out = self.ref_2(feats[2], out)
        out = self.ref_1(feats[1], out)
        out = self.ref_0(feats[0], out)

        if not self.resnet:
            out = F.interpolate(out, scale_factor=4)
        out = self.out_conv(out)
        out = F.interpolate(out, scale_factor=2)
        return out


def make_conv(input_dim, hidden_dim, output_dim, num_layers, kernel_size=1):
    if num_layers == 1:
        conv = nn.Conv2d(input_dim, output_dim, kernel_size)
    else:
        assert num_layers > 1
        modules = [nn.Conv2d(input_dim, hidden_dim, kernel_size), nn.ReLU(inplace=True)]
        for i in range(num_layers - 2):
            modules.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size))
            modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(hidden_dim, output_dim, kernel_size))
        conv = nn.Sequential(*modules)

    return conv


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1):
        super().__init__()
        if type(input_dim) is not int:
            input_dim = sum(input_dim)

        assert type(input_dim) is int
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding)

    def forward(self, feats):
        if type(feats) is list:
            feats = torch.cat(feats, dim=1)

        feats = interpolate(feats, scale_factor=4, mode="bilinear")
        return self.conv(feats)


class MultiscaleHead(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=1):
        super().__init__()

        self.convs = nn.ModuleList(
            [make_conv(in_d, None, hidden_dim, 1, kernel_size) for in_d in input_dims]
        )
        interm_dim = len(input_dims) * hidden_dim
        self.conv_mid = make_conv(interm_dim, hidden_dim, hidden_dim, 3, kernel_size)
        self.conv_out = make_conv(hidden_dim, hidden_dim, output_dim, 2, kernel_size)

    def forward(self, feats):
        num_feats = len(feats)
        feats = [self.convs[i](feats[i]) for i in range(num_feats)]

        h, w = feats[-1].shape[-2:]
        feats = [interpolate(feat, (h, w), mode="bilinear") for feat in feats]
        feats = torch.cat(feats, dim=1).relu()

        # upsample
        feats = interpolate(feats, scale_factor=2, mode="bilinear")
        feats = self.conv_mid(feats).relu()
        feats = interpolate(feats, scale_factor=4, mode="bilinear")
        return self.conv_out(feats)


class ClassificationHead(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 num_classes: int,
                 use_layernorm: bool = True,
                 dropout_rate: float = 0.0,
                 attention_map = None,
                 head_type: str = "linear"):
        super().__init__()
        self.name = f"cls_{head_type}"
        self.attention_map = attention_map
        self.dropout_rate = dropout_rate
        self.use_layernorm = use_layernorm
        self.norm = nn.LayerNorm(feat_dim) if use_layernorm else None
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, feats):
        # feats is expected to be [B, D] (e.g., CLS token)
        if isinstance(feats, (list, tuple)):
            # If provided as a list of tensors, concatenate along feature dim
            feats = torch.cat(feats, dim=-1)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        if self.norm is not None:
            feats = self.norm(feats)
        feats = self.dropout(feats)
        return self.classifier(feats)

class EfficientProbing(nn.Module):
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
        attention_map = None,
        head_type: str = "efficient",
    ):
        super().__init__()
        self.name = f'cls_{head_type}'
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.use_layernorm = use_layernorm
        self.feat_dim = feat_dim
        head_dim = feat_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.attention_map = attention_map
        self.d_out = d_out
        self.num_queries = num_queries
        self.norm = nn.LayerNorm(feat_dim // d_out) if use_layernorm else None
        self.v = nn.Linear(self.feat_dim, self.feat_dim // d_out, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, self.feat_dim) * 0.02)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(self.feat_dim // d_out, num_classes, bias=True)
        
    def forward(self, feats, cls=None):
        try:
            # from [B, (NxC)] to [B, N, C]
            feats = feats.view(feats.size(0),feats.size(1)//self.feat_dim,self.feat_dim)
        except:
            feats = feats
        cls_token = feats[:, 0]
        # feats = feats[:, 1:]
        B, N, C = feats.shape
        C_prime = C // self.d_out

        if cls is not None:
            cls_token = cls
        else:
            cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = (feats.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        q = q * self.scale
        v = (self.v(feats).reshape(B, N, self.num_queries, C // (self.d_out * self.num_queries)).permute(0, 2, 1, 3))

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attention_map = attn.squeeze(1)
        x_cls = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        x_cls = x_cls.view(B, C_prime)
        if self.norm is not None:
            x_cls = self.norm(x_cls)
        x_cls = self.proj_drop(x_cls)
        x_cls = self.classifier(x_cls)
        return x_cls

class ABMILPHead(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            num_classes: int,
            self_attention_apply_to: str = "none", # choices=["none", "map", "both"]
            activation: str= "relu",  # choices=["tanh", "relu"]
            depth: int = 1,
            cond: str="none", # choices=["none", "pe"]
            content: str = "all", # choices=["all", "patch"]
            num_patches: Optional[int] = None,
            use_layernorm: bool = False,
            dropout_rate: float = 0.0,
            attention_map = None,
            head_type: str = "abmilp",

        ):
        super().__init__()
        self.name = f'cls_{head_type}'
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        self.feat_dim = feat_dim
        self.cond = cond
        self.self_attention_apply_to = self_attention_apply_to
        self.content = content
        if self.cond == "pe":
            self.pos_embed = torch.nn.Parameter(
                torch.from_numpy(
                    get_2d_sincos_pos_embed(feat_dim, int(num_patches ** .5), cls_token=(content != "patch"))
                ).float().unsqueeze(0),
                requires_grad=False
            )
        else:
            self.pos_embed = None

        self.self_attn = Attention(feat_dim, num_heads=1) if self.self_attention_apply_to != "none" else nn.Identity()


        self.ATTENTION_BRANCHES = 1

        attn_pred_layers = []
        for i in range(depth-1):
            attn_pred_layers.extend([
                nn.Linear(feat_dim, feat_dim),
                (nn.Tanh() if activation == "tanh" else nn.ReLU()),
            ])

        attn_pred_layers.append(nn.Linear(feat_dim, self.ATTENTION_BRANCHES))
        self.attention_predictor = nn.Sequential(*attn_pred_layers)
        self.norm = nn.LayerNorm(self.feat_dim) if use_layernorm else None
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(self.feat_dim, num_classes)
        self.attention_map = attention_map
    def forward_with_attn_map(self, x):
        try:
            # from [B, (NxC)] to [B, N, C]
            x = x.view(x.size(0),x.size(1)//self.feat_dim,self.feat_dim)
        except:
            x = x

        if self.content == "patch":
            x = x[:, 1:] # keep patch tokens only

        x_attn = self.self_attn(x)
        if isinstance(x_attn, tuple):
            x_attn = x_attn[0]

        predictor_input = x_attn if self.self_attention_apply_to in ["map", "both"] else x

        if self.cond == "pe":
            predictor_input = predictor_input + self.pos_embed

        attn_map = self.attention_predictor(predictor_input)
        attn_map = F.softmax(attn_map, dim=1)
        attn_map = self.attn_drop(attn_map)
        x_out = x_attn if self.self_attention_apply_to in ["both"] else x
        out = (x_out * attn_map).sum(dim=1)
        return out, attn_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cls, _ = self.forward_with_attn_map(x)
        if self.norm is not None:
            x_cls = self.norm(x_cls)
        x_cls = self.proj_drop(x_cls)
        x_cls = self.classifier(x_cls)
        return x_cls