from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from torch.nn.utils import spectral_norm

recipes = {
    "S_16": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "S_8": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 8,
        "in_chans": 3,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "B_16": {
        "depth": 12,
        "key_depths": (2, 5, 8, 11),
        "norm_eps": 1e-6,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 768,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
}


class BatchNormLocal(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight[None, :, None] + self.bias[None, :, None]


class ResidualBlock(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.ratio = 1 / np.sqrt(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.fn(x)) / np.sqrt(2)


def make_conv(in_channels: int, out_channels: int, kernel_size: int, using_spec_norm: bool) -> nn.Module:
    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        padding_mode="circular",
    )
    return spectral_norm(conv, eps=1e-12) if using_spec_norm else conv


def make_block(channels: int, kernel_size: int, norm_type: str, norm_eps: float, using_spec_norm: bool) -> nn.Module:
    if norm_type == "bn":
        norm = BatchNormLocal(channels, eps=norm_eps)
    elif norm_type == "gn":
        norm = nn.GroupNorm(32, channels, eps=norm_eps)
    else:
        raise NotImplementedError(f"Unknown norm_type '{norm_type}'")

    return nn.Sequential(
        make_conv(channels, channels, kernel_size, using_spec_norm),
        norm,
        nn.LeakyReLU(0.2, inplace=True),
    )


class DinoDiscriminator(nn.Module):
    def __init__(
        self,
        dino_ckpt_path: str,
        ks: int,
        key_depths: Tuple[int, ...] = (2, 5, 8, 11),
        norm_type: str = "bn",
        using_spec_norm: bool = True,
        norm_eps: float = 1e-6,
        recipe: str = "S_16",
    ):
        super().__init__()
        assert recipe in recipes
        assert ks > 0 and ks % 2 == 1
        assert norm_type in ("bn", "gn")
        self.key_depths = tuple(key_depths)
        patch_size = recipes[recipe]["patch_size"]
        embed_dim = recipes[recipe]["embed_dim"]
        num_heads = recipes[recipe]["num_heads"]

        self.dino = VisionTransformer(
            img_size=224,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=12,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            num_classes=0,
            norm_layer=partial(nn.LayerNorm, eps=norm_eps),
            act_layer=partial(nn.GELU, approximate="tanh"),
        )
        state = torch.load(dino_ckpt_path, map_location="cpu", weights_only=True)
        for name in ("x_scale", "x_shift"):
            state.pop(name, None)
        for name, value in state.items():
            if name.endswith("attn.qkv.bias"):
                value.chunk(3)[1].zero_()
        self.dino.load_state_dict(state, strict=True)
        self.dino.requires_grad_(False).eval()
        for block in self.dino.blocks:
            block.attn.fused_attn = False

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.register_buffer("x_scale", (0.5 / std).reshape(1, 3, 1, 1))
        self.register_buffer("x_shift", ((0.5 - mean) / std).reshape(1, 3, 1, 1))

        block = partial(
            make_block,
            norm_type=norm_type,
            norm_eps=norm_eps,
            using_spec_norm=using_spec_norm,
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                block(embed_dim, 1),
                ResidualBlock(block(embed_dim, ks)),
                make_conv(embed_dim, 1, 1, using_spec_norm),
            )
            for _ in range(len(self.key_depths) + 1)
        ])

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.shape[1] == 3
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.dino.patch_embed(x * self.x_scale + self.x_shift)
        cls_token = self.dino.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1) + self.dino.pos_embed

        features = []
        for i, block in enumerate(self.dino.blocks):
            x = block(x)
            if i in self.key_depths:
                features.append(x[:, 1:].transpose(1, 2))

        features.insert(0, x[:, 1:].transpose(1, 2))
        return torch.cat([
            head(feature).flatten(1)
            for head, feature in zip(self.heads, features)
        ], dim=1)
