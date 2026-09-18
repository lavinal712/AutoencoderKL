from typing import Dict, Iterator, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ....util import default
from .discriminator_loss import GeneralLPIPSWithDiscriminator


class VFLossWithDiscriminator(GeneralLPIPSWithDiscriminator):
    def __init__(
        self,
        *args,
        foundation_model: str,
        embed_dim: int,
        reverse_proj: bool = False,
        vf_weight: float = 1e2,
        adaptive_vf: bool = False,
        cos_weight: float = 1.0,
        distmat_weight: float = 1.0,
        cos_margin: float = 0.0,
        distmat_margin: float = 0.0,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        foundation_model = default(
            foundation_model, "hf-hub:timm/vit_large_patch14_dinov2.lvd142m"
        )
        if checkpoint_path is None:
            self.foundation_model = timm.create_model(
                foundation_model, pretrained=True, dynamic_img_size=True
            )
        else:
            self.foundation_model = timm.create_model(
                foundation_model,
                pretrained=False,
                dynamic_img_size=True,
                checkpoint_path=checkpoint_path,
            )
        self.foundation_model.requires_grad_(False)
        self.foundation_model.eval()

        self.reverse_proj = reverse_proj
        if not self.reverse_proj:
            self.linear_proj = nn.Conv2d(
                self.foundation_model.num_features,
                embed_dim,
                kernel_size=1,
                bias=True,
            )
        else:
            self.linear_proj = nn.Conv2d(
                embed_dim,
                self.foundation_model.num_features,
                kernel_size=1,
                bias=False,
            )

        self.vf_weight = vf_weight
        self.adaptive_vf = adaptive_vf
        self.cos_weight = cos_weight
        self.distmat_weight = distmat_weight
        self.cos_margin = cos_margin
        self.distmat_margin = distmat_margin

        self.forward_keys += ["z", "encoder_last_layer"]

    def get_trainable_autoencoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from super().get_trainable_autoencoder_parameters()
        yield from self.linear_proj.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.foundation_model.eval()
        return self

    def calculate_adaptive_weight_vf(
        self, nll_loss: torch.Tensor, vf_loss: torch.Tensor, last_layer: torch.Tensor
    ) -> torch.Tensor:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        vf_grads = torch.autograd.grad(vf_loss, last_layer, retain_graph=True)[0]

        vf_weight = torch.norm(nll_grads) / (torch.norm(vf_grads) + 1e-4)
        vf_weight = torch.clamp(vf_weight, 0.0, 1e8).detach()
        vf_weight = vf_weight * self.vf_weight
        return vf_weight

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        *,  # added because I changed the order here
        regularization_log: Dict[str, torch.Tensor],
        optimizer_idx: int,
        global_step: int,
        last_layer: torch.Tensor,
        encoder_last_layer: torch.Tensor,
        z: torch.Tensor,
        split: str = "train",
        weights: Union[None, float, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        loss, log = super().forward(
            inputs,
            reconstructions,
            regularization_log=regularization_log,
            optimizer_idx=optimizer_idx,
            global_step=global_step,
            last_layer=last_layer,
            split=split,
            weights=weights,
        )

        if optimizer_idx == 0 and self.vf_weight > 0:
            with torch.no_grad():
                vf_inputs = F.interpolate(
                    inputs, size=(224, 224), mode="bilinear", align_corners=False
                )
                features = self.foundation_model.forward_features(vf_inputs)
                num_prefix_tokens = self.foundation_model.num_prefix_tokens
                features = features[:, num_prefix_tokens:]
                feature_size = int(features.shape[1] ** 0.5)
                assert feature_size ** 2 == features.shape[1]
                features = rearrange(
                    features, "b (h w) c -> b c h w", h=feature_size
                )

            if not self.reverse_proj:
                features = self.linear_proj(features)
            else:
                z = self.linear_proj(z)
            if features.shape[-2:] != z.shape[-2:]:
                features = F.interpolate(
                    features, size=z.shape[-2:], mode="bilinear", align_corners=False
                )

            z_flat = rearrange(z, "b c h w -> b c (h w)")
            features_flat = rearrange(features, "b c h w -> b c (h w)")
            z_norm = F.normalize(z_flat, dim=1)
            features_norm = F.normalize(features_flat, dim=1)
            z_sim = torch.einsum("bci,bcj->bij", z_norm, z_norm)
            features_sim = torch.einsum("bci,bcj->bij", features_norm, features_norm)

            distmat_loss = F.relu(
                torch.abs(z_sim - features_sim) - self.distmat_margin
            ).mean()
            cosine_similarity = F.cosine_similarity(z, features, dim=1)
            cos_loss = F.relu(
                1.0 - self.cos_margin - cosine_similarity
            ).mean()
            vf_loss = self.distmat_weight * distmat_loss + self.cos_weight * cos_loss

            if self.adaptive_vf and self.training:
                nll_loss = log[f"{split}/loss/nll"]
                vf_weight = self.calculate_adaptive_weight_vf(
                    nll_loss, vf_loss, last_layer=encoder_last_layer
                )
            else:
                vf_weight = torch.as_tensor(
                    self.vf_weight, device=inputs.device, dtype=inputs.dtype
                )

            loss = loss + vf_weight * vf_loss
            log.update(
                {
                    f"{split}/loss/vf": vf_loss.mean(),
                    f"{split}/loss/vf_cos": cos_loss.mean(),
                    f"{split}/loss/vf_distmat": distmat_loss.mean(),
                    f"{split}/scalars/vf_weight": vf_weight.mean(),
                }
            )
            log[f"{split}/loss/total"] = loss.mean()

        return loss, log
