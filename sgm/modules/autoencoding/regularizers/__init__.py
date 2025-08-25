from abc import abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....modules.distributions.distributions import \
    DiagonalGaussianDistribution
from .base import AbstractRegularizer


class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(
        self,
        sample: bool = True,
        deterministic: bool = False,
        normalize_latents: bool = False,
        patch_size: Optional[int] = None,
    ):
        super().__init__()
        self.sample = sample
        self.deterministic = deterministic
        self.normalize_latents = normalize_latents
        self.patch_size = patch_size

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(b, c, h_, p, w_, p)
        x = torch.einsum("bchpwq->bcpqhw", x)
        x = x.reshape(b, c * p ** 2, h_,  w_)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h_, w_ = x.shape
        p = self.patch_size
        c = x.shape[1] // (p ** 2)

        x = x.reshape(b, c, p, p, h_, w_)
        x = torch.einsum("bcpqhw->bchpwq", x)
        x = x.reshape(b, c, h_ * p, w_ * p)
        return x

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = dict()

        # src: https://github.com/stepfun-ai/NextStep-1/blob/main/nextstep/models/modeling_flux_vae.py
        mean, logvar = torch.chunk(z, 2, dim=1)
        if self.patch_size is not None:
            mean = self.patchify(mean)
        if self.normalize_latents:
            mean = mean.permute(0, 2, 3, 1)
            mean = F.layer_norm(mean, mean.shape[-1:], eps=1e-6)
            mean = mean.permute(0, 3, 1, 2)
        if self.patch_size is not None:
            mean = self.unpatchify(mean)
        z = torch.cat([mean, logvar], dim=1).contiguous()

        posterior = DiagonalGaussianDistribution(z, deterministic=self.deterministic)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log
