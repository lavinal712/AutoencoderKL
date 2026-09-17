from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL as DiffusersAutoencoderKL

from ....util import default
from .discriminator_loss import GeneralLPIPSWithDiscriminator


class DistillationLoss(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        latent_weight: float = 1.0,
        logvar_weight: float = 0.1,
        pixel_loss: str = "l1",
        **kwargs,
    ):
        super().__init__()
        pretrained_model_name_or_path = default(
            pretrained_model_name_or_path, "stabilityai/sd-vae-ft-mse"
        )
        self.teacher = DiffusersAutoencoderKL.from_pretrained(
            pretrained_model_name_or_path
        )
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        assert pixel_loss in ["l1", "l2"]
        if pixel_loss == "l1":
            self.pixel_loss = lambda x, y: F.l1_loss(x, y, reduction="none")
        else:
            self.pixel_loss = lambda x, y: F.mse_loss(x, y, reduction="none")
        self.latent_weight = latent_weight
        self.logvar_weight = logvar_weight

        self.forward_keys = ["split", "regularization_log", "autoencoder"]

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        *,  # added because I changed the order here
        regularization_log: Dict[str, torch.Tensor],
        split: str = "train",
        autoencoder: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, dict]:
        assert autoencoder is not None

        with torch.no_grad():
            teacher_posterior = self.teacher.encode(inputs).latent_dist
            teacher_mean = teacher_posterior.mean
            teacher_logvar = teacher_posterior.logvar
        student_rec = autoencoder.decode(teacher_mean)

        student_mean = regularization_log["mean"]
        student_logvar = regularization_log["logvar"]

        latent_loss = F.mse_loss(teacher_mean, student_mean)
        logvar_loss = F.mse_loss(teacher_logvar, student_logvar)
        rec_loss = self.pixel_loss(inputs.contiguous(), student_rec.contiguous())
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]

        loss = rec_loss + self.latent_weight * latent_loss + self.logvar_weight * logvar_loss
        log = dict()
        log.update(
            {
                f"{split}/loss/total": loss.clone().mean(),
                f"{split}/loss/latent": latent_loss.mean(),
                f"{split}/loss/logvar": logvar_loss.mean(),
                f"{split}/loss/rec": rec_loss.mean(),
            }
        )

        return loss, log


class DistillationLossWithDiscriminator(GeneralLPIPSWithDiscriminator):
    def __init__(
        self,
        *args,
        pretrained_model_name_or_path: Optional[str] = None,
        latent_weight: float = 1.0,
        logvar_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pretrained_model_name_or_path = default(
            pretrained_model_name_or_path, "stabilityai/sd-vae-ft-mse"
        )
        self.teacher = DiffusersAutoencoderKL.from_pretrained(
            pretrained_model_name_or_path
        )
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.latent_weight = latent_weight
        self.logvar_weight = logvar_weight

        self.forward_keys.append("autoencoder")

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        *,
        regularization_log: Dict[str, torch.Tensor],
        optimizer_idx: int,
        global_step: int,
        last_layer: torch.Tensor,
        autoencoder: Optional[nn.Module] = None,
        split: str = "train",
        weights: Union[None, float, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        assert autoencoder is not None

        with torch.no_grad():
            teacher_posterior = self.teacher.encode(inputs).latent_dist
            teacher_mean = teacher_posterior.mean
            teacher_logvar = teacher_posterior.logvar

        if optimizer_idx == 0:
            student_rec = autoencoder.decode(teacher_mean)
        else:
            with torch.no_grad():
                student_rec = autoencoder.decode(teacher_mean)

        loss, log = super().forward(
            inputs,
            student_rec,
            regularization_log=regularization_log,
            optimizer_idx=optimizer_idx,
            global_step=global_step,
            last_layer=last_layer,
            split=split,
            weights=weights,
        )

        if optimizer_idx == 0:
            student_mean = regularization_log["mean"]
            student_logvar = regularization_log["logvar"]

            latent_loss = F.mse_loss(teacher_mean, student_mean)
            logvar_loss = F.mse_loss(teacher_logvar, student_logvar)

            loss = loss + self.latent_weight * latent_loss + self.logvar_weight * logvar_loss
            log.update(
                {
                    f"{split}/loss/latent": latent_loss.mean(),
                    f"{split}/loss/logvar": logvar_loss.mean(),
                }
            )
            log[f"{split}/loss/total"] = loss.mean()

        return loss, log
