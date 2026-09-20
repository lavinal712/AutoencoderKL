from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import AutoencoderKL as DiffusersAutoencoderKL
from omegaconf import OmegaConf

from .autoencoder import AutoencodingEngine
from ..modules.ema import LitEma


class DiffusersAutoencoderKLWrapper(AutoencodingEngine):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        model_config_name_or_path: Optional[str] = None,
        subfolder: Optional[str] = None,
        regularizer_config: Optional[Dict] = None,
        **kwargs,
    ):
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")

        assert pretrained_model_name_or_path or model_config_name_or_path
        if pretrained_model_name_or_path is not None:
            vae = DiffusersAutoencoderKL.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder
            )
        else:
            config = DiffusersAutoencoderKL.load_config(model_config_name_or_path)
            vae = DiffusersAutoencoderKL.from_config(config)
        self.config = OmegaConf.to_container(
            OmegaConf.create(dict(vae.config)), resolve=True
        )

        super().__init__(
            encoder_config={"target": "torch.nn.Identity"},
            decoder_config={"target": "torch.nn.Identity"},
            regularizer_config=regularizer_config or {
                "target": (
                    "sgm.modules.autoencoding.regularizers"
                    ".DiagonalGaussianRegularizer"
                )
            },
            **kwargs,
        )

        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.quant_conv = vae.quant_conv
        self.post_quant_conv = vae.post_quant_conv

        if self.use_ema:
            self.model_ema = LitEma(self, decay=self.ema_decay)

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x)
        if self.quant_conv is not None:
            z = self.quant_conv(z)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z, **kwargs):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        return self.decoder(z, **kwargs)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def get_encoder_last_layer(self):
        return self.encoder.conv_out.weight

    def get_autoencoder_params(self):
        params = list(super().get_autoencoder_params())
        for m in (self.quant_conv, self.post_quant_conv):
            if m is not None:
                params += list(m.parameters())
        return params

    @torch.no_grad()
    def save_pretrained(self, save_dir, safe_serialization=True):
        vae = DiffusersAutoencoderKL.from_config(self.config)
        sd = self.state_dict()
        state = {k: sd[k].detach().cpu() for k in vae.state_dict() if k in sd}
        missing, unexpected = vae.load_state_dict(state, strict=False)
        vae.save_pretrained(save_dir, safe_serialization=safe_serialization)
        return missing, unexpected
