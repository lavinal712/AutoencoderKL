from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import AutoencoderDC as DiffusersAutoencoderDC
from diffusers import AutoencoderKL as DiffusersAutoencoderKL
from diffusers import AutoencoderKLWan as DiffusersAutoencoderKLWan
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

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
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


class DiffusersAutoencoderDCWrapper(AutoencodingEngine):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        model_config_name_or_path: Optional[str] = None,
        **kwargs,
    ):
        assert pretrained_model_name_or_path or model_config_name_or_path
        if pretrained_model_name_or_path is not None:
            ae = DiffusersAutoencoderDC.from_pretrained(pretrained_model_name_or_path)
        else:
            config = DiffusersAutoencoderDC.load_config(model_config_name_or_path)
            ae = DiffusersAutoencoderDC.from_config(config)
        self.config = OmegaConf.to_container(
            OmegaConf.create(dict(ae.config)), resolve=True
        )

        super().__init__(
            encoder_config={"target": "torch.nn.Identity"},
            decoder_config={"target": "torch.nn.Identity"},
            regularizer_config={
                "target": (
                    "sgm.modules.autoencoding.regularizers.base"
                    ".IdentityRegularizer"
                )
            },
            **kwargs,
        )

        self.encoder = ae.encoder
        self.decoder = ae.decoder

        if self.use_ema:
            self.model_ema = LitEma(self, decay=self.ema_decay)

    def get_last_layer(self):
        layer = self.decoder.conv_out
        if hasattr(layer, "conv"):
            return layer.conv.weight
        return layer.weight

    def get_encoder_last_layer(self):
        return self.encoder.conv_out.weight

    @torch.no_grad()
    def save_pretrained(self, save_dir, safe_serialization=True):
        ae = DiffusersAutoencoderDC.from_config(self.config)
        sd = self.state_dict()
        state = {k: sd[k].detach().cpu() for k in ae.state_dict() if k in sd}
        missing, unexpected = ae.load_state_dict(state, strict=False)
        ae.save_pretrained(save_dir, safe_serialization=safe_serialization)
        return missing, unexpected


class DiffusersAutoencoderKLWanWrapper(AutoencodingEngine):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        model_config_name_or_path: Optional[str] = None,
        subfolder: Optional[str] = None,
        regularizer_config: Optional[Dict] = None,
        **kwargs,
    ):
        assert pretrained_model_name_or_path or model_config_name_or_path
        if pretrained_model_name_or_path is not None:
            vae = DiffusersAutoencoderKLWan.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder
            )
        else:
            config = DiffusersAutoencoderKLWan.load_config(model_config_name_or_path)
            vae = DiffusersAutoencoderKLWan.from_config(config)
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

        # Precompute and cache conv counts for encoder and decoder for clear_cache speedup
        self._cached_conv_counts = {
            "decoder": sum(isinstance(m, nn.Conv3d) for m in self.decoder.modules())
            if self.decoder is not None
            else 0,
            "encoder": sum(isinstance(m, nn.Conv3d) for m in self.encoder.modules())
            if self.encoder is not None
            else 0,
        }

        if self.use_ema:
            self.model_ema = LitEma(self, decay=self.ema_decay)

    def clear_cache(self):
        # Use cached conv counts for decoder and encoder to avoid re-iterating modules each call
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        self.clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        out = self.quant_conv(out)
        self.clear_cache()
        return out

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self._encode(x)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z
    
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        self.clear_cache()
        iter_ = z.shape[2]
        x = self.post_quant_conv(z)
        out = None
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], dim=2)
        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        return out

    def decode(self, z, **kwargs):
        return self._decode(z)

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

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self.regularization, "sample"):
            self.regularization.sample = mode
        return self

    @torch.no_grad()
    def save_pretrained(self, save_dir, safe_serialization=True):
        vae = DiffusersAutoencoderKLWan.from_config(self.config)
        sd = self.state_dict()
        state = {k: sd[k].detach().cpu() for k in vae.state_dict() if k in sd}
        missing, unexpected = vae.load_state_dict(state, strict=False)
        vae.save_pretrained(save_dir, safe_serialization=safe_serialization)
        return missing, unexpected
