from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderDC as DiffusersAutoencoderDC
from diffusers import AutoencoderKL as DiffusersAutoencoderKL
from diffusers import AutoencoderKLQwenImage as DiffusersAutoencoderKLQwenImage
from diffusers import AutoencoderKLWan as DiffusersAutoencoderKLWan
from diffusers import AutoencoderRAE as DiffusersAutoencoderRAE
from omegaconf import OmegaConf
from transformers import Dinov2WithRegistersModel, SiglipVisionModel, ViTMAEModel

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
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")

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
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")

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

    def decode(self, z: torch.Tensor, **kwargs):
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


class DiffusersAutoencoderRAEWrapper(AutoencodingEngine):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        model_config_name_or_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
        freeze_encoder: bool = True,
        noise_tau: Optional[float] = None,
        **kwargs,
    ):
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")

        assert pretrained_model_name_or_path or model_config_name_or_path
        if pretrained_model_name_or_path is not None:
            rae = DiffusersAutoencoderRAE.from_pretrained(pretrained_model_name_or_path)
        else:
            config = DiffusersAutoencoderRAE.load_config(model_config_name_or_path)
            rae = DiffusersAutoencoderRAE.from_config(config)
            if encoder_path is not None:
                encoder_type = config.get("encoder_type", "dinov2")
                encoder = self._load_encoder(encoder_path, encoder_type)
                rae.encoder.load_state_dict(encoder.state_dict(), strict=False)
        self.config = OmegaConf.to_container(
            OmegaConf.create(dict(rae.config)), resolve=True
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

        self.encoder = rae.encoder
        self.decoder = rae.decoder

        self.encoder_input_size = rae.encoder_input_size
        self.noise_tau = rae.noise_tau if noise_tau is None else noise_tau
        self.reshape_to_2d = rae.reshape_to_2d
        self.encoder_type = self.config["encoder_type"]
        self.encoder_patch_size = self.config["encoder_patch_size"]
        self.scaling_factor = self.config.get("scaling_factor", 1.0)

        self.register_buffer("encoder_mean", rae.encoder_mean.detach().clone(), persistent=True)
        self.register_buffer("encoder_std", rae.encoder_std.detach().clone(), persistent=True)
        self.register_buffer("_latents_mean", rae._latents_mean.detach().clone(), persistent=True)
        self.register_buffer("_latents_std", rae._latents_std.detach().clone(), persistent=True)

        self.freeze_encoder = freeze_encoder
        self.encoder.requires_grad_(not self.freeze_encoder)
        if self.freeze_encoder:
            self.encoder.eval()

        if self.use_ema:
            self.model_ema = LitEma(self, decay=self.ema_decay)

    def _load_encoder(self, encoder_path: str, encoder_type: str):
        if encoder_type == "dinov2":
            try:
                encoder = Dinov2WithRegistersModel.from_pretrained(encoder_path, local_files_only=True)
            except (OSError, ValueError, AttributeError):
                encoder = Dinov2WithRegistersModel.from_pretrained(encoder_path, local_files_only=False)
        elif encoder_type == "siglip2":
            encoder = SiglipVisionModel.from_pretrained(encoder_path)
        elif encoder_type == "mae":
            encoder = ViTMAEModel.from_pretrained(encoder_path)
        else:
            raise ValueError(f"Unsupported encoder: {encoder_type}")
        return encoder

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = F.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )
        mean = self.encoder_mean.to(device=x.device, dtype=x.dtype)
        std = self.encoder_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _noising(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        # Per-sample random sigma in [0, noise_tau]
        noise_sigma = self.noise_tau * torch.rand(
            (x.size(0),) + (1,) * (x.ndim - 1),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        noise = torch.randn(
            x.shape, generator=generator, device=x.device, dtype=x.dtype
        )
        return x + noise_sigma * noise

    def _normalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        latents_mean = self._latents_mean.to(device=z.device, dtype=z.dtype)
        latents_std = self._latents_std.to(device=z.device, dtype=z.dtype)
        return (z - latents_mean) / (latents_std + 1e-5)

    def _denormalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        latents_mean = self._latents_mean.to(device=z.device, dtype=z.dtype)
        latents_std = self._latents_std.to(device=z.device, dtype=z.dtype)
        return z * (latents_std + 1e-5) + latents_mean

    def _encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "dinov2":
            outputs = self.encoder(x, output_hidden_states=True)
            unused_token_num = 5  # 1 CLS + 4 register tokens
            return outputs.last_hidden_state[:, unused_token_num:]
        elif self.encoder_type == "siglip2":
            outputs = self.encoder(
                x, output_hidden_states=True, interpolate_pos_encoding=True
            )
            return outputs.last_hidden_state
        elif self.encoder_type == "mae":
            h, w = x.shape[-2:]
            patch_num = h * w // self.encoder_patch_size ** 2
            if patch_num * self.encoder_patch_size ** 2 != h * w:
                raise ValueError("Image size should be divisible by patch size.")
            noise = (
                torch.arange(patch_num, device=x.device)
                .unsqueeze(0)
                .expand(x.shape[0], -1)
                .to(dtype=x.dtype)
            )
            outputs = self.encoder(x, noise, interpolate_pos_encoding=True)
            return outputs.last_hidden_state[:, 1:]  # remove cls token
        else:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        x = (x + 1.0) / 2.0
        x = self._resize_and_normalize(x)

        ctx = torch.no_grad() if self.freeze_encoder else nullcontext()
        with ctx:
            tokens = self._encoder_forward(x)
        
        if self.training and self.noise_tau > 0:
            tokens = self._noising(tokens, generator=generator)

        if self.reshape_to_2d:
            b, n, c = tokens.shape
            side = int(n ** 0.5)
            if side * side != n:
                raise ValueError(
                    f"Token length n={n} is not a perfect square; cannot reshape to 2D."
                )
            z = tokens.transpose(1, 2).contiguous().view(b, c, side, side)
        else:
            z = tokens

        z = self._normalize_latents(z)
        if self.scaling_factor != 1.0:
            z = z * self.scaling_factor

        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.scaling_factor != 1.0:
            z = z / self.scaling_factor
        z = self._denormalize_latents(z)

        if self.reshape_to_2d:
            b, c, h, w = z.shape
            tokens = z.view(b, c, h * w).transpose(1, 2).contiguous()
        else:
            tokens = z

        logits = self.decoder(tokens, return_dict=True).logits
        x = self.decoder.unpatchify(logits)

        mean = self.encoder_mean.to(device=x.device, dtype=x.dtype)
        std = self.encoder_std.to(device=x.device, dtype=x.dtype)
        x = x * std + mean
        return x * 2.0 - 1.0

    def get_last_layer(self):
        return self.decoder.decoder_pred.weight

    def get_encoder_last_layer(self):
        params = list(self.encoder.parameters())
        if not params:
            raise RuntimeError("RAE encoder has no parameters")
        return params[-1]

    def get_autoencoder_params(self):
        return [p for p in super().get_autoencoder_params() if p.requires_grad]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    @torch.no_grad()
    def save_pretrained(self, save_dir, safe_serialization=True):
        rae = DiffusersAutoencoderRAE.from_config(self.config)
        sd = self.state_dict()
        state = {k: sd[k].detach().cpu() for k in rae.state_dict() if k in sd}
        missing, unexpected = rae.load_state_dict(state, strict=False)
        rae.save_pretrained(save_dir, safe_serialization=safe_serialization)
        return missing, unexpected


class DiffusersAutoencoderKLQwenImageWrapper(AutoencodingEngine):
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
            vae = DiffusersAutoencoderKLQwenImage.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder
            )
        else:
            config = DiffusersAutoencoderKLQwenImage.load_config(
                model_config_name_or_path
            )
            vae = DiffusersAutoencoderKLQwenImage.from_config(config)
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
        squeeze_temporal = x.dim() == 4
        if squeeze_temporal:
            x = x.unsqueeze(2)
        z = self._encode(x)
        if unregularized:
            if squeeze_temporal:
                z = z.squeeze(2)
            return z, dict()
        z, reg_log = self.regularization(z)
        if squeeze_temporal:
            z = z.squeeze(2)
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

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        squeeze_temporal = z.dim() == 4
        if squeeze_temporal:
            z = z.unsqueeze(2)
        out = self._decode(z)
        if squeeze_temporal:
            out = out.squeeze(2)
        return out

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
        vae = DiffusersAutoencoderKLQwenImage.from_config(self.config)
        sd = self.state_dict()
        state = {k: sd[k].detach().cpu() for k in vae.state_dict() if k in sd}
        missing, unexpected = vae.load_state_dict(state, strict=False)
        vae.save_pretrained(save_dir, safe_serialization=safe_serialization)
        return missing, unexpected
