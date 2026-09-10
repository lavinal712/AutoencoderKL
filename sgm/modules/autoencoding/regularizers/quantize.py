import logging
from abc import abstractmethod
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from torch.distributed.nn.functional import all_reduce

from .base import AbstractRegularizer, measure_perplexity, compute_entropy_loss

logpy = logging.getLogger(__name__)


class AbstractQuantizer(AbstractRegularizer):
    def __init__(self):
        super().__init__()
        # Define these in your init
        # shape (N,)
        self.used: Optional[torch.Tensor]
        self.re_embed: int
        self.unknown_index: Union[Literal["random"], int]

    def remap_to_used(self, inds: torch.Tensor) -> torch.Tensor:
        assert self.used is not None, "You need to define used indices for remap"
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.Tensor) -> torch.Tensor:
        assert self.used is not None, "You need to define used indices for remap"
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    @abstractmethod
    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        raise NotImplementedError()

    def get_trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self.parameters()


class GumbelQuantizer(AbstractQuantizer):
    """
    credit to @karpathy:
    https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(
        self,
        num_hiddens: int,
        embedding_dim: int,
        n_embed: int,
        straight_through: bool = True,
        kl_weight: float = 5e-4,
        temp_init: float = 1.0,
        remap: Optional[str] = None,
        unknown_index: str = "random",
        loss_key: str = "loss/vq",
    ) -> None:
        super().__init__()

        self.loss_key = loss_key
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
        else:
            self.used = None
            self.re_embed = n_embed
        if unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        else:
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            self.unknown_index = unknown_index  # "random" or "extra" or integer
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )

    def forward(
        self, z: torch.Tensor, temp: Optional[float] = None, return_logits: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        # force hard = True when we are in eval mode, as we must quantize.
        # actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp
        out_dict = {}
        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, self.used, ...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:, self.used, ...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = (
            self.kl_weight
            * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        )
        out_dict[self.loss_key] = diff

        ind = soft_one_hot.argmax(dim=1)
        out_dict["indices"] = ind
        if self.remap is not None:
            ind = self.remap_to_used(ind)

        if return_logits:
            out_dict["logits"] = logits

        return z_q, out_dict

    def get_codebook_entry(self, indices, shape):
        # TODO: shape not yet optional
        b, h, w, c = shape
        assert b * h * w == indices.shape[0]
        indices = rearrange(indices, "(b h w) -> b h w", b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = (
            F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        )
        z_q = einsum("b n h w, n d -> b d h w", one_hot, self.embed.weight)
        return z_q


class VectorQuantizer(AbstractQuantizer):
    """
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term,
        beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        remap: Optional[str] = None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        legacy: bool = False,
        log_perplexity: bool = False,
        embedding_weight_norm: bool = False,
        entropy_loss_ratio: float = 0.0,
        l2_norm: bool = False,
        loss_key: str = "loss/vq",
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.loss_key = loss_key

        assert not (embedding_weight_norm and l2_norm)
        if not embedding_weight_norm:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.embedding = torch.nn.utils.parametrizations.weight_norm(
                nn.Embedding(self.n_e, self.e_dim), dim=1
            )

        if self.l2_norm:
            self.embedding.weight.data = F.normalize(
                self.embedding.weight.data, p=2, dim=-1
            )

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
        else:
            self.used = None
            self.re_embed = n_e
        if unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        else:
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            self.unknown_index = unknown_index  # "random" or "extra" or integer
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )

        self.sane_index_shape = sane_index_shape
        self.log_perplexity = log_perplexity

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        do_reshape = z.ndim == 4
        if do_reshape:
            # reshape z -> (batch, height, width, channel) and flatten
            z = rearrange(z, "b c h w -> b h w c").contiguous()
        else:
            assert z.ndim < 4, "No reshaping strategy for inputs > 4 dimensions defined"
            z = z.contiguous()

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.addmm(
            torch.sum(embedding**2, dim=1),
            z_flattened,
            embedding.t(),
            alpha=-2.0,
        )
        d.add_(torch.sum(z_flattened**2, dim=1, keepdim=True))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, embedding).view(z.shape)
        loss_dict = {}
        if self.log_perplexity:
            perplexity, cluster_usage = measure_perplexity(
                min_encoding_indices.detach(), self.n_e, self.training
            )
            loss_dict.update({"perplexity": perplexity, "cluster_usage": cluster_usage})

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + \
                self.beta * torch.mean((z_q - z.detach()) ** 2)
        if self.entropy_loss_ratio > 0.0:
            loss = loss + self.entropy_loss_ratio * compute_entropy_loss(-d)
        loss_dict[self.loss_key] = loss

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        if do_reshape:
            z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1
            )  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            if do_reshape:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3]
                )
            else:
                min_encoding_indices = rearrange(
                    min_encoding_indices, "(b s) 1 -> b s", b=z_q.shape[0]
                )

        loss_dict["min_encoding_indices"] = min_encoding_indices

        return z_q, loss_dict

    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            assert shape is not None, "Need to give shape for remap"
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
            z_q = F.embedding(indices, F.normalize(embedding, p=2, dim=-1))
        else:
            z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.reshape(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class EmbeddingEMA(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        codebook_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        self.register_buffer("weight", weight)
        self.register_buffer("cluster_size", torch.zeros(num_tokens))
        self.register_buffer("embed_avg", torch.zeros_like(weight))
        self.update = True

    def forward(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.weight)

    @torch.no_grad()
    def cluster_size_ema_update(self, new_cluster_size: torch.Tensor) -> None:
        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size, alpha=1.0 - self.decay
        )

    @torch.no_grad()
    def embed_avg_ema_update(self, new_embed_avg: torch.Tensor) -> None:
        self.embed_avg.data.mul_(self.decay).add_(
            new_embed_avg, alpha=1.0 - self.decay
        )

    @torch.no_grad()
    def weight_update(self, num_tokens: int) -> None:
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        # normalize embedding average with smoothed cluster size
        embed_normalized = (
            self.embed_avg / smoothed_cluster_size.clamp_min(self.eps).unsqueeze(1)
        )
        active = self.cluster_size > 0
        self.weight.copy_(
            torch.where(active.unsqueeze(1), embed_normalized, self.weight)
        )


class EMAVectorQuantizer(AbstractQuantizer):
    def __init__(
        self,
        n_embed: int,
        embedding_dim: int,
        beta: float,
        decay: float = 0.99,
        eps: float = 1e-5,
        remap: Optional[str] = None,
        unknown_index: str = "random",
        loss_key: str = "loss/vq",
    ):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.loss_key = loss_key

        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
        else:
            self.used = None
            self.re_embed = n_embed
        if unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        else:
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            self.unknown_index = unknown_index  # "random" or "extra" or integer
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.num_tokens} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )

    def forward(
        self,
        z: torch.Tensor,
        return_encodings: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        # reshape z -> (batch, height, width, channel) and flatten
        # z, 'b c h w -> b h w c'
        z = rearrange(z, "b c h w -> b h w c")
        z_flattened = z.reshape(-1, self.codebook_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.addmm(
            self.embedding.weight.pow(2).sum(dim=1),
            z_flattened,
            self.embedding.weight.t(),
            alpha=-2.0,
        )
        d.add_(z_flattened.pow(2).sum(dim=1, keepdim=True))

        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).to(z.dtype).view(z.shape)
        cluster_size = torch.bincount(encoding_indices, minlength=self.num_tokens)

        if self.training and self.embedding.update:
            embed_sum = torch.zeros(
                self.num_tokens, self.codebook_dim, device=z.device, dtype=z.dtype
            )
            embed_sum.index_add_(0, encoding_indices, z_flattened.detach())
            # synchronize across distributed processes if needed
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
                dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)
            # EMA cluster size
            self.embedding.cluster_size_ema_update(cluster_size)
            # EMA embedding average
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.num_tokens)

        avg_probs = cluster_size / cluster_size.sum().clamp_min(1.0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs.clamp_min(1e-10)))
        )

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, "b h w c -> b c h w")

        out_dict = {
            self.loss_key: loss,
            "perplexity": perplexity,
        }

        if return_encodings:
            out_dict["encodings"] = F.one_hot(
                encoding_indices, num_classes=self.num_tokens
            ).type(z.dtype)

        if self.remap is not None:
            encoding_indices = encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            encoding_indices = self.remap_to_used(encoding_indices)
            encoding_indices = encoding_indices.reshape(-1)  # flatten
        
        out_dict["encoding_indices"] = encoding_indices

        return z_q, out_dict

    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            assert shape is not None, "Need to give shape for remap"
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class ResidualVectorQuantizer(AbstractQuantizer):
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        num_quantizers: int = 4,
        shared_codebook: bool = False,
        log_perplexity: bool = False,
        embedding_weight_norm: bool = False,
        l2_norm: bool = False,
        loss_key: str = "loss/vq",
    ):
        super().__init__()
        assert num_quantizers > 0, "Number of quantizers must be positive"
        self.n_e = n_e
        self.e_dim = e_dim
        self.num_quantizers = num_quantizers
        self.shared_codebook = shared_codebook
        self.beta = beta
        self.log_perplexity = log_perplexity
        self.loss_key = loss_key

        if self.shared_codebook:
            codebook = VectorQuantizer(
                n_e=n_e,
                e_dim=e_dim,
                beta=beta,
                log_perplexity=log_perplexity,
                embedding_weight_norm=embedding_weight_norm,
                l2_norm=l2_norm,
                loss_key=loss_key,
            )
            self.codebooks = nn.ModuleList([codebook for _ in range(num_quantizers)])
        else:
            self.codebooks = nn.ModuleList([
                VectorQuantizer(
                    n_e=n_e,
                    e_dim=e_dim,
                    beta=beta,
                    log_perplexity=log_perplexity,
                    embedding_weight_norm=embedding_weight_norm,
                    l2_norm=l2_norm,
                    loss_key=loss_key,
                )
                for _ in range(num_quantizers)
            ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        residual = z
        quantized = torch.zeros_like(z)

        all_indices = []
        all_losses = []
        all_perplexities = []
        all_cluster_usages = []

        for i, codebook in enumerate(self.codebooks):
            z_q, out_dict = codebook(residual)

            residual = residual - z_q.detach()
            quantized = quantized + z_q.detach()

            all_indices.append(out_dict["min_encoding_indices"].reshape(-1))
            all_losses.append(out_dict[self.loss_key])

            if self.log_perplexity:
                all_perplexities.append(out_dict["perplexity"])
                all_cluster_usages.append(out_dict["cluster_usage"])

        min_encoding_indices = torch.stack(all_indices, dim=1)

        loss_dict = {
            self.loss_key: torch.stack(all_losses).mean(),
            "min_encoding_indices": min_encoding_indices,
        }

        if self.log_perplexity:
            loss_dict.update({
                "perplexity": torch.stack(all_perplexities).mean(),
                "cluster_usage": torch.stack(all_cluster_usages).float().mean(),
            })

        # preserve gradients
        z_q = z + (quantized - z).detach()

        return z_q, loss_dict

    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        quantized_levels = [
            codebook.get_codebook_entry(indices[:, i], shape)
            for i, codebook in enumerate(self.codebooks)
        ]
        z_q = torch.stack(quantized_levels, dim=0).sum(dim=0)

        return z_q


class FiniteScalarQuantizer(AbstractQuantizer):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        loss_key: str = "loss/vq",
    ):
        super().__init__()
        self.levels = levels
        self.codebook_dim = len(levels)
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = self.codebook_dim * num_codebooks

        if keep_num_codebooks_dim is None:
            keep_num_codebooks_dim = num_codebooks > 1
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        self.dim = self.effective_codebook_dim if dim is None else dim
        self.scale = scale
        self.loss_key = loss_key

        _levels = torch.tensor(levels, dtype=torch.long)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.long), dim=0)
        self.register_buffer("_basis", _basis, persistent=False)

        self.has_projections = self.dim != self.effective_codebook_dim
        if self.has_projections:
            self.project_in = nn.Linear(self.dim, self.effective_codebook_dim)
            self.project_out = nn.Linear(self.effective_codebook_dim, self.dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        self.codebook_size = int(self._levels.prod().item())
        self.n_e = self.codebook_size

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels.to(z) - 1) * (1 - eps) / 2
        offset = (self._levels % 2 == 0).to(z) * 0.5
        shift = torch.tan(offset / half_l)
        bounded = torch.tanh(z + shift) * half_l - offset
        return bounded

    @staticmethod
    def round_ste(z: torch.Tensor) -> torch.Tensor:
        """Round with straight through gradients."""
        zhat = z.round()
        return z + (zhat - z).detach()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        is_image = z.ndim == 4
        input_shape = z.shape
        if is_image:
            z = rearrange(z, "b d ... -> b ... d").contiguous()

        z = z.reshape(z.shape[0], -1, self.dim)
        z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        z_q = self.round_ste(self.bound(z))
        half_width = self._levels // 2
        z_q = z_q / half_width

        level_indices = (z_q * half_width + half_width).round().long()
        indices = (level_indices * self._basis).sum(dim=-1)

        z_q = rearrange(z_q, "b n c d -> b n (c d)")
        z_q = self.project_out(z_q)

        if is_image:
            z_q = z_q.reshape(input_shape[0], *input_shape[2:], self.dim)
            z_q = rearrange(z_q, "b ... d -> b d ...").contiguous()
            indices = indices.reshape(
                input_shape[0], *input_shape[2:], self.num_codebooks
            )
        else:
            z_q = z_q.reshape(input_shape)
            indices = indices.reshape(*input_shape[:-1], self.num_codebooks)

        if not self.keep_num_codebooks_dim:
            indices = indices.squeeze(-1)

        loss_dict = {
            self.loss_key: z_q.new_zeros(()),
            "indices": indices,
        }

        return z_q, loss_dict

    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        if shape is None:
            is_image = indices.ndim >= 3 + int(self.keep_num_codebooks_dim)
        else:
            assert len(shape) == 4, "Shape must be (batch, height, width, channels)"
            is_image = True

            if self.keep_num_codebooks_dim:
                indices = indices.reshape(-1, self.num_codebooks)
            else:
                indices = indices.reshape(-1)
        
        if self.keep_num_codebooks_dim:
            assert indices.shape[-1] == self.num_codebooks
        else:
            indices = indices.unsqueeze(-1)
        
        level_indices = (indices.unsqueeze(-1) // self._basis) % self._levels
        half_width = self._levels // 2
        z_q = (level_indices - half_width) / half_width

        z_q = z_q.flatten(-2)
        z_q = self.project_out(z_q)

        if shape is not None:
            z_q = z_q.reshape(shape)
        if is_image:
            z_q = rearrange(z_q, "b ... d -> b d ...").contiguous()

        return z_q


class LookupFreeQuantizer(AbstractQuantizer):
    def __init__(
        self,
        dim: Optional[int] = None,
        codebook_size: Optional[int] = None,
        entropy_loss_weight: float = 0.1,
        commitment_loss_weight: float = 0.0,
        diversity_gamma: float = 1.0,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        codebook_scale: float = 1.0,
        frac_per_sample_entropy: float = 1.0,
        has_projections: Optional[bool] = None,
        projection_has_bias: bool = True,
        soft_clamp_input_value: Optional[float] = None,
        spherical: bool = False,
        inv_temperature: float = 100.0,
        loss_key: str = "loss/vq",
    ):
        super().__init__()
        self.codebook_size = codebook_size if codebook_size is not None else 1 << dim
        assert self.codebook_size >= 2
        assert (self.codebook_size & (self.codebook_size - 1)) == 0
        self.codebook_dim = self.codebook_size.bit_length() -1
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = self.codebook_dim * num_codebooks
        self.dim = self.effective_codebook_dim if dim is None else dim

        if keep_num_codebooks_dim is None:
            keep_num_codebooks_dim = num_codebooks > 1
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.has_projections = (
            self.dim != self.effective_codebook_dim
            if has_projections is None else has_projections
        )
        assert self.has_projections or self.dim == self.effective_codebook_dim
        if self.has_projections:
            self.project_in = nn.Linear(
                self.dim, self.effective_codebook_dim, bias=projection_has_bias
            )
            self.project_out = nn.Linear(
                self.effective_codebook_dim, self.dim, bias=projection_has_bias
            )
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        self.entropy_loss_weight = entropy_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.diversity_gamma = diversity_gamma
        self.codebook_scale = codebook_scale
        self.frac_per_sample_entropy = frac_per_sample_entropy
        self.soft_clamp_input_value = soft_clamp_input_value
        self.spherical = spherical
        self.inv_temperature = inv_temperature
        self.loss_key = loss_key

        bit_mask = 2 ** torch.arange(self.codebook_dim - 1, -1, -1, dtype=torch.long)
        self.register_buffer("_bit_mask", bit_mask, persistent=False)

        all_indices = torch.arange(self.codebook_size, dtype=torch.long)
        bits = (all_indices.unsqueeze(-1) & bit_mask) != 0
        self.register_buffer(
            "codebook", self.bits_to_codes(bits.float()), persistent=False
        )

    def bits_to_codes(self, bits: torch.Tensor) -> torch.Tensor:
        return (bits * 2.0 - 1.0) * self.codebook_scale

    @staticmethod
    def entropy(probs: torch.Tensor) -> torch.Tensor:
        return -(probs * probs.clamp_min(1e-5).log()).sum(dim=-1)

    def compute_entropy(
        self, z: torch.Tensor, inv_temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: (tokens, num_codebooks, codebook_dim).
        if self.frac_per_sample_entropy < 1.0:
            num_tokens = z.shape[0]
            num_samples = max(1, int(num_tokens * self.frac_per_sample_entropy))
            sampled_indices = torch.randperm(num_tokens, device=z.device)[:num_samples]
            z = z[sampled_indices]

        codebook = self.codebook.float()
        if self.spherical:
            codebook = F.normalize(codebook, dim=-1) * self.codebook_scale

        logits = 2.0 * inv_temperature * einsum("n c d, k d -> n c k", z, codebook)
        probs = logits.softmax(dim=-1)
        prob_sum = probs.sum(dim=0)
        entropy_sum = self.entropy(probs).sum(dim=0)
        num_tokens = probs.new_tensor(probs.shape[0])

        if dist.is_available() and dist.is_initialized():
            # Use differentiable reductions for probability and entropy sums.
            prob_sum = all_reduce(prob_sum)
            entropy_sum = all_reduce(entropy_sum)
            dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

        per_sample_entropy = (entropy_sum / num_tokens).mean()
        codebook_entropy = self.entropy(prob_sum / num_tokens).mean()
        return per_sample_entropy, codebook_entropy

    def forward(
        self,
        z: torch.Tensor,
        inv_temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        inv_temperature = (
            self.inv_temperature if inv_temperature is None else inv_temperature
        )

        input_shape = z.shape
        is_image_or_video = z.ndim >= 4
        if is_image_or_video:
            z = rearrange(z, "b d ... -> b ... d").contiguous()

        z = self.project_in(z.reshape(-1, self.dim))
        z = z.reshape(-1, self.num_codebooks, self.codebook_dim)
        orig_dtype = z.dtype

        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            if self.soft_clamp_input_value is not None:
                clamp_value = self.soft_clamp_input_value
                z = torch.tanh(z / clamp_value) * clamp_value
            if self.spherical:
                z = F.normalize(z, dim=-1) * self.codebook_scale

            bits = z > 0
            indices = (bits.long() * self._bit_mask).sum(dim=-1)
            z_q = self.bits_to_codes(bits.to(z.dtype))
            if self.spherical:
                z_q = F.normalize(z_q, dim=-1) * self.codebook_scale

            per_sample_entropy = z.new_zeros(())
            codebook_entropy = z.new_zeros(())
            commitment_loss = z.new_zeros(())
            if self.training:
                if self.entropy_loss_weight > 0.0:
                    per_sample_entropy, codebook_entropy = self.compute_entropy(
                        z, inv_temperature
                    )
                if self.commitment_loss_weight > 0.0:
                    commitment_loss = F.mse_loss(z, z_q.detach())

                # Preserve gradients through the quantized representation.
                z_q = z + (z_q - z).detach()

            entropy_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
            loss = (
                self.entropy_loss_weight * entropy_loss
                + self.commitment_loss_weight * commitment_loss
            )

        z_q = z_q.to(orig_dtype).flatten(-2)
        z_q = self.project_out(z_q)

        if is_image_or_video:
            z_q = z_q.reshape(input_shape[0], *input_shape[2:], self.dim)
            z_q = rearrange(z_q, "b ... d -> b d ...").contiguous()
            indices = indices.reshape(
                input_shape[0], *input_shape[2:], self.num_codebooks
            )
        else:
            z_q = z_q.reshape(input_shape)
            indices = indices.reshape(*input_shape[:-1], self.num_codebooks)

        if not self.keep_num_codebooks_dim:
            indices = indices.squeeze(-1)

        loss_dict = {
            self.loss_key: loss,
            "indices": indices,
            "per_sample_entropy": per_sample_entropy.detach(),
            "codebook_entropy": codebook_entropy.detach(),
            "commitment_loss": commitment_loss.detach(),
        }

        return z_q, loss_dict

    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        # shape specifies (batch, height, width, channels), or the video analogue.
        if shape is not None:
            assert len(shape) >= 4 and shape[-1] == self.dim
            indices = indices.reshape(*shape[:-1], self.num_codebooks)
            is_image_or_video = True
        else:
            is_image_or_video = indices.ndim >= 3 + int(self.keep_num_codebooks_dim)
            if self.keep_num_codebooks_dim:
                assert indices.shape[-1] == self.num_codebooks
            else:
                indices = indices.unsqueeze(-1)

        bits = (indices.long().unsqueeze(-1) & self._bit_mask) != 0
        z_q = self.bits_to_codes(bits.float())
        if self.spherical:
            z_q = F.normalize(z_q, dim=-1) * self.codebook_scale

        z_q = z_q.to(self.codebook.dtype).flatten(-2)
        z_q = self.project_out(z_q)
        if is_image_or_video:
            z_q = rearrange(z_q, "b ... d -> b d ...").contiguous()
        return z_q
