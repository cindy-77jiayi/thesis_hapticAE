"""Temporal VQ-VAE for haptic waveform reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import clone_activation, make_activation, make_norm


class VectorQuantizer(nn.Module):
    """Straight-through vector quantizer for temporal latent sequences."""

    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_embeddings,
            1.0 / self.num_embeddings,
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantize latent tensor shaped ``(B, C, T)``."""
        z_btc = z.permute(0, 2, 1).contiguous()
        flat_z = z_btc.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view_as(z_btc)

        codebook_loss = F.mse_loss(quantized, z_btc.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z_btc)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = z_btc + (quantized - z_btc).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()

        encodings = F.one_hot(indices, self.num_embeddings).type(flat_z.dtype)
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "quantized": quantized,
            "indices": indices.view(z.shape[0], z.shape[-1]),
            "loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }


class ConvVQVAE(nn.Module):
    """1D convolutional VQ-VAE with temporal latent tokens.

    Unlike the global-vector VAE, this model keeps a downsampled time axis in
    the bottleneck so short events can be represented by local code tokens.
    """

    def __init__(
        self,
        T: int = 4000,
        channels: tuple[int, ...] = (64, 128, 256, 256),
        first_kernel: int = 31,
        kernel_size: int = 11,
        activation: str = "leaky_relu",
        norm: str = "group",
        embedding_dim: int = 64,
        codebook_size: int = 256,
        commitment_cost: float = 0.25,
        **_kwargs,
    ):
        super().__init__()
        self.T = int(T)
        self.embedding_dim = int(embedding_dim)
        self.codebook_size = int(codebook_size)

        act_fn = make_activation(activation)
        norm_fn = make_norm(norm)

        enc_layers = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            k = first_kernel if i == 0 else kernel_size
            enc_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=2, padding=k // 2),
                norm_fn(out_ch),
                clone_activation(act_fn),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)
        self.pre_quant = nn.Conv1d(channels[-1], self.embedding_dim, kernel_size=1)
        self.quantizer = VectorQuantizer(
            num_embeddings=self.codebook_size,
            embedding_dim=self.embedding_dim,
            commitment_cost=commitment_cost,
        )
        self.post_quant = nn.Conv1d(self.embedding_dim, channels[-1], kernel_size=1)

        dec_layers = []
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels)):
            in_ch = rev_channels[i]
            out_ch = rev_channels[i + 1] if i + 1 < len(rev_channels) else 1
            dec_layers.append(nn.Upsample(scale_factor=2, mode="linear", align_corners=False))
            dec_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            if out_ch > 1:
                dec_layers.append(norm_fn(out_ch))
                dec_layers.append(clone_activation(act_fn))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pre_quant(self.encoder(x))

    def decode(self, z_q: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        h = self.post_quant(z_q)
        x_hat = self.decoder(h)
        T = target_len or self.T
        if x_hat.shape[-1] > T:
            x_hat = x_hat[..., :T]
        elif x_hat.shape[-1] < T:
            x_hat = F.pad(x_hat, (0, T - x_hat.shape[-1]))
        return x_hat

    def forward(self, x: torch.Tensor):
        z_e = self.encode(x)
        q = self.quantizer(z_e)
        x_hat = self.decode(q["quantized"], target_len=x.shape[-1])
        return x_hat, q["quantized"], q["indices"], q["loss"], q["perplexity"]
