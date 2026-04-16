"""Haptic codec with residual vector quantization and a separate control branch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import clone_activation, make_activation, make_norm


class ResidualBlock1d(nn.Module):
    """Small residual block used in the encoder and decoder."""

    def __init__(self, channels: int, kernel_size: int = 7, activation: str = "leaky_relu", norm: str = "group"):
        super().__init__()
        norm_fn = make_norm(norm)
        act = make_activation(activation)
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            norm_fn(channels),
            clone_activation(act),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            norm_fn(channels),
        )
        self.act = clone_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class VectorQuantizer(nn.Module):
    """Straight-through vector quantizer for one codebook."""

    def __init__(self, codebook_size: int, code_dim: int, commitment_weight: float = 0.25):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.commitment_weight = float(commitment_weight)
        self.embedding = nn.Embedding(self.codebook_size, self.code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a sequence tensor of shape (B, C, T)."""
        bsz, channels, steps = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, channels)

        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view(bsz, steps, channels).permute(0, 2, 1).contiguous()

        codebook_loss = F.mse_loss(quantized, x.detach())
        commit_loss = self.commitment_weight * F.mse_loss(quantized.detach(), x)
        quantized_st = x + (quantized - x).detach()
        return quantized_st, indices.view(bsz, steps), codebook_loss + commit_loss


class ResidualVectorQuantizer(nn.Module):
    """Residual VQ stack, inspired by EnCodec-style bottlenecks."""

    def __init__(
        self,
        n_codebooks: int,
        codebook_size: int,
        code_dim: int,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    codebook_size=codebook_size,
                    code_dim=code_dim,
                    commitment_weight=commitment_weight,
                )
                for _ in range(int(n_codebooks))
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        residual = x
        quantized_total = torch.zeros_like(x)
        codes: list[torch.Tensor] = []
        stage_losses: list[torch.Tensor] = []

        for quantizer in self.quantizers:
            q, idx, loss = quantizer(residual)
            quantized_total = quantized_total + q
            residual = residual - q
            codes.append(idx)
            stage_losses.append(loss)

        codes_tensor = torch.stack(codes, dim=1)
        total_loss = torch.stack(stage_losses).sum()
        losses = {
            "vq_loss": total_loss,
            "vq_stage_losses": torch.stack(stage_losses),
        }
        return quantized_total, codes_tensor, losses


class UpsampleBlock1d(nn.Module):
    """Upsample + conv block that supports non-power-of-two strides."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        kernel_size: int = 9,
        activation: str = "leaky_relu",
        norm: str = "group",
        final: bool = False,
    ):
        super().__init__()
        norm_fn = make_norm(norm)
        act = make_activation(activation)
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=False),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        ]
        if not final:
            layers.extend([norm_fn(out_channels), clone_activation(act), ResidualBlock1d(out_channels, kernel_size, activation, norm)])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass
class HapticCodecOutputs:
    recon_seq: torch.Tensor
    recon_ctrl: torch.Tensor
    z_seq: torch.Tensor
    z_ctrl: torch.Tensor
    codes: torch.Tensor
    vq_losses: dict[str, torch.Tensor]


class HapticCodec(nn.Module):
    """Sequence codec for haptic waveforms with a separate control path."""

    def __init__(
        self,
        T: int = 40000,
        channels: tuple[int, ...] = (32, 64, 128, 128, 64),
        strides: tuple[int, ...] = (5, 4, 4, 2, 2),
        first_kernel: int = 25,
        kernel_size: int = 9,
        residual_kernel_size: int = 7,
        activation: str = "leaky_relu",
        norm: str = "group",
        code_dim: int = 64,
        n_codebooks: int = 4,
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
        control_dim: int = 16,
        control_hidden: int = 128,
        metric_dim: int = 8,
    ):
        super().__init__()
        if len(channels) != len(strides):
            raise ValueError("model.channels and model.strides must have the same length")
        if channels[-1] != code_dim:
            raise ValueError("The last encoder channel must equal code_dim for RVQ")

        self.T = int(T)
        self.code_dim = int(code_dim)
        self.control_dim = int(control_dim)
        self.metric_dim = int(metric_dim)
        self.strides = tuple(int(s) for s in strides)

        norm_fn = make_norm(norm)
        act = make_activation(activation)
        enc_layers: list[nn.Module] = []
        in_ch = 1
        for idx, (out_ch, stride) in enumerate(zip(channels, strides)):
            k = first_kernel if idx == 0 else kernel_size
            enc_layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=k // 2),
                    norm_fn(out_ch),
                    clone_activation(act),
                    ResidualBlock1d(out_ch, residual_kernel_size, activation, norm),
                ]
            )
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.T)
            enc = self.encoder(dummy)
        self.latent_frames = int(enc.shape[-1])
        self.encoder_channels = int(enc.shape[1])

        self.rvq = ResidualVectorQuantizer(
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
        )

        pooled_dim = self.encoder_channels * 2
        self.control_encoder = nn.Sequential(
            nn.Linear(pooled_dim, control_hidden),
            nn.LayerNorm(control_hidden),
            clone_activation(act),
            nn.Linear(control_hidden, control_dim),
        )
        self.control_decoder = nn.Sequential(
            nn.Linear(control_dim, control_hidden),
            clone_activation(act),
            nn.Linear(control_hidden, self.code_dim * self.latent_frames),
        )
        self.metric_head = nn.Sequential(
            nn.Linear(control_dim, control_hidden),
            clone_activation(act),
            nn.Linear(control_hidden, metric_dim),
        )

        dec_layers: list[nn.Module] = []
        rev_channels = list(reversed(channels))
        rev_strides = list(reversed(strides))
        for idx, stride in enumerate(rev_strides):
            in_channels = rev_channels[idx]
            out_channels = rev_channels[idx + 1] if idx + 1 < len(rev_channels) else 1
            final = idx == len(rev_strides) - 1
            dec_layers.append(
                UpsampleBlock1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    scale_factor=stride,
                    kernel_size=kernel_size,
                    activation=activation,
                    norm=norm,
                    final=final,
                )
            )
        self.decoder = nn.Sequential(*dec_layers)

    def _match_length(self, x: torch.Tensor, target_len: int | None) -> torch.Tensor:
        desired = int(target_len or self.T)
        if x.shape[-1] > desired:
            return x[..., :desired]
        if x.shape[-1] < desired:
            return F.pad(x, (0, desired - x.shape[-1]))
        return x

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def pool_control_features(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=-1)
        std = features.std(dim=-1, unbiased=False)
        pooled = torch.cat([mean, std], dim=1)
        return self.control_encoder(pooled)

    def encode_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode_features(x)
        z_seq, codes, _ = self.quantize_sequence(features)
        return z_seq, codes

    def quantize_sequence(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return self.rvq(features)

    def decode_sequence(self, z_seq: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        x_hat = self.decoder(z_seq)
        return self._match_length(x_hat, target_len)

    def encode_control(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode_features(x)
        return self.pool_control_features(features)

    def control_to_sequence(self, z_ctrl: torch.Tensor) -> torch.Tensor:
        coarse = self.control_decoder(z_ctrl).view(z_ctrl.shape[0], self.code_dim, self.latent_frames)
        return coarse

    def decode_control(self, z_ctrl: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        coarse = self.control_to_sequence(z_ctrl)
        return self.decode_sequence(coarse, target_len=target_len)

    def predict_metrics(self, z_ctrl: torch.Tensor) -> torch.Tensor:
        return self.metric_head(z_ctrl)

    def freeze_codec(self):
        for module in (self.encoder, self.rvq, self.decoder):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_codec(self):
        for module in (self.encoder, self.rvq, self.decoder):
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        features = self.encode_features(x)
        z_ctrl = self.pool_control_features(features)
        z_seq, codes, vq_losses = self.quantize_sequence(features)
        recon_seq = self.decode_sequence(z_seq, target_len=x.shape[-1])
        recon_ctrl = self.decode_control(z_ctrl, target_len=x.shape[-1])
        return recon_seq, recon_ctrl, z_seq, z_ctrl, codes, vq_losses
