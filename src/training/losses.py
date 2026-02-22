"""Loss functions for haptic signal reconstruction."""

import torch


def multi_scale_spectral_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Multi-scale spectral loss comparing STFT magnitudes at multiple resolutions.

    Computes both L1 magnitude distance and log-magnitude distance
    across FFT sizes [128, 256, 512, 1024].
    """
    loss = torch.tensor(0.0, device=x.device)
    x_1d = x.squeeze(1)
    xh_1d = x_hat.squeeze(1)

    for n_fft in [128, 256, 512, 1024]:
        window = torch.hann_window(n_fft, device=x.device)
        hop = n_fft // 4
        x_spec = torch.stft(x_1d, n_fft, hop_length=hop, return_complex=True, window=window)
        xh_spec = torch.stft(xh_1d, n_fft, hop_length=hop, return_complex=True, window=window)

        x_mag = torch.abs(x_spec)
        xh_mag = torch.abs(xh_spec)

        loss = loss + torch.mean(torch.abs(x_mag - xh_mag))
        loss = loss + torch.mean(torch.abs(
            torch.log(x_mag + 1e-7) - torch.log(xh_mag + 1e-7)
        ))

    return loss / 8  # 4 scales x 2 losses


def amplitude_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Penalize mismatch in RMS energy and peak amplitude."""
    rms_x = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
    rms_xh = torch.sqrt(torch.mean(x_hat ** 2, dim=-1, keepdim=True) + 1e-8)
    rms_diff = torch.mean((rms_x - rms_xh) ** 2)

    peak_x = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    peak_xh = torch.amax(torch.abs(x_hat), dim=-1, keepdim=True)
    peak_diff = torch.mean((peak_x - peak_xh) ** 2)

    return rms_diff + peak_diff


def kl_divergence_free_bits(
    mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.1
) -> torch.Tensor:
    """Free-bits KL divergence to prevent posterior collapse.

    Each latent dimension is allowed at least `free_bits` nats of information.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=1).mean()


def fft_mag_mse(
    x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8, use_log: bool = True
) -> torch.Tensor:
    """Single-scale FFT magnitude MSE (used by non-VAE AE baselines)."""
    x_fft = torch.fft.rfft(x)
    xh_fft = torch.fft.rfft(x_hat)
    x_mag = torch.abs(x_fft)
    xh_mag = torch.abs(xh_fft)

    if use_log:
        x_mag = torch.log(x_mag + eps)
        xh_mag = torch.log(xh_mag + eps)

    return torch.mean((x_mag - xh_mag) ** 2)
