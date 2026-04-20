"""Loss functions for haptic signal reconstruction."""

import torch
import torch.nn.functional as F
from collections.abc import Sequence


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


def multiscale_stft_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    stft_scales: Sequence[int],
    stft_hop_lengths: Sequence[int] | None = None,
    stft_win_lengths: Sequence[int] | None = None,
    stft_scale_weights: Sequence[float] | None = None,
    stft_linear_weight: float = 0.1,
    stft_log_weight: float = 0.1,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Configurable multi-scale STFT loss for reconstruction fidelity.

    Matching across multiple FFT resolutions encourages the decoder to
    reconstruct both fine transients and coarse envelopes. This often helps
    latent spaces organize into more stable, interpretable factors for PCA.
    """
    if len(stft_scales) == 0:
        raise ValueError("stft_scales must contain at least one FFT size")

    n_scales = len(stft_scales)
    hops = list(stft_hop_lengths) if stft_hop_lengths is not None else [s // 4 for s in stft_scales]
    wins = list(stft_win_lengths) if stft_win_lengths is not None else list(stft_scales)
    weights = list(stft_scale_weights) if stft_scale_weights is not None else [1.0] * n_scales

    if len(hops) != n_scales:
        raise ValueError("stft_hop_lengths length must match stft_scales length")
    if len(wins) != n_scales:
        raise ValueError("stft_win_lengths length must match stft_scales length")
    if len(weights) != n_scales:
        raise ValueError("stft_scale_weights length must match stft_scales length")

    x_1d = x.squeeze(1)
    xh_1d = x_hat.squeeze(1)
    total = torch.tensor(0.0, device=x.device)

    for n_fft, hop, win, scale_w in zip(stft_scales, hops, wins, weights):
        n_fft_i = int(n_fft)
        hop_i = int(hop)
        win_i = int(win)
        if n_fft_i <= 0 or hop_i <= 0 or win_i <= 0:
            raise ValueError("All STFT sizes/hops/windows must be positive integers")
        if win_i > n_fft_i:
            raise ValueError(f"win_length ({win_i}) must be <= n_fft ({n_fft_i})")

        window = torch.hann_window(win_i, device=x.device, dtype=x.dtype)
        x_spec = torch.stft(
            x_1d, n_fft_i, hop_length=hop_i, win_length=win_i,
            return_complex=True, window=window,
        )
        xh_spec = torch.stft(
            xh_1d, n_fft_i, hop_length=hop_i, win_length=win_i,
            return_complex=True, window=window,
        )

        x_mag = torch.abs(x_spec)
        xh_mag = torch.abs(xh_spec)

        scale_loss = torch.tensor(0.0, device=x.device)
        if stft_linear_weight > 0:
            scale_loss = scale_loss + stft_linear_weight * torch.mean(torch.abs(x_mag - xh_mag))
        if stft_log_weight > 0:
            scale_loss = scale_loss + stft_log_weight * torch.mean(
                torch.abs(torch.log(x_mag + eps) - torch.log(xh_mag + eps))
            )

        total = total + float(scale_w) * scale_loss

    return total


def amplitude_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Penalize mismatch in RMS energy and peak amplitude."""
    rms_x = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
    rms_xh = torch.sqrt(torch.mean(x_hat ** 2, dim=-1, keepdim=True) + 1e-8)
    rms_diff = torch.mean((rms_x - rms_xh) ** 2)

    peak_x = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    peak_xh = torch.amax(torch.abs(x_hat), dim=-1, keepdim=True)
    peak_diff = torch.mean((peak_x - peak_xh) ** 2)

    return rms_diff + peak_diff


def temporal_derivative_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    use_l1: bool = True,
) -> torch.Tensor:
    """Penalize mismatch in first-order temporal differences."""
    dx = x[..., 1:] - x[..., :-1]
    dx_hat = x_hat[..., 1:] - x_hat[..., :-1]

    if use_l1:
        return torch.mean(torch.abs(dx_hat - dx))
    return torch.mean((dx_hat - dx) ** 2)


def _check_odd_kernel(name: str, kernel_size: int) -> None:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")


def _local_abs_envelope(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    _check_odd_kernel("kernel_size", kernel_size)
    return F.avg_pool1d(torch.abs(x), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def event_weighted_recon_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    kernel_size: int = 41,
    emphasis: float = 2.0,
    floor: float = 1.0,
    use_l1: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weight reconstruction errors higher around target events.

    The target waveform defines the event mask, so quiet regions contribute
    less while true bursts are harder for the model to ignore.
    """
    envelope = _local_abs_envelope(x, kernel_size=kernel_size)
    peak = torch.amax(envelope, dim=-1, keepdim=True).clamp_min(eps)
    weights = float(floor) + float(emphasis) * (envelope / peak)
    weights = weights / torch.mean(weights, dim=-1, keepdim=True).clamp_min(eps)

    if use_l1:
        error = torch.abs(x_hat - x)
    else:
        error = (x_hat - x) ** 2
    return torch.mean(weights * error)


def local_energy_recall_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    kernel_size: int = 65,
    under_weight: float = 2.0,
    over_weight: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize missing local RMS energy more than overshooting it."""
    _check_odd_kernel("local_energy_kernel", kernel_size)
    x_energy = torch.sqrt(
        F.avg_pool1d(x ** 2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2) + eps
    )
    xh_energy = torch.sqrt(
        F.avg_pool1d(x_hat ** 2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2) + eps
    )

    missed = torch.relu(x_energy - xh_energy)
    overshot = torch.relu(xh_energy - x_energy)
    return float(under_weight) * torch.mean(missed) + float(over_weight) * torch.mean(overshot)


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
