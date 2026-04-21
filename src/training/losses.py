"""Loss functions for haptic signal reconstruction."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


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


def _same_length_avg_pool1d(x: torch.Tensor, window_size: int) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    left = (window_size - 1) // 2
    right = window_size // 2
    padded = F.pad(x, (left, right), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=window_size, stride=1)


def _as_window_list(windows: Sequence[int]) -> list[int]:
    if len(windows) == 0:
        raise ValueError("event windows must contain at least one window size")
    return [int(window) for window in windows]


def smooth_abs_envelope(
    x: torch.Tensor,
    window_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Smoothed absolute-value envelope with input length preserved."""
    return _same_length_avg_pool1d(torch.abs(x), window_size).clamp_min(eps)


def event_envelope_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    windows: Sequence[int] = (32, 64, 128),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compare log energy envelopes so events can match despite phase offsets."""
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    window_list = _as_window_list(windows)
    for window_size in window_list:
        x_env = torch.log(smooth_abs_envelope(x, window_size, eps=eps))
        xh_env = torch.log(smooth_abs_envelope(x_hat, window_size, eps=eps))
        total = total + torch.mean(torch.abs(xh_env - x_env))
    return total / len(window_list)


def event_local_rms_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    windows: Sequence[int] = (32, 64, 128),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compare multi-scale local RMS to preserve perceived event intensity."""
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    window_list = _as_window_list(windows)
    for window_size in window_list:
        x_rms = torch.sqrt(_same_length_avg_pool1d(x ** 2, window_size) + eps)
        xh_rms = torch.sqrt(_same_length_avg_pool1d(x_hat ** 2, window_size) + eps)
        total = total + torch.mean(torch.abs(xh_rms - x_rms))
    return total / len(window_list)


def event_onset_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    windows: Sequence[int] = (32, 64, 128),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compare positive log-envelope slopes to align event onsets."""
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    window_list = _as_window_list(windows)
    for window_size in window_list:
        x_env = torch.log(smooth_abs_envelope(x, window_size, eps=eps))
        xh_env = torch.log(smooth_abs_envelope(x_hat, window_size, eps=eps))
        x_onset = torch.relu(x_env[..., 1:] - x_env[..., :-1])
        xh_onset = torch.relu(xh_env[..., 1:] - xh_env[..., :-1])
        total = total + torch.mean(torch.abs(xh_onset - x_onset))
    return total / len(window_list)


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
