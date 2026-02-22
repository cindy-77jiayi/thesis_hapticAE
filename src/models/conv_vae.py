"""Conv1D Variational Autoencoder for haptic signals."""

import torch
import torch.nn as nn


def _group_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)


class ConvVAE(nn.Module):
    """1D Convolutional VAE with configurable depth and width.

    Encoder uses strided convolutions for downsampling.
    Decoder uses Upsample + Conv1d to avoid checkerboard artifacts.
    """

    def __init__(
        self,
        T: int = 4000,
        latent_dim: int = 64,
        channels: tuple[int, ...] = (32, 64, 128, 128),
        first_kernel: int = 25,
        kernel_size: int = 9,
        activation: str = "leaky_relu",
        norm: str = "group",
        logvar_clip: tuple[float, float] = (-10.0, 10.0),
    ):
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim
        self.logvar_clip = logvar_clip

        act_fn = nn.LeakyReLU(0.2) if activation == "leaky_relu" else nn.ReLU()
        norm_fn = _group_norm if norm == "group" else nn.BatchNorm1d

        # --- Encoder ---
        enc_layers = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            k = first_kernel if i == 0 else kernel_size
            p = k // 2
            enc_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=2, padding=p),
                norm_fn(out_ch),
                act_fn if i < len(channels) - 1 else type(act_fn)(act_fn.negative_slope if hasattr(act_fn, 'negative_slope') else True),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, T)
            h = self.encoder(dummy)
            self.enc_shape = h.shape[1:]
            self.enc_feat = h.numel()

        self.fc_mu = nn.Linear(self.enc_feat, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_feat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.enc_feat)

        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -1.0)

        # --- Decoder (Upsample + Conv to avoid checkerboard artifacts) ---
        dec_layers = []
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels)):
            in_ch = rev_channels[i]
            out_ch = rev_channels[i + 1] if i + 1 < len(rev_channels) else 1
            dec_layers.append(nn.Upsample(scale_factor=2, mode="linear", align_corners=False))
            dec_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            if out_ch > 1:
                dec_layers.append(norm_fn(out_ch))
                dec_layers.append(type(act_fn)(act_fn.negative_slope if hasattr(act_fn, 'negative_slope') else True))
        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        logvar = torch.clamp(logvar, self.logvar_clip[0], self.logvar_clip[1])
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar_raw = self.fc_logvar(h_flat)
        z, logvar = self.reparameterize(mu, logvar_raw)
        return z, mu, logvar

    def decode(self, z: torch.Tensor, target_len: int | None = None):
        h = self.fc_dec(z).view(z.size(0), *self.enc_shape)
        x_hat = self.decoder(h)
        T = target_len or self.T
        if x_hat.shape[-1] > T:
            x_hat = x_hat[..., :T]
        elif x_hat.shape[-1] < T:
            x_hat = torch.nn.functional.pad(x_hat, (0, T - x_hat.shape[-1]))
        return x_hat

    def forward(self, x: torch.Tensor):
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z, target_len=x.shape[-1])
        return x_hat, mu, logvar, z
