"""Conv1D Autoencoder (no variational component) for haptic signals."""

import torch
import torch.nn as nn


class ConvAE(nn.Module):
    """1D Convolutional Autoencoder.

    Similar architecture to ConvVAE but without the reparameterization trick.
    The bottleneck is a deterministic latent vector.
    """

    def __init__(
        self,
        T: int = 4000,
        latent_dim: int = 64,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 9,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim

        norm_fn = nn.BatchNorm1d if use_batchnorm else lambda c: nn.GroupNorm(8, c)

        # --- Encoder ---
        enc_layers = []
        in_ch = 1
        for out_ch in channels:
            enc_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
                norm_fn(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, T)
            h = self.encoder(dummy)
            self.enc_shape = h.shape[1:]
            self.enc_feat = h.numel()

        self.fc_enc = nn.Linear(self.enc_feat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.enc_feat)

        # --- Decoder ---
        dec_layers = []
        rev = list(reversed(channels))
        for i in range(len(rev)):
            in_ch = rev[i]
            out_ch = rev[i + 1] if i + 1 < len(rev) else 1
            dec_layers.extend([
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, output_padding=1),
            ])
            if out_ch > 1:
                dec_layers.extend([norm_fn(out_ch), nn.ReLU()])
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.fc_enc(h.view(h.size(0), -1))

    def decode(self, z: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), *self.enc_shape)
        x_hat = self.decoder(h)
        T = target_len or self.T
        if x_hat.shape[-1] > T:
            x_hat = x_hat[..., :T]
        elif x_hat.shape[-1] < T:
            x_hat = torch.nn.functional.pad(x_hat, (0, T - x_hat.shape[-1]))
        return x_hat

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z, target_len=x.shape[-1])
        return x_hat, z
