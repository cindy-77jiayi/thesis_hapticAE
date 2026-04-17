import importlib.util
import tempfile
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    from src.training.checkpointing import prepare_run_dir
    from src.training.trainer import Trainer


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in the current environment")
class DeterministicVAELossTest(unittest.TestCase):
    def test_validation_loss_uses_encode_decode_path_when_enabled(self):
        class SentinelVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(0.0))
                self.forward_calls = 0
                self.encode_calls = 0
                self.decode_calls = 0

            def forward(self, x):
                self.forward_calls += 1
                batch = x.shape[0]
                device = x.device
                mu = torch.zeros(batch, 2, device=device)
                logvar = torch.zeros_like(mu)
                z = mu
                return x + 1.0 + self.bias, mu, logvar, z

            def encode(self, x):
                self.encode_calls += 1
                batch = x.shape[0]
                device = x.device
                z = torch.zeros(batch, 2, device=device)
                mu = torch.zeros_like(z)
                logvar = torch.zeros_like(z)
                return z, mu, logvar

            def decode(self, z, target_len=None):
                self.decode_calls += 1
                target_len = target_len or 32
                return torch.zeros(z.shape[0], 1, target_len, device=z.device) + self.bias

        config = {
            "run_name": "deterministic_val_case",
            "output_dir": tempfile.gettempdir(),
            "model_type": "vae",
            "data": {"T": 32, "sr": 8000},
            "model": {
                "latent_dim": 2,
                "channels": [4, 8],
                "first_kernel": 5,
                "kernel_size": 3,
                "activation": "leaky_relu",
                "norm": "group",
            },
            "training": {
                "batch_size": 2,
                "epochs": 1,
                "patience": 5,
                "min_delta": 1e-4,
                "early_stop_start": 1,
                "grad_clip": 1.0,
                "print_every": 1,
            },
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
            "scheduler": {"factor": 0.5, "patience": 2},
            "loss": {
                "l1_weight": 0.1,
                "spectral_weight": 0.0,
                "amplitude_weight": 0.0,
                "fft_weight": 0.0,
                "clamp_range": 3.0,
                "recon_time_weight": 1.0,
            },
            "validation": {"sample_every": 0, "n_samples": 2, "deterministic_vae": True},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config["output_dir"] = tmpdir
            artifacts = prepare_run_dir(config)
            model = SentinelVAE()
            trainer = Trainer(model, config, torch.device("cpu"), artifacts=artifacts)
            x = torch.randn(2, 1, 32)

            trainer._compute_loss(x, epoch=1, deterministic_vae=True)

            self.assertEqual(model.forward_calls, 0)
            self.assertEqual(model.encode_calls, 1)
            self.assertEqual(model.decode_calls, 1)


if __name__ == "__main__":
    unittest.main()
