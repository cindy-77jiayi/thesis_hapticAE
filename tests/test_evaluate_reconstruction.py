import importlib.util

import pytest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader

    from src.eval.evaluate import evaluate_reconstruction


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed in the current environment")
def test_evaluate_reconstruction_can_use_deterministic_mu_decode():
    class DummyVAE:
        def eval(self):
            return self

        def encode(self, x):
            batch = x.shape[0]
            mu = torch.full((batch, 2), 2.0, dtype=x.dtype, device=x.device)
            logvar = torch.zeros_like(mu)
            z = mu + 1.0
            return z, mu, logvar

        def decode(self, z, target_len):
            value = z[:, :1].unsqueeze(-1)
            return value.repeat(1, 1, target_len)

        def __call__(self, x):
            z, mu, logvar = self.encode(x)
            x_hat = self.decode(z, target_len=x.shape[-1])
            return x_hat, mu, logvar, z

    loader = DataLoader(torch.zeros(2, 1, 4), batch_size=2, shuffle=False)
    device = torch.device("cpu")

    deterministic = evaluate_reconstruction(
        DummyVAE(),
        loader,
        device,
        n_samples=2,
        deterministic=True,
        clamp_range=None,
    )
    stochastic = evaluate_reconstruction(
        DummyVAE(),
        loader,
        device,
        n_samples=2,
        deterministic=False,
        clamp_range=None,
    )

    assert deterministic["deterministic"] is True
    assert stochastic["deterministic"] is False
    assert float(deterministic["xhat_np"][0, 0]) == pytest.approx(2.0)
    assert float(stochastic["xhat_np"][0, 0]) == pytest.approx(3.0)
