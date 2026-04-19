import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.training.losses import (
        envelope_loss,
        isolated_peak_loss,
        second_diff_loss,
        temporal_derivative_loss,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in the current environment")
class TemporalDerivativeLossTest(unittest.TestCase):
    def test_temporal_derivative_loss_is_zero_for_matching_waveforms(self):
        x = torch.tensor([[[0.0, 1.0, 3.0, 6.0]]], dtype=torch.float32)
        x_hat = x.clone()

        loss = temporal_derivative_loss(x_hat, x, use_l1=True)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=7)

    def test_temporal_derivative_loss_detects_shifted_spike(self):
        x = torch.tensor([[[0.0, 0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
        x_hat = torch.tensor([[[0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

        loss = temporal_derivative_loss(x_hat, x, use_l1=True)

        self.assertGreater(float(loss.item()), 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in the current environment")
class AdditionalLossesTest(unittest.TestCase):
    def test_second_diff_loss_is_zero_for_matching_waveforms(self):
        x = torch.tensor([[[0.0, 1.0, 3.0, 6.0, 10.0]]], dtype=torch.float32)

        loss = second_diff_loss(x, x, use_l1=True)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=7)

    def test_isolated_peak_loss_penalizes_extra_spike(self):
        x = torch.zeros((1, 1, 9), dtype=torch.float32)
        x_hat = x.clone()
        x_hat[..., 4] = 1.0

        loss = isolated_peak_loss(x_hat, x, kernel_size=3)

        self.assertGreater(float(loss.item()), 0.0)

    def test_envelope_loss_is_zero_for_matching_waveforms(self):
        x = torch.tensor([[[0.0, 1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)

        loss = envelope_loss(x, x, kernel_size=3, use_l1=True)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
