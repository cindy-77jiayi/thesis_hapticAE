import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.training.losses import (
        event_envelope_loss,
        event_local_rms_loss,
        event_onset_loss,
        smooth_abs_envelope,
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
class EventAwareLossTest(unittest.TestCase):
    def test_event_losses_are_zero_for_matching_waveforms(self):
        x = torch.tensor([[[0.0, 0.5, 1.0, 0.5, 0.0]]], dtype=torch.float32)
        x_hat = x.clone()

        self.assertAlmostEqual(
            float(event_envelope_loss(x_hat, x, windows=[3]).item()),
            0.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(event_local_rms_loss(x_hat, x, windows=[3]).item()),
            0.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(event_onset_loss(x_hat, x, windows=[3]).item()),
            0.0,
            places=7,
        )

    def test_event_losses_detect_missing_event_energy(self):
        x = torch.tensor([[[0.0, 0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
        x_hat = torch.zeros_like(x)

        self.assertGreater(float(event_envelope_loss(x_hat, x, windows=[3]).item()), 0.0)
        self.assertGreater(float(event_local_rms_loss(x_hat, x, windows=[3]).item()), 0.0)
        self.assertGreater(float(event_onset_loss(x_hat, x, windows=[3]).item()), 0.0)

    def test_smooth_abs_envelope_preserves_length_for_even_windows(self):
        x = torch.zeros((2, 1, 9), dtype=torch.float32)

        envelope = smooth_abs_envelope(x, window_size=4)

        self.assertEqual(envelope.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
