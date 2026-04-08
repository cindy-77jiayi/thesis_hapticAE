import importlib.util
import tempfile
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader

    from src.models.conv_ae import ConvAE
    from src.training.checkpointing import prepare_run_dir
    from src.training.trainer import Trainer


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in the current environment")
class TrainerResumeTest(unittest.TestCase):
    def test_trainer_saves_and_restores_resume_state(self):
        torch.manual_seed(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "run_name": "resume_case",
                "output_dir": tmpdir,
                "model_type": "ae",
                "data": {"T": 32, "sr": 8000},
                "model": {
                    "latent_dim": 4,
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
                "ema": {"use": True, "decay": 0.9},
                "checkpoint": {"save_last": True, "save_best": True, "save_every": 0, "keep_last": 0},
                "validation": {"sample_every": 0, "n_samples": 2},
            }

            artifacts = prepare_run_dir(config)
            device = torch.device("cpu")
            model = ConvAE(T=32, latent_dim=4, channels=(4, 8), first_kernel=5, kernel_size=3)
            trainer = Trainer(model, config, device, artifacts=artifacts)

            dataset = torch.randn(6, 1, 32)
            train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
            val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

            trainer.train(train_loader, val_loader)

            self.assertTrue(trainer.best_epoch >= 1)
            self.assertTrue(trainer.best_state is not None)

            restored = Trainer(
                ConvAE(T=32, latent_dim=4, channels=(4, 8), first_kernel=5, kernel_size=3),
                config,
                device,
                artifacts=artifacts,
            )
            restored.restore(artifacts.last_checkpoint_path)

            self.assertEqual(restored.start_epoch, 2)
            self.assertEqual(len(restored.history), 1)
            self.assertIsNotNone(restored.ema)
            self.assertAlmostEqual(restored.ema.decay, 0.9)


if __name__ == "__main__":
    unittest.main()
