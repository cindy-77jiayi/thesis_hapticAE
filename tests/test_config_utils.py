import os
import tempfile
import textwrap
import unittest

from src.utils.config import load_config


class ConfigUtilsTest(unittest.TestCase):
    def test_load_config_supports_inheritance_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "base.yaml")
            child_path = os.path.join(tmpdir, "child.yaml")

            with open(base_path, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent("""
                run_name: base_run
                training:
                  epochs: 5
                loss:
                  amplitude_weight: 0.1
                """).strip())

            with open(child_path, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent("""
                base_config: base.yaml
                training:
                  batch_size: 4
                ema:
                  use: true
                """).strip())

            config = load_config(
                child_path,
                overrides=["training.epochs=7", "loss.amplitude_weight=0.25"],
            )

        self.assertEqual(config["run_name"], "base_run")
        self.assertEqual(config["training"]["batch_size"], 4)
        self.assertEqual(config["training"]["epochs"], 7)
        self.assertAlmostEqual(config["loss"]["amplitude_weight"], 0.25)
        self.assertTrue(config["ema"]["use"])
        self.assertIn("checkpoint", config)


if __name__ == "__main__":
    unittest.main()
