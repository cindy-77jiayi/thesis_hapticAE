import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from src.data.audio_utils import ensure_audio_channels, load_audio


class AudioUtilsTest(unittest.TestCase):
    def test_ensure_audio_channels_downmix_and_upmix(self):
        stereo = np.array([[1.0, -1.0, 0.5], [-1.0, 1.0, 0.5]], dtype=np.float32)
        mono = ensure_audio_channels(stereo, channels=1)
        self.assertEqual(mono.shape, (1, 3))
        self.assertTrue(np.allclose(mono[0], np.array([0.0, 0.0, 0.5], dtype=np.float32)))

        upmixed = ensure_audio_channels(mono, channels=3)
        self.assertEqual(upmixed.shape, (3, 3))
        self.assertTrue(np.allclose(upmixed[0], mono[0]))
        self.assertTrue(np.allclose(upmixed[1], mono[0]))

    def test_load_audio_resamples_and_downmixes(self):
        sr = 4000
        samples = np.stack([
            np.sin(np.linspace(0, np.pi * 4, 20, dtype=np.float32)),
            np.cos(np.linspace(0, np.pi * 4, 20, dtype=np.float32)),
        ], axis=1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, samples, sr)
            audio, loaded_sr = load_audio(tmp_path, target_sr=8000, target_channels=1)
        finally:
            os.unlink(tmp_path)

        self.assertEqual(loaded_sr, 8000)
        self.assertEqual(audio.shape[0], 1)
        self.assertGreater(audio.shape[1], samples.shape[0])
        self.assertTrue(np.isfinite(audio).all())


if __name__ == "__main__":
    unittest.main()
