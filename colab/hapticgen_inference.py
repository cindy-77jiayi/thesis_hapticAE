"""Colab-friendly HapticGen inference script for the 3-image OpenAI validation flow.

Example usage in Colab after cloning this repo:

    pip install -r HapticGen/requirements.txt
    python colab/hapticgen_inference.py --prompt "A short sharp click followed by a soft fading buzz."

If direct HF loading fails, the script falls back to snapshot_download and retries with the local directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAPTICGEN_ROOT = PROJECT_ROOT / "HapticGen"
if str(HAPTICGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(HAPTICGEN_ROOT))

from audiocraft.data.audio import audio_write  # noqa: E402
from audiocraft.models import AudioGen  # noqa: E402


DEFAULT_MODEL_ID = "HapticGen/HapticGen-Weights"


def load_hapticgen_model(model_id: str, device: str | None = None) -> tuple[AudioGen, str]:
    """Load a HapticGen model from HF or a local snapshot fallback."""
    try:
        model = AudioGen.get_pretrained(model_id, device=device)
        return model, model_id
    except Exception:
        snapshot_dir = snapshot_download(repo_id=model_id)
        model = AudioGen.get_pretrained(snapshot_dir, device=device)
        return model, snapshot_dir


def save_waveform_plot(wav_tensor: torch.Tensor, sample_rate: int, output_path: Path) -> Path:
    """Save a waveform preview PNG for one generated sample."""
    waveform = wav_tensor.detach().cpu().squeeze().numpy()
    times = [idx / sample_rate for idx in range(len(waveform))]
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    ax.plot(times, waveform, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("HapticGen Output Waveform")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HapticGen inference for one text prompt")
    parser.add_argument("--prompt", type=str, default=None, help="Direct HapticGen prompt text")
    parser.add_argument(
        "--response_json",
        type=str,
        default=None,
        help="Path to an OpenAI response JSON containing `haptic_prompt`",
    )
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output_dir", type=str, default="outputs/hapticgen_colab")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    prompt = args.prompt
    if args.response_json:
        payload = json.loads(Path(args.response_json).read_text(encoding="utf-8"))
        prompt = payload.get("haptic_prompt", prompt)
    if not prompt:
        raise ValueError("Provide either --prompt or --response_json with a haptic_prompt field.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, loaded_from = load_hapticgen_model(args.model_id, device=args.device)
    model.set_generation_params(duration=args.duration)
    generated = model.generate([prompt])
    wav_tensor = generated[0].detach().cpu()

    wav_path = output_dir / "generated"
    audio_write(
        str(wav_path),
        wav_tensor,
        model.sample_rate,
        format="wav",
        add_suffix=False,
        normalize=False,
    )
    wav_file = output_dir / "generated.wav"
    waveform_path = save_waveform_plot(wav_tensor, model.sample_rate, output_dir / "generated_waveform.png")

    metadata = {
        "prompt": prompt,
        "model_id": args.model_id,
        "loaded_from": loaded_from,
        "duration": args.duration,
        "sample_rate": model.sample_rate,
        "wav_path": str(wav_file),
        "waveform_image_path": str(waveform_path),
    }
    (output_dir / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved wav: {wav_file}")
    print(f"Saved waveform: {waveform_path}")
    print(f"Saved metadata: {output_dir / 'generation_metadata.json'}")


if __name__ == "__main__":
    main()
