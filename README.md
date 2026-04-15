# Haptic Signal VAE

This repository trains a Conv1D VAE / AE for early-stage haptic representation learning and PCA control discovery.

The maintained data flow is now:

1. Start from WavCaps raw audio + metadata.
2. Apply a HapticGen-style preparation step:
   - automated amplitude filtering
   - prompt variant augmentation
   - audio-to-haptic conversion
3. Train VAE / AE on 10-second prepared haptic waveforms at 8 kHz.
4. Run latent extraction, PCA, and validation.

The `llm/` assets and `run_llm_to_haptic.py` remain outside this training pipeline.

## Maintained Scripts

- `scripts/prepare_wavcaps_haptic_dataset.py`: prepare WavCaps into a HapticGen-style haptic dataset.
- `scripts/train.py`: train VAE / AE on prepared haptic waveforms.
- `scripts/eval.py`: evaluate a trained checkpoint.
- `scripts/extract_and_pca.py`: extract latent vectors and fit PCA.
- `scripts/build_controls.py`: build control summaries from PCA sweeps.
- `scripts/validate_extended.py`: run extended validation.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare WavCaps:

```bash
python scripts/prepare_wavcaps_haptic_dataset.py \
    --metadata_dir /path/to/WavCaps/data/json_files \
    --audio_dir /path/to/WavCaps/audio \
    --output_dir data/wavcaps_haptic_prepared \
    --create_variants
```

Train:

```bash
python scripts/train.py \
    --config configs/vae_balanced.yaml \
    --data_dir data/wavcaps_haptic_prepared \
    --output_dir outputs
```

Extract latents and fit PCA:

```bash
python scripts/extract_and_pca.py \
    --config configs/vae_balanced.yaml \
    --data_dir data/wavcaps_haptic_prepared \
    --checkpoint outputs/vae_balanced/best_model.pt \
    --output_dir outputs/pca
```

Build controls:

```bash
python scripts/build_controls.py \
    --config configs/vae_balanced.yaml \
    --data_dir data/wavcaps_haptic_prepared \
    --checkpoint outputs/vae_balanced/best_model.pt \
    --output_dir outputs/controls \
    --pca_dir outputs/pca
```

Run extended validation:

```bash
python scripts/validate_extended.py \
    --config configs/vae_balanced.yaml \
    --data_dir data/wavcaps_haptic_prepared \
    --checkpoint outputs/vae_balanced/best_model.pt \
    --output_dir outputs/validation \
    --controls_dir outputs/controls \
    --pca_dir outputs/pca \
    --seed_configs configs/vae_balanced.yaml configs/vae_balanced_s123.yaml configs/vae_balanced_s456.yaml \
    --seed_output_base outputs
```

## Notes

- The preparation script mirrors HapticGen's published baseline ideas rather than copying their full training stack.
- Prepared samples are written as `.wav` files with `.am1.json` sidecar metadata and dataset-level `manifest.jsonl` / `rejected.jsonl`.
- Training and PCA scripts operate on the prepared output directory, not raw WavCaps directly.
- The maintained training setup now uses fixed 10-second windows (`T=80000` at `8 kHz`) with HapticGen-style random segment sampling instead of short energy-picked crops.
