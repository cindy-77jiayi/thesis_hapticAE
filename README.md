# Haptic Signal VAE — Master's Thesis

Conv1D Variational Autoencoder for haptic vibrotactile signal reconstruction and experimental controllable generation.

## Project Structure

```
thesis/
├── README.md
├── requirements.txt
├── semantic_control_schema.json  # Canonical semantic control schema
│
├── src/                    # Modular source code
│   ├── data/               # Dataset and preprocessing
│   ├── models/             # ConvVAE, ConvAE, shared building blocks
│   ├── training/           # Trainer, losses, KL schedulers
│   ├── eval/               # Evaluation, metrics, validation, visualization
│   ├── pipelines/          # PCA control, sweep, control specification
│   ├── semantic/           # Canonical semantic layer and semantic↔PCA mapping
│   └── utils/              # Seed, config loading
│
├── scripts/                # CLI entry points
│   ├── train.py            # Model training
│   ├── eval.py             # Standalone evaluation
│   ├── extract_and_pca.py  # Latent extraction + PCA fitting
│   ├── build_controls.py   # Control artifact generation
│   ├── validate_extended.py # Extended validation
│   └── cross_seed_stability.py  # Cross-seed PCA stability comparison
│
├── configs/                # YAML experiment configs
│   ├── vae_balanced.yaml   # Primary VAE (latent_dim=24, beta_max=0.003)
│   ├── vae_default.yaml    # Original VAE (latent_dim=64)
│   ├── vae_compact.yaml    # Compact VAE (latent_dim=16)
│   ├── vae_balanced_s123.yaml / s456.yaml  # Cross-seed variants
│   └── ae_matched.yaml     # Deterministic AE (same architecture, no KL)
│
├── controls/               # Compatibility wrappers for semantic mapping
│   ├── control_schema.json # Deprecated compatibility shim
│   └── mapping.py          # Backward-compatible semantic mapping wrapper
│
├── baseline/               # Rule-based haptic presets
│   └── rule_based_controls.py  # Action-type → attribute defaults
│
├── llm/                    # LLM prompt assets
│   └── prompt_template.md  # Multimodal prompt for attribute prediction
│
├── data/actions/            # Action dataset (frames + metadata)
│   └── example_action_dataset_schema.json
│
├── run_llm_to_haptic.py    # MVP: action → attributes → PC vector → haptic
│
├── colab/
│   └── train_colab.ipynb   # End-to-end Colab pipeline
│
├── notebooks_histories/    # Archived research notebooks
└── docs/                   # Documentation snapshots
```

## Quick Start

## Canonical Semantic Space

The PCA semantic source of truth is:

- `PC1 = frequency`
- `PC2 = intensity` (inverted in PCA space)
- `PC3 = envelope_modulation`
- `PC4 = temporal_grouping`
- `PC5 = sharpness`

Important inversion rule for `PC2`:

- lower `PC2` = higher physical amplitude
- higher semantic `intensity` = stronger vibration

The semantic mapping layer therefore exposes a normal-direction semantic field:

```json
{
  "frequency": 0.5,
  "intensity": 0.5,
  "envelope_modulation": 0.5,
  "temporal_grouping": 0.5,
  "sharpness": 0.5
}
```

but internally maps `intensity` to the inverted `PC2` axis.

### Google Colab (recommended)

Open `colab/train_colab.ipynb` — it handles everything: clone, install, train, PCA, validate.

### Local Training

```bash
pip install -r requirements.txt

python scripts/train.py \
    --config configs/vae_balanced.yaml \
    --data_dir /path/to/hapticgen-dataset/expertvoted \
    --output_dir outputs
```

### Training Quality-of-Life Improvements

The training CLI now supports lightweight config overrides and resumable checkpoints:

```bash
# override a few values without editing YAML
python scripts/train.py \
    --config configs/vae_balanced.yaml \
    --data_dir /path/to/hapticgen-dataset/expertvoted \
    --output_dir outputs \
    --set training.epochs=50 \
    --set ema.use=true \
    --set validation.sample_every=10

# resume an interrupted run
python scripts/train.py \
    --config configs/vae_balanced.yaml \
    --data_dir /path/to/hapticgen-dataset/expertvoted \
    --output_dir outputs \
    --resume outputs/vae_balanced/last_checkpoint.pt
```

Each run now saves:

- `resolved_config.yaml`: fully merged config after inheritance and CLI overrides
- `data_split.json`: persisted train/val split manifest for reproducible eval/resume
- `last_checkpoint.pt`: full resumable training state
- `best_model.pt`: best model weights for downstream PCA/eval scripts
- `validation_samples/epoch_XXX/`: periodic reconstruction previews when enabled
- `history.json` and `metrics.npz`: per-epoch metrics history

### Full Pipeline

```bash
# 1. Train
python scripts/train.py --config configs/vae_balanced.yaml --data_dir DATA --output_dir outputs

# 2. Extract latents + PCA
python scripts/extract_and_pca.py --config configs/vae_balanced.yaml --data_dir DATA \
    --checkpoint outputs/vae_balanced/best_model.pt --output_dir outputs/pca

# 3. Build control specification
python scripts/build_controls.py --config configs/vae_balanced.yaml --data_dir DATA \
    --checkpoint outputs/vae_balanced/best_model.pt --output_dir outputs/controls \
    --pca_dir outputs/pca

# 4. Extended validation (27 metrics, dual-reference, cross-seed alignment)
python scripts/validate_extended.py --config configs/vae_balanced.yaml --data_dir DATA \
    --checkpoint outputs/vae_balanced/best_model.pt --output_dir outputs/validation \
    --controls_dir outputs/controls --pca_dir outputs/pca \
    --seed_configs configs/vae_balanced.yaml configs/vae_balanced_s123.yaml configs/vae_balanced_s456.yaml \
    --seed_output_base outputs
```

## Model Architectures

| Model | Type | Latent Dim | Channels | Description |
|-------|------|-----------|----------|-------------|
| `ConvVAE` | VAE | 24 | 32→64→128→128 | Main model with Upsample+Conv decoder |
| `ConvAE` | AE | 24 | 32→64→128→128 | Deterministic baseline (same architecture) |

## Semantic-to-Haptic Pipeline

Canonical semantic prediction from UI actions, mapped to haptic signals via the trained model pipeline.

```
UI / multimodal input → LLM semantic interpretation → semantic controls → PCA vector → Haptic waveform
```

LLM-facing outputs should use canonical semantic keys only:

```json
{
  "frequency": 0.7,
  "intensity": 0.8,
  "envelope_modulation": 0.6,
  "temporal_grouping": 0.4,
  "sharpness": 0.5
}
```

LLM outputs should not emit raw `PC1..PC8` values or low-level waveform parameters.

### Quick Run (rule-based, no LLM needed)

```bash
python run_llm_to_haptic.py \
    --action_dir data/actions/action_001 \
    --output_dir outputs/llm_mvp \
    --use_rule_based
```

### With LLM Output

```bash
python run_llm_to_haptic.py \
    --action_dir data/actions/action_001 \
    --output_dir outputs/llm_mvp \
    --llm_output_path path/to/llm_response.json
```

### With Full Decoder (generates waveform)

```bash
python run_llm_to_haptic.py \
    --action_dir data/actions/action_001 \
    --output_dir outputs/llm_mvp \
    --llm_output_path path/to/llm_response.json \
    --config configs/vae_balanced.yaml \
    --checkpoint outputs/vae_balanced/best_model.pt \
    --pca_dir outputs/pca
```

## Loss Components (VAE)

- **MSE**: Primary reconstruction loss
- **L1** (×0.2): Robust reconstruction
- **Multi-scale spectral** (×0.15): STFT magnitude matching at 4 scales
- **Amplitude** (×0.5): RMS + peak matching
- **KL divergence**: Free-bits (0.1 nats) with cyclical annealing (β_max=0.003)
