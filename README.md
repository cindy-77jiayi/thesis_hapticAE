# Haptic Signal VAE — Master's Thesis

Conv1D Variational Autoencoder for haptic vibrotactile signal reconstruction and controllable generation via PCA-discovered control dimensions.

## Project Structure

```
thesis/
├── README.md
├── requirements.txt
│
├── src/                    # Modular source code
│   ├── data/               # Dataset and preprocessing
│   ├── models/             # ConvVAE, ConvAE, shared building blocks
│   ├── training/           # Trainer, losses, KL schedulers
│   ├── eval/               # Evaluation, metrics, validation, visualization
│   ├── pipelines/          # PCA control, sweep, control specification
│   └── utils/              # Seed, config loading
│
├── scripts/                # CLI entry points
│   ├── train.py            # Model training
│   ├── eval.py             # Standalone evaluation
│   ├── extract_and_pca.py  # Latent extraction + PCA fitting
│   ├── build_controls.py   # Control spec, sweep gallery, table
│   ├── validate_extended.py # Full validation (27 metrics, dual-reference, PCA alignment)
│   └── cross_seed_stability.py  # Cross-seed PCA stability comparison
│
├── configs/                # YAML experiment configs
│   ├── vae_balanced.yaml   # Primary VAE (latent_dim=24, beta_max=0.003)
│   ├── vae_default.yaml    # Original VAE (latent_dim=64)
│   ├── vae_compact.yaml    # Compact VAE (latent_dim=16)
│   ├── vae_balanced_s123.yaml / s456.yaml  # Cross-seed variants
│   └── ae_matched.yaml     # Deterministic AE (same architecture, no KL)
│
├── colab/
│   └── train_colab.ipynb   # End-to-end Colab pipeline
│
├── notebooks_histories/    # Archived research notebooks
└── docs/                   # Documentation snapshots
```

## Quick Start

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

## Control Pipeline

1. **VAE Training** → latent space (24D)
2. **PCA** → 8 interpretable control dimensions (~59% variance)
3. **Sweep analysis** → quantify what each PC controls
4. **Validation** → Spearman ρ monotonicity, orthogonality, effect sizes, cross-seed stability

### Signal Metrics (27 total)

| Category | Metrics |
|----------|---------|
| Intensity | RMS energy, peak amplitude, crest factor |
| Spectral | Centroid, rolloff, slope, flatness, HF ratio, band energies (3 bands) |
| Envelope | Decay slope, late/early ratio, attack time, transient ratio, duration, area, entropy |
| Rhythm | Onset density, IOI entropy, onset interval CV, modulation peak |
| Continuity | Zero-crossing rate, gap ratio |
| Texture | Short-term variance, AM modulation index |

## Loss Components (VAE)

- **MSE**: Primary reconstruction loss
- **L1** (×0.2): Robust reconstruction
- **Multi-scale spectral** (×0.15): STFT magnitude matching at 4 scales
- **Amplitude** (×0.5): RMS + peak matching
- **KL divergence**: Free-bits (0.1 nats) with cyclical annealing (β_max=0.003)
