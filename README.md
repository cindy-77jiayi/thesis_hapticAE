# Haptic Signal Autoencoder — Master's Thesis

Conv1D Variational Autoencoder for haptic vibrotactile signal reconstruction and generation.

## Project Structure

```
thesis/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                    # Modular source code
│   ├── data/               # Dataset classes and preprocessing
│   ├── models/             # ConvVAE, ConvAE, SimpleAE
│   ├── training/           # Trainer, losses, KL schedulers
│   ├── eval/               # Evaluation, visualization, audio playback
│   └── utils/              # Seed, config loading
│
├── scripts/                # CLI entry points
│   ├── train.py            # python scripts/train.py --config ... --data_dir ... --output_dir ...
│   └── eval.py             # python scripts/eval.py --config ... --checkpoint ... --data_dir ...
│
├── configs/                # YAML experiment configs
│   ├── vae_default.yaml    # Best VAE configuration
│   └── ae_baseline.yaml    # AE baseline (no KL)
│
├── colab/                  # Google Colab notebooks
│   └── train_colab.ipynb   # End-to-end Colab training
│
├── notebooks/              # Original research notebooks (archived)
│
├── data/                   # Local data (git-ignored)
├── outputs/                # Training outputs (git-ignored)
└── docs/                   # Documentation
```

## Quick Start

### Local Training

```bash
pip install -r requirements.txt

python scripts/train.py \
    --config configs/vae_default.yaml \
    --data_dir /path/to/hapticgen-dataset/expertvoted \
    --output_dir outputs
```

### Google Colab

Open `colab/train_colab.ipynb` in Colab, which will:
1. Mount Google Drive
2. Clone this repo
3. Install dependencies
4. Run `scripts/train.py` with Drive paths

### Evaluation

```bash
python scripts/eval.py \
    --config configs/vae_default.yaml \
    --data_dir /path/to/data \
    --checkpoint outputs/vae_default/best_model.pt
```

## Model Architectures

| Model | Type | Latent Dim | Channels | Description |
|-------|------|-----------|----------|-------------|
| `ConvVAE` | VAE | 64 | 32→64→128→128 | Main model with Upsample+Conv decoder |
| `ConvAE` | AE | 64 | 16→32→64 | Deterministic baseline |
| `SimpleAE` | AE | 16 | FC only | Sanity check |

## Loss Components (VAE)

- **MSE**: Primary reconstruction loss
- **L1** (×0.2): Robust reconstruction
- **Multi-scale spectral** (×0.15): STFT magnitude matching at 4 scales
- **Amplitude** (×0.5): RMS + peak matching to prevent amplitude compression
- **KL divergence**: Free-bits (0.1 nats) with cyclical annealing (β_max=0.0001)

## Experiment Management

Instead of maintaining multiple notebook versions, use:
- **Config files** in `configs/` to define experiments
- **Git commits** for version control
- **`outputs/`** for timestamped run results
