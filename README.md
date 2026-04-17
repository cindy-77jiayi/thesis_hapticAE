# Haptic Signal VAE - Master's Thesis

Conv1D Variational Autoencoder for vibrotactile waveform reconstruction, with the current branch focused on direct native-latent training and inspection.

## Project Structure

```text
thesis/
|-- README.md
|-- requirements.txt
|-- src/
|   |-- data/
|   |-- eval/
|   |-- models/
|   |-- pipelines/
|   `-- utils/
|-- scripts/
|-- baseline/
|   `-- rule_based_controls.py
|-- data/actions/
|-- colab/
|   `-- train_colab.ipynb
`-- docs/
```

## Current Workflow

This branch uses a direct latent workflow instead of a PCA-derived control space:

1. Train an 8D or other configured VAE with `scripts/train.py`
2. Load the trained checkpoint
3. Extract deterministic latent means with `src/pipelines/latent_extraction.py`
4. Analyze or sweep native latent axes with `src/pipelines/native_latent.py`
5. Inspect reconstruction quality and latent behavior in `colab/train_colab.ipynb`

## Key Entry Points

- `scripts/train.py`
- `configs/vae_balanced_8d.yaml`
- `src/pipelines/latent_extraction.py`
- `src/pipelines/native_latent.py`
- `colab/train_colab.ipynb`

## Notes

- The training loop, model definitions, and checkpoint flow remain unchanged.
- The Colab notebook is the main training-and-observation entry point on this branch.
- Older PCA-specific control and semantic labeling code has been removed from this branch to keep the workflow aligned with direct latent training.
