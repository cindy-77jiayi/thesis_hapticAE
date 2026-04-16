# Temporal Visual Event-to-Haptic Pipeline

Conv1D variational autoencoder for vibrotactile waveform reconstruction from temporally grounded visual events via a fixed semantic-first control space.

## Runtime Pipeline

```text
3 ordered images
-> OpenAI semantic interpretation
-> canonical semantic controls
-> PCA vector
-> local frozen VAE decode
-> generated wav + waveform preview
```

The runtime source of truth is:

- `semantic_control_schema.json`
- `src/semantic/pc_semantics.py`
- `src/semantic/mapping.py`
- `src/pipelines/semantic_reconstruction.py`

The current `data/actions/` path is a legacy name. Each folder should be treated as one visual-event window containing ordered keyframes plus metadata.

## Canonical Semantic Space

The semantic contract is fixed to 5 normalized controls:

- `frequency`
- `intensity`
- `envelope_modulation`
- `temporal_grouping`
- `sharpness`

These map onto PCA axes as:

- `PC1 = frequency`
- `PC2 = intensity` with inverted PCA direction
- `PC3 = envelope_modulation`
- `PC4 = temporal_grouping`
- `PC5 = sharpness`

Important inversion rule for `PC2`:

- lower `PC2` = higher physical amplitude
- higher semantic `intensity` = stronger vibration

LLM-facing outputs must use canonical semantic keys only. They must not emit raw `PC1..PC8` values, waveform parameters, or free-form text prompts.

## Main Entry Points

Start the local UI:

```bash
python launch_vae_ui.py --frozen_manifest <path-to-frozen_manifest.json>
```

Run the CLI over a segment manifest:

```bash
python run_llm_to_haptic.py --manifest <segment-manifest.json> --frozen_manifest <path-to-frozen_manifest.json>
```

Open the notebook walkthrough:

- `colab/vae_ui_pipeline.ipynb`

`--frozen_manifest` is required because the repository does not track a single canonical frozen runtime bundle.

## Frozen Runtime Contract

The frozen manifest must resolve these fields:

- `config_path`
- `checkpoint_path`
- `pca_dir`

The runtime uses the frozen config's `data.sr` and `data.T` as the decode sample rate and segment length.

## Training / Calibration

The generation stack remains:

1. Train or load the frozen VAE
2. Extract latent vectors
3. Fit PCA / Varimax controls
4. Map semantic controls into PCA coefficients
5. Decode waveform from the frozen model

Useful entrypoints:

- `scripts/train.py`
- `scripts/extract_and_pca.py`
- `scripts/build_controls.py`
- `scripts/validate_extended.py`
- `colab/frozen_vae_pca_workflow.ipynb`

## Rule-Based Fallback

`baseline/rule_based_controls.py` provides coarse fallback presets when no structured semantic output is available. It is a fallback only, not a second runtime pipeline.
