# Temporal Visual Event-to-Haptic Pipeline - Master's Thesis

Conv1D Variational Autoencoder for vibrotactile waveform reconstruction from temporally grounded visual events via semantic-first control mapping.

## Project Structure

```text
thesis/
|-- README.md
|-- requirements.txt
|-- semantic_control_schema.json
|-- src/
|   |-- data/
|   |-- eval/
|   |-- models/
|   |-- pipelines/
|   |-- semantic/
|   `-- utils/
|-- scripts/
|-- baseline/
|   `-- rule_based_controls.py
|-- llm/
|   `-- prompt_template.md
|-- data/actions/
|-- run_llm_to_haptic.py
|-- colab/
|   `-- frozen_vae_pca_workflow.ipynb
`-- docs/
```

The current `data/actions/` path is a legacy name. In the updated project framing, each folder should be treated as one visual-event window containing 3-5 keyframes plus metadata.

## Canonical Semantic Space

The semantic source of truth is:

- `PC1 = frequency`
- `PC2 = intensity` (inverted in PCA space)
- `PC3 = envelope_modulation`
- `PC4 = temporal_grouping`
- `PC5 = sharpness`

Important inversion rule for `PC2`:

- lower `PC2` = higher physical amplitude
- higher semantic `intensity` = stronger vibration

The runtime source of truth is:

- `semantic_control_schema.json`
- `src/semantic/pc_semantics.py`
- `src/semantic/mapping.py`
- `run_llm_to_haptic.py`

The Colab notebook is retained as labeling / calibration evidence and is not the runtime pipeline.

## Semantic-to-Haptic Pipeline

```text
temporal visual event window (3-5 keyframes)
-> LLM semantic interpretation
-> canonical semantic controls
-> PCA vector
-> haptic waveform
```

The intended runtime input is a short sequence of temporally ordered visual frames representing an event such as a water droplet falling, a phone swipe completing, or a person running past. Repeated LLM calls over consecutive frame windows can be used to assemble a final 2-3 second haptic vibration.

The semantic layer exposes a normal-direction control object:

```json
{
  "frequency": 0.5,
  "intensity": 0.5,
  "envelope_modulation": 0.5,
  "temporal_grouping": 0.5,
  "sharpness": 0.5
}
```

Internally, `intensity` is mapped to the inverted `PC2` axis.

LLM-facing outputs must use canonical semantic keys only. They must not emit raw `PC1..PC8` values or low-level waveform parameters.

The current semantic layer predicts one control vector per visual-event window. Longer vibrations are expected to be produced by chaining multiple temporally adjacent windows rather than by inferring a full 2-3 second waveform from a single still image.

## Training / Frozen Pipeline

The current generation stack remains:

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
- `run_llm_to_haptic.py`

## Rule-Based Fallback

`baseline/rule_based_controls.py` provides coarse semantic fallback presets when no structured LLM semantic output is available. It is a fallback only, not a second control system.
