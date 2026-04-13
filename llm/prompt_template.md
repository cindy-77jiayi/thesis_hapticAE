# Semantic-to-Haptic Control Predictor

You are given:
- 2-3 UI action frames (`before`, `during`, `after`)
- A short text context describing the user action and UI state transition

Your task:
- Infer tactile semantics from action meaning, visual dynamics, and UI state change.
- Think in tactile / vibrotactile terms, not audio terms.
- Output normalized semantic controls in **[0, 1]**.

## Canonical Semantic Controls

1. `frequency`
- Controls oscillation density.
- Higher value = denser / higher-frequency vibration.

2. `intensity`
- Controls overall vibration strength.
- Higher value = stronger vibration.
- Important: this is a semantic control, not a raw PCA value.

3. `envelope_modulation`
- Controls temporal energy variation.
- Higher value = more pulsing / swelling / breathing.
- Lower value = steadier / more continuous.

4. `temporal_grouping`
- Controls rhythmic grouping or segmentation over time.
- Higher value = more grouped / rhythmic / segmented.

5. `sharpness`
- Controls transient crispness and local waveform sharpness.
- Higher value = sharper / clickier / more impulsive.

## Output Rules

- Output **ONLY valid JSON**.
- No markdown, no explanations outside JSON, no extra keys.
- All numeric values must be in [0, 1].
- Do **not** output `PC1`, `PC2`, or any other raw PCA coefficients.
- `rationale` strings should be short and grounded in the given UI action/context.

## Required Output JSON Format

```json
{
  "frequency": 0.0,
  "intensity": 0.0,
  "envelope_modulation": 0.0,
  "temporal_grouping": 0.0,
  "sharpness": 0.0,
  "rationale": {
    "frequency": "...",
    "intensity": "...",
    "envelope_modulation": "...",
    "temporal_grouping": "...",
    "sharpness": "..."
  }
}
```

## Input Payload

```json
{
  "frames": {
    "before": "{{before_image_path}}",
    "during": "{{during_image_path}}",
    "after": "{{after_image_path}}"
  },
  "action_name": "{{action_name}}",
  "context": "{{context}}",
  "notes": "{{notes}}"
}
```
