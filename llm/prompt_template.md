# Segment-Level Visual-to-Haptic Semantic Predictor

You are given one temporal segment from a short visual event.

- The segment duration is always 0.5 seconds.
- You will receive 1-3 ordered keyframes for this segment.
- The image files are supplied separately as images in this request.
- The JSON payload below gives timing and context metadata for the same segment.

Your task:
- Infer vibrotactile semantics for this segment only.
- Think in tactile / haptic terms, not audio terms.
- Output normalized semantic controls in **[0, 1]**.
- Base your answer on the temporal change implied by the ordered keyframes plus the metadata.

## Canonical Semantic Controls

1. `frequency`
- Higher value = denser / higher-frequency vibration.

2. `intensity`
- Higher value = stronger vibration.
- This is a semantic intensity, not a raw PCA value.

3. `envelope_modulation`
- Higher value = more pulsing / swelling / breathing.
- Lower value = steadier / more continuous.

4. `temporal_grouping`
- Higher value = more grouped / rhythmic / segmented timing.

5. `sharpness`
- Higher value = sharper / clickier / more impulsive transients.

## Output Rules

- Output **ONLY valid JSON**.
- No markdown and no explanation outside JSON.
- Use only the required keys.
- Every control value must be numeric and in `[0, 1]`.
- Do **not** output raw PCA coefficients such as `PC1`, `PC2`, or waveform parameters.
- Keep each rationale string short and visually grounded.

## Required Output JSON

```json
{
  "semantic_controls": {
    "frequency": 0.0,
    "intensity": 0.0,
    "envelope_modulation": 0.0,
    "temporal_grouping": 0.0,
    "sharpness": 0.0
  },
  "rationale": {
    "frequency": "...",
    "intensity": "...",
    "envelope_modulation": "...",
    "temporal_grouping": "...",
    "sharpness": "..."
  }
}
```

## Segment Metadata

```json
{{segment_payload_json}}
```
