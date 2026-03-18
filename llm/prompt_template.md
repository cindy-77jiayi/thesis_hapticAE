# Semantic-to-Haptic Attribute Predictor (MVP)

You are given:
- 2-3 UI action frames (`before`, `during`, `after`)
- A short text context describing the user action and UI state transition

Your task:
- Infer tactile semantics from action meaning, visual dynamics, and UI state change.
- Think in tactile/haptic terms, not audio terms.
- Output normalized semantic attributes in **[0, 1]**.

## Attribute Definitions

1. `energy_roughness`
- stronger / rougher / more intense tactile feel

2. `temporal_irregularity`
- more irregular / jittery / less uniform timing structure

3. `modulation_texture`
- more modulated / textured / vibration-rich internal oscillation

4. `decay_envelope`
- longer / more sustained / more lingering envelope

## Output Rules

- Output **ONLY valid JSON**.
- No markdown, no explanations outside JSON, no extra keys.
- All numeric values must be in [0, 1].
- `rationale` strings should be short and grounded in the given UI action/context.

## Required Output JSON Format

```json
{
  "energy_roughness": 0.0,
  "temporal_irregularity": 0.0,
  "modulation_texture": 0.0,
  "decay_envelope": 0.0,
  "rationale": {
    "energy_roughness": "...",
    "temporal_irregularity": "...",
    "modulation_texture": "...",
    "decay_envelope": "..."
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
