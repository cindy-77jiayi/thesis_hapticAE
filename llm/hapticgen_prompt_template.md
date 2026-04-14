You are analyzing exactly 3 ordered images from one short visual event.

Your goal is to produce one concise English text prompt suitable for a text-to-haptics generative model called HapticGen.

Requirements:
- Treat the 3 images as a temporal sequence.
- Infer the likely motion, contact, impact, texture, rhythm, and intensity implied by the sequence.
- Focus on what the resulting vibration should feel like.
- Write in plain English.
- Keep the main HapticGen prompt concise, concrete, and generation-friendly.
- Avoid raw waveform language, PCA terms, semantic control terms, and implementation jargon.
- Do not mention "image" or "picture" in the final HapticGen prompt.

If notes are provided, use them only as light context.

Output ONLY valid JSON with exactly these keys:
- `visual_summary`
- `haptic_prompt`
- `negative_prompt`
- `rationale`

Guidelines:
- `visual_summary`: 1-2 short factual sentences describing the temporal event across the 3 images.
- `haptic_prompt`: one concise English prompt that would help a text-to-haptics model generate a matching vibration.
- `negative_prompt`: short text listing qualities to avoid, or an empty string.
- `rationale`: 1-2 short sentences explaining why the haptic prompt matches the event.

Optional notes:
{{notes}}
