# Haptic Control Dimensions (PC1–PC8)

**Model:** vae_balanced (latent_dim=24, beta_max=0.003)  
**Total explained variance:** 58.6%  
**Signal:** 4000 samples @ 8000 Hz (0.50s)

## Control Summary

| Control | Var% | Range (P5–P95) | Label | When control ↑ | When control ↓ | Subjective |
|---------|------|----------------|-------|----------------|----------------|------------|
| PC1 | 23.4 | [−6.41, +1.80] | **Intensity** | Onset density ↑, HF ratio ↑, RMS ↓ | RMS ↑, sustained energy | ↑ sparse/pulsive, ↓ intense/sustained |
| PC2 | 6.6 | [−2.07, +1.77] | **Sustain** | Decay slope ↑ (sustained), HF ↑, onsets ↓ | Sharp decay, fewer HF | ↑ sustained/ringing, ↓ percussive/damped |
| PC3 | 5.6 | [−1.90, +1.81] | **Warmth** | Decay slope ↑, HF ↓, RMS ↑ | Brighter, thinner | ↑ warm/full, ↓ thin/bright |
| PC4 | 5.1 | [−1.84, +1.83] | **Attack sharpness** | Decay slope ↓ (sharper), onsets ↑, HF ↑ | Smoother attacks | ↑ sharp/impulsive, ↓ smooth/gradual |
| PC5 | 4.9 | [−1.76, +1.99] | **Rhythmic complexity** | Decay slope ↓, onsets ↑, HF ↓ | Simpler rhythm, brighter | ↑ complex/busy, ↓ simple/steady |
| PC6 | 4.5 | [−1.56, +1.75] | **Event density** | Decay slope ↓, onsets ↑, HF ↓ | Fewer events, brighter | ↑ busy/active, ↓ sparse/clean |
| PC7 | 4.4 | [−1.66, +1.79] | **Continuity** | Decay slope ↑, onsets ↓, HF ↓ | More fragmented | ↑ continuous/flowing, ↓ choppy/fragmented |
| PC8 | 4.1 | [−1.39, +1.84] | **Temporal texture** | Decay slope ↑, onsets ↑, HF ↓ | Uniform texture | ↑ varied/textured, ↓ uniform/flat |

## Detailed Metric Profiles

### PC1 — Intensity (23.4% variance)

The dominant axis. Controls overall signal energy and transient character.  
Asymmetric range [−6.41, +1.80] indicates most signals cluster near low-energy end.

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Onset density | ↑ | 3.93× | [0, 10] /s |
| HF energy ratio | ↑ | 3.05× | [0.000, 0.023] |
| RMS energy | ↓ | 1.92× | [0.084, 1.621] |

### PC2 — Sustain (6.6% variance)

Controls the temporal sustain/decay profile. High values produce ringing, sustained signals; low values produce percussive, quickly-damped signals.

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Envelope decay slope | ↑ | 4.61× | [−11.16, 1.99] dB/s |
| HF energy ratio | ↑ | 1.49× | [0.004, 0.024] |
| Onset density | ↓ | 1.29× | [2, 6] /s |

### PC3 — Warmth (5.6% variance)

Trades off brightness for fullness. High values produce warmer, fuller signals with more energy but less high-frequency content.

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Envelope decay slope | ↑ | 4.05× | [−4.63, 3.14] dB/s |
| HF energy ratio | ↓ | 1.23× | [0.003, 0.017] |
| RMS energy | ↑ | 1.12× | [0.100, 0.263] |

### PC4 — Attack sharpness (5.1% variance)

Controls the onset attack profile. High values = sharper, more impulsive attacks; low values = smoother, more gradual onsets.

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Envelope decay slope | ↓ | 1.92× | [1.13, 14.90] dB/s |
| Onset density | ↑ | 1.10× | [2, 6] /s |
| HF energy ratio | ↑ | 1.04× | [0.005, 0.017] |

### PC5 — Rhythmic complexity (4.9% variance)

Controls the density and complexity of temporal events. High values produce busier, more rhythmically complex signals with darker tone.

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Envelope decay slope | ↓ | 2.50× | [−27.44, 0.93] dB/s |
| Onset density | ↑ | 1.57× | [2, 10] /s |
| HF energy ratio | ↓ | 1.25× | [0.003, 0.017] |

### PC6 — Event density (4.5% variance)

Similar to PC5 but controls overall busyness at a different temporal scale.

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Envelope decay slope | ↓ | 3.29× | [−6.37, 3.29] dB/s |
| Onset density | ↑ | 1.74× | [2, 8] /s |
| HF energy ratio | ↓ | 1.32× | [0.003, 0.019] |

### PC7 — Continuity (4.4% variance)

Controls signal continuity. High values produce smoother, more continuous signals; low values produce more fragmented, segmented output.

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Envelope decay slope | ↑ | 1.35× | [1.22, 8.28] dB/s |
| Onset density | ↓ | 1.29× | [2, 6] /s |
| HF energy ratio | ↓ | 0.91× | [0.006, 0.017] |

### PC8 — Temporal texture (4.1% variance)

Controls fine temporal texture variation. High values add temporal variation while maintaining sustain.

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Envelope decay slope | ↑ | 3.25× | [−5.00, 5.79] dB/s |
| Onset density | ↑ | 1.83× | [2, 8] /s |
| HF energy ratio | ↓ | 0.87× | [0.006, 0.017] |

---

## API Usage

```python
# Default: all controls at 0 → neutral/average signal
signal = decode_controls([0, 0, 0, 0, 0, 0, 0, 0])

# High intensity, sustained, warm
signal = decode_controls([-4.0, 1.5, 1.5, 0, 0, 0, 0, 0])

# Low intensity, percussive, bright, sharp attacks
signal = decode_controls([1.0, -1.5, -1.5, 1.5, 0, 0, 0, 0])
```

Each control should stay within its P5–P95 range for realistic output.
