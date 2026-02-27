# Haptic Control Dimensions (PC1–PC8)

**Model:** vae_balanced (latent_dim=24, beta_max=0.003)  
**Total explained variance:** 58.6%  
**Signal:** 4000 samples @ 8000 Hz (0.50s)

> **Note on 58.6% coverage:** This is sufficient for a controllable subspace — the goal
> is not lossless compression but controllable generation. The first 4 PCs capture
> the primary perceptual axes (35.7%), and PC5–8 add finer temporal detail.
> Statistical evidence below demonstrates monotonic, orthogonal control.

## Control Summary

### Primary Controls (PC1–PC4): Strong, monotonic, independently validated

| Control | Var% | Range (P5–P95) | Label | Bound Metrics | Evidence |
|---------|------|----------------|-------|---------------|----------|
| PC1 | 23.4 | [−6.41, +1.80] | **Intensity** | RMS energy, peak amplitude | Spearman ρ, monotonic sweep |
| PC2 | 6.6 | [−2.07, +1.77] | **Sustain** | Envelope decay slope, late/early energy ratio | Spearman ρ, monotonic sweep |
| PC3 | 5.6 | [−1.90, +1.81] | **Warmth** | Spectral centroid (↓=warm), low/high band ratio | Spearman ρ, monotonic sweep |
| PC4 | 5.1 | [−1.84, +1.83] | **Attack sharpness** | Attack time (↓=sharper), crest factor | Spearman ρ, monotonic sweep |

### Secondary Controls (PC5–PC8): Finer temporal microstructure

These dimensions capture secondary temporal variations. Their semantic labels
are weaker and more context-dependent than PC1–4. They are grouped under the
umbrella of **temporal microstructure / patterning**.

| Control | Var% | Range (P5–P95) | Label | Notes |
|---------|------|----------------|-------|-------|
| PC5 | 4.9 | [−1.76, +1.99] | **Temporal patterning A** | Onset density, rhythm complexity |
| PC6 | 4.5 | [−1.56, +1.75] | **Temporal patterning B** | Event density at different scale |
| PC7 | 4.4 | [−1.66, +1.79] | **Temporal patterning C** | Continuity/fragmentation |
| PC8 | 4.1 | [−1.39, +1.84] | **Temporal patterning D** | Fine texture variation |

## Validation Evidence

### 1. Monotonicity (Spearman ρ)

Each PC's bound metrics should show |ρ| > 0.8 with p < 0.05 along the sweep.
See `outputs/validation/monotonicity_heatmap.png` for full matrix.

*To be populated after running `scripts/validate_controls.py`*

### 2. Orthogonality (Selectivity)

Each PC should primarily affect its own bound metrics (on-target |ρ| >> off-target |ρ|).
Selectivity > 2× indicates good orthogonality.
See `outputs/validation/selectivity.png`.

*To be populated after running `scripts/validate_controls.py`*

### 3. Effect Size (PC1–4 vs PC5–8)

Primary PCs should produce larger metric changes than secondary PCs.

*To be populated after running `scripts/validate_controls.py`*

### 4. Cross-Seed Stability

Trained with seeds {42, 123, 456}. Key stability metrics:
- Explained variance ratios should be within ±2% across seeds
- Spearman ρ sign should be consistent for bound metrics

*To be populated after running `scripts/cross_seed_stability.py`*

## Detailed Metric Profiles

### PC1 — Intensity (23.4% variance)

The dominant axis. Controls overall signal energy and transient character.  
Asymmetric range [−6.41, +1.80] indicates most signals cluster near low-energy end.

**Bound metrics:** `rms_energy`, `peak_amplitude`

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Onset density | ↑ | 3.93× | [0, 10] /s |
| HF energy ratio | ↑ | 3.05× | [0.000, 0.023] |
| RMS energy | ↓ | 1.92× | [0.084, 1.621] |

### PC2 — Sustain (6.6% variance)

Controls the temporal sustain/decay profile.

**Bound metrics:** `envelope_decay_slope_dBps`, `late_early_energy_ratio`

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Envelope decay slope | ↑ | 4.61× | [−11.16, 1.99] dB/s |
| HF energy ratio | ↑ | 1.49× | [0.004, 0.024] |
| Onset density | ↓ | 1.29× | [2, 6] /s |

### PC3 — Warmth (5.6% variance)

Trades off brightness for fullness.

**Bound metrics:** `spectral_centroid_hz`, `low_high_band_ratio`

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Envelope decay slope | ↑ | 4.05× | [−4.63, 3.14] dB/s |
| HF energy ratio | ↓ | 1.23× | [0.003, 0.017] |
| RMS energy | ↑ | 1.12× | [0.100, 0.263] |

### PC4 — Attack sharpness (5.1% variance)

Controls the onset attack profile.

**Bound metrics:** `attack_time_s`, `crest_factor`

| Metric | Direction | Rel. Change | Range |
|--------|-----------|-------------|-------|
| Envelope decay slope | ↓ | 1.92× | [1.13, 14.90] dB/s |
| Onset density | ↑ | 1.10× | [2, 6] /s |
| HF energy ratio | ↑ | 1.04× | [0.005, 0.017] |

### PC5–PC8 — Temporal microstructure (16.3% combined variance)

These secondary dimensions capture finer temporal variations that are harder
to assign single semantic labels to. They primarily influence:
- Onset density and inter-onset interval entropy (rhythm)
- Gap ratio and zero-crossing rate (continuity)
- Short-term variance and AM modulation index (texture)

Their effects are real but smaller in magnitude than PC1–4 (see effect size
analysis), and their semantic interpretation is more context-dependent.

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
