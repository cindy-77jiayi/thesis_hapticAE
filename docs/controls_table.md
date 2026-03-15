# Haptic Control Dimensions (PC1–PC8)

**Model:** vae_balanced (latent_dim=24, beta_max=0.003)  
**Total explained variance:** 58.9% (mean across 3 seeds: 60.6% ± 4.3%)  
**Signal:** 4000 samples @ 8000 Hz (0.50s)

> **Note on 58.9% coverage:** The goal is controllable generation, not lossless
> compression. PC1–PC2 alone capture 29.5% and show strong monotonic control
> (|ρ| > 0.8) over multiple signal metrics. PC3–PC4 add moderate control (|ρ| > 0.7).
> PC5–PC8 provide fine-grained adjustment with limited individual semantics.

## Validation Summary

| Evidence | Result | Interpretation |
|----------|--------|----------------|
| **Monotonicity** | PC1: 4 metrics |ρ|>0.7; PC2: 4 metrics |ρ|>0.8 | Strong data-driven labels |
| **Orthogonality** | Mean selectivity 3.1× (all >1×) | Each PC primarily affects its own bound metrics |
| **Effect size** | PC1-4 avg relative change > PC5-8 on key metrics | Primary/secondary split is justified |
| **Cross-seed** | EVR stable ±1.6%; PC1/PC2 sign consistency 87% | Structure not an artifact of single seed |

## Control Specification

### Primary Controls: Strong monotonic, cross-seed stable

| Control | Var% | Range (P5–P95) | Label | Bound Metrics (Spearman ρ) | Seed Stability |
|---------|------|----------------|-------|---------------------------|----------------|
| PC1 | 22.7 | [−1.60, +6.14] | **Intensity** | AM mod ↓(−0.85), decay slope ↑(+0.78), crest ↓(−0.78), RMS ↑(+0.74) | 87% |
| PC2 | 6.8 | [−2.16, +2.00] | **Brightness** | spec centroid ↑(+0.97), short-term var ↓(−0.97), AM mod ↓(−0.91), decay slope ↑(+0.84) | 87% |

### Moderate Controls: Fewer bound metrics, partially stable

| Control | Var% | Range (P5–P95) | Label | Bound Metrics (Spearman ρ) | Seed Stability |
|---------|------|----------------|-------|---------------------------|----------------|
| PC3 | 5.6 | [−1.84, +1.99] | **Sustain** | decay slope ↑(+0.76), spec centroid ↑(+0.67) | 47% |
| PC4 | 5.3 | [−1.80, +1.94] | **Rhythmic variation** | IOI entropy ↑(+0.75), short-term var ↑(+0.68) | 67% |

### Secondary Controls: Temporal microstructure

These dimensions capture finer temporal variations. Their individual semantic
labels are weaker (no non-peak metric with |ρ| > 0.7 except PC7 AM mod and
PC8 attack time) and their cross-seed direction consistency ranges from 47% to
93%. They are best characterized collectively as **temporal microstructure**.

| Control | Var% | Range (P5–P95) | Notable Metric | Seed Stability |
|---------|------|----------------|----------------|----------------|
| PC5 | 5.0 | [−2.05, +1.90] | — | 47% |
| PC6 | 4.7 | [−1.94, +1.81] | decay slope ↑(+0.56) | 93% |
| PC7 | 4.5 | [−1.68, +1.69] | AM mod ↑(+0.86) | 60% |
| PC8 | 4.3 | [−1.69, +1.69] | attack time ↓(−0.73) | 93% |

## Detailed Evidence

### PC1 — Intensity (22.7% variance)

The dominant control axis. As PC1 increases: overall energy rises (RMS ↑),
signal becomes more sustained and uniform (AM mod ↓, crest factor ↓),
with flatter envelope (decay slope ↑).

Asymmetric range [−1.60, +6.14] reflects the data distribution: most signals
are low-energy, with a tail of high-energy samples.

| Metric | Spearman ρ | p-value | Direction |
|--------|-----------|---------|-----------|
| AM modulation index | −0.853 | <0.01 | ↓ (less modulation) |
| Envelope decay slope | +0.777 | <0.05 | ↑ (more sustained) |
| Crest factor | −0.775 | <0.05 | ↓ (less impulsive) |
| RMS energy | +0.743 | <0.05 | ↑ (more energy) |

### PC2 — Brightness (6.8% variance)

Controls spectral content and temporal smoothness. As PC2 increases:
signal shifts toward higher frequencies (centroid ↑) with less temporal
variation (short-term var ↓, AM mod ↓) and more sustain (decay slope ↑).

| Metric | Spearman ρ | p-value | Direction |
|--------|-----------|---------|-----------|
| Spectral centroid | +0.969 | <0.01 | ↑ (brighter) |
| Short-term variance | −0.971 | <0.01 | ↓ (smoother) |
| AM modulation index | −0.910 | <0.01 | ↓ (less modulation) |
| Envelope decay slope | +0.836 | <0.01 | ↑ (more sustained) |

### PC3 — Sustain (5.6% variance)

Weaker semantic dimension. Primarily associated with envelope sustain
characteristics. Lower cross-seed consistency (47%) suggests this dimension
may swap with adjacent PCs under different training conditions.

| Metric | Spearman ρ | p-value | Direction |
|--------|-----------|---------|-----------|
| Envelope decay slope | +0.755 | <0.05 | ↑ (more sustained) |
| Spectral centroid | +0.665 | <0.05 | ↑ (brighter) |

### PC4 — Rhythmic variation (5.3% variance)

Associated with temporal rhythm complexity. Higher values produce more
irregular inter-onset intervals and greater short-term energy variation.

| Metric | Spearman ρ | p-value | Direction |
|--------|-----------|---------|-----------|
| IOI entropy | +0.746 | <0.05 | ↑ (more complex rhythm) |
| Short-term variance | +0.677 | <0.05 | ↑ (more variation) |
| Envelope decay slope | +0.643 | <0.05 | ↑ (more sustained) |

### PC5–PC8 — Temporal microstructure (18.5% combined)

These secondary dimensions capture fine-grained temporal patterning that
resists stable per-axis labeling. Their effects are real but small, and
their semantic interpretation is context-dependent. Notable individual
bindings:

- **PC7**: AM modulation index (ρ=+0.86) — controls amplitude modulation depth
- **PC8**: Attack time (ρ=−0.73) — sharper attacks at higher values

## Cross-Seed Stability

Trained with 3 random seeds (42, 123, 456). Results confirm structural stability:

| Metric | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|--------|---------|----------|----------|------------|
| PC1 variance | 22.7% | 26.3% | 26.0% | 25.0% ± 1.6% |
| PC2 variance | 6.8% | 6.0% | 9.4% | 7.4% ± 1.4% |
| PC3–PC8 avg | 4.9% | 4.5% | 4.3% | 4.5% ± 0.2% |
| Total (8 PCs) | 58.9% | 60.1% | 62.7% | 60.6% ± 1.6% |

PC1 consistently captures 22–26% of variance across all seeds, confirming
it as the dominant intensity axis regardless of initialization.

## API Usage

```python
# Default: all controls at 0 → neutral/average signal
signal = decode_controls([0, 0, 0, 0, 0, 0, 0, 0])

# High intensity, bright, sustained
signal = decode_controls([4.0, 1.5, 1.5, 0, 0, 0, 0, 0])

# Low intensity, dark, percussive, complex rhythm
signal = decode_controls([-1.0, -1.5, -1.5, 1.5, 0, 0, 0, 0])
```

Each control should stay within its P5–P95 range for realistic output.

## Methodological Note

The `peak_amplitude` metric shows |ρ| ≈ 1.0 for nearly all PCs, which is
a measurement artifact: the decoder produces monotonically changing peak
values along any sweep direction. This metric was excluded from semantic
binding analysis. All reported bindings use metrics with genuine
discriminative power.
