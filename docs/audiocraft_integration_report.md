# AudioCraft Integration Report

## 1. Summary of Repo Architecture

### Core packages and responsibilities

- `src/data`
  - Dataset discovery, quality filtering, preprocessing, and dataloader construction for haptic WAV signals.
- `src/models`
  - `ConvVAE` and `ConvAE` plus shared Conv1D building blocks.
- `src/training`
  - Training loop, losses, and KL scheduling.
- `src/eval`
  - Reconstruction metrics, PC validation, and plotting helpers.
- `src/pipelines`
  - Latent extraction, PCA fitting, control sweeps, and control specification generation.
- `scripts`
  - Entry points for train, eval, latent extraction, validation, and control artifact generation.

### Main execution paths

1. `scripts/train.py`
   - Load config
   - Set seed
   - Build train/val dataloaders
   - Build model
   - Run trainer
2. `scripts/eval.py`
   - Load config and checkpoint
   - Rebuild validation data
   - Run reconstruction metrics and plotting
3. `scripts/extract_and_pca.py`
   - Load trained model
   - Extract latent vectors
   - Fit PCA
4. `scripts/validate_extended.py`
   - Load model and PCA
   - Sweep PCs
   - Compute extended control validation metrics

### Data flow

- WAV files are filtered by metadata in `src/data/preprocessing.py`.
- Train/val splits are created in `src/data/loaders.py`.
- `HapticWavDataset` samples energy-rich segments.
- Training and evaluation operate on tensors with shape `(B, 1, T)`.
- Downstream PCA and control tools consume model latents and decoder outputs.

### Configuration flow

- YAML configs were previously merged only with hard-coded defaults.
- Training and evaluation scripts consumed the resolved dict directly.
- There was no inheritance, no CLI override support, and no persisted resolved config snapshot.

### Extension points and technical debt found

- Training logic bundled model optimization, checkpointing, and reporting inside one monolithic trainer.
- No resumable full-state checkpoint for optimizer/scheduler/history.
- Best model tracking existed, but only as raw weights.
- Validation lacked periodic sample emission.
- Train/val split reproducibility depended on re-running with the same seed rather than persisted manifests.
- Audio loading relied on `librosa.load(..., mono=True)`, which hid channel handling and made audio IO less explicit/testable.
- Test coverage for the top-level thesis code was effectively missing.

### Tests and CI expectations discovered

- No top-level CI configuration was present.
- No top-level `tests/` directory existed before this change.
- A separate `HapticGen/` subtree contains its own extensive tests, but it behaves as an adjacent project rather than the active thesis codebase.

## 2. AudioCraft Components Reviewed

### Upstream AudioCraft files reviewed

- `README.md`
  - High-level project structure and training/documentation entry points.
- `docs/TRAINING.md`
  - Solver design, stage separation, experiment management principles, and checkpoint lifecycle.
- `audiocraft/solvers/base.py`
  - `StandardSolver`, train/valid/evaluate/generate stage boundaries, best-state handling, EMA swaps, and restore flow.
- `audiocraft/solvers/builders.py`
  - Builders for optimizers, schedulers, EMA, and datasets.
- `audiocraft/data/audio_utils.py`
  - Channel conversion, resampling, and normalization utilities.
- `tests/data/test_audio_utils.py`
  - Focused tests around audio utility behavior.

### Local adjacent reference also inspected

- `HapticGen/README.md`
  - Confirmed that `HapticGen/` is an AudioCraft-derived side project, not the top-level thesis training stack to modify directly.
- `HapticGen/audiocraft/...`
  - Used only as supporting context for how AudioCraft-style patterns had previously been adapted in a haptics setting.

## 3. Candidate Improvements Found

| Candidate | AudioCraft inspiration | Fit for this repo | Cost | Benefit | Risk | Decision |
|---|---|---|---:|---:|---:|---|
| Solver-like separation of builders/checkpoint/loop | `docs/TRAINING.md`, `solvers/base.py`, `solvers/builders.py` | High | Medium | High | Low | Implemented |
| Resumable full-state checkpointing | `solvers/base.py`, `utils/checkpoint.py` | High | Medium | High | Low | Implemented |
| Best-state tracking as a first-class artifact | `solvers/base.py` | High | Low | High | Low | Implemented |
| Optional EMA for validation/best export | `solvers/base.py`, `solvers/builders.py` | Medium | Medium | Medium | Low | Implemented |
| Periodic validation sample generation | `solvers/base.py` generate/evaluate stages | High | Medium | Medium | Low | Implemented |
| Config inheritance and CLI delta overrides | `docs/TRAINING.md` Hydra/Dora principles | High | Medium | High | Low | Implemented in lightweight form |
| Explicit audio channel/resample utilities | `data/audio_utils.py` | High | Low | Medium | Low | Implemented |
| Dataset abstraction with train/valid/generate/evaluate splits | `solvers/base.py`, `solvers/builders.py` | Medium | Medium | Medium | Medium | Partially implemented via persisted split manifests |
| Full Hydra/Dora experiment manager | `docs/TRAINING.md` | Low | High | Medium | High | Skipped |
| FSDP/distributed checkpoint semantics | `solvers/base.py`, `utils/checkpoint.py` | Low | High | Low | High | Skipped |
| EnCodec-style tokenizer/codec interfaces | broader AudioCraft model stack | Low for current VAE/PCA path | High | Low | High | Skipped |
| Heavy evaluate/generate stage framework | `solvers/base.py` | Medium | High | Medium | Medium | Skipped for now |

## 4. Implemented Changes

### Training architecture and state management

- Added `src/training/builders.py`
  - Centralized optimizer, scheduler, and EMA construction.
- Added `src/training/checkpointing.py`
  - Run artifact paths
  - Resumable full-state checkpoint management
  - Periodic epoch checkpoint retention
  - Resolved config persistence
- Added `src/training/ema.py`
  - Lightweight EMA wrapper with temporary swap support
- Reworked `src/training/trainer.py`
  - Metric dictionaries per epoch instead of only scalar loss
  - Optional EMA-based validation/best export
  - `last_checkpoint.pt` save/restore flow
  - `best_model.pt` preservation for downstream compatibility
  - Periodic validation preview export
  - Persistent history and metrics artifacts

### Config and experiment reproducibility

- Extended `src/utils/config.py`
  - `base_config` inheritance
  - CLI override application via `key=value`
  - Resolved config dump helper
- Updated `scripts/train.py`
  - `--resume`
  - repeated `--set key=value`
  - run directory preparation before dataloader construction
- Updated `scripts/eval.py`
  - repeated `--set key=value`
  - automatic reuse of `data_split.json` from the checkpoint directory when available

### Data and audio handling

- Added `src/data/audio_utils.py`
  - explicit channel conversion
  - explicit resampling
  - robust float32 loading with NaN/Inf cleanup
- Updated `src/data/preprocessing.py`
  - now uses the shared audio utility instead of implicit mono loading
- Updated `src/data/loaders.py`
  - persisted train/val split manifests
  - optional reuse of prior split manifest
  - dataloader worker/pinning options from config
  - checkpoint loader compatibility with both raw state dicts and richer checkpoint payloads

### Evaluation and observability

- Updated `src/eval/visualize.py`
  - save-only plotting now closes figures cleanly instead of always showing them
- Training can now emit:
  - `validation_samples/epoch_XXX/original.npy`
  - `validation_samples/epoch_XXX/reconstructed.npy`
  - `validation_samples/epoch_XXX/waveforms.png`
  - `validation_samples/epoch_XXX/summary.json`

### Tests

- Added `tests/test_audio_utils.py`
- Added `tests/test_config_utils.py`
- Added `tests/test_trainer_resume.py`
  - this test is skipped automatically when `torch` is unavailable in the environment

## 5. Why Each Implemented Change Was Chosen

- Solver-style builders/checkpoint separation
  - This captures the most useful AudioCraft idea without importing Flashy, Hydra, or Dora.
- Full resumable checkpointing
  - Highest practical payoff for long-running training and experimentation.
- EMA
  - Low-risk enhancement for validation stability and best-model export.
- Validation sample hooks
  - Gives fast qualitative visibility similar to AudioCraft's stage-based design.
- Config inheritance and CLI overrides
  - Improves experiment hygiene and reduces one-off config editing.
- Persisted split manifests
  - Directly improves reproducibility for resume and later evaluation.
- Shared audio utilities
  - Makes channel/sample-rate assumptions explicit and testable.
- Focused tests
  - Adds minimal safety around the new behavior in the active codebase.

## 6. What Was Intentionally Not Integrated and Why

- Hydra/Dora/Flashy stack
  - Too heavy for this repo and would create a framework migration rather than a safe improvement sweep.
- AudioCraft's full `train/valid/evaluate/generate` execution framework
  - The top-level thesis scripts do not yet need a multi-stage scheduling system at AudioCraft scale.
- FSDP, distributed rank-aware checkpointing, and cluster abstractions
  - Non-applicable for the current single-project, local-script workflow.
- EnCodec/MusicGen tokenizer or codec abstractions
  - Your active project works on continuous haptic waveforms plus PCA controls, not discrete codec tokens.
- Alternative decoding/model families from AudioCraft
  - Would change modeling behavior substantially and is outside the low-risk scope.

## 7. Risks / Follow-up Suggestions

- Environment gap
  - The current machine does not have `torch` importable, so full end-to-end training execution could not be validated here.
- Resume compatibility
  - New `last_checkpoint.pt` files are richer than old checkpoints; raw `best_model.pt` compatibility was kept intentionally to avoid breaking downstream scripts.
- Config inheritance adoption
  - Existing configs continue to work unchanged, but new configs should prefer `base_config` rather than manual copy/paste.
- Next useful follow-up
  - Add a small `scripts/resume_latest.py` or `--auto_resume` mode if interrupted runs are common.
- Next observability follow-up
  - Export per-epoch metrics as CSV in addition to JSON/NPZ if spreadsheet analysis matters for the thesis workflow.

## 8. Exact Commands Used for Validation

```bash
python -m compileall src scripts tests
python -m unittest discover -s tests
```

### Validation outcome

- `compileall`: passed
- `unittest`: passed
- `tests/test_trainer_resume.py`: skipped because `torch` is not installed in the current environment
