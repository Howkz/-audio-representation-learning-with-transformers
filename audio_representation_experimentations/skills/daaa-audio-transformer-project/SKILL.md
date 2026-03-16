---
name: daaa-audio-transformer-project
description: Build and evaluate a DAAA audio representation Transformer project under strict reproducibility and 6GB VRAM constraints. Use when asked to set up the template-ml workflow, implement AudioPatchEmbedding and Transformer audio modules, run MAE-style self-supervised pretraining, fine-tune ASR tasks with CTC, report 5-seed mean plus std metrics, optimize frugality, and prepare a compliant final report and submission package.
---

# DAAA Audio Transformer Project

## Overview

Use this skill to execute the full DAAA project lifecycle from repository bootstrap to report-ready experiments.
Prioritize modular code in `src/`, reproducible runs, and explicit compliance with vdigpu evaluation constraints.

## Workflow Decision Tree

1. If repository is not initialized from template-ml, start with `Bootstrap`.
2. If data processing is missing or unstable, run `Data Pipeline`.
3. If core model blocks are missing, run `Model Implementation`.
4. If pretraining/fine-tuning loops are incomplete, run `Training Loops`.
5. If metrics or ablations are missing, run `Evaluation Protocol`.
6. If results exist but submission artifacts are incomplete, run `Report And Packaging`.

## Bootstrap

1. Create or reuse a `daaa` workspace and clone the session repository.
2. Pull template files from `template/main` using `git archive`.
3. Commit initialization with a clear message.
4. Confirm `Makefile`, `src/`, `data/`, and experiment output folders exist.
5. Enforce deterministic setup (fixed seeds, config-driven runs, tracked dependencies).

## Data Pipeline

1. Load datasets only through Hugging Face `datasets`.
2. Standardize sample rate (target 16 kHz unless justified otherwise).
3. Apply duration filtering plus crop/pad to control memory usage.
4. Implement feature extraction for log-Mel and optional alternatives (raw wave or spectrogram).
5. Cache prepared features when beneficial to training throughput.
6. Keep preprocessing configurable in `src/data/dataset.py` and `src/data/features.py`.

Read [spec-checklist.md](references/spec-checklist.md) before changing dataset policy.

## Model Implementation

1. Implement `AudioPatchEmbedding` first, with a clear choice of patching strategy:
   - Time-only patches for simpler memory profile.
   - Time-frequency patches for richer locality at higher cost.
2. Implement `AudioTransformerEncoder` with configurable depth, heads, dim, MLP ratio, dropout, and positional embedding type.
3. Implement MAE pretraining wrapper:
   - Mask generator (`make_mae_mask`) for frame or region masking.
   - Lightweight decoder used only during pretraining.
   - Reconstruction loss computed only on masked regions.
4. Implement ASR head (`AudioTransformerCTC`) for downstream fine-tuning.
5. Keep components reusable and independent (encoder shared across MAE and CTC).

Read [experiment-playbook.md](references/experiment-playbook.md) to pick baseline and ablations.

## Training Loops

1. Implement `make train` to support:
   - Self-supervised pretraining.
   - Downstream fine-tuning.
   - Resume from checkpoints.
2. Enforce frugal defaults for 6GB VRAM:
   - Mixed precision.
   - Gradient accumulation.
   - Modest sequence lengths and batch sizes.
3. Track runtime and memory metrics during both training and inference.
4. Save checkpoints and logs outside submission package directories.

## Evaluation Protocol

1. Implement `make test` to evaluate all downstream tasks.
2. Report metrics as mean plus std over 5 seeds for each important setting.
3. Use WER for ASR tasks and document metric computation path.
4. For optional TTS bonus:
   - Report reconstruction losses (for example L1 on log-Mel).
   - Include qualitative spectrogram comparisons.
5. Store machine-readable outputs in `results.txt` or equivalent table artifacts.

## Report And Packaging

1. Write a PDF report focused on decisions and trade-offs, not raw code screenshots.
2. Include:
   - Architecture choices (embedding, masking, decoder, CTC setup).
   - Frugality optimizations and measured impact.
   - Ablation study (at least one focused comparison).
   - Main benchmark table with uncertainty.
3. Verify submission format `daaa_nomduprojet1_nom2.zip`.
4. Exclude checkpoints and datasets from final archive.

## Output Contract

When using this skill, always return:
1. A compact plan mapped to `make data`, `make train`, `make test`.
2. Explicit statement of assumptions for memory budget and dataset subsets.
3. A list of files created or modified in `src/` and config folders.
4. Verification evidence (commands run, metrics produced, artifacts written).
5. Remaining risks and next experiments.

## References

- [spec-checklist.md](references/spec-checklist.md)
- [experiment-playbook.md](references/experiment-playbook.md)
