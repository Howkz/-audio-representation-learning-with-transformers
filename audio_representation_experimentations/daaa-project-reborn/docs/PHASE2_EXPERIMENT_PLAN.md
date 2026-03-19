# Phase 2 Experiment Plan

## Purpose

Phase 2 is a corrective campaign.

Its goal is not to multiply ablations immediately. Its first goal is to exit the
current degenerate regime where `WER ~= 1.0` across nearly all variants.

Only once a non-degenerate baseline exists should we resume architecture and
pretraining comparisons.

## Why We Need Phase 2

Phase 1 produced three strong signals:

1. The pipeline runs.
2. Resource usage differences are measurable.
3. ASR quality does not separate meaningfully between variants.

The third point is the blocker. We therefore need a new protocol centered on
diagnosis and recovery instead of immediate leaderboard comparison.

## Phase 2 Objectives

### Objective A

Obtain at least one baseline that transcribes better than the current
near-degenerate regime.

### Objective B

Verify that predictions are not dominated by blank or near-empty outputs.

### Objective C

Create real contrast on ASR quality before resuming fine architectural ablations.

## Phase 2 Rules

1. Do not reuse `E00 -> E11` naming.
2. Do not mix phase 2 outputs with the legacy `results/experiments/` phase 1
   interpretation.
3. Prefer fewer runs with stronger diagnostics over many weak ablations.
4. Inspect predictions explicitly during phase 2.
5. Only re-open `MAE vs NoMAE` or architecture comparisons after a baseline
   exits the `WER ~= 1.0` regime.

## Proposed Naming

Use a new prefix for phase 2, for example:

- `P2Dxx` for diagnostic runs
- `P2Bxx` for baseline recovery runs
- `P2Axx` for later ablations

This prevents confusion with phase 1 experiment IDs.

## Minimal Phase 2 Run List

The first phase 2 batch should be small and diagnostic.

### P2D01 - Diagnostic CTC Baseline

Goal:

- inspect whether predictions are blank, empty, repetitive, or badly aligned

Main changes:

- keep one simple baseline architecture
- add qualitative prediction dumps on validation
- log input lengths, target lengths, prediction lengths, and blank dominance

Expected value:

- identifies whether the failure is truly a CTC collapse

### P2B01 - Budget Recovery Baseline

Goal:

- test whether the current supervised budget is simply too small

Main changes:

- increase `finetune.max_steps`
- optionally increase effective supervised data budget
- keep the rest as stable as possible

Expected value:

- if `WER` decreases materially, the main issue is budget rather than exotic
  architecture design

### P2B02 - NoMAE Recovery Baseline

Goal:

- compare a simpler supervised-only recovery against the MAE-based recovery

Main changes:

- same training budget as `P2B01`
- disable pretraining

Expected value:

- meaningful `MAE vs NoMAE` comparison, but only after the baseline is no longer
  fully degenerate

### P2A01 - Lower Compression Variant

Goal:

- test whether the encoder compresses time too aggressively for CTC

Main changes:

- reduce temporal compression or patch size along time
- keep the rest close to the best recovery baseline

Expected value:

- checks whether alignment failure is partly caused by sequence compression

## Success Criteria

Phase 2 should not be judged only on runtime.

The first acceptable success criteria are:

1. `WER < 1.0` in a stable and repeatable way.
2. Validation predictions are visibly non-empty and not dominated by blank.
3. At least one baseline shows a quality delta large enough to justify a real
   follow-up comparison.

Only after those criteria are met should we resume:

- `MAE` vs `NoMAE`
- capacity comparisons
- positional embedding comparisons
- mask-ratio comparisons

## Recommended Phase 2 Folder Usage

- phase 1 launcher archive: `scripts/experiments_phase1/`
- phase 2 launcher workspace: `scripts/experiments_phase2/`
- phase 1 config archive: `configs/phase1/`
- phase 2 config workspace: `configs/phase2/`
- phase 1 suite snapshot: `results/phase1/`
- phase 2 outputs: `results/phase2/`

## Immediate Next Implementation Tasks

1. Add a diagnostic mode that logs a small batch of references and predictions.
2. Create one clean phase 2 baseline config.
3. Create one extended-budget variant.
4. Create one matched `NoMAE` variant.
5. Keep all phase 2 outputs under phase 2 specific paths.
