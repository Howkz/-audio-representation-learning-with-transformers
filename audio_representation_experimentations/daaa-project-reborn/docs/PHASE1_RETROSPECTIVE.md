# Phase 1 Retrospective

## Scope

Phase 1 corresponds to the current `E00 -> E11` campaign and its derived
selection runs `SEL01 -> SEL05`.

This phase should now be treated as an exploratory campaign:

- pipeline validation
- low-storage / streaming stabilization
- checkpointing and resume stabilization
- first screening of MAE / NoMAE / capacity / positional / patching / mask ratio

## What Phase 1 Successfully Established

- The end-to-end pipeline `data -> train -> test` can run on the target setup.
- The low-storage strategy is operational:
  - streaming datasets
  - lazy data step
  - `soundfile` / `ffmpeg` fallback when `torchaudio` is absent
- The suite runner can generate screening, selection, and final-stage runtime
  configs.
- Resource differences between model variants are measurable.

## Main Empirical Observation

The central result of phase 1 is negative:

- screening runs `E01 -> E08` are all at `WER ~= 1.0`
- selection runs remain at `WER ~= 1.0` as well
- the current rankings mostly reflect runtime / memory differences, not useful
  ASR quality differences

This means phase 1 does not yet support strong scientific conclusions about:

- the real contribution of MAE pretraining
- the best architectural choice for ASR quality
- the robustness of the variants under seeds

## Why Phase 1 Is Not Enough

At the moment, the campaign mostly compares failed or near-failed ASR models.
That makes the contrast weak:

- `MAE` vs `NoMAE` does not separate clearly on quality
- capacity changes mostly separate on cost
- selection and final stages inherit a quality regime that is already degenerate

As a result, continuing to patch and extend phase 1 would mix:

- exploratory results
- pipeline fixes
- corrective hypotheses
- new experimental goals

This would reduce traceability.

## Working Failure Hypotheses

The main hypotheses to investigate in phase 2 are:

1. CTC collapse toward blank or near-empty predictions.
2. Fine-tuning budget too small for the supervised head to recover useful
   transcriptions.
3. Temporal compression too aggressive for CTC alignment.
4. Tokenization / transcript normalization / decoding mismatch.
5. Quality diagnostics are too weak because predictions are not inspected
   qualitatively during training.

## Decision

Phase 1 is frozen as an exploratory campaign.

It remains useful for:

- documenting the environment constraints
- documenting the stabilization work already done
- reporting resource trends
- motivating the design of a corrective phase 2

It should not be extended with additional corrective runs under the same naming
and interpretation scheme.

## Phase 1 Assets

- launchers snapshot: `scripts/experiments_phase1/`
- config snapshot: `configs/phase1/`
- suite snapshot: `results/phase1/suite_snapshot/`

Detailed legacy run artifacts still exist in the live project tree under:

- `results/experiments/`
- `results/suite/`
- `outputs/checkpoints/`

Those directories should be considered legacy phase 1 artifacts unless
explicitly re-archived elsewhere later.
