# Phase 7 Hypothesis Snapshot

## Purpose

This note freezes the current Phase 7 diagnostic state before the next patch run overwrites the artifacts in `results/phase7/...`.

The goal is to preserve:

- the latest stable numerical results for `P7D01` and `P7B01`;
- the intermediate validation behavior observed during the previous KD patch run;
- the current main hypothesis on the failure mode.

## Current Main Hypothesis

The most credible hypothesis is that the remaining failure mode comes from a **mismatch between framewise logit distillation and the sparse sequence geometry of CTC**.

In the current implementation:

- teacher outputs are remapped from the external Wav2Vec2 vocabulary to the student character vocabulary in [teachers.py:103](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/src/training/teachers.py:103) and [teachers.py:173](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/src/training/teachers.py:173);
- teacher values are interpolated in time to match the student sequence length in [loops.py:164](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/src/training/loops.py:164);
- KD is then applied with a framewise KL on the aligned distributions in [loops.py:202](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/src/training/loops.py:202).

CTC, however, is not a dense frame classification objective. It wants:

- many `blank` frames;
- a small number of well-placed non-blank peaks;
- global sequence consistency rather than local framewise similarity.

This explains the observed evolution of failure modes:

- first `blank collapse`;
- then transient non-empty outputs;
- now stable **babbling collapse** with too many non-blank emissions and no correct transcription.

## Preserved Results

### 1. `P7D01` after the optimization stabilization patch

Sources:

- [train_audio_transformer_P7D01_final.json](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7D01/benchmark_results/train_audio_transformer_P7D01_final.json)
- [asr_checkpoint_variants_P7D01_final.json](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7D01/benchmark_results/asr_checkpoint_variants_P7D01_final.json)

Key training metrics:

| Metric | Value |
|---|---:|
| `train_loss` | `25.0787` |
| `train_ctc_loss` | `22.4432` |
| `train_kd_loss` | `10.5418` |
| `train_kd_active_ratio` | `0.3710` |
| `train_runtime_sec` | `10.81` |
| `train_peak_gpu_mem_mb` | `597.22` |
| `finetune_optimizer_steps` | `63` |
| `finetune_early_stopped` | `1` |
| `finetune_early_stop_epoch` | `2` |
| `best_valid_epoch` | `0` |

Validation metrics:

| Metric | Value |
|---|---:|
| `valid_wer` | `1.0000` |
| `valid_accuracy` | `0.0000` |
| `valid_blank_ratio` | `0.0311` |
| `valid_empty_pred_ratio` | `0.0000` |
| `valid_pred_to_ref_char_ratio` | `3.3782` |
| `valid_selection_score` | `1.0031` |

Test metrics (`best` and `final` were identical):

| Metric | Value |
|---|---:|
| `wer` | `1.0000` |
| `accuracy` | `0.0000` |
| `blank_ratio` | `0.0515` |
| `empty_pred_ratio` | `0.0000` |
| `pred_to_ref_char_ratio` | `2.9607` |
| `avg_pred_chars` | `63.77` |
| `avg_ref_chars` | `21.54` |

Reading:

- optimization no longer collapses to blank;
- outputs are non-empty;
- but they are much too long and still wrong;
- this is the clearest Phase 7 example of **babbling collapse**.

### 2. `P7B01` after the optimization stabilization patch

Sources:

- [train_audio_transformer_P7B01_final.json](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7B01/benchmark_results/train_audio_transformer_P7B01_final.json)
- [asr_checkpoint_variants_P7B01_final.json](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7B01/benchmark_results/asr_checkpoint_variants_P7B01_final.json)

Aggregate training metrics:

| Metric | Value |
|---|---:|
| `train_loss` mean | `23.3140` |
| `train_ctc_loss` mean | `20.4652` |
| `train_kd_loss` mean | `11.3954` |
| `train_kd_active_ratio` mean | `0.4103` |
| `train_runtime_sec` mean | `99.99` |
| `train_peak_gpu_mem_mb` mean | `653.68` |
| `finetune_optimizer_steps` mean | `188` |
| `finetune_early_stopped` mean | `1` |
| `finetune_early_stop_epoch` mean | `2` |
| `best_valid_epoch` mean | `0` |

Aggregate validation metrics:

| Metric | Value |
|---|---:|
| `valid_wer` mean | `1.0000` |
| `valid_accuracy` mean | `0.0334` |
| `valid_blank_ratio` mean | `0.0000` |
| `valid_empty_pred_ratio` mean | `0.0000` |
| `valid_pred_to_ref_char_ratio` mean | `1.4962` |
| `valid_selection_score` mean | `1.2265` |

Aggregate test metrics (`best` and `final` identical):

| Metric | Value |
|---|---:|
| `wer` mean | `1.0000` |
| `accuracy` mean | `0.0317` |
| `blank_ratio` mean | `0.0000` |
| `empty_pred_ratio` mean | `0.0000` |
| `pred_to_ref_char_ratio` mean | `1.5613` |
| `short_pred_ratio` mean | `0.2633` |
| `avg_pred_chars` mean | `44.42` |
| `avg_ref_chars` mean | `28.45` |

Per-seed test behavior:

| Seed | WER | Accuracy | Blank Ratio | Empty Pred Ratio | Pred/Ref Char Ratio | Reading |
|---|---:|---:|---:|---:|---:|---|
| `42` | `1.0000` | `0.0000` | `0.0000` | `0.0000` | `2.6491` | strong over-generation / babbling |
| `123` | `1.0000` | `0.0634` | `0.0000` | `0.0000` | `0.4734` | shorter outputs, still wrong |

Reading:

- the stabilization patch fixed late blank collapse;
- it did not fix transcription quality;
- the failure mode moved from silence to over-generation;
- `P7B01` remains unusable as ASR despite cleaner optimization.

### 3. Intermediate run preserved from console logs

These metrics were observed during the previous KD patch run, before the current optimization stabilization patch, and are preserved here because they may not survive the next overwrite.

Observed on `P7B01`, `seed=123`:

| Epoch | WER | Accuracy | Blank Ratio | Empty Pred Ratio | Pred/Ref Char Ratio | Selection Score |
|---|---:|---:|---:|---:|---:|---:|
| `0` | `1.0000` | `0.0705` | `0.360` | `0.000` | `0.181` | `1.6329` |
| `1` | `1.0000` | `0.0732` | `0.644` | `0.000` | `0.363` | `1.5236` |
| `2` | `1.0000` | `0.0000` | `1.000` | `1.000` | `0.000` | `2.8500` |
| `3` | `1.0000` | `0.0000` | `1.000` | `1.000` | `0.000` | `2.8500` |
| `4` | `1.0000` | `0.0000` | `1.000` | `1.000` | `0.000` | `2.8500` |
| `5` | `1.0000` | `0.0000` | `1.000` | `1.000` | `0.000` | `2.8500` |

This run was important because it showed:

- a real transient improvement at `epoch 0` and `epoch 1`;
- then a rapid recollapse to blank at `epoch 2+`.

This is the run that motivated the optimization stabilization patch.

## Interpretation

The sequence of observations now looks like this:

1. Initial Phase 7 runs:
   - blank collapse or token collapse.

2. KD informative-frame patch:
   - better early behavior;
   - still unstable;
   - recollapse after a few epochs.

3. Optimization stabilization patch:
   - training becomes stable;
   - early stopping works;
   - blank collapse disappears;
   - but ASR quality does not improve;
   - the new failure mode is babbling collapse.

This evolution strongly supports the hypothesis that the main bottleneck is no longer low-level plumbing, but the **objective mismatch between framewise KD and CTC sequence learning**.

## Why This Hypothesis Is More Credible Than the Alternatives

It is more credible than:

- a pure optimizer bug:
  - warmup, calibrated scheduler, and early stopping are now in place;
  - the runs are stable.

- a pure padding/length bug:
  - `invalid_length_ratio = 0.0`;
  - padding masks are already used.

- a pure “model too small” explanation:
  - the model does learn something;
  - it just learns the wrong emission behavior.

The most plausible remaining explanation is therefore:

- the student is being trained to imitate local teacher distributions frame by frame;
- CTC needs sparse sequence-level structure;
- the resulting compromise produces dense non-blank emissions and poor decoded text.

## Consequence for the Next Patch

The next patch should not start with “more data” or “more epochs”.

The most defensible next step is to change the distillation target itself, for example:

1. hidden-state distillation instead of framewise logit KL;
2. or an explicit regularization against over-emission of non-blank symbols.

At this point, another large sweep of learning-rate or architecture ablations would be weaker than a targeted change to the KD objective.
