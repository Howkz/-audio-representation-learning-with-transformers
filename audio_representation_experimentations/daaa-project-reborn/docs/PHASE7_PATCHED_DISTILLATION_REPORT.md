# Phase 7 Patched Distillation Report

## Purpose

This report freezes the results of the three post-diagnostic Phase 7 variants before any further patch changes the artifacts again.

The objective of these runs was to test, in a controlled way, two ideas derived from the previous diagnosis:

1. stop relying on framewise logit KD and move to hidden-state distillation;
2. keep logit KD but add an explicit anti-overemission constraint.

The three evaluated runs were:

- `P7C01`: hidden-state distillation only;
- `P7C02`: logit KD + anti-overemit;
- `P7C03`: hidden-state distillation + anti-overemit.

## Sources

- [P7B01 train](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7B01/benchmark_results/train_audio_transformer_P7B01_final.json)
- [P7B01 test](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7B01/benchmark_results/asr_checkpoint_variants_P7B01_final.json)
- [P7C01 train](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7C01/benchmark_results/train_audio_transformer_P7C01_final.json)
- [P7C01 test](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7C01/benchmark_results/asr_checkpoint_variants_P7C01_final.json)
- [P7C02 train](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7C02/benchmark_results/train_audio_transformer_P7C02_final.json)
- [P7C02 test](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7C02/benchmark_results/asr_checkpoint_variants_P7C02_final.json)
- [P7C03 train](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7C03/benchmark_results/train_audio_transformer_P7C03_final.json)
- [P7C03 test](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7C03/benchmark_results/asr_checkpoint_variants_P7C03_final.json)

## Baseline Reference

The relevant baseline is the latest `P7B01`, i.e. the clean LibriSpeech distillation baseline before the new target/regularization ablations.

Aggregate test metrics for `P7B01`:

| Metric | Value |
|---|---:|
| `wer` mean | `1.0000` |
| `accuracy` mean | `0.0317` |
| `blank_ratio` mean | `0.0000` |
| `empty_pred_ratio` mean | `0.0000` |
| `pred_to_ref_char_ratio` mean | `1.5613` |
| `short_pred_ratio` mean | `0.2633` |

Interpretation:

- `P7B01` no longer collapses to blank;
- but it over-generates strongly;
- its main failure mode is babbling / over-emission.

## Results Overview

### Aggregate comparison

| Run | Training change | Test WER | Test Accuracy | Blank Ratio | Empty Pred Ratio | Pred/Ref Char Ratio | Short Pred Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| `P7B01` | logit KD baseline | `1.0000` | `0.0317` | `0.0000` | `0.0000` | `1.5613` | `0.2633` |
| `P7C01` | hidden-state KD only | `1.0055` | `0.0713` | `0.0994` | `0.0000` | `0.1022` | `0.8978` |
| `P7C02` | logit KD + anti-overemit | `1.0000` | `0.0539` | `0.2518` | `0.0000` | `0.1262` | `0.8738` |
| `P7C03` | hidden-state KD + anti-overemit | `1.0000` | `0.0515` | `0.2143` | `0.0000` | `0.1085` | `0.8915` |

## Detailed Reading

### 1. `P7C01` - Hidden-state distillation only

Key training metrics:

| Metric | Value |
|---|---:|
| `train_loss` mean | `7.2987` |
| `train_ctc_loss` mean | `7.2514` |
| `train_kd_loss` mean | `0.2362` |
| `train_kd_active_ratio` mean | `1.0000` |
| `train_overemit_loss` mean | `0.0000` |
| `best_valid_epoch` mean | `3.0` |
| `finetune_early_stop_epoch` mean | `5.0` |

Validation metrics:

| Metric | Value |
|---|---:|
| `valid_accuracy` mean | `0.0520` |
| `valid_blank_ratio` mean | `0.2080` |
| `valid_empty_pred_ratio` mean | `0.0000` |
| `valid_pred_to_ref_char_ratio` mean | `0.0975` |

Per-seed best test metrics:

| Seed | WER | Accuracy | Blank Ratio | Pred/Ref Char Ratio | Short Pred Ratio |
|---|---:|---:|---:|---:|---:|
| `42` | `1.0000` | `0.0398` | `0.1987` | `0.1138` | `0.8862` |
| `123` | `1.0110` | `0.1029` | `0.0000` | `0.0906` | `0.9094` |

Interpretation:

- this is the best run of the three;
- it clearly improves `accuracy` over `P7B01`;
- it also strongly reduces the over-generation problem of `P7B01`;
- however, the model now under-emits badly:
  - `pred_to_ref_char_ratio ≈ 0.10`
  - `short_pred_ratio ≈ 0.90`

So `P7C01` replaces babbling collapse with a new failure mode:

- sparse, too-short predictions;
- still not usable ASR, but diagnostically better.

### 2. `P7C02` - Logit KD + anti-overemit

Key training metrics:

| Metric | Value |
|---|---:|
| `train_loss` mean | `9.6284` |
| `train_ctc_loss` mean | `6.7498` |
| `train_kd_loss` mean | `10.6339` |
| `train_kd_active_ratio` mean | `0.4103` |
| `train_overemit_loss` mean | `0.4403` |
| `best_valid_epoch` mean | `2.5` |
| `finetune_early_stop_epoch` mean | `4.5` |

Validation metrics:

| Metric | Value |
|---|---:|
| `valid_accuracy` mean | `0.0303` |
| `valid_blank_ratio` mean | `0.6343` |
| `valid_empty_pred_ratio` mean | `0.5000` |
| `valid_pred_to_ref_char_ratio` mean | `0.0454` |

Interpretation:

- anti-overemit does push the model away from babbling;
- but in this configuration it is too aggressive;
- it damages the output regime more than it helps;
- the model becomes both short and partially blank again.

Conclusion:

- the anti-overemit idea is not absurd;
- but the current weight/configuration is too strong.

### 3. `P7C03` - Hidden-state KD + anti-overemit

Key training metrics:

| Metric | Value |
|---|---:|
| `train_loss` mean | `7.6875` |
| `train_ctc_loss` mean | `7.3687` |
| `train_kd_loss` mean | `0.2463` |
| `train_kd_active_ratio` mean | `1.0000` |
| `train_overemit_loss` mean | `0.5392` |
| `best_valid_epoch` mean | `2.5` |
| `finetune_early_stop_epoch` mean | `4.5` |

Validation metrics:

| Metric | Value |
|---|---:|
| `valid_accuracy` mean | `0.0493` |
| `valid_blank_ratio` mean | `0.1894` |
| `valid_empty_pred_ratio` mean | `0.0000` |
| `valid_pred_to_ref_char_ratio` mean | `0.1071` |

Interpretation:

- combining the two mechanisms does not beat `P7C01`;
- the anti-overemit term partially cancels the gain from hidden-state KD;
- the model remains in a strong under-emission regime.

Conclusion:

- the combined variant is not the best direction at this stage;
- the extra regularization is not paying for itself.

## Main Conclusion

The promising advance is **real**, but limited:

- the hidden-state distillation hypothesis is partially validated;
- `P7C01` is the best Phase 7 variant so far;
- it improves `accuracy` from `0.0317` (`P7B01`) to `0.0713`;
- it also suppresses the strong babbling behavior of `P7B01`.

However:

- this does not produce a good ASR system;
- `WER` remains around `1`;
- the dominant failure mode has changed again.

The sequence is now:

1. earlier Phase 7 baseline:
   - non-empty babbling / over-emission;
2. `P7C01`:
   - much less babbling;
   - but severe under-emission.

This is still progress, because the new failure mode is easier to reason about than babbling collapse.

## Most Defensible Interpretation

The most defensible interpretation is:

- hidden-state KD is a better supervision signal than framewise logit KD for this student/CTC setup;
- anti-overemit in its current form is too blunt;
- the next minimal step should probably keep the `P7C01` objective unchanged and improve checkpoint selection before touching the loss again.

In other words:

- `P7C01` should be treated as the new base;
- `P7C02` and `P7C03` do not justify replacing it.
