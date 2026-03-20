# P7C04 - Selection Score v2 Report

## Objet

`P7C04` avait un objectif volontairement limité :

- ne **pas** changer la loss ;
- ne **pas** changer le student ;
- ne **pas** changer la distillation ;
- tester uniquement un nouveau `checkpoint_selection` sur la base `P7C01`.

L'hypothèse testée était :

- un meilleur score de sélection pourrait choisir un `ctc_best.pt` plus intéressant,
- sans avoir besoin de retoucher l'entraînement lui-même.

## Références

- [train_audio_transformer_P7C04_final.json](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7C04/benchmark_results/train_audio_transformer_P7C04_final.json)
- [asr_checkpoint_variants_P7C04_final.json](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7C04/benchmark_results/asr_checkpoint_variants_P7C04_final.json)
- [asr_checkpoint_variants_P7C01_final.json](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/results/phase7/experiments/P7C01/benchmark_results/asr_checkpoint_variants_P7C01_final.json)

## Résultats principaux

### `P7C04 best`

| Métrique | Valeur |
|---|---:|
| `wer` | `1.0000` |
| `accuracy` | `0.0267` |
| `blank_ratio` | `0.0000` |
| `empty_pred_ratio` | `0.0000` |
| `pred_to_ref_char_ratio` | `0.0351` |
| `short_pred_ratio` | `0.9649` |
| `length_deviation_ratio` | `0.9649` |
| `adjacent_repeat_ratio` | `0.0000` |
| `dominant_char_ratio` | `1.0000` |

### `P7C04 final`

| Métrique | Valeur |
|---|---:|
| `wer` | `1.0055` |
| `accuracy` | `0.0705` |
| `blank_ratio` | `0.0792` |
| `empty_pred_ratio` | `0.0000` |
| `pred_to_ref_char_ratio` | `0.1025` |
| `short_pred_ratio` | `0.8975` |
| `length_deviation_ratio` | `0.8975` |
| `adjacent_repeat_ratio` | `0.7339` |
| `dominant_char_ratio` | `1.0000` |

### Référence utile : `P7C01 best`

| Métrique | Valeur |
|---|---:|
| `wer` | `1.0055` |
| `accuracy` | `0.0713` |
| `blank_ratio` | `0.0994` |
| `empty_pred_ratio` | `0.0000` |
| `pred_to_ref_char_ratio` | `0.1022` |
| `short_pred_ratio` | `0.8978` |

## Conclusion

`P7C04` n'a pas marché.

Plus précisément :

- le nouveau score n'a pas amélioré le choix du meilleur checkpoint ;
- il a au contraire sélectionné un modèle **plus mauvais** que `P7C01 best` ;
- `P7C04 best` est tombé sur une solution très sous-émissive ;
- `P7C04 final` est presque équivalent à `P7C01 best`, ce qui montre que le problème vient bien de la **sélection**, pas de l'entraînement.

## Interprétation

Le score v2 a trop favorisé :

- `WER = 1.0`,
- `blank_ratio = 0`,
- `empty_pred_ratio = 0`,

et n'a pas suffisamment pénalisé :

- les sorties extrêmement courtes ;
- les sorties dominées par un seul caractère.

Le signal le plus clair est :

- `pred_to_ref_char_ratio = 0.0351`
- `dominant_char_ratio = 1.0000`

Donc le score v2 a préféré une solution quasiment mono-caractère, trop courte, mais non vide.

## Décision

`P7C04` ne doit pas devenir la nouvelle base.

La meilleure base Phase 7 reste :

- `P7C01`

Autrement dit :

- l'idée "améliorer uniquement le score de sélection" n'est pas validée en l'état ;
- la prochaine étape rationnelle doit repartir de `P7C01`, pas de `P7C04`.
