# P7R01 - Echec instructif apres alignement temporel

## Objet

Ce rapport documente l'echec instructif de `P7R01`, c'est-a-dire la variante qui :

- conserve la hidden-state distillation de `P7C01`,
- corrige la resolution temporelle du student avec `patch_time=2`,
- desserre fortement les filtres dataset ASR.

Sources :

- `results/phase7/experiments/P7R01/benchmark_results/train_audio_transformer_P7R01_final.json`
- `results/phase7/experiments/P7R01/benchmark_results/asr_forensics_P7R01_final.json`
- `results/phase7/experiments/P7R01/benchmark_results/asr_checkpoint_variants_P7R01_final.json`
- comparaison avec `P7F01`

## Resume net

`P7R01` a reussi les deux corrections structurelles visees :

- alignement temporel student/teacher ;
- augmentation du dataset effectif apres filtrage.

Mais cet alignement plus propre a revele un nouvel echec beaucoup plus net :

- **blank collapse integral et immediat** sur les deux seeds.

Le patch est donc un echec de performance, mais un succes de diagnostic.

## Ce que `P7R01` a corrige

### 1. Le dataset effectif n'est plus derisoire

Comparaison train :

- `P7F01`
  - `train_dataset_size_after_filters = 127`
  - `train_examples_seen mean = 762`
  - `train_optimizer_steps_completed mean = 95`

- `P7R01`
  - `train_dataset_size_after_filters = 577`
  - `train_examples_seen mean = 1500`
  - `train_optimizer_steps_completed mean = 188`

Conclusion :

- le desserrage des filtres a bien change le regime experimental ;
- on n'est plus dans un train quasi miniature de `127` exemples.

### 2. L'alignement temporel teacher/student est effectivement corrige

Comparaison `test_best` :

- `P7F01`
  - `student_frames_per_second_estimate = 100.2711`
  - `teacher_frames_per_second_estimate = 49.7585`
  - `alignment_interpolation_ratio_mean = 2.0152`

- `P7R01`
  - `student_frames_per_second_estimate = 50.1627`
  - `teacher_frames_per_second_estimate = 49.8340`
  - `alignment_interpolation_ratio_mean = 1.0066`

Conclusion :

- `patch_time=2` aligne bien le student sur le teacher autour de `50 Hz` ;
- la correction structurelle de resolution temporelle fonctionne.

### 3. Le student est beaucoup plus proche du teacher

Comparaison `test_best` :

- `P7F01`
  - `teacher_student_argmax_agreement = 0.1244`
  - `teacher_student_feature_cosine_mean = 0.2722`

- `P7R01`
  - `teacher_student_argmax_agreement = 0.5566`
  - `teacher_student_feature_cosine_mean = 0.4956`

Conclusion :

- la hidden-state KD devient beaucoup plus efficace quand l'alignement temporel est propre.

## Ce qui casse

### 1. Blank collapse total

Sur `test_best`, pour les deux seeds :

- `accuracy = 0.0`
- `blank_ratio = 1.0`
- `empty_pred_ratio = 1.0`
- `pred_to_ref_char_ratio = 0.0`

Sur `train_probe` aussi :

- `blank_ratio = 1.0`
- `empty_pred_ratio = 1.0`
- `raw_blank_argmax_ratio = 1.0`
- `raw_token_switch_ratio = 0.0`

Conclusion :

- le collapse ne vient pas d'un mauvais checkpoint tardif ;
- il est deja present des les logits et tres tot pendant le fine-tune.

### 2. Le regime est reproductible, pas accidentel

Indices convergents :

- seed `42` et seed `123` tombent dans le meme mode d'echec ;
- `best_valid_epoch = 0` pour les deux seeds ;
- `finetune_early_stop_epoch = 2` pour les deux seeds ;
- `best_valid_score = 2.85` pour les deux seeds.

Conclusion :

- on n'est pas face a un bruit experimental marginal ;
- `P7R01` stabilise un **blank collapse propre et reproductible**.

## Interpretation

Le diagnostic le plus plausible est le suivant :

1. le mismatch `100 Hz -> 50 Hz` nuisait bien a la distillation dans `P7F01` ;
2. une fois l'alignement corrige, la hidden-state KD transmet beaucoup mieux le signal teacher ;
3. dans le regime actuel, ce signal mieux aligne devient trop compatible avec une solution
   blank-dominante, que le student adopte immediatement.

Autrement dit :

- `P7R01` ne dit pas que `patch_time=2` est une mauvaise idee ;
- `P7R01` dit que **la hidden-state KD actuelle devient trop aggressive trop tot**
  quand l'alignement temporel est bon.

## Decision rationnelle

Il ne faut pas revenir en arriere sur les acquis structurels de `P7R01` :

- conserver `patch_time=2` ;
- conserver le dataset plus large ;
- conserver l'observabilite forensic.

Le patch suivant doit viser la dynamique de la KD, pas l'architecture :

- **KD plus faible au debut**
- **KD warmup explicite sur les optimizer steps**

## Patch suivant retenu

`P7R02` garde le coeur de `P7R01`, mais ajoute :

- un warmup lineaire de `lambda_kd` sur le debut du fine-tune ;
- une KD qui commence plus faible puis remonte progressivement vers sa valeur cible.

L'objectif est simple :

- eviter le blank collapse immediat en tout debut d'entrainement,
- tout en conservant les benefices structurels de `P7R01`.
