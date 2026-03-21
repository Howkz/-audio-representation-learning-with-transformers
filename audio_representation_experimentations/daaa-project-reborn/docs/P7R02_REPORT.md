# P7R02 - Rapport de resultat

## Objet

Ce rapport documente `P7R02`, c'est-a-dire la variante qui conserve :

- `patch_time=2`
- le dataset ASR relache de `P7R01`
- la hidden-state KD

et ajoute :

- une rampe lineaire de KD au debut du fine-tune
- un budget d'entrainement plus large

Sources :

- `results/phase7/experiments/P7R02/benchmark_results/train_audio_transformer_P7R02_final.json`
- `results/phase7/experiments/P7R02/benchmark_results/asr_forensics_P7R02_final.json`
- `results/phase7/experiments/P7R02/benchmark_results/asr_checkpoint_variants_P7R02_final.json`

Comparaisons :

- `P7F01`
- `P7R01`

## Resume net

`P7R02` est **meilleur que `P7R01`**, mais reste **un echec massif en ASR**.

La rampe de KD a legerement desserre le blank collapse, sans faire revenir le regime
non-vide observe dans `P7F01`.

## Ce que `P7R02` changeait

Par rapport a `P7R01`, `P7R02` garde la meme base structurelle :

- student a `~50 Hz`
- teacher a `~50 Hz`
- dataset effectif train plus large

mais modifie la dynamique de KD :

- `lambda_kd target = 0.20`
- `distillation.warmup_steps = 60`
- `distillation.warmup_start_factor = 0.0`

Budget d'entrainement :

- `epochs = 10`
- `max_steps = 3000`
- `early_stopping_min_epochs = 5`
- `early_stopping_patience = 5`

## Validation du patch

Le patch a bien ete applique :

- `train_kd_lambda_effective mean = 0.1837`
- `distillation_lambda_kd_target = 0.2`
- `distillation_kd_warmup_steps = 60`
- `train_optimizer_steps_completed = 375`
- `finetune_early_stopped = 0`

Conclusion :

- la KD a bien commence plus faible puis est montee ;
- le run n'a pas ete coupe trop tot ;
- l'experience est interpretable.

## Resultats principaux

### Train

`P7R02`

- `train_dataset_size_after_filters = 577`
- `train_examples_seen = 3000`
- `train_effective_epochs_completed = 5.1993`
- `train_optimizer_steps_completed = 375`
- `train_kd_loss mean = 0.1080`
- `train_kd_lambda_effective mean = 0.1837`

Comparaison :

- `P7F01`
  - `dataset_size = 127`
  - `examples_seen = 762`
  - `optimizer_steps = 95`

- `P7R01`
  - `dataset_size = 577`
  - `examples_seen = 1500`
  - `optimizer_steps = 188`

Lecture :

- `P7R02` a bien donne plus de budget et plus de temps a la dynamique KD.

### Test `best`

`P7R02`

- `WER mean = 1.0`
- `accuracy mean = 0.0029`
- `blank_ratio mean = 0.9983`
- `empty_pred_ratio mean = 0.8659`
- `pred_to_ref_char_ratio mean = 0.0036`
- `raw_blank_argmax_ratio mean = 0.9986`
- `raw_dominant_token_ratio mean = 0.9986`
- `raw_token_switch_ratio mean = 0.0014`

Comparaison :

- `P7F01`
  - `accuracy = 0.0713`
  - `blank_ratio = 0.0994`
  - `empty_pred_ratio = 0.0`
  - `pred_to_ref_char_ratio = 0.1022`

- `P7R01`
  - `accuracy = 0.0`
  - `blank_ratio = 1.0`
  - `empty_pred_ratio = 1.0`
  - `pred_to_ref_char_ratio = 0.0`

Lecture :

- `P7R02` est meilleur que `P7R01`
- mais reste tres loin de `P7F01`
- on a encore un blank collapse quasi complet

## Ce qui s'est ameliore par rapport a `P7R01`

`P7R01`

- `blank_ratio = 1.0`
- `empty_pred_ratio = 1.0`
- `accuracy = 0.0`
- `pred_to_ref_char_ratio = 0.0`

`P7R02`

- `blank_ratio = 0.9983`
- `empty_pred_ratio = 0.8659`
- `accuracy = 0.0029`
- `pred_to_ref_char_ratio = 0.0036`

Conclusion :

- la rampe KD a desserre legerement le collapse ;
- elle n'a pas change le regime de fond.

## Ce qui reste structurellement bon

Les acquis de `P7R01` restent presents :

- `alignment_interpolation_ratio_mean = 1.0066`
- `teacher_student_argmax_agreement = 0.5565`
- `teacher_student_feature_cosine_mean = 0.5684`

Lecture :

- l'alignement temporel teacher/student est toujours bon ;
- le student reste bien plus proche du teacher que dans `P7F01`.

## Interpretation

`P7R02` permet de conclure que :

1. le blank collapse de `P7R01` ne venait pas seulement d'une KD trop forte au tout
   debut ;
2. la warmup lineaire aide un peu, mais insuffisamment ;
3. avec `patch_time=2`, la hidden-state KD actuelle reste tres compatible avec une
   solution blank-dominante.

Autrement dit :

- le probleme n'est plus le mismatch temporel ;
- le probleme n'est plus un run trop court ;
- le probleme devient la **forme meme de la KD hidden-state dans ce regime aligne**.

## Conclusion

`P7R02` est un **echec instructif utile** :

- il valide que la simple warmup lineaire de la KD ne suffit pas ;
- il conserve les progres structurels de `P7R01` ;
- il clarifie que la prochaine piste doit aller au-dela d'un simple amortissement de
  `lambda_kd`.

La bonne lecture est donc :

- `P7R01` : blank collapse total apres correction structurelle
- `P7R02` : blank collapse legerement desserre, mais toujours dominant

## Decision rationnelle

La suite ne devrait plus etre :

- un nouveau micro-tuning de `lambda_kd`
- ni un nouveau simple warmup lineaire

La suite devrait plutot tester :

1. une phase initiale **CTC seule** avant activation de la KD ;
2. ou une **KD hidden-state selective**, appliquee seulement sur certains frames ;
3. ou une autre forme de supervision teacher/student moins blank-compatible.
