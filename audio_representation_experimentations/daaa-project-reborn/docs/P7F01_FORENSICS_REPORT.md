# P7F01 - Forensics Report

## Objet

Ce rapport fige ce que `P7F01` a permis de trancher sur la meilleure base actuelle
`P7C01`, sans changer la loss ni l'architecture de reference.

Sources principales :

- `results/phase7/experiments/P7F01/benchmark_results/asr_forensics_P7F01_final.json`
- `results/phase7/experiments/P7F01/benchmark_results/train_audio_transformer_P7F01_final.json`
- `results/phase7/experiments/P7F01/tables/asr_forensics_overview_P7F01.md`
- `results/phase7/experiments/P7F01/tables/asr_forensics_alignment_P7F01.md`

## Resume net

`P7F01` montre que le goulot principal actuel n'est pas un simple probleme de
selection de checkpoint ou de decodage greedy.

Le student collapse deja au niveau des logits vers un token quasi unique, puis le
CTC greedy ne fait que reveler ce collapse.

Deux facteurs structurels ressortent :

1. le student opere a environ `100 Hz` alors que le teacher est a environ `50 Hz`,
   ce qui impose un upsampling temporel d'environ `x2` ;
2. le dataset effectif apres filtres est beaucoup trop petit (`127` exemples train),
   donc la conclusion "le modele ne sait pas apprendre l'ASR" serait trop forte dans
   cette configuration de donnees.

## Ce que `P7F01` a permis de trancher

### 1. La configuration actuelle n'est pas "a moins d'une demi-epoque"

Le diagnostic d'exposition train montre :

- `train_dataset_size_after_filters = 127`
- seed `42` :
  - `train_examples_seen = 889`
  - `train_effective_epochs_completed = 7.0`
  - `train_optimizer_steps_completed = 111`
- seed `123` :
  - `train_examples_seen = 635`
  - `train_effective_epochs_completed = 5.0`
  - `train_optimizer_steps_completed = 79`

Conclusion :

- l'hypothese "on n'a presque rien entraine" est fausse **dans le monde reel de la
  config actuelle**, car le train ne contient en pratique que `127` exemples ;
- en revanche, `127` exemples restent beaucoup trop peu pour juger serieusement un
  pipeline ASR from scratch.

### 2. Le collapse est visible avant le decodage CTC

Sur `test_best` :

- `accuracy mean = 0.0713`
- `WER mean = 1.0055`
- `pred_to_ref_char_ratio mean = 0.1022`
- `raw_dominant_token_ratio mean = 0.8495`
- `raw_token_switch_ratio mean = 0.0182`
- `collapsed_dominant_char_ratio mean = 1.0`

Lecture :

- environ `85%` des frames valides sont deja occupees par un seul token top-1 ;
- le token dominant change tres peu (`~1.8%` de switchs) ;
- apres collapse CTC, la sortie est entierement dominee par un seul caractere.

Conclusion :

- le probleme n'est pas d'abord le greedy ;
- le probleme est un **monotoken collapse pre-CTC**.

### 3. Les seeds n'apprennent pas des transcriptions, elles collapsent differemment

Sur `test_best` :

- seed `42` :
  - `accuracy = 0.0398`
  - `blank_ratio = 0.1987`
  - top token brut dominant : `r`
  - `raw_dominant_token_ratio = 0.7994`
  - sortie collapsée : `100% r`

- seed `123` :
  - `accuracy = 0.1029`
  - `blank_ratio = 0.0`
  - top token brut dominant : `e`
  - `raw_dominant_token_ratio = 0.8996`
  - second token non-negligeable : espace
  - sortie collapsée : melange `e` / espace, mais toujours degenere

Conclusion :

- la seed `123` parait meilleure non parce qu'elle transcrit, mais parce qu'elle
  collapse vers un token un peu moins absurde que `42`.

### 4. Le mismatch de resolution teacher/student est reel

Les tables d'alignement montrent, de facon stable sur train/valid/test :

- `student_frames_per_second_estimate ~= 100.27`
- `teacher_frames_per_second_estimate ~= 49.77`
- `teacher_to_student_time_ratio_mean ~= 0.496`
- `alignment_interpolation_ratio_mean ~= 2.015`

Conclusion :

- le student voit environ deux fois plus de frames temporelles que le teacher ;
- la distillation hidden-state repose donc sur une interpolation temporelle `x2`,
  qui constitue un facteur aggravant plausible.

### 5. Le student n'est pas bien aligne au teacher

Sur `test_best` :

- `teacher_student_argmax_agreement mean = 0.1244`
- `teacher_student_feature_cosine_mean = 0.2722`
- `teacher_blank_prob_mean = 0.6140`

Lecture :

- l'accord top-1 teacher/student reste tres faible ;
- la similarite de features reste faible egalement ;
- le student ne reproduit pas un signal teacher riche et coherent.

Conclusion :

- la distillation actuelle n'est pas en train de transmettre une structure
  acoustico-linguistique exploitable ;
- elle coexiste avec un collapse non-blank vers un token dominant.

## Decision d'ingenierie

La suite rationnelle n'est plus :

- d'ajouter une nouvelle penalite locale ;
- de retoucher encore le `selection_score` ;
- ou de tuner a la marge `lambda_kd`.

La suite rationnelle est :

1. **corriger la resolution temporelle du student**
   - `patch_time = 2`
   - objectif : ramener le student a environ `50 Hz`, donc l'aligner sur le teacher

2. **desserrer fortement le filtrage dataset**
   - les filtres actuels (`8s`, `50 chars`, `10 mots`) reduisent le train reel a `127`
     exemples
   - ce regime est trop etroit pour juger l'ASR from scratch

## Patch suivant retenu

Le patch suivant garde la loss et la distillation intactes, et change seulement :

- `model.patch_time: 2`
- filtres dataset ASR plus permissifs

Ce choix isole deux hypotheses structurelles, sans relancer une nouvelle phase de
bricolage sur la loss :

- alignement temporel teacher/student ;
- taille effective du train apres filtrage.

## Limite assumee

Ce patch ne pretend pas encore resoudre tout seul la question du budget
d'entrainement. Il vise d'abord a sortir du regime artificiellement etroit de
`P7C01/P7F01`, de sorte que le prochain run soit interpretable.
