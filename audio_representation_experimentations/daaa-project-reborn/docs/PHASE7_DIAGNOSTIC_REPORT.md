# Phase 7 - Diagnostic intermediaire avant patch KD

## Objet

Ce document synthétise l'état réel de la phase 7 avant toute correction supplémentaire
du pipeline de distillation. L'objectif est de figer proprement :

- ce qui a été implémenté ;
- ce que montrent `P7D01` et `P7B01` ;
- ce qui fonctionne effectivement ;
- ce qui échoue encore ;
- l'hypothèse technique principale qui motive le prochain patch.

Ce rapport sert de point d'arrêt méthodologique : on ne corrige pas le pipeline sans
avoir documenté l'état précédent.

## Rappel de l'intention de la phase 7

La phase 7 introduit une nouvelle voie "clean", distincte des phases 1 à 6, plus
alignée avec les consignes du TP :

- `make data`, `make train`, `make test` restent les seuls points d'entrée ;
- LibriSpeech `clean` devient le benchmark critique ;
- la distillation par teacher externe devient le chemin principal ;
- le student reste un encodeur Transformer frugal avec tête CTC ;
- le MAE devient optionnel et sort du chemin critique.

La configuration de référence a été pensée pour rester exécutable sur un
environnement contraint, avec `streaming: true`.

## Implémentation réellement en place

### Student

- entrée log-Mel 80 bandes ;
- embedding linéaire par trame (`patch_time=1`) ;
- encodeur Transformer avec `src_key_padding_mask` ;
- tête CTC simple.

### Teacher externe

- `facebook/wav2vec2-base-960h` ;
- utilisé uniquement pendant l'entraînement ;
- sortie teacher remappée sur le vocabulaire caractère du student ;
- perte supervisée totale :

`L = lambda_ctc * CTC + lambda_kd * KL`

### Sélection de checkpoint

La sélection du `best checkpoint` ne repose plus sur le `WER` seul. Elle intègre :

- `WER`
- `empty_pred_ratio`
- `blank_ratio`
- `short_pred_ratio`
- `accuracy`

Le but est d'éviter qu'un checkpoint vide soit systématiquement préféré.

## Résultats observés

### P7D01 - Smoke test

Le smoke test valide que la nouvelle voie fonctionne de bout en bout :

- le teacher externe est chargé ;
- le fine-tuning distillé s'exécute ;
- les checkpoints `best` et `final` sont produits ;
- l'évaluation complète tourne.

Mais en performance ASR, `P7D01` reste un échec :

- `WER = 1.0`
- `accuracy = 0.0`

Le point positif est que le `best checkpoint` n'est plus un simple collapse vide :

- le `best` produit des sorties non vides ;
- le `final` recollapse plutôt vers blank/vide.

Le smoke test a donc validé la chaîne logicielle, pas la qualité du modèle.

### P7B01 - Baseline clean distillée

`P7B01` est la première expérience Phase 7 un peu plus sérieuse.

Constats train :

- `train_invalid_length_ratio = 0.0` sur les deux seeds ;
- mémoire GPU train autour de `664-685 MB` ;
- runtime train autour de `387 s` par seed ;
- `train_ctc_loss` et `train_kd_loss` restent élevés.

Constats validation :

- `valid_wer = 1.0` sur les deux seeds ;
- `valid_accuracy` reste quasi nulle ;
- une seed retombe massivement vers blank ;
- l'autre sort partiellement du vide, mais sans apprendre à transcrire.

Constats test (`best`) :

- `WER mean = 1.0`
- `accuracy mean = 0.0404`
- `blank_ratio mean = 0.8126`
- `empty_pred_ratio mean = 0.4725`

Constats test (`final`) :

- `WER mean = 1.0`
- `accuracy mean = 0.0011`
- `blank_ratio mean = 0.9998`
- `empty_pred_ratio mean = 0.9633`

Conclusion immédiate :

- le `best checkpoint` est clairement plus informatif que le `final` ;
- mais même le `best` reste très mauvais ;
- la phase 7 ne bat pas encore la phase 6.

## Analyse qualitative du mode d'échec

Les deux seeds de `P7B01` ne convergent pas vers le même mauvais régime.

### Seed 42

Cette seed reste très proche d'un collapse blank :

- `empty_pred_ratio` très élevé ;
- prédictions souvent vides ;
- `accuracy` presque nulle.

### Seed 123

Cette seed produit des sorties non vides, mais dégénérées.

Exemple observé :

- référence : `concord returned to its place amidst the tents`
- prédiction : `tttttttttttttt`

On n'observe donc pas une transcription partielle utile, mais un
`token collapse` vers le caractère `t`.

Ce point est important :

- on n'est plus dans un simple collapse vide ;
- mais on n'est pas encore dans un apprentissage linguistique réel ;
- on est dans un optimum dégénéré non vide.

## Hypothèse technique principale

L'hypothèse la plus crédible est que la distillation logit-level actuelle pousse le
student vers un optimum trivial.

### 1. La composante KD est trop forte

Dans `P7B01`, la configuration utilise :

- `lambda_ctc = 1.0`
- `lambda_kd = 0.75`

Or les métriques train montrent environ :

- `train_ctc_loss ≈ 3.72`
- `train_kd_loss ≈ 5.67`

Donc, en contribution pondérée, la KD domine légèrement la CTC.

Autrement dit :

- la transcription de référence n'est pas le seul signal dominant ;
- l'imitation du teacher devient le centre de gravité de l'entraînement ;
- cela est risqué pour un student entraîné from scratch.

### 2. La KD est appliquée sur tous les frames valides

Le teacher Wav2Vec2 CTC produit naturellement :

- beaucoup de blank ;
- quelques caractères fortement localisés ;
- une distribution très déséquilibrée dans le temps.

Dans la boucle actuelle, la KD est calculée sur l'ensemble des frames valides après
alignement temporel teacher/student, sans filtrage explicite des frames blank-dominés.

Conséquence probable :

- une seed apprend surtout à prédire blank ;
- une autre seed apprend blank + un caractère fréquent ;
- l'optimisation trouve une solution locale simple, pas une vraie transcription.

### 3. Le remapping teacher -> student reste très local

Le teacher externe est remappé vers le vocabulaire caractère du student par agrégation
des probabilités sur les caractères compatibles.

Ce choix n'est pas absurde, mais il fournit une supervision très locale :

- caractère par caractère ;
- frame par frame ;
- sans contrainte linguistique de haut niveau ;
- sans filtrage des positions les plus ambiguës.

Cela favorise les collapses vers :

- blank ;
- ou un petit sous-ensemble de caractères fréquents.

### 4. Le score de sélection ne pénalise pas encore les sorties répétitives

Le nouveau score évite mieux les checkpoints vides qu'avant, mais il ne pénalise pas
encore explicitement :

- la faible diversité des caractères ;
- les séquences répétitives du type `tttttttt`.

Cela explique qu'un checkpoint non vide mais dégénéré puisse être retenu comme
`best checkpoint`.

## Ce que la phase 7 a déjà validé

Malgré l'échec en performance, la phase 7 a déjà établi plusieurs points utiles :

1. la nouvelle voie clean est exécutable sur l'environnement réel ;
2. la plomberie teacher/student fonctionne ;
3. la sélection `best` est moins absurde qu'aux phases précédentes ;
4. le pipeline n'est plus cassé au niveau des longueurs CTC ;
5. le mode d'échec a changé, ce qui est un signal méthodologiquement utile.

## Ce que la phase 7 n'a pas encore validé

La phase 7 n'a pas encore prouvé que :

1. la distillation améliore réellement l'ASR ;
2. la supervision KD actuelle est correctement formulée ;
3. la voie clean surpasse la phase 6 ;
4. le student apprend une structure de transcription exploitable.

## Décision avant patch

La bonne décision n'est pas de multiplier immédiatement les nouveaux runs.

Le prochain levier rationnel est un patch ciblé de la distillation :

1. réduire le poids de la KD ;
2. distiller seulement les frames teacher réellement informatifs ;
3. diminuer ou supprimer l'effet de blank dans la KD ;
4. ajouter un critère de sélection qui pénalise les sorties répétitives.

Ce patch devra être traité comme une correction ciblée d'objectif d'entraînement,
pas comme une nouvelle campagne d'ablation large.

## Conclusion

La phase 7, dans son état actuel, est une amélioration d'ingénierie mais pas encore
une amélioration de performance.

Elle montre que :

- la chaîne clean/distillée est désormais opérationnelle ;
- le mode d'échec ne relève plus d'un simple bug de pipeline ;
- la forme actuelle de la distillation logit-level est probablement le nouveau goulot
  d'étranglement principal.

Le patch suivant sera donc motivé non par un simple essai empirique, mais par un
diagnostic technique déjà documenté.
