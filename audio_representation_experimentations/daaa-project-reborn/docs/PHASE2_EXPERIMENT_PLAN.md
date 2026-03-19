# Plan d'expérimentation Phase 2

## Objectif général

La phase 2 est une campagne corrective.

Son but n'est pas de multiplier immédiatement les ablations. Son premier but
est de sortir du régime actuel dégénéré où `WER ~= 1.0` pour presque toutes les
variantes.

Ce n'est qu'une fois une baseline non dégénérée obtenue qu'il faudra reprendre
les comparaisons d'architecture et de pré-entraînement.

## Pourquoi une phase 2 est nécessaire

La phase 1 a produit trois signaux forts :

1. le pipeline tourne
2. les différences de consommation de ressources sont mesurables
3. la qualité ASR ne se sépare pas de façon significative entre variantes

Le troisième point est bloquant. Il faut donc un nouveau protocole centré sur
le diagnostic et la récupération, pas sur la comparaison immédiate de
leaderboards.

## Objectifs de la phase 2

### Objectif A

Obtenir au moins une baseline qui transcrit mieux que le régime actuel quasi
dégénéré.

### Objectif B

Vérifier que les prédictions ne sont pas dominées par `blank` ni par des sorties
quasi vides.

### Objectif C

Créer un vrai contraste sur la qualité ASR avant de reprendre des ablations
architecturales fines.

## Règles de la phase 2

1. ne pas réutiliser le nommage `E00 -> E11`
2. ne pas mélanger les sorties phase 2 avec l'interprétation héritée de phase 1
   dans `results/experiments/`
3. préférer peu de runs, mais fortement diagnostiqués, à beaucoup d'ablations
   faibles
4. inspecter explicitement les prédictions pendant la phase 2
5. ne rouvrir `MAE vs NoMAE` ou les comparaisons d'architecture qu'après la
   sortie du régime `WER ~= 1.0`

## Nommage proposé

Utiliser un nouveau préfixe pour la phase 2, par exemple :

- `P2Dxx` pour les runs diagnostiques
- `P2Bxx` pour les baselines de récupération
- `P2Axx` pour les ablations ultérieures

Cela évite toute confusion avec les identifiants de phase 1.

## Première liste minimale de runs phase 2

Le premier lot phase 2 doit rester petit et orienté diagnostic.

### P2D01 - Baseline diagnostique CTC

But :

- inspecter si les prédictions sont vides, dominées par `blank`, répétitives ou
  mal alignées

Modifications principales :

- conserver une architecture baseline simple
- ajouter des dumps qualitatifs de prédictions sur la validation
- logguer les longueurs d'entrée, longueurs de cible, longueurs de prédiction et
  la dominance de `blank`

Valeur attendue :

- identifier si l'échec actuel correspond bien à un collapse CTC

### P2B01 - Baseline de récupération par budget

But :

- tester si le budget supervisé actuel est simplement trop faible

Modifications principales :

- augmenter `finetune.max_steps`
- éventuellement augmenter le budget de données supervisées effectif
- garder le reste aussi stable que possible

Valeur attendue :

- si le `WER` baisse de façon nette, le problème principal est le budget plutôt
  qu'un choix architectural exotique

### P2B02 - Baseline de récupération NoMAE

But :

- comparer une récupération supervisée simple à la récupération basée sur MAE

Modifications principales :

- même budget d'entraînement que `P2B01`
- pré-entraînement désactivé

Valeur attendue :

- comparaison `MAE vs NoMAE` réellement informative, mais seulement une fois la
  baseline sortie du régime dégénéré

### P2A01 - Variante à compression plus faible

But :

- tester si l'encodeur compresse trop agressivement le temps pour CTC

Modifications principales :

- réduire la compression temporelle ou la taille de patch sur l'axe temps
- garder le reste proche de la meilleure baseline de récupération

Valeur attendue :

- vérifier si l'échec d'alignement vient en partie de la compression de séquence

## Critères de succès

La phase 2 ne doit pas être jugée seulement sur le runtime.

Les premiers critères de succès acceptables sont :

1. `WER < 1.0` de manière stable et répétable
2. des prédictions de validation visiblement non vides et non dominées par
   `blank`
3. au moins une baseline présentant un delta de qualité assez grand pour
   justifier une vraie comparaison de suivi

Ce n'est qu'après validation de ces critères qu'il faudra reprendre :

- `MAE` vs `NoMAE`
- comparaisons de capacité
- comparaisons d'embeddings positionnels
- comparaisons de mask ratio

## Usage recommandé des dossiers phase 2

- archive des launchers phase 1 : `scripts/experiments_phase1/`
- espace de travail des launchers phase 2 : `scripts/experiments_phase2/`
- archive des configs phase 1 : `configs/phase1/`
- espace de travail des configs phase 2 : `configs/phase2/`
- snapshot de suite phase 1 : `results/phase1/`
- sorties phase 2 : `results/phase2/`

## Prochaines tâches d'implémentation

1. ajouter un mode diagnostic qui loggue un petit lot de références et de
   prédictions
2. créer une baseline phase 2 propre
3. créer une variante à budget étendu
4. créer une variante `NoMAE` alignée
5. garder toutes les sorties phase 2 sous des chemins explicitement dédiés

## État d'implémentation actuel

Les premières briques de phase 2 sont maintenant en place :

1. de nouveaux launchers séparés sous `scripts/experiments_phase2/`
2. de nouvelles configs séparées sous `configs/phase2/`
3. des sorties séparées sous `results/phase2/experiments/`
4. des diagnostics CTC persistés pendant la validation et le test

## Correctif pipeline déjà intégré

Un point de risque d'implémentation a été corrigé avant les runs phase 2 :

- le décodage greedy respecte désormais `out_lengths`

Avant ce correctif, l'évaluation décodait toute la longueur temporelle des
logits, y compris les pas au-delà de la longueur utile estimée par le modèle.
Cela pouvait polluer les transcriptions décodées et donc le `WER`.

Ce correctif ne garantit pas à lui seul une récupération complète, mais il
retire une source crédible d'erreur d'implémentation dans le pipeline ASR.
