# Plan d'expérimentation Phase 4

## Objet

La phase 4 vise à tester l'hypothèse technique restante la plus crédible après
les phases 2 et 3 :

1. le Transformer traitait encore du padding comme du vrai signal, faute de
   `padding mask` ;
2. les log-mels restaient peu conditionnés pour un apprentissage CTC direct ;
3. le protocole final doit désormais suivre à la fois le `WER` et une
   `accuracy` caractère normalisée pour détecter un progrès partiel avant même
   qu'il devienne visible au niveau mot.

## Diagnostic hérité de la phase 3

La phase 3 a validé que :

1. le problème de longueurs CTC était réel ;
2. la suppression du tronquage dur et la réduction de la compression
   temporelle étaient nécessaires ;
3. malgré cela, le système restait bloqué soit en `blank collapse`, soit en
   `token collapse`.

La conclusion opérationnelle est la suivante :

- le pipeline n'est plus structurellement invalide pour CTC ;
- mais l'encodeur traite encore probablement trop de padding, et l'entrée
  acoustique reste fragile à optimiser.

## Corrections intégrées en phase 4

Les changements techniques intégrés dans le code et utilisés par les runs phase
4 sont :

1. **padding mask propagé dans le Transformer**
   - `src_key_padding_mask` passé à l'encodeur ;
   - positions paddées remises à zéro après encodage ;
   - masque également utilisé côté MAE pour éviter de reconstruire du padding.

2. **normalisation d'utterance des log-mels**
   - centrage-réduction par bande de Mel sur chaque exemple ;
   - objectif : rendre l'optimisation CTC plus stable et réduire les biais
     d'échelle.

3. **benchmark enrichi**
   - `WER` conservé comme métrique principale ;
   - `accuracy` ajoutée comme métrique secondaire ;
   - définition retenue : accuracy caractère normalisée après
     `normalize_transcript`.

## Hypothèses phase 4

### H4.1

Si le padding mask était une cause importante, alors la phase 4 doit au moins
réduire :

- `blank_ratio`
- `empty_pred_ratio`
- la fréquence des sorties constantes de type `"g"`

### H4.2

Si la normalisation des features aide vraiment, alors une progression peut être
visible d'abord sur `accuracy`, même si le `WER` reste dur à faire baisser.

### H4.3

Si aucune progression n'apparaît ni sur `WER`, ni sur `accuracy`, alors il
restera une cause plus profonde liée à la recette d'entraînement ou à
l'architecture elle-même, et non plus seulement à l'implémentation.

## Expériences prévues

### `P4D01`

Run diagnostic rapide sans MAE.

But :

- vérifier si `padding mask + feature_norm` suffisent à faire sortir le système
  du régime dégénéré ;
- mesurer si `accuracy` bouge avant le `WER`.

### `P4B01`

Baseline corrective phase 4 avec MAE.

But :

- réévaluer la chaîne `MAE -> CTC` avec le pipeline le plus propre disponible ;
- observer si le préentraînement devient enfin utile une fois les principaux
  défauts de conditionnement retirés.

### `P4B02`

Baseline corrective phase 4 sans MAE.

But :

- mesurer si la progression éventuelle vient surtout des correctifs de
  pipeline ;
- conserver une comparaison `MAE` vs `NoMAE` sous protocole désormais plus sain.

## Critères de succès

La phase 4 sera considérée comme encourageante si au moins un run montre :

1. `WER < 1.0`
2. `accuracy` en hausse nette par rapport à la phase 3
3. `empty_pred_ratio < 0.5`
4. disparition du collapse vers un token unique sur les exemples sauvegardés

## Règle de lecture

Le `WER` reste la métrique prioritaire pour le rapport final.

L'`accuracy` ajoutée en phase 4 a un rôle de diagnostic :

- si `accuracy` reste nulle ou quasi nulle, le système n'apprend toujours rien
  d'exploitable ;
- si `accuracy` monte alors que `WER` reste mauvais, cela indiquera un début
  d'apprentissage encore insuffisant au niveau mot, mais réel au niveau
  caractère.
