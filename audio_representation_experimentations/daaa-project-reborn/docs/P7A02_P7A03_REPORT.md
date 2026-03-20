# P7A02 / P7A03 - Rapport court

## Objet

Ce rapport clot deux pistes ouvertes pour mieux couvrir les consignes du TP :

- `P7A02` : ajout d'augmentations legeres ;
- `P7A03` : petite ablation d'architecture (`depth=4` au lieu de `depth=6`).

La reference de comparaison reste `P7C01`, qui est la meilleure base Phase 7 a ce stade.

## Reference

Run de reference :

- `P7C01`
- config : `configs/phase7/hidden_states_librispeech.yaml`

Resultats test `best` :

- `WER = 1.0055`
- `accuracy = 0.07133`
- `blank_ratio = 0.09937`
- `empty_pred_ratio = 0.0`
- `pred_to_ref_char_ratio = 0.10223`

## P7A02 - Augmentations legeres

Config :

- `configs/phase7/hidden_states_aug_librispeech.yaml`

Ajouts :

- gain aleatoire ;
- bruit additif faible ;
- SpecAugment leger.

Resultats test `best` :

- `WER = 1.0055`
- `accuracy = 0.07106`
- `blank_ratio = 0.10855`
- `empty_pred_ratio = 0.0`
- `pred_to_ref_char_ratio = 0.10271`

Lecture :

- le `WER` ne bouge pas ;
- l'`accuracy` baisse legerement ;
- le `blank_ratio` augmente legerement ;
- le regime global reste identique.

Conclusion `P7A02` :

- les augmentations legeres ont ete implementees proprement ;
- elles n'ont pas apporte d'amelioration mesurable dans cette recette.

## P7A03 - Ablation profondeur

Config :

- `configs/phase7/hidden_states_depth4_librispeech.yaml`

Changement unique :

- `depth=4` au lieu de `depth=6`

Resultats test `best` :

- `WER = 1.0041`
- `accuracy = 0.07080`
- `blank_ratio = 0.10373`
- `empty_pred_ratio = 0.0`
- `pred_to_ref_char_ratio = 0.09916`

Lecture :

- le `WER` est tres legerement meilleur ;
- l'`accuracy` est tres legerement plus basse ;
- la sous-emission reste presente ;
- le meilleur seed retrouve le meme ordre de grandeur que `P7C01` (`~10.3%` de char accuracy),
  mais la moyenne sur deux seeds n'est pas superieure.

Informations complementaires :

- `P7C01` : `2,839,133` parametres
- `P7A03` : `1,949,405` parametres

Conclusion `P7A03` :

- c'est une ablation propre et defensable pour le rapport ;
- elle montre qu'un modele plus petit peut conserver un regime comparable ;
- elle ne bat pas `P7C01` de facon nette.

## Conclusion generale

1. `P7A02` n'a pas aide.
2. `P7A03` est comparable, mais pas meilleur.
3. `P7C01` reste la meilleure base operationnelle.

La bonne lecture n'est donc pas :

- "il manque seulement des augmentations" ;
- "il suffit de reduire la profondeur".

La bonne lecture est :

- le verrou principal reste la sous-emission du student.

## Prochain patch propose

Le patch suivant doit partir de `P7C01` et viser explicitement la sous-emission,
sans reintroduire le babbling de `P7B01`.

### Intention

Aujourd'hui, les symptomes dominants sont :

- `pred_to_ref_char_ratio ~ 0.10`
- `short_pred_ratio ~ 0.90`
- sorties non vides, mais beaucoup trop courtes

Le prochain patch doit donc encourager une emission legerement plus dense.

### Proposition concrete

Ajouter une penalite douce de sous-emission sur la densite non-blank du student :

- conserver `hidden-state KD`
- ne pas remettre `anti_overemit`
- ajouter un terme `anti_underemit` leger

Idee :

- estimer la densite cible via `target_lengths / out_lengths`
- estimer la densite non-blank du student
- penaliser seulement quand la densite student est trop en dessous de la densite cible

Forme possible :

- `loss = lambda * relu((target_density - margin) - nonblank_density)^2`

### Pourquoi ce patch est rationnel

- il cible directement le mode d'echec actuel ;
- il reste minimal et compatible avec la boucle existante ;
- il n'oblige pas a refaire la distillation ;
- il est plus coherent avec le diagnostic actuel qu'un nouveau micro-tuning de `lambda_kd`.

### Nom d'experience suggere

- `P7N01`
- base : `P7C01`
- changement unique : ajout d'une contrainte douce contre la sous-emission

## Decision recommandee

1. figer `P7A02` comme resultat negatif utile ;
2. garder `P7A03` comme ablation propre pour le rapport ;
3. repartir de `P7C01` pour le prochain patch ;
4. implementer un test `P7N01` centre sur la sous-emission.
