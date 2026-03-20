# Rétrospective Phase 4

## Objet

La phase 4 visait à tester deux hypothèses restantes après la phase 3 :

1. le Transformer exploitait encore du padding comme s'il s'agissait de vrai
   signal ;
2. les log-mels étaient insuffisamment conditionnés pour un apprentissage CTC
   stable.

Elle a également introduit une nouvelle métrique secondaire :

- `accuracy` caractère normalisée, utilisée comme indicateur précoce d'un
  éventuel apprentissage partiel.

## Changements effectivement testés

La phase 4 a intégré :

1. `padding mask` dans le Transformer encodeur ;
2. prise en compte du masque valide dans le MAE ;
3. normalisation d'utterance des log-mels ;
4. benchmarks enrichis avec `accuracy`.

## Expériences exécutées

- `P4D01` : diagnostic sans MAE
- `P4B01` : baseline corrective avec MAE
- `P4B02` : baseline corrective sans MAE

## Résumé exécutif

La phase 4 est un résultat négatif propre.

Elle montre que :

1. la correction de phase 3 sur les longueurs CTC tient toujours ;
2. `padding mask + feature_norm` ne suffisent pas à faire émerger un ASR
   utilisable ;
3. l'`accuracy` ajoutée confirme que l'apprentissage utile reste quasi nul ;
4. la seule légère variation observée concerne `P4B02`, mais elle reste très
   loin d'un comportement satisfaisant.

## Tableau de synthèse

| Run | WER | Accuracy | Blank ratio | Empty pred ratio | Invalid length ratio | Lecture |
|---|---|---|---|---|---|---|
| `P4D01` | `1.0` | `0.0` | `1.0` | `1.0` | `0.0` | Collapse `blank` complet |
| `P4B01` | `1.0` | `0.0` | `0.999996` | `1.0` | `0.0` | MAE n'apporte rien de visible |
| `P4B02` | `1.0` | `0.002575` | `0.998547` | `0.898333` | `0.0` | Micro-signal sur une seed, encore inutilisable |

## Analyse détaillée

### `P4D01`

Le run diagnostic ne montre aucune progression :

- `valid_wer = 1.0`
- `valid_accuracy = 0.0`
- `valid_blank_ratio = 1.0`
- `valid_empty_pred_ratio = 1.0`
- `valid_invalid_length_ratio = 0.0`

Interprétation :

- le correctif `padding mask + feature_norm` ne suffit pas, à lui seul, à sortir
  le système du régime dégénéré ;
- le problème de longueurs CTC n'est plus présent ;
- le mode d'échec reste un `blank collapse` complet.

### `P4B01`

Le run avec MAE reste totalement bloqué :

- `test_wer = 1.0`
- `test_accuracy = 0.0`
- `blank_ratio ≈ 1.0`
- `empty_pred_ratio = 1.0`

Interprétation :

- le préentraînement MAE ne compense pas le problème restant ;
- dans le protocole actuel, `MAE -> CTC` ne donne toujours aucun signal
  exploitable.

### `P4B02`

Le run sans MAE est le seul à bouger légèrement :

- `test_accuracy ≈ 0.002575`
- `test_blank_ratio ≈ 0.9985`
- `test_empty_pred_ratio ≈ 0.8983`

Détail important :

- seed `42` reste en collapse presque total ;
- seed `123` génère quelques sorties non vides, mais celles-ci restent
  extrêmement pauvres, typiquement un seul caractère.

Exemple observé :

- référence : phrase complète
- prédiction : `s`

Interprétation :

- ce n'est pas une amélioration ASR utile ;
- c'est seulement un micro-signal montrant que la phase 4 ne produit plus
  exclusivement du vide sur toutes les seeds.

## Ce que la phase 4 valide

1. Le problème de longueurs identifié en phase 2 et corrigé en phase 3 est bien
   derrière nous.

2. Le `padding mask` était une hypothèse raisonnable, mais il ne constitue pas
   la clé principale du problème restant.

3. La normalisation d'utterance des log-mels, seule, ne suffit pas non plus.

4. L'`accuracy` ajoutée au benchmark est utile :
   - elle confirme l'absence d'apprentissage sur `P4D01` et `P4B01` ;
   - elle quantifie le micro-signal, encore insignifiant, de `P4B02`.

## Ce que la phase 4 n'a pas réussi à obtenir

La phase 4 n'a pas réussi à produire :

1. un `WER < 1.0`
2. une `accuracy` réellement significative
3. des sorties non vides stables et informatives
4. une validation positive de l'hypothèse `padding mask` comme solution
   principale

## Conclusion

La phase 4 est un échec de performance, mais un échec informatif.

Elle réduit encore l'espace des causes plausibles :

- le pipeline n'est plus structurellement invalide pour CTC ;
- le `padding mask` n'était pas la cause principale restante ;
- le problème est probablement désormais dans la recette d'apprentissage
  supervisé elle-même, ou dans la capacité du système à apprendre sur une tâche
  encore trop difficile.

La décision logique qui en découle est de passer à une **phase 5 de test
d'apprenabilité** :

1. augmenter franchement le fine-tuning supervisé ;
2. réduire la difficulté du problème ;
3. vérifier si la tête CTC sait apprendre sur un sous-cas simple.
