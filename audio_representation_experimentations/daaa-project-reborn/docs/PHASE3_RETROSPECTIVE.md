# Rétrospective Phase 3

## Objet

La phase 3 avait un objectif plus ciblé que la phase 2 :

- appliquer les corrections déduites du diagnostic phase 2 ;
- vérifier si ces corrections suffisaient à lever le principal blocage du
  pipeline ;
- observer si un apprentissage ASR utile réapparaissait ensuite.

La phase 3 ne devait donc pas seulement produire de nouveaux runs. Elle devait
servir de validation expérimentale du diagnostic précédent.

## Rappel du diagnostic phase 2

La phase 2 avait mis en évidence deux problèmes majeurs :

1. une incompatibilité CTC massive due à la compression temporelle et au
   traitement des longueurs ;
2. un échec persistant de la transcription utile, même après augmentation du
   budget d'entraînement.

L'hypothèse de départ phase 3 était donc :

- corriger le problème de longueurs CTC ;
- vérifier si cette correction suffit à faire baisser le `WER`.

## Corrections appliquées en phase 3

La phase 3 a introduit les corrections suivantes :

1. `patch_time = 2` pour réduire la compression temporelle ;
2. `max_len` élargi pour accepter des séquences plus longues ;
3. suppression du tronquage fixe sur les datasets ASR :
   - `max_duration_sec: null`
   - `length_policy: none`
4. conservation d'un prétrain borné si nécessaire, mais sans imposer ce
   tronquage aux jeux ASR supervisés.

## Périmètre des expériences

Trois expériences ont été exécutées :

- `P3D01` : validation directe du diagnostic ;
- `P3B01` : baseline corrective avec MAE ;
- `P3B02` : baseline corrective sans MAE.

Les artefacts sont stockés sous :

- `results/phase3/experiments/P3D01/`
- `results/phase3/experiments/P3B01/`
- `results/phase3/experiments/P3B02/`

## Résumé exécutif

La phase 3 valide clairement le diagnostic principal de la phase 2 :

- les problèmes de longueurs CTC ont bien disparu ;
- les ratios `invalid_length_ratio` tombent à `0.0` sur train, validation et
  test dans les expériences phase 3.

En revanche, la phase 3 ne produit toujours pas une implémentation ASR viable :

- `WER = 1.0` pour `P3D01`, `P3B01` et `P3B02` ;
- le modèle ne reste plus seulement bloqué dans le collapse `blank` pur ;
- un second mode d'échec apparaît : collapse vers un token non vide constant.

Conclusion synthétique :

- le diagnostic phase 2 était correct ;
- la correction phase 3 supprime bien un blocage réel ;
- ce blocage n'était pas le seul.

## Résultats quantitatifs principaux

| Expérience | Adaptation | WER valid. | WER test | Invalid length train | Invalid length valid. | Blank ratio test | Empty pred. test |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `P3D01` | MAE + CTC | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| `P3B01` | MAE + CTC | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5978 | 0.9967 |
| `P3B02` | CTC seul | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |

Le point important n'est pas seulement que le `WER` reste à `1.0`. Le point
important est que les métriques de longueur sont désormais saines, alors que le
comportement du modèle reste dégénéré.

## Lecture par expérience

### `P3D01`

`P3D01` est l'expérience de validation la plus importante de la phase 3.

Observations :

- `train_invalid_length_ratio = 0.0`
- `valid_invalid_length_ratio = 0.0`
- `test_invalid_length_ratio = 0.0`
- `valid_blank_ratio = 1.0`
- `valid_empty_pred_ratio = 1.0`
- `test_blank_ratio = 1.0`
- `test_empty_pred_ratio = 1.0`

Interprétation :

- la correction pipeline supprime bien le problème structurel de longueurs ;
- malgré cela, le modèle reste en collapse complet vers `blank`.

Conclusion :

- `P3D01` valide le diagnostic de phase 2 ;
- mais il invalide l'idée que cette seule correction suffirait à récupérer
  l'ASR.

### `P3B01`

`P3B01` réévalue la chaîne `MAE -> CTC` après correction du pipeline.

Observations :

- longueurs CTC saines (`invalid_length_ratio = 0.0`) ;
- `WER = 1.0` ;
- la plupart des prédictions restent vides ;
- une seed produit très légèrement moins de `blank`, mais sans effet utile sur
  la transcription.

Interprétation :

- MAE ne suffit pas à compenser le problème résiduel ;
- le pipeline corrigé ne débouche pas encore sur une vraie récupération du
  signal ASR.

Conclusion :

- la correction phase 3 améliore les conditions d'apprentissage ;
- elle ne rend pas pour autant la variante `MAE + CTC` performante.

### `P3B02`

`P3B02` est la plus informative sur le nouveau mode d'échec.

Observations :

- longueurs CTC saines (`invalid_length_ratio = 0.0`) ;
- `WER = 1.0` ;
- forte bifurcation selon la seed :
  - seed `123` : collapse `blank` / vide complet ;
  - seed `42` : plus de `blank`, plus de sorties vides, mais prédiction quasi
    constante `"g"`.

Interprétation :

- la phase 3 a bien changé le mode d'échec ;
- le modèle n'est plus seulement incapable de produire une sortie ;
- il peut produire une sortie non vide, mais totalement dégénérée.

Conclusion :

- `P3B02` montre que la correction pipeline a eu un effet réel ;
- mais cet effet n'est pas encore une récupération de la transcription utile.

## Exemples qualitatifs

### `P3D01`

Exemple validation :

```text
Référence :
he was in a fevered state of mind owing to the blight his wife's action threatened to cast upon his entire future

Prédiction :
""

Longueurs :
out_length = 330, target_length = 113
```

Ce cas est important :

- les longueurs sont largement suffisantes ;
- pourtant la sortie reste vide.

### `P3B01`

Exemple test :

```text
Référence :
from the respect paid her on all sides she seemed like a queen ...

Prédiction :
"e"
```

Le modèle ne reste pas totalement muet dans tous les cas, mais il ne produit
pas non plus une transcription utile.

### `P3B02`

Exemple test, seed `42` :

```text
Référence :
concord returned to its place amidst the tents

Prédiction :
"g"
```

Ce motif se répète sur de nombreux exemples.

Ce comportement indique un **token collapse** :

- la sortie n'est plus vide ;
- elle n'est pas davantage correcte.

## Ce que la phase 3 valide

La phase 3 valide de façon forte les points suivants :

1. le problème de longueurs CTC identifié en phase 2 était réel ;
2. les corrections appliquées ont bien supprimé ce problème ;
3. le pipeline initial avait donc effectivement un défaut structurel de
   conditionnement pour CTC.

En ce sens, la phase 3 est un succès de diagnostic expérimental.

## Ce que la phase 3 ne valide pas

La phase 3 ne valide pas l'hypothèse plus forte selon laquelle :

- corriger les longueurs suffirait à restaurer un apprentissage ASR utile.

Les résultats montrent au contraire que :

- le `WER` ne baisse pas ;
- le modèle oscille encore entre deux régimes dégénérés :
  - collapse vide / `blank`
  - collapse vers un token constant non vide.

## Comparaison avec la phase 2

La différence avec la phase 2 est importante.

### En phase 2

Le système échouait dans un cadre où :

- les longueurs CTC étaient souvent invalides ;
- les sorties étaient majoritairement vides ;
- l'interprétation restait partiellement ambiguë.

### En phase 3

Le système échoue dans un cadre où :

- les longueurs CTC sont valides ;
- le problème de longueurs ne peut plus servir d'explication principale ;
- le mode d'échec résiduel est plus proprement observable.

La phase 3 réduit donc l'espace des causes plausibles.

## Lecture méthodologique

La phase 3 n'est pas une réussite de performance, mais elle améliore
significativement la valeur scientifique du protocole :

- elle transforme une intuition diagnostique en constat validé ;
- elle sépare un problème de pipeline d'un problème d'apprentissage résiduel ;
- elle justifie que la suite ne doit plus porter d'abord sur les longueurs, mais
  sur le comportement du modèle une fois ces longueurs corrigées.

## Conclusion

La phase 3 permet d'affirmer :

1. la phase 2 avait correctement identifié un problème majeur du pipeline ;
2. ce problème a été effectivement corrigé ;
3. malgré cette correction, l'ASR reste totalement non exploitable ;
4. il existe donc au moins un second blocage, distinct du problème de longueurs.

La phase 3 constitue ainsi une **validation partielle du diagnostic global** :

- validation complète sur le sous-problème des longueurs ;
- échec persistant sur la récupération finale de la transcription.

## Décision pour la suite

La suite ne doit plus se concentrer prioritairement sur :

- la compression temporelle seule ;
- le tronquage fixe seul ;
- l'augmentation brute du budget seule.

La prochaine étape doit cibler le nouveau mode d'échec observé :

1. quantifier la distribution des tokens prédits ;
2. comprendre pourquoi une seed reste en `blank` et une autre en token collapse
   (`"g"`) ;
3. corriger le comportement du modèle une fois les longueurs devenues saines.

En pratique, la phase 3 clôt proprement une étape :

- **le diagnostic de longueurs est validé** ;
- **la performance ASR reste à reconstruire**.
