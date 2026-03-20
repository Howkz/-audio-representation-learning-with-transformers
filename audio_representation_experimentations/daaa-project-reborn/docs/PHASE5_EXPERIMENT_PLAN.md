# Plan d'expérimentation Phase 5

## Objet

La phase 5 n'est pas une nouvelle vague d'ablations.

C'est une campagne de **test d'apprenabilité** destinée à répondre à une seule
question :

> la tête CTC et l'encodeur savent-ils apprendre quelque chose de réel si on
> simplifie franchement la tâche et qu'on augmente nettement le fine-tuning
> supervisé ?

## Constat de départ

Après les phases 2, 3 et 4 :

1. le problème de longueurs CTC a été identifié puis corrigé ;
2. `padding mask` et `feature_norm` n'ont pas suffi à sauver la performance ;
3. le modèle reste globalement en collapse vide ou quasi vide ;
4. les benchmarks montrent toujours `WER = 1.0`.

La suite logique n'est donc plus de multiplier les ablations fines, mais de
réduire franchement la difficulté du problème pour savoir si le pipeline peut
apprendre dans un cas simple.

## Correctifs intégrés suite à audit

Un audit externe du code a mis en évidence deux points mal conçus qui ont été
corrigés avant la phase 5 :

1. **le MAE n'était pas implémenté dans l'esprit frugal attendu**
   - l'encodeur voyait encore les tokens masqués ;
   - cela se rapprochait d'un masquage de type BERT plutôt que d'un vrai MAE ;
   - le correctif appliqué force désormais l'encodeur à ne traiter que les
     tokens visibles, puis à réinjecter les `mask_tokens` uniquement au niveau
     du décodeur léger.

2. **le WER était entièrement recodé à la main**
   - ce n'était pas faux mathématiquement, mais inutilement artisanal ;
   - le calcul s'appuie maintenant sur `torchmetrics` quand la dépendance est
     disponible, avec fallback interne pour préserver l'exécution si besoin.

Ces corrections ne garantissent pas une amélioration des résultats, mais elles
retirent deux sources de dette technique et rendent la phase 5 plus défendable
méthodologiquement.

## Principe de la phase 5

La phase 5 applique trois décisions explicites :

1. **augmenter fortement le fine-tuning supervisé**
   - davantage de `max_steps`
   - ETA visible pendant l'entraînement

2. **réduire la difficulté du problème**
   - filtrer les phrases courtes uniquement
   - conserver un problème ASR réel, mais plus facile

3. **tester l'apprenabilité de la tête CTC**
   - d'abord sans MAE sur un sous-cas simple
   - puis comparer à une version avec MAE

## Simplification retenue

La réduction de difficulté se fait par filtrage des transcriptions :

- `max_transcript_words`
- `max_transcript_chars`

Cela évite :

- les cibles trop longues pour un premier test de capacité ;
- la variabilité extrême des phrases longues ;
- les conclusions ambiguës de type “le modèle n'apprend rien ou bien la tâche
  est juste trop difficile”.

## Expériences prévues

### `P5D01`

Diagnostic d'apprenabilité sans MAE sur sous-cas très simple.

But :

- vérifier si la tête CTC sait apprendre quand on lui donne des phrases très
  courtes ;
- augmenter fortement le fine-tuning pour tester un vrai scénario de
  mémorisation/apprentissage sur cas simple.

### `P5B01`

Baseline simple avec MAE.

But :

- comparer `MAE -> CTC` à la baseline simple une fois la difficulté réduite ;
- voir si le préentraînement aide enfin dans un régime plus apprenable.

### `P5B02`

Baseline simple sans MAE.

But :

- mesurer l'effet des correctifs de pipeline et du protocole simple sans
  préentraînement ;
- garder une comparaison directe avec `P5B01`.

## Critères de succès

La phase 5 sera considérée comme utile si au moins une expérience montre :

1. `WER < 1.0`
2. `accuracy` en hausse nette et cohérente
3. `empty_pred_ratio` clairement en baisse
4. des sorties qualitatives non vides et non réduites à un caractère unique

## Critère de décision

Deux cas sont possibles :

1. **Le modèle apprend sur le sous-cas simple**
   - alors le pipeline n'est pas fondamentalement mort ;
   - le problème venait surtout du niveau de difficulté et du budget
     supervisé.

2. **Le modèle n'apprend toujours pas**
   - alors il faudra assumer qu'il reste un problème plus profond dans la
     recette d'entraînement ou l'architecture, et non plus seulement dans le
     pipeline de données ou de masquage.
