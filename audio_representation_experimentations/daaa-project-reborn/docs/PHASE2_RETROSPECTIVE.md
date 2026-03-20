# Rétrospective Phase 2

## Objet

La phase 2 avait pour but de sortir du régime d'échec observé en phase 1
(`WER ~= 1.0` partout) et de distinguer deux explications possibles :

1. un problème principalement expérimental, par exemple un budget
   d'entraînement trop faible ;
2. un problème d'implémentation ou de conditionnement du pipeline ASR/CTC.

La phase 2 n'avait donc pas pour objectif principal de produire immédiatement
les meilleurs résultats finaux, mais d'identifier ce qui empêchait
l'apprentissage utile.

## Périmètre

Quatre expériences ont été exécutées :

- `P2D01` : baseline diagnostique proche de la phase 1 ;
- `P2B01` : récupération par budget avec MAE ;
- `P2B02` : récupération par budget sans MAE ;
- `P2A01` : ablation à compression temporelle plus faible.

Les artefacts de cette phase sont stockés sous :

- `results/phase2/experiments/P2D01/`
- `results/phase2/experiments/P2B01/`
- `results/phase2/experiments/P2B02/`
- `results/phase2/experiments/P2A01/`

## Correctif d'implémentation déjà intégré avant l'analyse

Avant l'interprétation de la phase 2, un correctif de décodage a été appliqué :

- le décodage greedy respecte désormais `out_lengths`.

Avant ce correctif, l'évaluation décodait toute la longueur temporelle des
logits, y compris les pas au-delà de la longueur utile estimée par le modèle.
Ce point pouvait biaiser les transcriptions décodées et donc le `WER`.

La phase 2 a montré que ce correctif était nécessaire, mais non suffisant.

## Résumé exécutif

La phase 2 n'a pas encore produit une baseline ASR exploitable :

- tous les runs restent à `WER = 1.0` en validation et en test ;
- augmenter le budget seul n'a pas permis de casser ce plafond ;
- l'ablation à compression plus faible a levé un problème structurel de
  longueurs CTC, mais n'a pas suffi à faire sortir le modèle du collapse vers
  `blank`.

En revanche, la phase 2 a rempli son objectif diagnostique :

1. elle a montré que le problème n'était pas seulement un manque de budget ;
2. elle a mis en évidence une incompatibilité CTC massive dans les variantes de
   base ;
3. elle a montré qu'une réduction de la compression temporelle corrige en
   grande partie ce problème de longueurs ;
4. elle a révélé qu'un second problème persiste ensuite : un collapse vers des
   sorties vides ou quasi vides ;
5. elle pointe fortement vers un problème de supervision ASR lié au
   prétraitement audio, en particulier au tronquage fixe des signaux.

## Résultats quantitatifs principaux

| Expérience | Adaptation | WER valid. | WER test | Blank ratio valid. | Empty pred. valid. | Invalid length train | Invalid length valid. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `P2D01` | MAE + CTC | 1.0000 | 1.0000 | 0.7077 | 0.9767 | 0.7773 | 0.1467 |
| `P2B01` | MAE + CTC | 1.0000 | 1.0000 | 0.4795 | 0.8517 | 0.8000 | 0.1900 |
| `P2B02` | CTC seul | 1.0000 | 1.0000 | 0.6560 | 0.6608 | 0.8000 | 0.1900 |
| `P2A01` | MAE + CTC, compression réduite | 1.0000 | 1.0000 | 0.9836 | 1.0000 | 0.0000 | 0.0150 |

Lecture immédiate :

- aucune expérience n'a réellement appris à transcrire ;
- `P2B01` et `P2B02` réduisent parfois la dominance de `blank`, mais sans gain
  sur le `WER` ;
- `P2A01` corrige presque totalement l'invalidité CTC liée aux longueurs, mais
  le modèle s'effondre alors presque entièrement vers `blank`.

## Lecture détaillée par expérience

### `P2D01`

Cette expérience confirme que le régime de phase 1 était structurellement
pathologique.

Observations :

- `WER = 1.0` partout ;
- `empty_pred_ratio` très proche de `1.0` ;
- `train_invalid_length_ratio` très élevé (`~ 0.78`).

Conclusion :

- l'échec n'est pas marginal ;
- une très grande partie des exemples de train ne respecte pas la contrainte
  CTC `out_lengths >= target_lengths`.

### `P2B01`

Cette expérience teste l'hypothèse d'un budget simplement trop faible.

Observations :

- plus de données et plus de steps ne changent pas le `WER` ;
- `blank_ratio` baisse par rapport à `P2D01`, mais les sorties restent
  majoritairement vides ;
- `train_invalid_length_ratio` reste massif (`~ 0.80`).

Conclusion :

- le manque de budget n'est pas l'explication principale ;
- augmenter le budget dans un pipeline mal conditionné ne suffit pas.

### `P2B02`

Cette expérience rejoue la tentative de récupération, mais sans MAE.

Observations :

- `WER = 1.0` également ;
- quelques sorties non vides apparaissent, mais elles sont dégénérées ;
- les rares prédictions non vides sont typiquement réduites à `"e"`.

Conclusion :

- dans l'état actuel du pipeline, `NoMAE` n'apporte pas de solution ;
- la comparaison `MAE vs NoMAE` reste non concluante scientifiquement, car les
  deux variantes échouent encore à la tâche.

### `P2A01`

Cette expérience est la plus instructive de la phase 2.

Observations :

- `train_invalid_length_ratio = 0.0` ;
- `valid_invalid_length_ratio ~= 0.015` ;
- `test_invalid_length_ratio ~= 0.02` ;
- malgré cela, `blank_ratio` devient extrêmement élevé et
  `empty_pred_ratio = 1.0`.

Conclusion :

- la compression temporelle baseline était bien trop agressive pour CTC ;
- une fois ce problème largement corrigé, un second problème apparaît plus
  clairement : collapse quasi total vers `blank`.

## Exemples qualitatifs

Des diagnostics sauvegardés montrent explicitement le comportement du modèle.

Exemple `P2D01`, validation :

```text
Référence :
he was in a fevered state of mind owing to the blight his wife's action threatened to cast upon his entire future

Prédiction :
""

Longueurs :
out_length = 151, target_length = 113
```

Exemple `P2B02`, test :

```text
Référence :
concord returned to its place amidst the tents

Prédiction :
"e"

Longueurs :
out_length = 151, target_length = 46
```

Exemple `P2A01`, validation :

```text
Référence :
he was in a fevered state of mind owing to the blight his wife's action threatened to cast upon his entire future

Prédiction :
""

Longueurs :
out_length = 301, target_length = 113
```

Ces exemples confirment que le problème n'est pas seulement métrique :

- soit les prédictions sont vides ;
- soit elles se réduisent à un caractère isolé ;
- elles ne constituent pas encore une transcription ASR utile.

## Ce que la phase 2 a infirmé

La phase 2 permet d'écarter plusieurs explications insuffisantes :

1. **"Il suffit d'augmenter les données et les steps."**
   Non. `P2B01` et `P2B02` restent à `WER = 1.0`.

2. **"Le problème venait seulement du décodage."**
   Non. Le correctif `out_lengths` était nécessaire, mais il ne résout pas à lui
   seul le collapse observé.

3. **"Le problème venait seulement de l'absence de MAE."**
   Non. `P2B01` et `P2B02` échouent tous les deux, malgré des profils
   diagnostiques un peu différents.

## Ce que la phase 2 a établi positivement

### 1. Le pipeline baseline viole massivement les contraintes CTC

Les variantes proches de la phase 1 montrent des ratios très élevés de
`out_lengths < target_lengths` en train. Cela rend une grande partie de
l'apprentissage supervisé incohérente ou inutile.

### 2. La compression temporelle est un facteur important

L'expérience `P2A01` montre qu'en réduisant la compression temporelle, on
ramène presque à zéro l'invalidité CTC de longueurs.

Cela signifie que la conception initiale n'était pas seulement sous-optimale ;
elle était structurellement défavorable à CTC.

### 3. Un second problème persiste après correction des longueurs

Une fois l'essentiel du problème de longueur corrigé, le modèle reste en
collapse vers `blank`.

La phase 2 montre donc que le problème global est au moins double :

- un problème de longueurs / compression ;
- un problème supplémentaire de supervision ou de conditionnement du pipeline.

## Hypothèse principale issue de la phase 2

L'hypothèse la plus crédible à ce stade est la suivante :

- le pipeline tronque l'audio à durée fixe ;
- mais conserve la transcription complète ;
- ce désalignement rend l'apprentissage CTC incohérent sur une partie
  importante des exemples ;
- même lorsqu'on corrige en partie la compression temporelle, le modèle reste
  piégé dans un régime de prédictions vides ou quasi vides.

Cette hypothèse est cohérente avec l'implémentation actuelle :

- l'audio est forcé à une longueur maximale fixe ;
- le cropping coupe explicitement les signaux trop longs ;
- la transcription de référence n'est pas raccourcie en conséquence.

## Conclusion

La phase 2 est un échec en performance, mais une réussite en diagnostic.

Elle n'a pas produit de système ASR viable pour le rendu final. En revanche,
elle a apporté des conclusions techniques solides :

1. le problème n'est pas réductible à un simple manque de budget ;
2. le pipeline baseline viole massivement les contraintes CTC de longueur ;
3. la réduction de compression temporelle est nécessaire ;
4. elle n'est pas suffisante, car un collapse vers `blank` persiste ;
5. le prétraitement audio et la cohérence supervision audio/texte doivent être
   reconsidérés avant toute nouvelle consolidation.

## Décision pour la phase 3

La phase 3 ne doit pas repartir de la baseline de phase 1.

Elle doit :

1. corriger le pipeline de supervision ASR, en particulier le traitement des
   signaux trop longs ;
2. repartir d'une base proche de `P2A01`, qui corrige déjà le problème majeur
   de longueurs CTC ;
3. ne reprendre `MAE vs NoMAE` et les comparaisons fines qu'après obtention
   d'une baseline réellement non dégénérée.

En pratique, la prochaine itération doit être pensée comme une **phase 3 de
correction pipeline**, pas comme une simple extension de budget.
