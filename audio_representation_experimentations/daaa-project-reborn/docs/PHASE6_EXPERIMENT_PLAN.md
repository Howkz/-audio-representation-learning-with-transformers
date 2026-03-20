# Plan d'expérimentation Phase 6

## Objet

La phase 6 part d'un constat simple issu de la phase 5 :

- `P5D01` a enfin montré un signal d'apprentissage non trivial en validation ;
- mais le checkpoint sélectionné comme `best` restait dégénéré en test ;
- le protocole précédent testait surtout l'apprenabilité, pas une vraie généralisation simple.

La phase 6 vise donc trois corrections ciblées :

1. évaluer explicitement `ctc_final.pt` en plus de `ctc_best.pt` ;
2. changer la sélection du meilleur checkpoint pour pénaliser les `empty_pred_ratio` élevés ;
3. lancer un run de généralisation simple sur un dataset plus large, avec peu d'époques.

## Changement méthodologique principal

Le meilleur checkpoint n'est plus choisi sur le `WER` seul.

La sélection utilise désormais un score pénalisé :

- `selection_score = wer + empty_pred_penalty * empty_pred_ratio`

Conséquence attendue :

- un modèle qui obtient un `WER` artificiellement bas parce qu'il prédit surtout du vide ou des espaces ne doit plus gagner automatiquement ;
- un checkpoint qui produit un signal non vide mais encore imparfait peut enfin être conservé comme `best`.

## Vérification désormais exigée

Les benchmarks phase 6 évaluent explicitement plusieurs variantes de checkpoint :

- `best`
- `final`

L'objectif n'est plus de supposer que `best` suffit, mais de vérifier si la dynamique d'entraînement continue d'apporter un signal utile jusqu'au dernier checkpoint.

## Expérience prévue

### `P6G01`

Run de généralisation simple sans MAE.

Choix :

- dataset ASR plus large que `P5D01`
- phrases encore filtrées pour rester raisonnablement simples
- peu d'époques
- mêmes correctifs de pipeline que les phases 3, 4 et 5
- comparaison `best` vs `final` au test

## Différences avec `P5D01`

`P6G01` n'est pas un test d'overfit volontaire.

Il cherche un compromis :

- tâche encore simplifiée
- mais dataset suffisamment moins minuscule pour que la généralisation commence à avoir un sens
- budget d'entraînement encore contrôlé pour rester exécutable

## Critères de succès

La phase 6 sera considérée utile si `P6G01` montre au moins un des signaux suivants :

1. `ctc_final.pt` meilleur que `ctc_best.pt` ou inversement, avec preuve explicite ;
2. baisse claire de `empty_pred_ratio` sur la variante retenue ;
3. `accuracy` non triviale sur validation et test ;
4. `WER < 1.0` sur au moins une variante de checkpoint ;
5. sorties qualitatives non vides et moins dégénérées.

## Décision attendue

Deux cas sont possibles :

1. **La généralisation simple commence à apparaître**
   - alors la phase 6 confirme que le pipeline est apprenable au-delà du pur overfit ;
   - on pourra ensuite réintroduire progressivement des comparaisons plus riches.

2. **Le signal reste uniquement local ou dégénéré**
   - alors il faudra assumer que le pipeline reste trop faible même après toutes les corrections structurelles ;
   - la phase 6 servira alors de borne haute honnête sur ce qui a été tenté.
