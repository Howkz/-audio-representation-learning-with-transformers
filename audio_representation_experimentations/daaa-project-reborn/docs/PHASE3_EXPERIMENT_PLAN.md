# Plan d'expérimentation Phase 3

## Objet

La phase 3 est une campagne de validation corrective.

Son but n'est pas d'ajouter de nouvelles ablations fines, mais d'appliquer les
corrections déduites du diagnostic de phase 2 afin de vérifier si ce diagnostic
était juste.

## Hypothèse issue de la phase 2

La phase 2 a mis en évidence deux causes majeures d'échec :

1. la compression temporelle baseline rendait une grande partie des exemples
   incompatibles avec CTC ;
2. le pipeline ASR tronquait l'audio à durée fixe tout en conservant la
   transcription complète, ce qui introduisait une supervision incohérente.

## Corrections phase 3

La phase 3 applique explicitement les corrections suivantes :

1. **plus de tronquage dur pour les datasets ASR**
   - `length_policy: none`
   - `max_duration_sec: null`

2. **compression temporelle réduite**
   - `patch_time: 2`

3. **max_len élargi**
   - pour accepter les séquences ASR plus longues

4. **batches ASR plus petits**
   - pour absorber les longueurs variables sans retomber dans les OOM

## Objectifs de validation

La phase 3 sera considérée comme validante si elle montre au moins :

1. `train_invalid_length_ratio` proche de `0`
2. `valid_invalid_length_ratio` proche de `0`
3. `empty_pred_ratio` en baisse nette
4. des prédictions qualitatives non vides et non réduites à un seul caractère
5. un `WER < 1.0`

## Expériences prévues

### `P3D01`

Run de validation du diagnostic.

But :

- vérifier que les corrections pipeline suffisent à faire disparaître le gros
  problème de longueurs CTC ;
- observer si des prédictions non vides apparaissent enfin.

### `P3B01`

Baseline corrective avec MAE.

But :

- réévaluer la chaîne `MAE -> CTC` après correction du pipeline supervisé.

### `P3B02`

Baseline corrective sans MAE.

But :

- comparer `MAE` vs `NoMAE` seulement après correction des problèmes
  identifiés.

## Critère de lecture

Si les métriques de longueur deviennent saines mais que le modèle continue à
sortir majoritairement `blank`, alors le diagnostic de phase 2 n'était que
partiel.

Si au contraire les sorties deviennent non vides et que le `WER` baisse, alors
la phase 2 aura été validée et la phase 3 servira de nouvelle base crédible
pour la consolidation finale.
