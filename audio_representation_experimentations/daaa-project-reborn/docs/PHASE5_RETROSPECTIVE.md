# Rétrospective Phase 5

## Objet

La phase 5 visait à répondre à une question plus primitive que les phases
précédentes :

> le pipeline est-il encore capable d'apprendre quelque chose si l'on réduit
> fortement la difficulté de la tâche et que l'on augmente franchement le
> fine-tuning supervisé ?

Cette phase n'avait donc pas pour objectif principal de mesurer une bonne
généralisation ASR sur LibriSpeech complet, mais de tester l'apprenabilité
minimale de la tête CTC sur un sous-cas simple.

## Périmètre réellement exécuté

Le plan de phase 5 prévoyait trois expériences :

- `P5D01` : diagnostic d'apprenabilité sans MAE
- `P5B01` : baseline simple avec MAE
- `P5B02` : baseline simple sans MAE

Dans les artefacts actuellement disponibles, seule `P5D01` a été exécutée
complètement.

La présente rétrospective porte donc sur :

- l'expérience `P5D01`
- le correctif de planification d'entraînement appliqué pendant la phase

## Changements effectivement testés

La phase 5 a combiné plusieurs décisions :

1. augmentation forte du fine-tuning supervisé ;
2. filtrage des phrases courtes uniquement ;
3. conservation des correctifs des phases 3 et 4 ;
4. ajout des corrections issues de l'audit de code :
   - MAE visible-only côté encodeur ;
   - WER standardisé via `torchmetrics` quand disponible.

Un point important est apparu pendant l'exécution :

- le premier `P5D01` s'arrêtait après seulement `2` steps réels ;
- la boucle d'entraînement a donc été corrigée pour forcer l'atteinte du
  budget `max_steps` même sur un dataset filtré minuscule.

Le rerun final de `P5D01` est bien celui pris en compte ici.

## Résumé exécutif

La phase 5 apporte le signal le plus intéressant depuis le début de la
campagne, mais elle ne produit toujours pas de résultat ASR satisfaisant.

Elle montre que :

1. le correctif sur la planification des steps fonctionne ;
2. le modèle peut sortir du collapse `blank` complet sur un sous-cas simple ;
3. la tête CTC produit enfin des sorties non triviales en validation ;
4. mais la qualité reste très faible ;
5. le critère de sélection du meilleur checkpoint favorise encore un modèle
   dégénéré en test.

Autrement dit :

- la phase 5 ne valide pas encore une solution ;
- elle valide en revanche que le pipeline n'est plus totalement incapable de
  produire un signal d'apprentissage.

## Tableau de synthèse

| Run | Steps réels | Valid WER | Valid accuracy | Test WER | Test accuracy | Lecture |
|---|---:|---:|---:|---:|---:|---|
| `P5D01` | `3000` | `1.5` | `0.1633` | `1.0` | `0.0` | Signal d'apprentissage faible en validation, checkpoint best encore dégénéré en test |

## Résultat principal : `P5D01`

### Budget effectivement consommé

Le rerun final a bien exécuté tout le budget prévu :

- checkpoint final : `checkpoint_epoch_e1500_s0003000.pt`
- runtime train : `1323.58 s`
- `train_loss = 0.2239`

Cela valide le correctif de planification appliqué en phase 5 :

- le modèle ne s'est plus arrêté artificiellement à la fin d'une seule époque ;
- `max_steps` est désormais réellement respecté même quand le dataset filtré
  est très petit.

### Ce qui a changé par rapport aux phases précédentes

En validation finale, `P5D01` ne présente plus les symptômes les plus triviaux
observés auparavant :

- `valid_blank_ratio = 0.0`
- `valid_empty_pred_ratio = 0.0`
- `valid_nonempty_pred_ratio = 1.0`
- `valid_accuracy = 0.1633`

Ce point est important :

- la tête CTC ne produit plus seulement du vide ;
- elle émet des séquences de caractères non vides ;
- cela constitue un signal d'apprenabilité réel, même si encore insuffisant.

### Limites observées

Malgré ce progrès, la qualité reste faible :

- `valid_wer = 1.5`
- `valid_exact_match_ratio = 0.0`

Exemples observés en validation :

- référence : `otto winked at me`
- prédiction : `fe i`

- référence : `yes how many`
- prédiction : `ic s n e en n`

- référence : `twenty thirty enough`
- prédiction : `a u s a en`

Ces sorties montrent que :

- le système ne reste plus bloqué au niveau vide/blank ;
- mais il n'apprend toujours pas une transcription correcte ;
- il produit plutôt un charabia partiellement structuré.

## Incohérence apparente entre validation et test

Le résultat test final peut sembler contradictoire :

- `test_wer = 1.0`
- `test_accuracy = 0.0`
- `blank_ratio = 0.0`
- `empty_pred_ratio = 1.0`

Cette combinaison n'indique pas un retour au `blank collapse`.

L'interprétation la plus probable est la suivante :

1. le checkpoint sélectionné pour le test est `ctc_best.pt` ;
2. ce checkpoint a été choisi uniquement sur le `valid_wer` ;
3. il semble prédire majoritairement le token correspondant à l'espace ;
4. après décodage puis normalisation, ces sorties deviennent vides.

Cette lecture est cohérente avec :

- les `decoded_token_ids` observés dans les diagnostics test ;
- la définition du tokenizer caractère CTC ;
- le fait que `blank_ratio = 0.0` alors que `empty_pred_ratio = 1.0`.

Donc :

- le modèle testé n'est pas dominé par `blank` ;
- il est dominé par une sortie dégénérée en espaces ;
- le critère de sélection du meilleur checkpoint reste méthodologiquement
  insuffisant sur ce protocole simplifié.

## Ce que la phase 5 valide

1. **Le correctif d'entraînement est valide**
   - le budget `max_steps` est désormais réellement consommé.

2. **Le pipeline peut produire un signal d'apprentissage non trivial**
   - sur un sous-cas simple, la validation n'est plus entièrement vide ;
   - `accuracy` devient non nulle ;
   - les prédictions ne sont plus systématiquement réduites à `blank`.

3. **Le test d'apprenabilité était justifié**
   - le régime simplifié a révélé un comportement que les phases précédentes
     masquaient.

## Ce que la phase 5 ne valide pas

1. **Le modèle ne transcrit toujours pas correctement**
   - `WER` reste très mauvais ;
   - aucune correspondance utile n'apparaît encore.

2. **La généralisation n'est pas démontrée**
   - `P5D01` est un test d'apprenabilité, pas un benchmark final de
     généralisation.

3. **Le critère de meilleur checkpoint n'est pas satisfaisant**
   - un modèle dégénéré peut être retenu comme `best` ;
   - cela brouille la lecture du progrès réel.

4. **La campagne phase 5 n'est pas complète**
   - `P5B01` et `P5B02` n'ont pas encore fourni d'artefacts finaux.

## Conclusion

La phase 5 n'est pas un succès de performance, mais elle constitue un tournant
diagnostique utile.

Elle montre que :

- le pipeline corrigé peut enfin sortir du collapse le plus trivial ;
- la tête CTC sait produire un début de signal sur un sous-cas simple ;
- mais ce signal reste trop faible et trop instable pour conclure à une
  solution viable.

La conclusion rigoureuse est donc :

- **oui, le système semble apprenable dans une certaine mesure** ;
- **non, il n'est pas encore exploitable comme système ASR correct**.

## Décision logique pour la suite

La prochaine étape rationnelle n'est pas de multiplier de nouvelles grandes
campagnes.

Les deux priorités sont :

1. corriger la sélection du meilleur checkpoint pour éviter de favoriser les
   sorties dégénérées en espaces/vide ;
2. lancer un test plus proche d'une généralisation simple :
   - dataset simplifié mais moins minuscule ;
   - peu d'époques ;
   - mêmes correctifs de pipeline ;
   - comparaison éventuelle `avec / sans MAE` seulement si le signal persiste.
