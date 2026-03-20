# Rétrospective Phase 6

## Objet

La phase 6 visait à corriger deux limites apparues en phase 5 :

1. le test ne regardait pas explicitement `ctc_final.pt`, alors que le
   comportement final du modèle semblait parfois plus intéressant que celui du
   checkpoint sélectionné comme `best` ;
2. la sélection du meilleur checkpoint reposait encore trop fortement sur le
   `WER` seul, ce qui favorisait des solutions dégénérées avec beaucoup de
   sorties vides ou quasi vides.

Elle introduisait également un nouveau protocole :

- un run de **généralisation simple** ;
- plus large que `P5D01` ;
- mais encore filtré pour conserver une difficulté raisonnable.

## Expérience exécutée

La phase 6 a été exécutée via :

- `P6G01` : généralisation simple sans MAE

Cette expérience évalue explicitement deux variantes de checkpoint :

- `best`
- `final`

## Résumé exécutif

La phase 6 est le meilleur résultat obtenu jusqu'ici, mais elle ne produit
toujours pas un ASR réellement satisfaisant.

Elle montre que :

1. le patch de sélection du `best checkpoint` corrige bien le problème des
   sorties vides ;
2. l'évaluation de `ctc_final.pt` était nécessaire ;
3. le modèle produit désormais des sorties non triviales en test ;
4. le `WER` reste néanmoins voisin de `1.0` ;
5. les prédictions restent trop courtes, très inexactes et qualitativement
   faibles.

## Tableau de synthèse

| Variante | WER moyen | Accuracy moyenne | Empty pred ratio | Nonempty pred ratio | Avg pred chars | Lecture |
|---|---:|---:|---:|---:|---:|---|
| `best` | `1.0047` | `0.1478` | `0.0128` | `0.9872` | `4.29` | meilleur `WER`, mais sorties encore très pauvres |
| `final` | `1.0059` | `0.1673` | `0.0` | `1.0` | `5.15` | légèrement meilleur signal caractère, toujours très insuffisant |

## Résultat d'entraînement

Le run de phase 6 n'est pas bloqué structurellement :

- `train_invalid_length_ratio = 0.0`
- `valid_invalid_length_ratio = 0.0`
- `train_loss` autour de `3.16` pour la seed `42` et `2.83` pour la seed `123`

Cela confirme que :

- le problème de longueurs CTC est bien derrière nous ;
- le pipeline peut exécuter un fine-tuning stable sur ce protocole ;
- l'échec restant n'est plus un échec trivial de type “sorties impossibles”.

## Analyse `best` vs `final`

### Variante `best`

Le checkpoint `best` produit :

- `WER mean = 1.0047`
- `accuracy mean = 0.1478`
- `empty_pred_ratio = 0.0128`
- `nonempty_pred_ratio = 0.9872`

Ce résultat est important :

- le patch de sélection a bien réduit le risque de choisir un checkpoint presque
  vide ;
- on n'est plus dans la situation de phase 5 où le `best` pouvait encore être
  dominé par des sorties équivalentes à du vide.

En revanche, qualitativement, les sorties restent faibles.

Exemples observés :

- référence : `i can perceive love clearly enough`
- prédiction : `e ah ee`

- référence : `what was that`
- prédiction : `e oa`

- référence : `a sound of voices a flash of light`
- prédiction : `a oeso e`

Lecture :

- le modèle émet des caractères et des segments non vides ;
- mais la transcription reste très éloignée de la cible ;
- on observe surtout des fragments courts, récurrents, et très peu informatifs.

### Variante `final`

Le checkpoint `final` produit :

- `WER mean = 1.0059`
- `accuracy mean = 0.1673`
- `empty_pred_ratio = 0.0`
- `nonempty_pred_ratio = 1.0`

Le point essentiel est ici :

- `final` est légèrement moins bon que `best` sur le `WER` moyen ;
- mais il est meilleur sur l'`accuracy` ;
- et il supprime complètement les sorties vides.

Exemples observés :

- référence : `i can perceive love clearly enough`
- seed `42` : `e o ih e`
- seed `123` : `n rse`

- référence : `what was that`
- seed `42` : `e o o`
- seed `123` : `es`

Lecture :

- `ctc_final.pt` est plus informatif que le simple `WER` ne le laisse penser ;
- la fin d'entraînement continue parfois à faire émerger un signal caractère ;
- ce signal reste cependant trop faible pour parler de transcription utile.

## Ce que la phase 6 valide

1. **Le changement de sélection du `best checkpoint` était justifié**
   - les checkpoints retenus ne retombent plus dans un régime quasi vide.

2. **L'évaluation explicite de `ctc_final.pt` était nécessaire**
   - `final` apporte une lecture complémentaire réelle ;
   - le `WER` seul ne suffisait pas à décrire correctement la dynamique.

3. **Le pipeline atteint désormais une généralisation simple non triviale**
   - on dépasse le pur test d'overfit de phase 5 ;
   - les sorties test sont presque toujours non vides ;
   - le système produit un signal caractère stable sur les deux seeds.

## Ce que la phase 6 ne valide pas

1. **Le système ne transcrit toujours pas correctement**
   - `WER` reste supérieur à `1.0` ;
   - `exact_match_ratio = 0.0` ;
   - aucune phrase n'est correctement reconstruite.

2. **La qualité des sorties reste trop faible**
   - `avg_pred_chars` reste très inférieur à `avg_ref_chars` ;
   - les prédictions sont trop courtes et trop bruitées.

3. **Le problème n'est pas résolu**
   - la phase 6 améliore la lecture du modèle ;
   - elle ne transforme pas encore le pipeline en solution ASR viable.

## Conclusion

La phase 6 est une amélioration méthodologique et diagnostique nette.

Elle permet d'affirmer plus solidement que :

- le système n'est plus enfermé dans le collapse vide ;
- la tête CTC apprend désormais un signal caractère non trivial ;
- la comparaison `best` vs `final` est indispensable pour interpréter
  correctement les progrès ;
- mais la qualité de transcription reste encore trop faible pour conclure à une
  implémentation satisfaisante.

La formulation la plus honnête est donc :

- **oui, la phase 6 montre un progrès réel par rapport aux phases précédentes** ;
- **non, elle ne suffit pas encore à produire un ASR défendable en performance**.

## Décision logique pour la suite

La phase 6 réduit encore l'espace des causes plausibles, mais elle ne justifie
pas une nouvelle grande campagne d'ablations.

Si une suite devait être lancée, elle devrait être très ciblée :

1. améliorer la longueur et la richesse des sorties décodées ;
2. confirmer si `accuracy` et longueur prédite continuent à progresser avec un
   budget un peu plus grand ;
3. éviter de repartir dans des variantes architecturales tant que le niveau
   transcription reste aussi faible.
