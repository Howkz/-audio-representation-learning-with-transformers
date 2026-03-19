# Rétrospective Phase 1

## Périmètre

La phase 1 correspond à la campagne actuelle `E00 -> E11` ainsi qu'aux runs de
sélection dérivés `SEL01 -> SEL05`.

Cette phase doit désormais être traitée comme une campagne exploratoire :

- validation du pipeline
- stabilisation du mode low-storage / streaming
- stabilisation du checkpointing et de la reprise
- premier screening de `MAE / NoMAE / capacité / positionnel / patching / mask ratio`

## Ce que la phase 1 a effectivement établi

- Le pipeline complet `data -> train -> test` peut tourner sur l'environnement cible.
- La stratégie low-storage est opérationnelle :
  - datasets en streaming
  - étape data lazy
  - fallback `soundfile` / `ffmpeg` en l'absence de `torchaudio`
- Le runner de suite sait générer les configs runtime pour le screening, la
  sélection et les expériences finales.
- Les différences de coût entre variantes sont mesurables.

## Observation empirique principale

Le résultat central de la phase 1 est négatif :

- les runs de screening `E01 -> E08` sont tous à `WER ~= 1.0`
- les runs de sélection restent eux aussi à `WER ~= 1.0`
- les classements actuels reflètent surtout des différences de runtime et de
  mémoire, pas une différence utile de qualité ASR

Cela signifie que la phase 1 ne permet pas encore de tirer des conclusions
scientifiques solides sur :

- l'apport réel du pré-entraînement MAE
- le meilleur choix d'architecture du point de vue qualité ASR
- la robustesse des variantes sous différentes seeds

## Pourquoi la phase 1 ne suffit pas

À ce stade, la campagne compare surtout des modèles ASR en échec ou quasi en
échec. Le contraste obtenu est donc faible :

- `MAE` vs `NoMAE` ne se sépare pas clairement sur la qualité
- les changements de capacité se séparent surtout sur le coût
- la sélection et la finale héritent déjà d'un régime de qualité dégénéré

Continuer à patcher et prolonger la phase 1 mélangerait :

- résultats exploratoires
- correctifs de pipeline
- hypothèses correctives
- nouveaux objectifs expérimentaux

Cela dégraderait la traçabilité.

## Hypothèses de défaillance à investiguer

Les principales hypothèses à investiguer en phase 2 sont :

1. collapse CTC vers le `blank` ou vers des prédictions quasi vides
2. budget de fine-tuning trop faible pour que la tête supervisée récupère des
   transcriptions utiles
3. compression temporelle trop agressive pour l'alignement CTC
4. décalage entre tokenisation, normalisation de transcript et décodage
5. diagnostics de qualité trop faibles parce que les prédictions ne sont pas
   inspectées qualitativement pendant l'entraînement

## Décision

La phase 1 est figée comme campagne exploratoire.

Elle reste utile pour :

- documenter les contraintes d'environnement
- documenter le travail de stabilisation déjà réalisé
- rapporter les tendances de coût
- motiver la conception d'une phase 2 corrective

Elle ne doit plus être prolongée par de nouveaux runs correctifs sous le même
schéma de nommage et d'interprétation.

## Actifs de la phase 1

- snapshot des launchers : `scripts/experiments_phase1/`
- snapshot des configs : `configs/phase1/`
- snapshot de suite : `results/phase1/suite_snapshot/`

Les artefacts détaillés des anciens runs restent présents dans l'arborescence
active sous :

- `results/experiments/`
- `results/suite/`
- `outputs/checkpoints/`

Ces dossiers doivent être considérés comme des artefacts hérités de phase 1,
sauf ré-archivage explicite ultérieur.
