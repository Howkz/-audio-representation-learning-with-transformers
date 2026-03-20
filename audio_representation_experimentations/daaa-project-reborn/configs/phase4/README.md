# Espace de travail des configs Phase 4

Ce dossier contient les configs de la campagne corrective phase 4.

Principes :

- garder la séparation avec les phases 1, 2 et 3 ;
- tester d'abord le correctif technique le plus plausible restant ;
- enrichir les benchmarks avec une `accuracy` caractère en plus du `WER`.

Configs présentes :

- `P4D01.yaml` : run diagnostic sans MAE
- `P4B01.yaml` : baseline corrective avec MAE
- `P4B02.yaml` : baseline corrective sans MAE

Corrections et hypothèses intégrées :

- `padding mask` dans le Transformer
- `feature_norm: utterance`
- `patch_time: 2`
- `max_len` plus large
- pas de tronquage fixe pour les datasets ASR
