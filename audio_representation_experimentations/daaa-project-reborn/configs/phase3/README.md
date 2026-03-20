# Espace de travail des configs Phase 3

Ce dossier contient les configs de la campagne corrective phase 3.

Principes :

- garder les sorties séparées de la phase 2
- valider d'abord les corrections de pipeline
- ne comparer `MAE` vs `NoMAE` qu'après correction des problèmes de supervision

Configs présentes :

- `P3D01.yaml` : validation du diagnostic
- `P3B01.yaml` : baseline corrective avec MAE
- `P3B02.yaml` : baseline corrective sans MAE

Corrections intégrées dans ces configs :

- `patch_time: 2`
- `max_len` élargi
- pas de tronquage fixe pour les datasets ASR
- diagnostics CTC activés
