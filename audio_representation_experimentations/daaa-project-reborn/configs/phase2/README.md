# Espace de travail des configs Phase 2

Ce dossier contient les configs dédiées à la campagne corrective de phase 2.

Configs présentes :

- `P2D01.yaml` : baseline diagnostique au plus proche de la phase 1
- `P2B01.yaml` : tentative de récupération par budget avec MAE
- `P2B02.yaml` : tentative de récupération par budget sans MAE
- `P2A01.yaml` : ablation à compression temporelle plus faible

Principes de conception :

- identifiants séparés de `E00 -> E11`
- sorties séparées sous `results/phase2/experiments/<ID>`
- checkpoints séparés sous `outputs/phase2/checkpoints/<ID>`
- diagnostics CTC activés dans toutes les configs

Diagnostics activés :

- exemples de prédictions de validation et de test
- `blank_ratio`
- `empty_pred_ratio`
- `invalid_length_ratio`
- statistiques de longueurs `out_lengths` / `target_lengths`
