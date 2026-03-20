# Résultats Phase 4

Ce dossier est réservé aux sorties de la campagne corrective phase 4.

Organisation attendue :

- `results/phase4/experiments/P4D01/`
- `results/phase4/experiments/P4B01/`
- `results/phase4/experiments/P4B02/`

Chaque expérience doit produire :

- `benchmark_results/`
- `diagnostics/`
- `tables/`

Les benchmarks phase 4 doivent désormais inclure :

- `WER`
- `accuracy`
- `blank_ratio`
- `empty_pred_ratio`
- `invalid_length_ratio`

Les exemples `référence -> prédiction` restent indispensables pour détecter un
collapse vers un token unique.
