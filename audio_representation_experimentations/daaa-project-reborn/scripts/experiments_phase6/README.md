# Lanceurs Phase 6

Cette campagne exécute une généralisation simple après le test d'apprenabilité phase 5.

Script principal :

- `run_single_phase6_experiment.sh`

Launcher présent :

- `run_P6G01.sh`

Particularité :

- le test est lancé avec `--checkpoint-variant all` pour comparer `ctc_best.pt` et `ctc_final.pt`.
