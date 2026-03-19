# Launchers Phase 2

Ce dossier contient les launchers dédiés à la campagne corrective de phase 2.

Règles :

- ne pas réutiliser `E00 -> E11`
- garder les sorties phase 2 sous `results/phase2/`
- lancer explicitement `data -> train -> test`
- activer les diagnostics CTC dans les configs phase 2

Scripts présents :

- `run_single_phase2_experiment.sh` : exécuteur générique
- `run_P2D01.sh` : baseline diagnostique
- `run_P2B01.sh` : baseline de récupération avec MAE
- `run_P2B02.sh` : baseline de récupération sans MAE
- `run_P2A01.sh` : ablation à compression temporelle plus faible

Exemples :

```bash
bash scripts/experiments_phase2/run_P2D01.sh
bash scripts/experiments_phase2/run_P2B01.sh --resume
bash scripts/experiments_phase2/run_single_phase2_experiment.sh P2A01 --materialize-data
```

Options principales :

- `--resume` : reprend l'entraînement via `run_train.py --continue-completed`
- `--dry-run` : valide les trois étapes sans lancer les calculs
- `--materialize-data` : force la matérialisation des jeux de données à l'étape `data`
- `--skip-data` : saute l'étape `data`
- `--skip-test` : saute l'étape `test`
