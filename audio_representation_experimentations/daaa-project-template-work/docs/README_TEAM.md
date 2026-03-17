# README Equipe - Fonctionnement Clair

Objectif de ce fichier: expliquer simplement comment lancer, reprendre, et lire la campagne.

## 1) Vue d'ensemble

Pipeline technique:

1. `run_data.py`: prepare/valide les datasets HF.
2. `run_train.py`: pretrain MAE puis finetune CTC.
3. `run_test.py`: evalue ASR et agregation des metriques.

Campagne complete (`run_experiment_suite.py`):

1. `E00` (smoke de validation).
2. `E01..E08` (screening 1 seed).
3. `SEL01..SEL05` (selection Top-5 en 2 seeds).
4. `E09..E11` (final Top-3 en 5 seeds).

## 2) Ce qui est automatise

Le runner de suite fait automatiquement:

- generation des runtime configs par experience,
- execution `data -> train -> test`,
- classement screening (WER, runtime, memoire),
- phase selection Top-5 en 2 seeds,
- resolution des finales depuis la selection,
- archivage des artefacts utiles,
- nettoyage des checkpoints d'experience,
- generation du template LaTeX final.

## 3) Commandes Linux recommandees

Depuis la racine du projet:

```bash
cd audio_representation_experimentations/daaa-project-template-work
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x scripts/linux_experiments.sh
```

Lancement:

```bash
./scripts/linux_experiments.sh smoke
./scripts/linux_experiments.sh suite
./scripts/linux_experiments.sh resume
./scripts/linux_experiments.sh suite --clean
```

## 4) Quand utiliser quoi

- Vous voulez juste verifier la stack: `smoke`.
- Vous demarrez la vraie experimentation: `suite`.
- Session coupee/crash: `resume`.
- Disque plein ou cache sale: `suite --clean`.

## 5) Resultats a consulter

- `results/suite/leaderboard_screening.csv`
- `results/suite/leaderboard_selection.csv`
- `results/suite/suite_summary.csv`
- `results/experiments/<ID>/benchmark_results/*.json`
- `results/suite/rapport_final.tex`

## 6) Points importants pour le rapport

- Expliquer la contrainte vdigpu (6 Go VRAM / budget disque).
- Expliquer le compromis:
  - screening sous-echantillonne,
  - selection top-5 en 2 seeds,
  - final top-3 full-data en 5 seeds.
- Rapporter WER + runtime + memoire GPU.
- Utiliser les checklists:
  - `docs/REPORT_CHECKLIST.md`
  - `docs/SUBMISSION_CHECKLIST.md`
