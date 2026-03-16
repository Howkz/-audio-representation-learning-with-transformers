# README Collègues

Ce document permet de reprendre vite le projet sans relire tout l’historique.

## 1) Ce qui a été fait et pourquoi

- Pipeline complet opérationnel: `data -> train -> test`.
- Suite expérimentale E00→E11 implémentée pour raconter une progression scientifique:
  - E00 validation technique.
  - E01 baseline.
  - E02 effet du pré-entraînement MAE (NoMAE).
  - E03–E08 ablations architecture/hyperparamètres.
  - E09–E11 consolidation Top-3 en 5 seeds.
- Reprise après crash robuste:
  - checkpoints périodiques + fin d’époque,
  - état modèle/optim/scheduler/scaler/RNG,
  - marqueurs de run terminé.
- Contrainte stockage respectée:
  - conservation de `data/cache` (évite re-téléchargement),
  - nettoyage des checkpoints entre expériences,
  - archivage des artefacts finaux.
- Génération automatique du template LaTeX du rapport scientifique (avec sections résultats/discussion laissées vides).

Objectif global: satisfaire les exigences du TP avec une méthodologie reproductible, comparée et défendable.

## 2) Principe du script expérimental (et pourquoi)

Script principal: `scripts/run_experiment_suite.py`

- Lit un manifest unique: `configs/suite_e00_e11.yaml`.
- Exécute les expériences dans l’ordre avec isolement des sorties par ID (`E00`, `E01`, ...).
- Ranke automatiquement le screening (WER, puis runtime, puis mémoire) pour déterminer Top-1/2/3.
- Lance ensuite E09/E10/E11 en 5 seeds à partir des meilleures configs.
- Affiche l’état de progression et le stockage en temps réel.
- Applique une garde d’espace disque (`--disk-guard-gb`) pour stop propre.
- Génère à la fin:
  - `results/suite/leaderboard_screening.csv`
  - `results/suite/suite_summary.csv`
  - `results/suite/rapport_final.tex`

Pourquoi ce design:
- reproductibilité (manifest + seeds + reprise),
- comparabilité (protocole identique),
- robustesse opérationnelle (crash tolerance),
- frugalité (cache conservé, checkpoints purgés).

## 3) Comment exécuter les scripts

## Windows (PowerShell)

```powershell
cd DAAA/daaa-project
$env:PYTHONPATH='.'

# 1) Pipeline simple
python scripts/run_data.py --config configs/low_storage.yaml
python scripts/run_train.py --config configs/low_storage.yaml
python scripts/run_test.py --config configs/low_storage.yaml

# 2) Suite complète E00->E11
python scripts/run_experiment_suite.py --suite-config configs/suite_e00_e11.yaml --verbose --disk-guard-gb 1.5

# 3) Reprise après interruption
python scripts/run_experiment_suite.py --suite-config configs/suite_e00_e11.yaml --resume --verbose --disk-guard-gb 1.5

# 4) Générer uniquement le rapport LaTeX
python scripts/generate_report_template.py --suite-config configs/suite_e00_e11.yaml --output results/suite/rapport_final.tex
```

## Linux/macOS

```bash
cd DAAA/daaa-project
export PYTHONPATH=.

make data CONFIG=configs/low_storage.yaml
make train CONFIG=configs/low_storage.yaml
make test CONFIG=configs/low_storage.yaml

make suite
make report-template
```

## 4) Cartographie docs

- `docs/TP_consignes.md`: sujet original.
- `docs/PROJECT_CADRAGE_TP.md`: cadrage scientifique.
- `docs/REPORT_CHECKLIST.md`: checklist du rapport.
- `docs/SUBMISSION_CHECKLIST.md`: checklist de rendu.
- `docs/README_COLLEGUES.md`: ce document.
