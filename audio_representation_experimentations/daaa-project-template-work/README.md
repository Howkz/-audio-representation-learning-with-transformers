# DAAA Audio Transformer - Guide Rapide

Ce projet implemente un pipeline audio conforme au TP:
- pretrain MAE
- finetune CTC
- evaluation WER
- campagnes experimentales reproductibles avec reprise

## Demarrage en 3 etapes

```bash
cd audio_representation_experimentations/daaa-project-template-work
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Le plus simple sous Linux (vdigpu)

Un script unique est disponible:

```bash
chmod +x scripts/linux_experiments.sh
```

Puis:

```bash
./scripts/linux_experiments.sh smoke
./scripts/linux_experiments.sh suite
./scripts/linux_experiments.sh resume
./scripts/linux_experiments.sh suite --clean
./scripts/linux_experiments.sh resume --cache-root /path/with/space --clean-hf
```

## Que fait chaque mode

- `smoke`: mini run technique rapide (pipeline complet) pour verifier que tout tourne.
- `suite`: lance toute la campagne depuis le debut.
- `resume`: reprend la campagne sans relancer ce qui est deja termine.
- `--clean`: vide caches/artefacts projet avant le lancement.
- `--cache-root <path>`: place les caches HF/TMP sur un disque plus grand.
- `--clean-hf`: vide les caches HF/TMP.

## Logique experimentale de la suite

La suite suit ce protocole:

1. `E00`: validation technique (smoke interne a la suite).
2. `E01..E08`: screening en 1 seed.
3. `SEL01..SEL05`: selection Top-5 en 2 seeds.
4. `E09..E11`: final Top-3 en 5 seeds.

Politique de donnees:
- screening: sous-echantillon fixe (comparabilite interne forte, cout borne),
- final: sous-echantillon elargi pour renforcer la validite sans depasser le budget temps 24h.

## Commandes manuelles (sans script Linux)

```bash
export PYTHONPATH=.
python3 scripts/run_data.py --config configs/low_storage.yaml
python3 scripts/run_train.py --config configs/low_storage.yaml
python3 scripts/run_test.py --config configs/low_storage.yaml
```

Suite complete:

```bash
PYTHONPATH=. python3 scripts/run_experiment_suite.py \
  --suite-config configs/suite_e00_e11.yaml \
  --verbose \
  --disk-guard-gb 1.5
```

Reprise:

```bash
PYTHONPATH=. python3 scripts/run_experiment_suite.py \
  --suite-config configs/suite_e00_e11.yaml \
  --resume \
  --verbose \
  --disk-guard-gb 1.5
```

## Ajustement si ETA > 24h

Exemple de relance avec budget reduit (a partir de E05):

```bash
PYTHONPATH=. python3 scripts/run_experiment_suite.py \
  --suite-config configs/suite_e00_e11.yaml \
  --from-id E05 \
  --resume \
  --set training.finetune.max_steps=160 \
  --verbose
```

## Ou lire les resultats

- `results/suite/leaderboard_screening.csv`
- `results/suite/leaderboard_selection.csv`
- `results/suite/suite_summary.csv`
- `results/experiments/*/benchmark_results/*.json`
- `results/suite/rapport_final.tex`

## Notes pratiques vdigpu

- Evitez de lancer `smoke` puis `suite` si vous voulez gagner du temps: `suite` contient deja `E00`.
- Si un run est interrompu, utilisez `resume`.
- Si le stockage sature, utilisez `--cache-root` vers un point de montage avec plus d'espace.
- Si le cache HF est corrompu, relancez avec `--clean-hf`.
- Checklists rapport/rendu:
  - `docs/REPORT_CHECKLIST.md`
  - `docs/SUBMISSION_CHECKLIST.md`
