# VDIGPU Runbook

## Hypothèses d'environnement

- Python avec `requirements.txt` installé.
- GPU `vdigpu` avec environ `6 Go` de VRAM.
- Accès réseau sortant pour `datasets` et modèles Hugging Face.
- Lancement depuis la racine `audio_representation_experimentations/daaa-project-reborn`.

## Installation minimale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Vérification de conformité

```bash
make compliance CONFIG=configs/final_tp/core_librispeech.yaml
```

Artefact attendu :

- `results/compliance/tp_compliance_report.json`

## Smoke core path

```bash
make data CONFIG=configs/final_tp/core_librispeech.yaml
make train CONFIG=configs/final_tp/core_librispeech.yaml
make test CONFIG=configs/final_tp/core_librispeech.yaml
```

Artefacts attendus :

- `results/final_tp/experiments/TP5COREL/benchmark_results/train_audio_transformer_TP5COREL_final.json`
- `results/final_tp/experiments/TP5COREL/benchmark_results/asr_benchmark_TP5COREL_final.json`
- `results/final_tp/experiments/TP5COREL/benchmark_results/probe_benchmark_TP5COREL_final.json`

## Campagne 5 seeds

```bash
make suite-tp
make report-tp
```

Sorties attendues :

- `results/experiments/<ID>/...` pour chaque expérience `R0..R4` et `A1`
- `results/suite/tp5/tp5_main_performance_table.md`
- `results/suite/tp5/tp5_frugality_table.md`
- `results/suite/tp5/tp5_ablation_table.md`

## Exécution parallèle sur 2 VDI

VDI 1 :

```bash
make suite-tp-pair
```

VDI 2 :

```bash
make suite-tp-triple
```

Répartition :

- `pair` : seeds `42, 123`
- `triple` : seeds `456, 789, 1024`

Après les deux runs :

1. copier le dossier `results/experiments` de la seconde VDI dans `results_peer/experiments` sur la machine principale ;
2. lancer la fusion :

```bash
make merge-tp-shards
```

La fusion reconstruit les JSON finaux agrégés sur 5 seeds et relance la génération des tables finales.

## Reprise d'entraînement

Support statique :

- `scripts/run_train.py --continue-completed`
- `src/training/checkpointing.py`

Smoke recommandé sur VDI :

```bash
python scripts/test_resume_training.py --config configs/final_tp/core_librispeech.yaml
```

Le test doit produire un JSON sous `results/compliance/` et confirmer qu'un run court reprend bien depuis `latest.pt`.

## Calibration mémoire

Point de départ recommandé :

- LibriSpeech : `batch_size=8`, `grad_accum_steps=2`
- VoxPopuli : `batch_size=6`, `grad_accum_steps=2`
- Pré-entraînement MAE : `batch_size=16`, `grad_accum_steps=1`

Si OOM :

1. réduire `batch_size` par 2 ;
2. doubler `grad_accum_steps` ;
3. garder `amp: true`.

## Limites connues

- VoxPopuli est traité ici par filtrage strict, pas par chunking.
- Le linear probe FSC est embarqué sur les runs LibriSpeech, pas sur les runs VoxPopuli.
- La distillation `A1` reste une annexe expérimentale, pas le cœur du protocole.
