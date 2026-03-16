# DAAA Audio Transformer Project

Implementation skeleton aligned with TP constraints:
- `make data`
- `make train`
- `make test`

Core goals:
- Hugging Face `datasets` only
- MAE pretraining + CTC fine-tuning
- 5-seed reproducible evaluation with WER mean/std
- robust checkpointing and resume after crash
- 6GB VRAM frugality tracking

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

make data
make train
make test
```

## Low-storage mode (vdigpu constrained disk)
Use the dedicated profile:
```bash
make CONFIG=configs/low_storage.yaml data
make CONFIG=configs/low_storage.yaml train
make CONFIG=configs/low_storage.yaml test
```

This profile reduces:
- dataset sample caps
- model size
- saved checkpoint count (rotation)
- retained pretrain checkpoints after each seed

## Ultra quick smoke mode
```bash
make CONFIG=configs/smoke.yaml data
make CONFIG=configs/smoke.yaml train
make CONFIG=configs/smoke.yaml test
```

## Full E00->E11 suite runner
```bash
make suite
```

Manual runner invocation:
```bash
python scripts/run_experiment_suite.py \
  --suite-config configs/suite_e00_e11.yaml \
  --verbose \
  --disk-guard-gb 1.5
```

Optional deterministic override (base + experiment overrides + CLI):
```bash
python scripts/run_experiment_suite.py \
  --suite-config configs/suite_e00_e11.yaml \
  --set training.finetune.max_steps=120 \
  --set model.mae_mask_ratio=0.55
```

Key behavior:
- one global sequence `data -> train -> test -> archive -> cleanup` per experiment
- automatic top-3 selection after screening
- keep `data/cache` between experiments
- clean only experiment checkpoints after archival
- generate final LaTeX template at `results/suite/rapport_final.tex`

## Outputs
- `results/benchmark_results/*.json` : partial and final aggregated metrics
- `outputs/checkpoints/*` : training checkpoints + run markers
- `results/tables/*.md` : compact report-ready tables

## Notes
- Do not include `outputs/checkpoints` or raw datasets in final submission zip.
- Use `configs/baseline.yaml` to control datasets, seeds, and memory-sensitive hyperparameters.
