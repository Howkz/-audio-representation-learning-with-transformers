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

## Outputs
- `results/benchmark_results/*.json` : partial and final aggregated metrics
- `outputs/checkpoints/*` : training checkpoints + run markers
- `results/tables/*.md` : compact report-ready tables

## Notes
- Do not include `outputs/checkpoints` or raw datasets in final submission zip.
- Use `configs/baseline.yaml` to control datasets, seeds, and memory-sensitive hyperparameters.
