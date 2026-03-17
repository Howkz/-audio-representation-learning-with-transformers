# Submission Checklist (Final Zip)

## Naming
- Archive name follows: `daaa_nomduprojet1_nom2.zip`

## Must include
- `src/`
- `scripts/`
- `configs/`
- `results/` (lightweight summaries/tables)
- `rapport.pdf`

## Must exclude
- Raw datasets / cached datasets
- Model checkpoints
- Temporary outputs / runtime caches

## Verification steps
1. Run `make data`, `make train`, `make test` on target env.
2. Confirm 5 seeds completed and aggregated mean/std files exist.
3. Confirm WER tables are generated.
4. Confirm report includes ablation and frugality section.
5. Build archive with `make package` (or manual zip with same exclusion rules).
