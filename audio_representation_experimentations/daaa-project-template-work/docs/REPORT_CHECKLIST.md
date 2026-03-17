# Report Checklist (TP DAAA)

## Required sections
- Problem framing and constraints (vdigpu, 6GB VRAM, reproducibility).
- Final architecture and rationale (`AudioPatchEmbedding`, encoder, MAE decoder, CTC head).
- Data pipeline policy (HF datasets only, resample, crop/pad, log-Mel).
- Pretraining objective and masking strategy.
- Fine-tuning protocol and ASR evaluation (WER).
- 5-seed mean/std result tables.
- Explain 3-phase protocol: screening (1 seed), selection top-5 (2 seeds), final top-3 (5 seeds full-data).
- Frugality analysis (train/inference time, effective batch, peak GPU memory).
- Ablation study (at least one controlled comparison).
- Limitations and next steps.

## Forbidden / mandatory formatting
- No code screenshots.
- Keep focus on methodology, tradeoffs, and empirical evidence.
- Include exact dataset splits and seed list.
- Explicitly distinguish screening subset policy vs final full-data policy.
- Explain resume/checkpoint strategy and crash tolerance.

## Evidence to reference
- `results/benchmark_results/train_audio_transformer_final.json`
- `results/benchmark_results/asr_benchmark_final.json`
- `results/benchmark_results/asr_benchmark_by_dataset_final.json`
- `results/tables/asr_overall_table.md`
- `results/tables/asr_dataset_breakdown.md`
