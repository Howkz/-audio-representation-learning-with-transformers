# Experiment Playbook (Audio Transformer, 6GB VRAM)

Use this file to plan short, defensible experiments that satisfy course grading criteria.

## Baseline First

Start with one stable baseline and lock it before ablations:

- Input: log-Mel (`n_mels` around 80)
- Patch strategy: time-only patches
- Encoder: moderate depth and width (fit 6GB VRAM)
- Pretraining: MAE-style masking on time frames
- Downstream: CTC fine-tuning on ASR

Record:

- parameter count
- train step time
- inference time
- peak GPU memory
- WER mean plus std (5 seeds)

## Suggested Ablation Grid

Run focused ablations, one variable at a time:

1. Positional embeddings:
   - sinusoidal vs learned
2. Patch embedding:
   - time-only vs time-frequency
3. Masking ratio:
   - low/medium/high (for example 0.4 / 0.6 / 0.75)
4. Encoder capacity:
   - shallow vs deeper stack
5. Input space:
   - log-Mel baseline vs alternative representation

Keep all other hyperparameters fixed during each ablation.

## Frugality Tactics To Evaluate

- mixed precision (`torch.autocast` + GradScaler if needed)
- gradient accumulation to emulate larger effective batch
- duration filtering/chunking for long utterances
- dynamic padding at batch collation
- lightweight decoder for MAE pretraining only

For each tactic, show gain/cost:

- memory saved
- speed impact
- metric impact

## Results Table Template

Use one table format consistently:

- Model name
- Architecture variant
- Adaptation protocol (linear probe / full fine-tune / etc.)
- WER mean plus std
- Optional extra metric (accuracy or loss)
- Inference time
- Parameter count

## Report Narrative Structure

1. Problem framing and constraints.
2. Final architecture and rationale.
3. Data pipeline and preprocessing decisions.
4. Pretraining objective and masking strategy.
5. Downstream setup and metrics.
6. Ablation outcomes and interpretation.
7. Frugality analysis under 6GB VRAM.
8. Limitations and next steps.
