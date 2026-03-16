# Spec Checklist (DAAA Audio Transformer)

Use this file as a hard-constraint checklist before coding and before final submission.

## Non-Negotiable Constraints

- Keep compatibility with the template-ml project structure and `Makefile` workflow.
- Load datasets only with Hugging Face `datasets` API.
- Keep training and inference runnable on one GPU with about 6GB VRAM (vdigpu target).
- Implement modular reusable components in `src/` (especially audio embedding and encoder blocks).
- Implement checkpointing and resume for training loops.
- Report downstream results as mean plus std over 5 random seeds.
- Write a structured PDF report without code screenshots.
- Exclude datasets and checkpoints from final zip.

## Mandatory Make Targets

- `make data`: Download, preprocess, filter, resample, feature extraction, cache.
- `make train`: Self-supervised pretraining and downstream fine-tuning.
- `make test`: Evaluation protocols and metric reporting.

## Required Model Scope

- `AudioPatchEmbedding` adapted for audio tokens.
- Audio Transformer encoder (multi-head self-attention plus FFN blocks).
- MAE-style pretraining objective with masking and masked-region reconstruction.
- CTC fine-tuning head for ASR.

## Allowed/Disallowed Pretraining Rules

- Do not use external pretrained backbones as the main model.
- Allow teacher-model distillation as an optional enhancement.

## Datasets Mentioned In Subject

- Fluent Speech Commands (self-supervised pretraining candidate):
  - `load_dataset("Codec-SUPERB/fluent_speech_commands_synth")`
- LibriSpeech ASR clean subset (pretraining/fine-tuning candidate):
  - `load_dataset("openslr/librispeech_asr", "clean", split="train.clean.100")`
- VoxPopuli EN (ASR benchmark candidate):
  - `load_dataset("facebook/voxpopuli", "en", split="train")`
- Gemini Flash 2.0 Speech (optional TTS bonus):
  - `load_dataset("shb777/gemini-flash-2.0-speech")`

## Evaluation Expectations

- ASR metric: WER on validation/test split.
- Optional TTS bonus: reconstruction loss and qualitative spectrogram comparison.
- Report memory usage, effective batch size, and runtime considerations.
- Include at least one ablation study tied to a key architectural decision.

## Packaging Checklist

- Zip naming convention follows course instructions.
- Include code, configs, logs/results summaries, and report.
- Exclude heavy artifacts:
  - raw data directories
  - cached datasets
  - model checkpoints
  - temporary outputs
