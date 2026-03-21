# TP 5-Seeds Main Performance Table

| Experiment | Benchmark | Task | Input | Pretraining | Aug | Patch time | Distillation | Metric A | Metric B |
|---|---|---|---|---|---|---|---|---|---|
| R0_LIBRI | librispeech | ASR | logmel | none | no | 4 | none | 1.0000 +- 0.0000 | 0.0750 +- 0.0645 |
| R1_LIBRI | librispeech | ASR | logmel | mae | no | 4 | none | 1.0000 +- 0.0000 | 0.0592 +- 0.0612 |
| R2_LIBRI | librispeech | ASR | logmel | mae | yes | 4 | none | 1.0000 +- 0.0000 | 0.0504 +- 0.0525 |
| R4_LIBRI | librispeech | ASR | logmel | mae | no | 2 | none | 1.0019 +- 0.0037 | 0.0734 +- 0.0665 |
| R0_VOX | voxpopuli | ASR | logmel | none | no | 4 | none | 0.9989 +- 0.0023 | 0.0072 +- 0.0060 |
| R1_VOX | voxpopuli | ASR | logmel | mae | no | 4 | none | 1.0000 +- 0.0000 | 0.0028 +- 0.0000 |
| R2_VOX | voxpopuli | ASR | logmel | mae | yes | 4 | none | 1.0000 +- 0.0000 | 0.0028 +- 0.0000 |
| R4_VOX | voxpopuli | ASR | logmel | mae | no | 2 | none | 1.0138 +- 0.0248 | 0.0019 +- 0.0012 |
