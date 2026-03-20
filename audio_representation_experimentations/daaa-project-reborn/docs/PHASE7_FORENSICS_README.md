# Phase 7 Forensics (`P7F01`)

`P7F01` est une variante strictement diagnostique de `P7C01`.

Ce run ne modifie pas :
- la loss,
- le teacher,
- l'architecture student,
- le scheduler,
- la selection de checkpoint,
- le decodage officiel.

Il ajoute :
- un `train_probe` deterministe borne ;
- des metriques d'exposition reelle a l'entrainement ;
- des metriques d'alignement teacher/student ;
- des metriques frame-level sur les logits CTC ;
- des metriques token-level avant et apres collapse CTC ;
- des traces bornees par exemple.

## Config

- [hidden_states_forensics_librispeech.yaml](/c:/Users/maelc_wa38p7e/OneDrive/Universite/M2%20IA%20Informatique/Developpement%20avanc%C3%A9%20appren%20auto/-audio-representation-learning-with-transformers/audio_representation_experimentations/daaa-project-reborn/configs/phase7/hidden_states_forensics_librispeech.yaml)

## Commandes

```bash
rm -rf outputs/phase7/checkpoints/P7F01
rm -rf results/phase7/experiments/P7F01
rm -rf data/processed/phase7/P7F01

make data CONFIG=configs/phase7/hidden_states_forensics_librispeech.yaml
make train CONFIG=configs/phase7/hidden_states_forensics_librispeech.yaml
make test CONFIG=configs/phase7/hidden_states_forensics_librispeech.yaml
```

## Artefacts principaux

Diagnostics par seed :
- `results/phase7/experiments/P7F01/diagnostics/forensics_train_probe_seed_<seed>.json`
- `results/phase7/experiments/P7F01/diagnostics/forensics_valid_seed_<seed>.json`
- `results/phase7/experiments/P7F01/diagnostics/forensics_test_seed_<seed>_<variant>_<dataset>.json`

Bundle final :
- `results/phase7/experiments/P7F01/benchmark_results/asr_forensics_P7F01_final.json`

Tables :
- `results/phase7/experiments/P7F01/tables/asr_forensics_overview_P7F01.md`
- `results/phase7/experiments/P7F01/tables/asr_forensics_alignment_P7F01.md`
- `results/phase7/experiments/P7F01/tables/asr_forensics_teacher_student_P7F01.md`

## Questions que `P7F01` doit trancher

- le modele est-il surtout sous-entraine ?
- le mismatch temporel teacher/student est-il significatif ?
- le collapse apparait-il deja au niveau logits frame-level ?
- la pauvrete vient-elle d'un manque de diversite token-level avant le collapse CTC ?
- le greedy decode perd-il de l'information ou reflète-t-il simplement un logit deja pauvre ?
- pourquoi `seed=123` est-il meilleur que `seed=42` ?
