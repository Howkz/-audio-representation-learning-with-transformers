# Espace de travail des configs Phase 5

Ce dossier contient les configs de la campagne de test d'apprenabilité phase 5.

Objectifs :

- augmenter fortement le fine-tuning supervisé ;
- réduire la difficulté de la tâche ;
- vérifier si la tête CTC apprend enfin sur un sous-cas simple.

Configs présentes :

- `P5D01.yaml` : diagnostic d'apprenabilité sans MAE
- `P5B01.yaml` : baseline simple avec MAE
- `P5B02.yaml` : baseline simple sans MAE

Choix structurants :

- `padding mask` conservé
- `feature_norm: utterance`
- `patch_time: 2`
- pas de tronquage fixe sur les datasets ASR
- filtrage par longueur de transcription
- fine-tuning supervisé fortement augmenté
