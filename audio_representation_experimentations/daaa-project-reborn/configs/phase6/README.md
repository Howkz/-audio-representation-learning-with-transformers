# Espace de travail des configs Phase 6

Ce dossier contient les configs de la campagne phase 6.

Objectif principal :

- tester une généralisation simple après la phase 5.

Choix structurants :

- sélection du `best checkpoint` pénalisée par `empty_pred_ratio`
- évaluation explicite de `best` et `final`
- dataset plus large que `P5D01`
- peu d'époques
- mêmes correctifs de pipeline conservés

Config présente :

- `P6G01.yaml` : généralisation simple sans MAE
