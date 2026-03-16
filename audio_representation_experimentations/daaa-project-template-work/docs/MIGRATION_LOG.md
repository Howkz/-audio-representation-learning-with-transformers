# Migration Log - Template UE -> Projet TP

Ce dossier est construit selon la demande: partir d'un template UE vide puis greffer l'ancien travail.

## Source template

- Origine: `https://git.unicaen.fr/pantin241/daaa-template-ml`
- Snapshot local utilisé: `C:\tmp\daaa-template-bootstrap`

## Cible de migration

- Dossier de travail: `audio_representation_experimentations/daaa-project-template-work`

## Étapes de greffe effectuées

1. **Phase 0 - Template vide**
   - Copie intégrale du template dans le dossier cible.

2. **Phase 1 - Cœur applicatif**
   - Copie de `configs/`
   - Copie de `scripts/`
   - Copie de `src/`
   - Copie de `requirements.txt`

3. **Phase 2 - Documentation**
   - Copie de `docs/` du projet actuel
   - Copie de `09_audio_machine_learning.ipynb`

4. **Phase 3 - Artefacts utiles**
   - Copie de `results/suite/` (si présent)
   - Copie de `Makefile`
   - Copie de `README.md`

## Intention

- Conserver la traçabilité "on part du template prof".
- Réintégrer progressivement le travail existant sans modifier le dossier d'origine.
