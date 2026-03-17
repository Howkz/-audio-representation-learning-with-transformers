# Template UE - Provenance et conformité

Objectif: documenter explicitement l'alignement avec la procédure "Before starting your project" du sujet.

## 1) Procédure officielle reproduite

La séquence suivante a été exécutée pour récupérer le template UE:

1. Création d'un dépôt sandbox dédié.
2. Ajout du remote template:
   - `https://git.unicaen.fr/pantin241/daaa-template-ml`
3. `git fetch template`
4. Extraction du contenu `template/main` (méthode équivalente à `git archive ... | tar -x`).
5. Commit d'initialisation:
   - `feat(initialization): init project with template`

Sandbox de vérification:
- `C:\tmp\daaa-template-bootstrap`

## 2) Constat technique

Le template UE récupéré contient des placeholders Cookiecutter non rendus (`{{ cookiecutter.* }}`) dans plusieurs fichiers (`src/__init__.py`, `src/dataset.py`, `src/features.py`, `src/modeling/*`, `docs/mkdocs/*`).

Conséquence:
- utilisation "brute" non directement exécutable sans phase de rendu/adaptation.

## 3) Stratégie adoptée dans ce projet

Pour rester conforme au sujet et conserver un pipeline exécutable:

- la structure attendue du template a été respectée et complétée:
  - `data/raw`, `data/external`, `data/interim`, `data/processed`
  - `models`, `notebooks`, `references`, `reports`, `reports/figures`
  - `setup.cfg`
- l'implémentation audio/Transformer du TP a été conservée dans la structure modulaire (`src/`, `scripts/`, `configs/`) avec routines `make data/train/test`.
- les adaptations techniques (vdigpu, contraintes dépendances/stockage) sont documentées dans le cadrage et le rapport.

## 4) Position défendable devant l'examinateur

- la procédure template UE a bien été reproduite et auditée;
- la structure projet est alignée sur le template;
- les écarts sont des adaptations nécessaires pour rendre le projet réellement exécutable et conforme aux contraintes du TP.
