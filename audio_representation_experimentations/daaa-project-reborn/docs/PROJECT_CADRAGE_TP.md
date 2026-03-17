# Cadrage Scientifique du TP DAAA Audio Transformer

## 1. Concept du TP (version simple)
- Construire un encodeur Transformer audio pre-entraine en auto-supervise (MAE), puis transfere vers ASR (CTC + WER).
- Priorite: implementation modulaire, reproductible, frugale et argumentee, plutot qu'un score absolu isole.

### Definitions rapides
- **ASR**: conversion audio parole -> texte.
- **CTC**: perte pour aligner une sequence audio et une transcription sans alignement frame-a-frame annote.
- **WER**: taux d'erreur mot pour ASR.
- **MAE**: pre-entrainement par reconstruction de regions masquees.
- **TTS**: texte -> parole (hors scope principal, bonus).

## 2. Obligatoire vs optionnel
### Obligatoire
- Workflow `make data`, `make train`, `make test`.
- Datasets charges via Hugging Face `datasets`.
- Execution correcte sur vdigpu (frugalite, robustesse, reprise).
- Briques audio Transformer: embedding, encodeur, MAE pretrain, tete CTC.
- Evaluation ASR avec WER.
- Resultats principaux en moyenne + ecart-type sur 5 seeds.
- Rapport PDF structure, sans capture d'ecran de code.
- Ablation courte mais defendable.
- Archive finale sans checkpoints ni donnees.

### Optionnel
- Bonus TTS.
- Bonus reproduction papier ICLR.

## 3. Contraintes reelles (verrou de cadrage)
- **Contrainte dure**: le temps total disponible est borne a `24h` de travail GPU utile pour notre execution.
- **Contrainte locale variable**: le stockage disponible (par exemple `~18 Go` sur notre machine) sert au pilotage local, mais ne doit pas etre fige comme limite scientifique universelle.
- **Contrainte methode**: protocole identique entre machines (meme logique d'experiences, memes regles de selection, memes seeds finales).
- **Contrainte d'evaluation**: le professeur peut executer sur une machine avec plus de stockage/temps; le protocole reste valide sans modification conceptuelle.

## 4. Budget tenu par design (pas par variables globales)
Le respect des 24h est porte par un profil technique fixe, pas par des variables runtime globales.

### Profil technique frugal par defaut
- Modele baseline: `dim=192`, `depth=4`, `num_heads=6`, `mlp_ratio=3.0`, `max_len=1536`.
- Entrainement: AMP active, gradient accumulation, clipping, checkpoints limites.
- Boucle: `pretrain.max_steps` et `finetune.max_steps` calibres pour tenir la trajectoire 24h.

### Quotas datasets calibres par phase
- Screening (`E01-E08`): `pretrain=900`, `asr_train=1100`, `asr_valid=300`.
- Selection (`SEL01-SEL05`, 2 seeds): memes quotas que screening.
- Finale (`E09-E11`, 5 seeds): sous-echantillon elargi fixe (`pretrain=2000`, `asr_train=3000`, test non tronque si possible).

Ces choix sont figes dans les configs d'experiences et dans les overrides par experience, pour que le comportement soit stable et defendable.

## 5. Protocole experimental verrouille E00->E11
1. `E00` (smoke): validation pipeline complet.
2. `E01-E08` (screening): 1 seed par variante sur sous-echantillons calibres.
3. `SEL01-SEL05` (selection): top-5 screening reevalues sur 2 seeds.
4. `E09-E11` (final): top-3 selection consolides sur 5 seeds, sous-echantillon elargi si budget le permet.

### Regles de classement et de pilotage
- Classement screening/selection: tri lexicographique `(WER asc, runtime inference asc, memoire GPU asc)`.
- Si ETA > 24h apres `E04`: reduire `finetune.max_steps` de `15-20%` pour `E05-E11`.
- Si ETA > 24h apres selection: reduire uniquement `pretrain.max_steps` sur finale.
- Interdit en finale: changer les seeds finales ou les splits de test.

## 6. Plan d'execution 24h (operatoire)
- `make data` (1h30-2h):
  - preparer/cache uniquement les splits requis,
  - appliquer les caps calibres par phase,
  - journaliser les tailles de splits effectives.
- `make train` (15h-17h):
  - sequence `E00 -> E01..E08 -> SEL01..SEL05 -> E09..E11`,
  - checkpointing + reprise auto.
- `make test` (4h-5h):
  - WER + mean/std,
  - generation des tableaux screening/selection/final,
  - verification coherence `partial -> final`.

## 7. Interface d'execution Linux unique (a livrer)
Script cible: `scripts/linux_experiments.sh`

### CLI publique
- `smoke`
- `suite`
- `resume`
- `dry-run`
- `clean`

### Parametrage public
- `EXPERIMENT_CACHE_ROOT` (racine des caches)
- Les choix de budget se font dans les hyperparametres/configs (steps, seeds, quotas), pas via des variables globales de budget.

### Sorties attendues
- `results/suite/leaderboard_screening.csv`
- `results/suite/leaderboard_selection.csv`
- `results/suite/suite_summary.csv`
- `results/suite/runtime_configs/*.yaml`
- artefacts de run `partial` et `final` par experience

## 8. Justification methodologique (reference TAL)
Le protocole reprend les mecanismes qui ont bien fonctionne sur le projet TAL:
- progression en phases (smoke -> screening -> final),
- sauvegarde incremental `partial` puis consolidation `final`,
- gestion multi-seeds et reprise,
- comparaison controllee avec un protocole stable entre variantes.

Transposition audio:
- memes garanties de reproductibilite et de tolerance aux interruptions,
- mais metriques et pipeline adaptes a ASR (WER, cout inference, memoire GPU).

## 9. Criteres d'acceptation et preuves attendues
1. `make data`, `make train`, `make test` passent sans erreur bloquante.
2. `resume` reprend une suite interrompue sans perte de runs finalises.
3. Fichiers `partial` puis `final` produits et coherents.
4. WER trace avec mean/std sur les experiences finales.
5. Mesures frugales presentes (VRAM pic, temps train/inference, batch effectif).
6. Rapport PDF conforme et explicite sur les compromis.
7. Archive finale conforme, sans donnees/checkpoints.

## 10. Limite epistemique assumee
- Une conformite totale n'est prouvable qu'apres execution complete sur vdigpu et verification de tous les artefacts.
- Ce cadrage rend les hypotheses et regles de decision explicites pour une evaluation defendable.
