# Cadrage Scientifique du TP DAAA Audio Transformer

## 1. Concept du TP (version simple)
- Le TP demande de construire un encodeur Transformer audio entrainable en auto-supervise.
- Le modele apprend une representation generale sur l'audio, puis cette representation est transferee vers des taches ASR (CTC + WER).
- Le coeur attendu n'est pas "avoir le meilleur score absolu", mais "faire une implementation modulaire, reproductible, frugale et justifiee".

### Definitions rapides des termes cles
- **ASR (Automatic Speech Recognition)**: tache qui transforme un signal audio de parole en texte.
- **CTC (Connectionist Temporal Classification)**: fonction de perte pour aligner des sorties temporelles audio avec une transcription, sans alignement frame-a-frame annote.
- **WER (Word Error Rate)**: metrique ASR basee sur les erreurs de mots (substitutions, insertions, suppressions) rapportees au nombre de mots de reference.
- **MAE (Masked AutoEncoder)**: pre-entrainement auto-supervise ou une partie de l'entree est masquee puis reconstruite par le modele.
- **TTS (Text-To-Speech)**: tache inverse de l'ASR, qui transforme un texte en signal audio de parole.

## 2. Ce qui est obligatoire vs optionnel
### Obligatoire
- Respect structure template-ml et workflow `make data`, `make train`, `make test`.
- Chargement datasets uniquement via Hugging Face `datasets`.
- Contrainte execution vdigpu, budget cible ~6 Go VRAM.
- Implementation des briques audio Transformer (embedding, encodeur, MAE pretrain, tete CTC).
- Evaluation ASR avec WER.
- Resultats en moyenne + ecart-type sur 5 seeds.
- Rapport PDF structure, sans capture ecran de code.
- Petite etude d'ablation.
- Zip final sans checkpoints ni donnees.

### Optionnel (bonus)
- Bonus TTS.
- Bonus reproduction papier ICLR 2026.

## 3. Enjeux et contraintes fortes
- **Contrainte materielle**: 1 GPU avec VRAM limitee.
- **Temps dispo (definition operationnelle)**: nombre d'heures GPU effectivement disponibles avant le rendu, en incluant interruptions et reprises de session.
- **Contrainte methodologique**: reproductibilite stricte, comparabilite des protocoles.
- **Contrainte eval**: WER + incertitude statistique (5 seeds), frugalite mesuree.
- **Contrainte architecture**: code modulaire et reutilisable dans `src/`.
- **Contrainte operationnelle**: tolerance aux crashs et reprise propre des runs.

## 4. Defis techniques et strategie de mitigation
1. **OOM et instabilite GPU**
- Mitigation: AMP, gradient accumulation, padding dynamique, filtrage duree, fallback batch-size.

2. **Perte de progression en cas de crash/session coupee**
- Mitigation: checkpoints reguliers (steps + fin epoch), reprise auto, marqueur `run_completed`.

3. **Variance experimentale**
- Mitigation: seeds fixes, protocole identique entre runs, aggregation mean/std.

4. **Comparaison non rigoureuse des variantes**
- Mitigation: baseline verrouillee, ablation un facteur a la fois.

5. **Divergence train/test entre phases MAE et CTC**
- Mitigation: contrat de donnees unique (features, lengths, masks, labels).

## 5. Ce que nous devons faire concretement
1. Preparer les donnees audio (`make data`) via HF datasets only.
2. Implementer extraction log-Mel + normalisation duree.
3. Implementer `AudioPatchEmbedding`, `AudioTransformerEncoder`, `AudioMAEPretrain`, `AudioTransformerCTC`.
4. Entrainer en 2 etapes (`make train`): MAE puis fine-tuning CTC.
5. Evaluer (`make test`) sur ASR avec WER, 5 seeds, et mesures frugalite.
6. Appliquer un protocole en trois phases: screening 1-seed sur sous-echantillons, selection Top-5 en 2 seeds, puis final Top-3 en 5 seeds sur sous-echantillon elargi compatible avec un budget de 24h.
7. Produire artefacts de preuve: JSON partial/final, tableaux, logs memoire/temps.
8. Rediger rapport final centre sur choix, compromis et ablation en explicitant la difference screening (subsample) vs final (subsample elargi) et sa justification temporelle.

## 6. Resultats theoriquement attendus (sans promesse chiffrree)
- Le pretraining MAE doit stabiliser le fine-tuning CTC par rapport a un encodeur non pre-entraine.
- Un mask ratio trop faible ou trop fort degrade en general le transfert.
- Une capacite encodeur plus grande peut ameliorer WER mais augmente cout memoire/temps.
- Les optimisations frugales doivent reduire le cout (memoire/temps) avec compromis potentiel sur WER.
- Les metriques 5 seeds doivent montrer une variance raisonnable et interpretable.

## 6bis. Plan operationnel 24h (realiste vdigpu)
- Budget cible: 24h GPU utile (interruptions incluses via resume).
- Decoupage:
  - E00 (smoke): <= 20 min.
  - E01-E08 (screening 1 seed): objectif 6-8h cumule.
  - SEL01-SEL05 (selection 2 seeds): objectif 7-9h cumule.
  - E09-E11 (final 5 seeds, sous-echantillon elargi): objectif 7-9h cumule.
- Regles de pilotage:
  - Si ETA depasse 24h apres E04: reduire `max_steps` finetune de 15-20% pour E05-E11.
  - Si ETA depasse 24h apres selection: conserver top-3 mais reduire uniquement `pretrain.max_steps` en finale (pas les seeds, pas les splits de test).
  - Ne jamais changer les seeds en finale (contrainte statistique du TP).
- Principe de validite:
  - Comparabilite interne forte (meme pipeline, memes seeds par phase, memes regles de classement).
  - Comparabilite externe explicitee comme limite (absence de full-data total).

## 7. Criteres d'acceptation et preuves attendues
1. `make data`, `make train`, `make test` passent sans erreur.
2. Reprise apres interruption validee avec checkpoints.
3. Fichiers `partial` puis `final` generes avec mean/std sur 5 seeds.
4. WER calcule sur split fixe et trace dans les resultats.
5. Metrage frugalite present (VRAM max, batch effectif, temps train/inference).
6. Rapport PDF conforme (pas de screenshot de code, ablation incluse).
7. Zip final conforme, sans data/checkpoints.

## 8. Limite epistemique assumee
- Une conformite "100% garantie" n'existe qu'apres execution complete sur vdigpu et verification des artefacts finaux.
- Ce cadrage vise a couvrir 100% des contraintes connues du sujet, avec gates de preuve explicites.
