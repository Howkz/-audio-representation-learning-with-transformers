# Phase 7 - Pipeline ASR distille propre

## Intention

La phase 7 ouvre une nouvelle voie propre, en parallele des phases 1 a 6, pour produire
un pipeline ASR plus conforme au sujet et plus defensable pour le rendu :

- `make data`, `make train`, `make test` restent les points d'entree uniques ;
- la distillation ASR devient le chemin principal ;
- LibriSpeech `clean` devient le benchmark critique ;
- VoxPopuli sort du chemin critique tant que la chaine LibriSpeech n'est pas stabilisee ;
- le MAE est conserve comme option, mais il n'est plus necessaire a la voie principale.
- les configs Phase 7 sont reglees en `streaming: true` pour rester compatibles avec
  un poste local contraint en espace disque.

## Configs

- `configs/phase7/smoke.yaml`
  - `P7D01`
  - smoke test avec teacher externe Wav2Vec2
  - 1 seed
  - objectif : verifier que la chaine tient et que les predictions cessent d'etre triviales

- `configs/phase7/distill_librispeech.yaml`
  - `P7B01`
  - baseline clean distillee sur LibriSpeech
  - teacher externe `facebook/wav2vec2-base-960h`
  - evalue `best` et `final`

- `configs/phase7/self_distill_librispeech.yaml`
  - `P7B02`
  - self-distillation a partir d'un checkpoint student phase 7
  - le chemin `teacher.checkpoint_path` doit pointer vers un checkpoint deja produit par `P7B01`

- `configs/phase7/hidden_states_librispeech.yaml`
  - `P7C01`
  - variante hidden-state distillation
  - meme teacher externe Wav2Vec2

- `configs/phase7/logits_anti_overemit_librispeech.yaml`
  - `P7C02`
  - conserve la KD logits
  - ajoute une contrainte explicite contre la sur-emission non-blank

- `configs/phase7/combined_hidden_anti_overemit_librispeech.yaml`
  - `P7C03`
  - combine hidden-state distillation et anti-overemit
  - utile seulement apres les deux runs separes

- `configs/phase7/hidden_states_selection_v2_librispeech.yaml`
  - `P7C04`
  - reprend strictement l'objectif de `P7C01`
  - change uniquement le `checkpoint_selection` pour tester si un meilleur `ctc_best.pt` existe deja

- `configs/phase7/hidden_states_kd_stronger_librispeech.yaml`
  - `P7C05`
  - repart de `P7C01`
  - change uniquement l'intensite de la hidden-state KD
  - objectif : tester si une KD un peu plus forte reduit la sous-emission

- `configs/phase7/hidden_states_underemit_librispeech.yaml`
  - `P7N01`
  - repart de `P7C01`
  - ajoute une contrainte douce contre la sous-emission
  - objectif : augmenter legerement la densite non-blank sans retomber dans le babbling

- `configs/phase7/hidden_states_aug_librispeech.yaml`
  - `P7A02`
  - repart de `P7C01`
  - ajoute une vraie augmentation legere sur `asr_train` :
    gain aleatoire, bruit faible et SpecAugment leger

- `configs/phase7/hidden_states_depth4_librispeech.yaml`
  - `P7A03`
  - ablation simple `depth=4` contre `depth=6`
  - meme teacher, meme loss, meme dataset

- `configs/phase7/hidden_states_forensics_librispeech.yaml`
  - `P7F01`
  - reprend `P7C01` sans changer la loss ni le teacher
  - ajoute une couche forensic riche pour mesurer :
    - exposition train reelle
    - alignement teacher/student
    - collapse token-level avant et apres CTC
    - similarite hidden-state teacher/student

- `configs/phase7/hidden_states_patch2_relaxed_librispeech.yaml`
  - `P7R01`
  - repart du constat forensic de `P7F01`
  - garde la hidden-state KD et la loss inchangees
  - corrige la resolution temporelle student avec `patch_time=2`
  - desserre les filtres ASR (`12s`, `120 chars`, `25 mots`) pour sortir du regime
    artificiellement etroit qui ne gardait que `127` exemples train

- `configs/phase7/hidden_states_patch2_kd_warmup_librispeech.yaml`
  - `P7R02`
  - repart strictement de `P7R01`
  - garde `patch_time=2` et le dataset relache
  - ajoute un warmup lineaire de la KD au debut du fine-tune
  - budget volontairement plus large pour laisser la KD warmup s'exprimer :
    `epochs=10`, `max_steps=3000`, `early_stopping_min_epochs=5`
  - objectif : eviter le blank collapse immediat observe dans `P7R01`

## Choix techniques

- Student :
  - log-Mel 80 bandes
  - embedding lineaire par trame (`patch_time=1` sur la baseline `P7C01/P7F01`)
  - encodeur Transformer avec `src_key_padding_mask`
  - tete CTC simple
  - une variante `patch_time=2` est maintenant preparee pour `P7R01` afin de
    rapprocher le student du teacher Wav2Vec2 autour de `50 Hz`

- Distillation externe :
  - teacher utilise uniquement pendant l'entrainement
  - deux cibles sont maintenant supportees :
    - logits CTC Wav2Vec2 remappes sur le vocabulaire caractere du student
    - hidden states du teacher, alignes temporellement et compares aux features du student
  - pertes :
    - logits KD : `lambda_ctc * CTC + lambda_kd * KL`
    - hidden-state KD : `lambda_ctc * CTC + lambda_kd * MSE`
  - a partir du diagnostic intermediaire, la phase 7 utilise une KD plus conservative :
    - `lambda_kd` reduit
    - KD appliquee uniquement sur les frames teacher juges informatifs
    - un frame est considere informatif si la masse non-blank du teacher depasse un seuil
      et si l'argmax teacher n'est pas `blank`
  - pour la hidden-state KD, le student projette ses features vers la dimension teacher
    uniquement pour la distillation ; cette projection n'est pas utilisee a l'inference finale
  - une contrainte `anti_overemit` est maintenant disponible :
    - elle penalise une densite non-blank trop elevee par rapport au ratio cible
      `target_lengths / out_lengths`
    - elle est activee seulement dans les configs dediees
  - la phase 7 utilise aussi une optimisation plus prudente :
    - warmup explicite
    - LR fine-tune reduit
    - early stopping base sur le `selection_score`
  - une rampe de KD est maintenant supportee :
    - `distillation.warmup_steps`
    - `distillation.warmup_start_factor`
    - la KD peut donc commencer plus faible puis remonter progressivement vers
      `lambda_kd`

- Distillation interne :
  - meme interface teacher
  - teacher recharge depuis `ctc_best.pt` ou `ctc_final.pt`

- Selection de checkpoint :
  - `WER`
  - penalite `empty_pred_ratio`
  - penalite de predictions trop courtes
  - bonus d'`accuracy`
  - une variante `selection v2` est maintenant disponible pour `P7C04` :
    - penalite symetrique sur les ecarts de longueur
    - penalite de repetition
    - bonus d'accuracy un peu plus fort
  - l'objectif est de mieux choisir `ctc_best.pt` sans toucher a la loss

- Augmentations :
  - la phase 7 supporte maintenant des augmentations de train legeres par config
  - elles s'appliquent uniquement au split `asr_train`
  - support disponible :
    - gain aleatoire sur waveform
    - bruit additif faible via SNR cible
    - SpecAugment leger sur log-Mel
  - les splits validation/test restent strictement non augmentes

- Contraintes de densite :
  - `anti_overemit` penalise une densite non-blank trop elevee
  - `anti_underemit` penalise une densite non-blank trop basse par rapport a
    `target_lengths / out_lengths`
  - la version `P7N01` n'active que `anti_underemit`

## Commandes

```bash
make data CONFIG=configs/phase7/smoke.yaml
make train CONFIG=configs/phase7/smoke.yaml
make test CONFIG=configs/phase7/smoke.yaml
```

Puis :

```bash
make data CONFIG=configs/phase7/distill_librispeech.yaml
make train CONFIG=configs/phase7/distill_librispeech.yaml
make test CONFIG=configs/phase7/distill_librispeech.yaml
```

Ensuite seulement :

```bash
make data CONFIG=configs/phase7/self_distill_librispeech.yaml
make train CONFIG=configs/phase7/self_distill_librispeech.yaml
make test CONFIG=configs/phase7/self_distill_librispeech.yaml
```

Variantes post-diagnostic :

```bash
make data CONFIG=configs/phase7/hidden_states_librispeech.yaml
make train CONFIG=configs/phase7/hidden_states_librispeech.yaml
make test CONFIG=configs/phase7/hidden_states_librispeech.yaml
```

```bash
make data CONFIG=configs/phase7/logits_anti_overemit_librispeech.yaml
make train CONFIG=configs/phase7/logits_anti_overemit_librispeech.yaml
make test CONFIG=configs/phase7/logits_anti_overemit_librispeech.yaml
```

```bash
make data CONFIG=configs/phase7/combined_hidden_anti_overemit_librispeech.yaml
make train CONFIG=configs/phase7/combined_hidden_anti_overemit_librispeech.yaml
make test CONFIG=configs/phase7/combined_hidden_anti_overemit_librispeech.yaml
```

```bash
make data CONFIG=configs/phase7/hidden_states_selection_v2_librispeech.yaml
make train CONFIG=configs/phase7/hidden_states_selection_v2_librispeech.yaml
make test CONFIG=configs/phase7/hidden_states_selection_v2_librispeech.yaml
```

```bash
make data CONFIG=configs/phase7/hidden_states_kd_stronger_librispeech.yaml
make train CONFIG=configs/phase7/hidden_states_kd_stronger_librispeech.yaml
make test CONFIG=configs/phase7/hidden_states_kd_stronger_librispeech.yaml
```

```bash
make data CONFIG=configs/phase7/hidden_states_underemit_librispeech.yaml
make train CONFIG=configs/phase7/hidden_states_underemit_librispeech.yaml
make test CONFIG=configs/phase7/hidden_states_underemit_librispeech.yaml
```

```bash
make data CONFIG=configs/phase7/hidden_states_aug_librispeech.yaml
make train CONFIG=configs/phase7/hidden_states_aug_librispeech.yaml
make test CONFIG=configs/phase7/hidden_states_aug_librispeech.yaml
```

```bash
make data CONFIG=configs/phase7/hidden_states_depth4_librispeech.yaml
make train CONFIG=configs/phase7/hidden_states_depth4_librispeech.yaml
make test CONFIG=configs/phase7/hidden_states_depth4_librispeech.yaml
```

```bash
make data CONFIG=configs/phase7/hidden_states_forensics_librispeech.yaml
make train CONFIG=configs/phase7/hidden_states_forensics_librispeech.yaml
make test CONFIG=configs/phase7/hidden_states_forensics_librispeech.yaml
```

```bash
make data CONFIG=configs/phase7/hidden_states_patch2_relaxed_librispeech.yaml
make train CONFIG=configs/phase7/hidden_states_patch2_relaxed_librispeech.yaml
make test CONFIG=configs/phase7/hidden_states_patch2_relaxed_librispeech.yaml
```

```bash
make data CONFIG=configs/phase7/hidden_states_patch2_kd_warmup_librispeech.yaml
make train CONFIG=configs/phase7/hidden_states_patch2_kd_warmup_librispeech.yaml
make test CONFIG=configs/phase7/hidden_states_patch2_kd_warmup_librispeech.yaml
```

## Criteres de validation

- `make data` doit refuser toute configuration ASR incoherente qui croppe l'audio tout en gardant la transcription complete ;
- les artefacts de test doivent rapporter `WER`, `accuracy`, `blank_ratio`,
  `empty_pred_ratio`, `pred_to_ref_char_ratio`, temps, memoire GPU et nombre de parametres ;
- `best` et `final` doivent etre compares systematiquement pour eviter les faux meilleurs checkpoints degeneres.
- `P7F01` a etabli deux contraintes structurelles a corriger :
  - le student baseline est a environ `100 Hz` contre `50 Hz` pour le teacher ;
  - le train reel apres filtres est tombe a `127` exemples ;
  `P7R01` est la reponse experimentale minimale a ces deux constats.
- `P7R01` a ensuite montre que ces corrections structurelles, sans amortissement
  de la KD, menent a un blank collapse integral et reproductible ;
  `P7R02` teste donc une KD plus faible au debut, sans revenir en arriere sur
  `patch_time=2` ni sur le dataset plus large.
