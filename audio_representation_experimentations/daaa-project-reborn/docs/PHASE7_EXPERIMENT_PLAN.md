# Phase 7 - Pipeline ASR distille propre

## Intention

La phase 7 ouvre une nouvelle voie propre, en parallele des phases 1 a 6, pour produire
un pipeline ASR plus conforme au sujet et plus defensable pour le rendu :

- `make data`, `make train`, `make test` restent les points d'entree uniques ;
- la distillation ASR devient le chemin principal ;
- LibriSpeech `clean` devient le benchmark critique ;
- VoxPopuli sort du chemin critique tant que la chaine LibriSpeech n'est pas stabilisee ;
- le MAE est conserve comme option, mais il n'est plus necessaire a la voie principale.

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

## Choix techniques

- Student :
  - log-Mel 80 bandes
  - embedding lineaire par trame (`patch_time=1`)
  - encodeur Transformer avec `src_key_padding_mask`
  - tete CTC simple

- Distillation externe :
  - teacher utilise uniquement pendant l'entrainement
  - cible principale : logits CTC Wav2Vec2 remappes sur le vocabulaire caractere du student
  - perte : `lambda_ctc * CTC + lambda_kd * KL`

- Distillation interne :
  - meme interface teacher
  - teacher recharge depuis `ctc_best.pt` ou `ctc_final.pt`

- Selection de checkpoint :
  - `WER`
  - penalite `empty_pred_ratio`
  - penalite de predictions trop courtes
  - bonus d'`accuracy`

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

## Criteres de validation

- `make data` doit refuser toute configuration ASR incoherente qui croppe l'audio tout en gardant la transcription complete ;
- les artefacts de test doivent rapporter `WER`, `accuracy`, `blank_ratio`,
  `empty_pred_ratio`, `pred_to_ref_char_ratio`, temps, memoire GPU et nombre de parametres ;
- `best` et `final` doivent etre compares systematiquement pour eviter les faux meilleurs checkpoints degeneres.
