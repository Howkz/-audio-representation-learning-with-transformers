# TP 2h Stage Plan

Objectif : exécuter une campagne conforme au TP par **petits jobs pair/triple** d'environ 2h chacun, au lieu d'une suite monolithique irréaliste sur `vdigpu`.

Principe :
- `5 seeds` conservés au total : `42, 123, 456, 789, 1024`
- shard `pair` : `42, 123`
- shard `triple` : `456, 789, 1024`
- une **mécanique clé par stage**
- datasets et budgets réduits, mais comparaisons cohérentes et identiques à l'intérieur d'un stage

## Couverture du sujet

Ce plan couvre :
- pré-entraînement masqué `MAE`
- fine-tuning aval `ASR`
- augmentation de données
- petite ablation `patch_time=4 vs 2`
- `5 seeds`
- deux benchmarks `LibriSpeech` et `VoxPopuli`
- contraintes `vdigpu`

Ce plan ne couvre pas :
- linear probing FSC dans la campagne finale compacte
- distillation comme résultat principal
- comparaison `spectrogram vs log-Mel`

## Stages

LibriSpeech :
- `R0_LIBRI` : baseline CTC sans MAE
- `R1_LIBRI` : MAE + CTC
- `R2_LIBRI` : MAE + CTC + augmentations
- `R4_LIBRI` : MAE + CTC avec `patch_time=2`

VoxPopuli :
- `R0_VOX` : baseline CTC sans MAE
- `R1_VOX` : MAE + CTC
- `R2_VOX` : MAE + CTC + augmentations
- `R4_VOX` : MAE + CTC avec `patch_time=2`

## Commandes

VDI 1 :

```bash
make suite-tp-2h-pair STAGE=R1_LIBRI
```

VDI 2 :

```bash
make suite-tp-2h-triple STAGE=R1_LIBRI
```

Ou directement :

```bash
python scripts/run_tp2h_pair.py --stage R1_LIBRI --verbose
python scripts/run_tp2h_triple.py --stage R1_LIBRI --verbose
```

## Ordre recommandé

1. `R0_LIBRI`
2. `R1_LIBRI`
3. `R2_LIBRI`
4. `R4_LIBRI`
5. `R0_VOX`
6. `R1_VOX`
7. `R2_VOX`
8. `R4_VOX`

## Budget visé

Les configs `final_tp_2h` ont été réduites pour viser des jobs pair/triple d'environ 2h :
- prétrain FSC `original` : `1000` steps
- fine-tune Libri : `700` steps
- fine-tune Vox : `600` steps
- sous-ensembles datasets réduits

Ce n'est pas une garantie stricte. Le but est de rendre la campagne **opérationnelle** sur `vdigpu`, pas de maximiser la performance brute.
