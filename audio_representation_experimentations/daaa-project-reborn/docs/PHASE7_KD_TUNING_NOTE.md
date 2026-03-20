# Phase 7 - Note sur la stabilite du tuning KD

## Objet

Cette note fige le constat suivant avant d'ouvrir une nouvelle famille de patchs :
la calibration actuelle de la hidden-state KD autour de `P7C01` semble deja etre
dans une zone stable, au sens ou les recalibrages simples ne produisent pas
d'amelioration exploitable.

## Resultat de reference

Run de reference :

- `P7C01`
- config : `configs/phase7/hidden_states_librispeech.yaml`

Resultats test `best` :

- `WER = 1.0055`
- `accuracy = 0.07133`
- `blank_ratio = 0.09937`
- `empty_pred_ratio = 0.0`
- `pred_to_ref_char_ratio = 0.10223`

## Recalibrage teste

Run de recalibrage :

- `P7C05`
- config : `configs/phase7/hidden_states_kd_stronger_librispeech.yaml`
- changement unique : `lambda_kd` passe de `0.20` a `0.30`

Resultats test `best` :

- `WER = 1.0048`
- `accuracy = 0.07106`
- `blank_ratio = 0.09901`
- `empty_pred_ratio = 0.0`
- `pred_to_ref_char_ratio = 0.10190`

## Lecture

Le recalibrage ne change pas le regime :

- le modele ne retombe pas dans le `blank collapse` ;
- il reste fortement sous-emissif ;
- l'accuracy ne monte pas ;
- les variations observees sont trop faibles pour etre interpretees comme un vrai gain.

Autrement dit :

- `P7C05` n'est pas meilleur que `P7C01` ;
- le simple ajustement de `lambda_kd` ne deplace pas utilement le point d'equilibre.

## Conclusion

La bonne lecture n'est pas :

- "il faut encore micro-tuner `lambda_kd`".

La bonne lecture est :

- "la base `P7C01` est localement stable, et le prochain levier doit venir d'ailleurs".

Les pistes rationnelles suivantes sont donc :

1. couvrir ce qui manque vis-a-vis des consignes du TP ;
2. ajouter une vraie augmentation de donnees ;
3. faire une petite ablation propre et simple ;
4. documenter correctement le sujet du chunking/VoxPopuli.
