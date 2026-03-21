# Submission Checklist

- [ ] `make compliance CONFIG=configs/final_tp/core_librispeech.yaml` passe.
- [ ] `make suite-tp` a tourné jusqu'au bout.
- [ ] `make report-tp` a généré les trois tableaux finaux.
- [ ] Les résultats 5 seeds contiennent bien `mean +- std`.
- [ ] Les tableaux couvrent FSC linear probe, LibriSpeech ASR et VoxPopuli ASR.
- [ ] Le rapport PDF final est présent dans `docs/` ou à la racine avant zippage.
- [ ] `data/processed/` est vide ou supprimé.
- [ ] `outputs/` est vide ou supprimé.
- [ ] Aucun cache lourd Hugging Face local n'est embarqué.
- [ ] `results/` ne contient que les agrégats utiles à la remise.
- [ ] Les fichiers temporaires et logs locaux ont été exclus.
- [ ] `make package CONFIG=configs/final_tp/core_librispeech.yaml` a produit un zip propre.
- [ ] Le zip final a été ouvert et vérifié manuellement avant dépôt.
