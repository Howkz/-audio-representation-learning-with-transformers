Ingénierie Avancée pour l'Apprentissage AutomatiqueProjet : Apprentissage de Représentation Audio avec des TransformersLe projet porte sur l’apprentissage auto-supervisé à partir de données audio. À l’instar des TPs consacrés à l’adaptation de l’architecture Transformer à différents types de données, la première implémentation à réaliser concerne le bloc de plongement (embedding) des données audio.Il est fortement recommandé d’avoir terminé l’ensemble des TPs de l’UE avant de commencer ce projet. Il est également conseillé de débuter par le Jupyter notebook dédié aux données audio afin de bien comprendre les spécificités du pré-entraînement et de la représentation des signaux.L’évaluation du projet portera principalement sur l’implémentation du Transformer audio ainsi que sur les différents choix techniques effectués, notamment :la stratégie de patch embedding,le type d’entrée (onde brute, spectrogramme, log-Mel, etc.),la stratégie de pré-entraînement (MAE, augmentations, etc.),les optimisations en termes de temps de calcul et d’empreinte mémoire,ainsi que toute autre décision architecturale pertinente.Vous devrez respecter la méthodologie du template présenté en TP et y intégrer l’ensemble de vos composants de manière modulaire et reproductible.Il vous est également demandé de rédiger un rapport (.pdf) structuré, justifiant et présentant l’ensemble des éléments mis en place. Le rapport ne doit pas contenir de captures d’écran de code. Il est attendu que vous présentiez les résultats de plusieurs modèles évalués à l’aide de métriques reconnues. Enfin, vos choix d’implémentation devront être motivés par de courtes expérimentations mettant en évidence leurs effets empiriques.Groupes : 2 à 3 étudiants (aucun monôme)Rendu : nom1_nom2.zipContrainte d'évaluation : vdigpuAvant de commencer votre projetComme lors des sessions précédentes, nous travaillerons avec git pour le contrôle de version et la maintenance de notre projet. Dans un premier temps, vous devez organiser chaque session sous un dépôt (à l'emplacement de votre choix) nommé daaa. Chaque session pratique doit être numérotée comme suit : daaa/tpn, où n est le numéro de la session.Allez d'abord sur votre tableau de bord principal (https://git.unicaen.fr/dashboard/home). Cliquez ensuite sur le bouton Projects, dans le menu de navigation à gauche. Appuyez maintenant sur le bouton New project et créez un projet nommé daaa-project (où n est le numéro de la session pratique). Note : vous pouvez laisser l'initialisation du README.Copiez maintenant votre URL de clonage, qui devrait avoir le format suivant :https://git.unicaen.fr/votre_nom_utilisateur/daaa-project.gitUne routine habituelle pour chaque session pratique devrait maintenant être :# aller dans votre dépôt principal
cd daaa

# cloner votre nouveau dépôt git
git clone [https://git.unicaen.fr/votre_nom_utilisateur/daaa-tpn.git](https://git.unicaen.fr/votre_nom_utilisateur/daaa-tpn.git)

# aller dans la session du TP
cd daaa-tpn

# préparer le chargement du template depuis le git de l'UE
git remote add template [https://git.unicaen.fr/pantin241/daaa-template-ml](https://git.unicaen.fr/pantin241/daaa-template-ml)
git fetch template

# depuis la branche principale, télécharger tous les fichiers
git archive --format tar template/main | tar -x

# nettoyage
git remote remove template

# ajouter l'initialisation du template à l'historique git
git add .
git commit
Par défaut, l'éditeur de texte VI s'ouvre à ce moment-là. Tapez :feat (initialization): init project with templateEnsuite, appuyez sur : $CTRL+X -> y -> ENTER$Vous êtes maintenant prêt à commencer votre projet.Modalités du projetDans ce projet, nous vous demanderons d'implémenter un modèle audio pré-entraîné basé sur l'architecture des transformers. En suivant la méthodologie des sessions pratiques, nous souhaitons que vous implémentiez un dossier src/ qui recycle autant que possible les composants déjà développés de vos transformers. Un bloc important à adapter sera le AudioPatchEmbedding.Vous serez évalués sur votre capacité à structurer votre code, à tirer parti de tous les concepts du cours, à réutiliser les connaissances des sessions pratiques et sur votre pédagogie pour présenter les résultats et les composants importants de votre approche. Un autre point important est la façon dont vous avez réussi à utiliser le template que nous avons vu lors des sessions pratiques.ImportantVous soumettrez l'intégralité de votre dépôt de projet sur ecampus avec le standard suivant :daaa_nomduprojet1_nom2.zipLes checkpoints et les données ne doivent pas être inclus !TâcheTravail attenduOrganisation du projetSuivez la structure et les conventions du dépôt template-ml pour la reproductibilité et la modularité.make dataTéléchargez, prétraitez et préparez entièrement tous les jeux de données dans le répertoire data/. Vous pouvez utiliser src/data/dataset.py et src/data/features.py pour implémenter le chargement des jeux de données, le filtrage, le rééchantillonnage, l'extraction de caractéristiques et la mise en cache.make trainmake testImplémentez toutes les procédures d'entraînement, y compris le pré-entraînement auto-supervisé et le fine-tuning en aval. Sauvegardez les checkpoints du modèle et assurez-vous que l'entraînement peut être repris. Toutes les exigences de modélisation doivent être gérées à cette étape.Exécutez tous les protocoles d'évaluation et rapportez les résultats soit sur la sortie standard (stdout) soit dans un fichier results.txt. Pour chaque tâche en aval, les résultats doivent être moyennés sur 5 exécutions et rapportés avec l'écart type.Rapport finalRédigez un rapport décrivant les choix méthodologiques, les stratégies d'optimisation et les décisions de conception. Le rapport doit se concentrer sur les techniques et les compromis plutôt que sur les détails du code, et inclure une petite étude d'ablation (par exemple, comparaison avec/sans embeddings positionnels sinusoïdaux sur deux tâches en aval, nombre de blocs, etc...).Contrainte de groupeLe projet doit être réalisé en groupes de 2 ou 3 étudiants. Les projets individuels (groupes de 1) ne seront pas acceptés.Contrainte d'environnementPlus d'informationsLe vdigpu sera utilisé comme environnement d'évaluation. Toutes les expériences et tous les scripts doivent s'exécuter correctement sur cette plateforme. Tout code qui ne parvient pas à s'exécuter sans erreurs sur vdigpu ne recevra aucun point.Parmi les optimisations que vous avez abordées, l'apprentissage automatique frugal (frugal machine learning) est également un composant important qui sera évalué. Ainsi, le temps d'inférence et le temps d'entraînement, par exemple, seront évalués.Critères d'évaluationLes points dans le tableau sont fournis à titre indicatif uniquement.Critère d'évaluationPointsLe code ne fonctionne pas sur vdigpu-15Implémentation du Transformer Audio/3Objectif de pré-entraînement et stratégie de masquage/3Optimisations post-hoc (frugalité, augmentation de données,...)/3Respect de la structure du template et des routines du Makefile/3Tâches en aval (fine-tuning, ablations, résultats expérimentaux,...)/3Qualité du rapport : pédagogie, explications, justifications techniques, présentation des résultats et de l'ablation/5Bonus 1 : Benchmark sur la synthèse vocale (TTS - text-to-speech)/2Bonus 2 : Implémentation du papier ICLR 2026/5[Image du schéma d'architecture Voxtral de MistralAI]Figure 1 : Voxtral de MistralAI (Liu et. al. 2025)1. Aperçu du projetL'objectif de ce projet est de concevoir, d'entraîner et d'évaluer un modèle de représentation audio à usage général basé sur un encodeur Transformer. L'objectif principal n'est pas une tâche unique, mais la capacité d'une représentation audio apprise à être transférée efficacement à travers plusieurs tâches audio en aval. Il est attendu que vous exploriez des stratégies d'apprentissage supervisé, de pré-entraînement et d'apprentissage auto-supervisé, tout en respectant une contrainte computationnelle stricte d'un seul GPU avec 6 Go de VRAM.Contrairement à la vision ou au traitement du langage naturel, l'audio se situe à l'intersection des signaux continus et de la structure discrète. Ce projet vise à fournir une compréhension approfondie de la façon dont les données audio sont gérées dans les systèmes d'apprentissage automatique, comment les Transformers peuvent être adaptés à cette modalité, et comment la qualité de la représentation peut être évaluée au-delà d'un seul benchmark.2. Modèle de représentation audioDans ce projet, nous concevons un modèle de représentation audio basé sur un encodeur Transformer, en suivant les mêmes principes architecturaux que lors des sessions pratiques précédentes. L'objectif de ce modèle est d'apprendre des représentations transférables à partir de signaux audio bruts qui peuvent être réutilisées pour plusieurs tâches en aval, notamment la reconnaissance automatique de la parole (ASR) et la synthèse vocale (TTS).Les signaux audio sont d'abord convertis en représentations temps-fréquence (spectrogrammes log-Mel), résultant en des séquences d'entrée :$X \in \mathbb{R}^{T \times M}$où $T$ désigne le nombre de trames temporelles et $M$ le nombre de bandes de fréquences Mel. Chaque trame temporelle est projetée dans un espace latent de dimension $d$ à l'aide d'une couche d'intégration linéaire (linear embedding layer), produisant une séquence de tokens appropriée pour le traitement par le Transformer.Le cœur du modèle est un encodeur Transformer composé de blocs de self-attention multi-têtes (multi-head self-attention) et de feed-forward. Pendant le pré-entraînement, cet encodeur est couplé à un décodeur léger (dans l'esprit des auto-encodeurs masqués - masked autoencoders) pour reconstruire les portions masquées de l'entrée. Ce décodeur n'est utilisé que pendant le pré-entraînement et doit être jeté par la suite. L'encodeur pré-entraîné constitue la représentation audio réutilisable partagée entre les tâches en aval.Tous les choix architecturaux doivent respecter la contrainte matérielle d'un seul GPU avec environ 6 Go de VRAM.3. Jeux de donnéesTous les jeux de données doivent être chargés en utilisant la bibliothèque datasets du Hub Hugging Face. Les téléchargements externes ou les chargeurs de jeux de données personnalisés ne sont pas autorisés. Un taux d'échantillonnage cohérent (par exemple, 16 kHz) doit être appliqué à tous les jeux de données pour garantir une extraction de caractéristiques cohérente et une comparaison équitable des modèles. Les clips audio doivent être rognés ou complétés (padding) à une durée maximale fixe afin de contrôler l'utilisation de la mémoire et la taille du batch. Vous êtes libre de les utiliser pour n'importe quelle étape, mais vous devez être clair sur la façon dont vous les avez préparés et utilisés.3.1 Pré-entraînement : Fluent Speech CommandsEn tant que jeu de données de pré-entraînement auto-supervisé, vous pouvez utiliser le jeu de données Fluent Speech Commands. Ce jeu de données se compose d'énoncés parlés courts correspondant à des commandes vocales structurées. Vous prendrez la répartition d'origine (original split).Pour les besoins de ce projet, seule la modalité audio doit être utilisée pendant le pré-entraînement. Toutes les étiquettes doivent être ignorées. L'objectif est d'apprendre des représentations acoustiques générales en utilisant une stratégie de prédiction masquée sur les spectrogrammes log-Mel.Le jeu de données est relativement petit et contient des énoncés courts et propres. Sa variabilité acoustique contrôlée permet une convergence rapide et une analyse claire du comportement de l'apprentissage des représentations.Vous devez charger le jeu de données en utilisant la bibliothèque datasets de Hugging Face et appliquer un prétraitement cohérent (normalisation du taux d'échantillonnage, rognage et padding). L'encodeur pré-entraîné obtenu à partir de cette étape sera ensuite transféré vers les tâches ASR en aval en utilisant des jeux de données étiquetés.from datasets import load_dataset
dataset = load_dataset("Codec-SUPERB/fluent_speech_commands_synth")
3.2 Pré-entraînement/ASR : LibriSpeechEn complément et/ou comme alternative pour le pré-entraînement auto-supervisé, vous utiliserez le jeu de données LibriSpeech (audio non étiqueté uniquement). Seule la modalité audio est utilisée pour le pré-entraînement ; les transcriptions textuelles seraient utilisées pour la tâche en aval. Un sous-ensemble réduit de LibriSpeech (par exemple, train-clean-100 ou un sous-ensemble davantage filtré en fonction de la durée) doit être utilisé pour satisfaire les contraintes de mémoire et de temps d'exécution. Concernant le pré-entraînement, le modèle est entraîné en utilisant un objectif de prédiction masquée, où un sous-ensemble de trames temporelles (ou régions temps-fréquence) est masqué et reconstruit. La perte de reconstruction est calculée uniquement sur les régions masquées. Cette étape vise à apprendre des représentations acoustiques à usage général sans aucune supervision textuelle.from datasets import load_dataset
dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.clean.100")
3.3 ASR : Vox PopuliComme deuxième benchmark de reconnaissance automatique de la parole, vous pouvez utiliser le jeu de données Vox Populi. Vox Populi se compose d'enregistrements vocaux issus des sessions du Parlement européen associés à des transcriptions normalisées, et est largement utilisé dans les études ASR à grande échelle et l'apprentissage auto-supervisé.Par rapport à LibriSpeech, Vox Populi présente des énoncés plus longs, une plus grande diversité de locuteurs et des conditions d'enregistrement plus réalistes. La parole est moins contrôlée et inclut souvent des bruits de fond et une variabilité dans le style d'élocution. Ce jeu de données fournit donc un test robuste de la solidité du domaine et du transfert de représentation.Le jeu de données doit être chargé à l'aide de la bibliothèque datasets de Hugging Face. En raison de la durée potentiellement longue des enregistrements, vous êtes tenu d'appliquer un filtrage de durée strict, des stratégies de rognage ou de découpage (chunking) pour respecter la contrainte de 6 Go de VRAM. Le fine-tuning doit suivre le même protocole basé sur CTC que les autres jeux de données ASR, et l'évaluation doit être rapportée en utilisant le Taux d'Erreur sur les Mots (Word Error Rate - WER).from datasets import load_dataset
dataset = load_dataset("facebook/voxpopuli", "en", split="train")
3.4 ASR Long/TTS : Gemini SpeechLe TTS est optionnel, mais vous pouvez utiliser ce jeu de données pour n'importe quelle phase de votre modélisation. En guise de bonus optionnel, vous pouvez réaliser une expérience de synthèse vocale en utilisant le jeu de données gemini-flash-2.0-speech. Ce jeu de données contient des textes appariés et des enregistrements vocaux d'un seul locuteur et est couramment utilisé dans la recherche TTS. Vous devez implémenter une architecture encodeur-décodeur légère avec cross-attention, en réutilisant l'encodeur audio pré-entraîné et un décodeur basé sur Transformer. Le modèle est entraîné pour prédire des caractéristiques acoustiques (spectrogrammes log-Mel) à partir de séquences de texte. La reconstruction des formes d'onde brutes n'est pas requise. Cette tâche est optionnelle et évaluée comme un bonus.from datasets import load_dataset
dataset = load_dataset("shb777/gemini-flash-2.0-speech")
4. Comment évaluer les tâches ASR et TTSPour les tâches ASR, l'évaluation doit être effectuée en utilisant le Taux d'Erreur sur les Mots (WER) sur une division de validation ou de test. Vous pouvez utiliser torchmetrics. Les résultats doivent être rapportés comme la moyenne et l'écart type sur au moins cinq exécutions, en utilisant différentes graines aléatoires (random seeds).Pour la tâche TTS optionnelle, l'évaluation est principalement basée sur les caractéristiques et qualitative. Vous devez rapporter les pertes de reconstruction (par exemple, perte $L_1$ sur les spectrogrammes log-Mel) et inclure des comparaisons visuelles entre les spectrogrammes prédits et la vérité terrain (ground-truth). Une discussion claire des résultats qualitatifs et des limites est attendue.Dans tous les cas, vous devez explicitement rapporter l'utilisation de la mémoire GPU, la taille de batch effective, et toutes les stratégies d'optimisation utilisées pour respecter la contrainte de 6 Go de VRAM.Une évaluation plus approfondie pourrait considérer la robustesse de la reconstruction face au niveau de bruit, aux signaux nuls ou à différents types d'espaces d'entrée (log-Mel, wav brut, ...).Pour tous les résultats importants, vous enregistrerez les résultats dans un tableau similaire à celui-ci :ModèleArchitectureAdaptationWERAcc.Temps d'inférenceTaille (#par.)Nom du modèleEncodeur-Décodeur*$0.42 \pm 0.01$$0.42 \pm 0.01$$42.0s$$4.2B$Nom du modèleEncodeur-Décodeur*$0.42 \pm 0.01$$0.42 \pm 0.01$$42.0s$$4.2B$Nom du modèleEncodeur-Décodeur*$0.42 \pm 0.01$$0.42 \pm 0.01$$42.0s$$4.2B$Nom du modèleDécodeur uniquement*$0.42 \pm 0.01$$0.42 \pm 0.01$$42.0s$$4.2B$Nom du modèleDécodeur uniquement*$0.42 \pm 0.01$$0.42 \pm 0.01$$42.0s$$4.2B$* : le linear-probing (sondage linéaire) est un exemple.5. Utilisation de modèles pré-entraînésBien que les modèles pré-entraînés ne puissent pas être utilisés comme backbones (structures de base), les étudiants sont encouragés à utiliser des modèles audio pré-entraînés existants comme professeurs pour la distillation de connaissances. De tels modèles peuvent fournir des cibles douces (soft targets) ou des représentations intermédiaires qui aident à stabiliser l'entraînement et à améliorer les performances.Une partie de vos ablations et résultats principaux abordera le problème de la frugalité en affichant les gains et éventuellement les inconvénients.6. Quelques conseils techniques (indices, pas des exigences exhaustives)Voici une structure de base suggérée pour guider votre implémentation (tout le code fourni ci-dessous est intégralement repris du sujet) :from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn

def decode_audio(
    audio_dict: Dict[str, Any],
    target_sr: int
) -> Tuple[torch.Tensor, int]:
    # un exemple
    pass

def extract_logmel(
    waveform: torch.Tensor,
    sr: int,
    n_mels: int,
    win_length: int,
    hop_length: int
) -> torch.Tensor:
    pass

def make_mae_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float,
    device: torch.device
) -> torch.Tensor:
    pass

def pad_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    pass

class AudioPatchEmbedding(nn.Module):
    pass

class AudioTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        patch_size: int,
        max_len: int,
        pos_embed: str
    ):
        super().__init__()
        pass

class AudioMAEPretrain(nn.Module):
    def __init__(
        self,
        encoder: AudioTransformerEncoder,
        n_mels: int,
        dec_dim: int,
        dec_depth: int,
        dec_heads: int,
        dropout: float,
    ):
        super().__init__()
        pass

    def forward(
        self,
        x_logmel: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pass

class AudioTransformerCTC(nn.Module):
    def __init__(
        self,
        encoder: AudioTransformerEncoder,
        vocab_size: int,
    ):
        super().__init__()
        pass

    def forward(
        self,
        x_logmel: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
