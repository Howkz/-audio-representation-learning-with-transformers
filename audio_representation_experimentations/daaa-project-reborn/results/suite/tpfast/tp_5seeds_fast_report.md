\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{lmodern}
\usepackage{geometry}
\usepackage{setspace}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{float}
\usepackage{xcolor}
\usepackage{hyperref}
\geometry{margin=2.4cm}
\onehalfspacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.35em}
\hypersetup{
  colorlinks=true,
  linkcolor=blue!40!black,
  urlcolor=blue!40!black,
  citecolor=blue!40!black
}

\title{Campagne TP 5 seeds ultra-compacte}
\author{CLAIR Mael, SABINE Hugo, CHANTELOUP William}
\date{\today}

\begin{document}

\begin{titlepage}
\centering
{\Large Master 2 Algorithmiques et Syst?mes Intelligents\par}
\vspace{0.35cm}
{\large Projet DAAA\par}
\vspace{1.3cm}
{\LARGE\bfseries Campagne TP 5 seeds ultra-compacte\par}
\vspace{0.8cm}
{\large Suite unique rapide pour vdigpu\par}
\vfill
\begin{tabular}{ll}
\textbf{Étudiants} : & CLAIR Mael \\
 & SABINE Hugo \\
 & CHANTELOUP William \\
\textbf{Date} : & \today \\
\end{tabular}
\vfill
\end{titlepage}

\tableofcontents
\newpage

\section{Introduction}
Ce rapport présente une étude expérimentale progressive sur un encodeur Transformer audio pré-entraîné par MAE puis adapté en ASR via CTC.
L'objectif scientifique est triple: (i) vérifier l'apport du pré-entraînement MAE, (ii) identifier les compromis entre WER, temps d'inférence et mémoire GPU, (iii) produire un benchmark multi-variantes avec consolidation statistique finale.

Contraintes de conception: environnement vdigpu, budget de calcul borné (cible 24h GPU utile), exécution reproductible, et suivi frugal des ressources.

\paragraph{Choix globaux assumés}
\begin{itemize}
\item Suite unique compacte con?ue pour rester exploitable sur vdigpu.
\item Couvre les m?caniques obligatoires: MAE, augmentation, patch embedding, LibriSpeech, VoxPopuli.
\item Le linear probe FSC et la distillation annexe sont exclus de cette suite rapide.
\end{itemize}


\section{Questions de recherche et hypothèses globales}
\subsection{Questions de recherche}
\begin{enumerate}
\item \textbf{RQ1}: Le pré-entraînement MAE améliore-t-il la performance ASR (WER) par rapport à un entraînement CTC sans pré-entraînement ?
\item \textbf{RQ2}: Quel compromis est obtenu entre qualité (WER), coût temporel (runtime/débit) et coût matériel (mémoire GPU) selon la capacité du modèle ?
\item \textbf{RQ3}: Quels choix d'architecture (positionnel, stratégie de patching) et d'hyperparamètres MAE (mask ratio) sont les plus robustes ?
\item \textbf{RQ4}: Les conclusions restent-elles stables sous variabilité aléatoire (5 seeds sur les meilleures variantes) ?
\end{enumerate}

\subsection{Hypothèses globales justifiées}
\begin{itemize}
\item \textbf{H1 (apport MAE)}: une représentation auto-supervisée pré-entraînée améliore la généralisation ASR, car l'encodeur apprend des régularités acoustiques indépendantes de l'alignement texte.
\item \textbf{H2 (capacité)}: augmenter la capacité du Transformer peut réduire le WER, mais au prix d'une hausse de runtime et de mémoire.
\item \textbf{H3 (inductive bias)}: des choix de patching et d'encodage positionnel modifient le biais inductif; certains réglages peuvent mieux capturer la structure temps-fréquence.
\item \textbf{H4 (mask ratio)}: le ratio de masquage MAE gouverne la difficulté de prétexte; trop faible sous-contraint l'apprentissage, trop fort peut dégrader l'optimisation.
\end{itemize}


\section{Méthodologie générale}
\subsection{Définitions opératoires}
\begin{itemize}
\item \textbf{ASR (Automatic Speech Recognition)}: conversion d'un signal de parole en transcription texte.
\item \textbf{CTC (Connectionist Temporal Classification)}: fonction de perte permettant d'apprendre sans alignement frame-à-frame explicite.
\item \textbf{WER (Word Error Rate)}: métrique de transcription basée sur substitutions, insertions et suppressions.
\item \textbf{MAE (Masked AutoEncoder)}: pré-entraînement auto-supervisé par reconstruction de portions masquées du signal.
\item \textbf{TTS (Text-To-Speech)}: tâche inverse de l'ASR (texte vers audio), non incluse dans le cœur de cette suite.
\end{itemize}

\subsection{Pipeline}
Le protocole suit les étapes \texttt{make data}, \texttt{make train}, \texttt{make test}.
Chaque expérience est exécutée dans un namespace dédié, avec nettoyage des checkpoints intermédiaires après archivage des résultats, tout en conservant le cache de données.

\subsection{Politique dataset et périmètre}
\begin{itemize}
\item \textbf{Screening}: non spécifié.
\item \textbf{Consolidation finale}: non spécifié.
\item VoxPopuli reste une extension possible, hors chemin critique de la présente étude (contraintes disque/temps).
\end{itemize}

\subsection{Contrôle expérimental}
\begin{itemize}
\item Même pipeline de preprocessing audio pour toutes les variantes.
\item Même protocole d'évaluation et mêmes splits pour la comparabilité.
\item Une seule variable manipulée à la fois durant le screening autant que possible.
\item Checkpointing complet (modèle, optimiseur, scheduler, scaler AMP, état RNG) pour reprise fiable.
\end{itemize}

\subsection{Métriques}
Les métriques principales sont:
\begin{itemize}
\item WER (Word Error Rate) pour la performance ASR.
\item Temps d'inférence et débit (samples/s) pour la frugalité temporelle.
\item Mémoire GPU pic pour la frugalité matérielle.
\item Taille modèle (\#paramètres) pour l'analyse capacité/coût.
\end{itemize}

\subsection{Règles statistiques}
Le screening initial est réalisé en 1 seed par variante (E01--E08).
Une phase de sélection intermédiaire est ensuite exécutée sur le Top-5 du screening avec 2 seeds (42, 123).
La consolidation finale est réalisée sur 5 seeds pour les 3 meilleures variantes issues de la sélection (E09--E11), avec reporting moyenne~$\pm$~écart-type.
Les résultats finaux rapportés suivent la politique de consolidation finale déclarée dans la configuration de suite (sous-échantillon élargi sous contrainte temps).

\subsection{Règle de décision Top-3}
Le classement screening puis sélection est effectué par tri lexicographique sur:
\begin{enumerate}
\item WER (ascendant),
\item runtime d'inférence (ascendant),
\item mémoire GPU pic (ascendant).
\end{enumerate}


\section{Expérience R0\_LIBRI --- R0 LibriSpeech CTC rapide}
\subsection{Objectif scientifique local}
À préciser.

\subsection{Choix méthodologiques et justification}
\textbf{Phase}: final\_tp\_fast\\
\textbf{Seeds prévues}: 456, 789, 1024\\


\paragraph{Justification du plan}


\paragraph{Mécanisme scientifique visé}
À préciser.

\paragraph{Variables manipulées}
\begin{itemize}
\item \texttt{pretrain.mode=none}
\item \texttt{training.pretrain.enabled=False}
\end{itemize}

\subsection{Hypothèse causale et résultats attendus}
\begin{itemize}[leftmargin=1.2cm]
\item \textbf{Hypothèse}: 
\item \textbf{Mécanisme attendu}: À préciser.
\item \textbf{Tendance attendue}: À préciser.
\item \textbf{Critère de décision}: À préciser.
\end{itemize}

\subsection{Résultats}
\textit{Section volontairement laissée vide avant exécution.}

\begin{table}[H]
\centering
\caption{Résultats quantitatifs --- Expérience R0\_LIBRI (à compléter)}
\begin{tabular}{lcccc}
\toprule
Seed & WER $\downarrow$ & Runtime inf. (s) $\downarrow$ & Débit (samples/s) $\uparrow$ & Mémoire GPU (MB) $\downarrow$ \\
\midrule
\multicolumn{5}{c}{\textit{À compléter après exécution des runs.}} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Discussion des résultats}
\textit{Section volontairement laissée vide avant interprétation finale.}


\section{Expérience R1\_LIBRI --- R1 LibriSpeech MAE rapide}
\subsection{Objectif scientifique local}
À préciser.

\subsection{Choix méthodologiques et justification}
\textbf{Phase}: final\_tp\_fast\\
\textbf{Seeds prévues}: 456, 789, 1024\\


\paragraph{Justification du plan}


\paragraph{Mécanisme scientifique visé}
À préciser.

\paragraph{Variables manipulées}
\begin{itemize}
\item \textit{Aucune surcharge explicite (configuration de base conservée).}
\end{itemize}

\subsection{Hypothèse causale et résultats attendus}
\begin{itemize}[leftmargin=1.2cm]
\item \textbf{Hypothèse}: 
\item \textbf{Mécanisme attendu}: À préciser.
\item \textbf{Tendance attendue}: À préciser.
\item \textbf{Critère de décision}: À préciser.
\end{itemize}

\subsection{Résultats}
\textit{Section volontairement laissée vide avant exécution.}

\begin{table}[H]
\centering
\caption{Résultats quantitatifs --- Expérience R1\_LIBRI (à compléter)}
\begin{tabular}{lcccc}
\toprule
Seed & WER $\downarrow$ & Runtime inf. (s) $\downarrow$ & Débit (samples/s) $\uparrow$ & Mémoire GPU (MB) $\downarrow$ \\
\midrule
\multicolumn{5}{c}{\textit{À compléter après exécution des runs.}} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Discussion des résultats}
\textit{Section volontairement laissée vide avant interprétation finale.}


\section{Expérience R2\_LIBRI --- R2 LibriSpeech MAE+Aug rapide}
\subsection{Objectif scientifique local}
À préciser.

\subsection{Choix méthodologiques et justification}
\textbf{Phase}: final\_tp\_fast\\
\textbf{Seeds prévues}: 456, 789, 1024\\


\paragraph{Justification du plan}


\paragraph{Mécanisme scientifique visé}
À préciser.

\paragraph{Variables manipulées}
\begin{itemize}
\item \texttt{datasets.asr\_train.augmentations.enabled=True}
\item \texttt{datasets.asr\_train.augmentations.gain\_prob=0.5}
\item \texttt{datasets.asr\_train.augmentations.gain\_db\_max=6.0}
\item \texttt{datasets.asr\_train.augmentations.noise\_prob=0.35}
\item \texttt{datasets.asr\_train.augmentations.noise\_snr\_db\_min=20.0}
\item \texttt{datasets.asr\_train.augmentations.noise\_snr\_db\_max=35.0}
\item \texttt{datasets.asr\_train.augmentations.specaugment\_prob=0.35}
\item \texttt{datasets.asr\_train.augmentations.num\_time\_masks=2}
\item \texttt{datasets.asr\_train.augmentations.max\_time\_mask\_frames=16}
\item \texttt{datasets.asr\_train.augmentations.num\_freq\_masks=1}
\item \texttt{datasets.asr\_train.augmentations.max\_freq\_mask\_bins=8}
\end{itemize}

\subsection{Hypothèse causale et résultats attendus}
\begin{itemize}[leftmargin=1.2cm]
\item \textbf{Hypothèse}: 
\item \textbf{Mécanisme attendu}: À préciser.
\item \textbf{Tendance attendue}: À préciser.
\item \textbf{Critère de décision}: À préciser.
\end{itemize}

\subsection{Résultats}
\textit{Section volontairement laissée vide avant exécution.}

\begin{table}[H]
\centering
\caption{Résultats quantitatifs --- Expérience R2\_LIBRI (à compléter)}
\begin{tabular}{lcccc}
\toprule
Seed & WER $\downarrow$ & Runtime inf. (s) $\downarrow$ & Débit (samples/s) $\uparrow$ & Mémoire GPU (MB) $\downarrow$ \\
\midrule
\multicolumn{5}{c}{\textit{À compléter après exécution des runs.}} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Discussion des résultats}
\textit{Section volontairement laissée vide avant interprétation finale.}


\section{Expérience R4\_LIBRI --- R4 LibriSpeech Patch-2 rapide}
\subsection{Objectif scientifique local}
À préciser.

\subsection{Choix méthodologiques et justification}
\textbf{Phase}: final\_tp\_fast\\
\textbf{Seeds prévues}: 456, 789, 1024\\


\paragraph{Justification du plan}


\paragraph{Mécanisme scientifique visé}
À préciser.

\paragraph{Variables manipulées}
\begin{itemize}
\item \texttt{model.patch\_time=2}
\item \texttt{model.max\_len=2048}
\end{itemize}

\subsection{Hypothèse causale et résultats attendus}
\begin{itemize}[leftmargin=1.2cm]
\item \textbf{Hypothèse}: 
\item \textbf{Mécanisme attendu}: À préciser.
\item \textbf{Tendance attendue}: À préciser.
\item \textbf{Critère de décision}: À préciser.
\end{itemize}

\subsection{Résultats}
\textit{Section volontairement laissée vide avant exécution.}

\begin{table}[H]
\centering
\caption{Résultats quantitatifs --- Expérience R4\_LIBRI (à compléter)}
\begin{tabular}{lcccc}
\toprule
Seed & WER $\downarrow$ & Runtime inf. (s) $\downarrow$ & Débit (samples/s) $\uparrow$ & Mémoire GPU (MB) $\downarrow$ \\
\midrule
\multicolumn{5}{c}{\textit{À compléter après exécution des runs.}} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Discussion des résultats}
\textit{Section volontairement laissée vide avant interprétation finale.}


\section{Expérience R0\_VOX --- R0 VoxPopuli CTC rapide}
\subsection{Objectif scientifique local}
À préciser.

\subsection{Choix méthodologiques et justification}
\textbf{Phase}: final\_tp\_fast\\
\textbf{Seeds prévues}: 456, 789, 1024\\


\paragraph{Justification du plan}


\paragraph{Mécanisme scientifique visé}
À préciser.

\paragraph{Variables manipulées}
\begin{itemize}
\item \texttt{pretrain.mode=none}
\item \texttt{training.pretrain.enabled=False}
\end{itemize}

\subsection{Hypothèse causale et résultats attendus}
\begin{itemize}[leftmargin=1.2cm]
\item \textbf{Hypothèse}: 
\item \textbf{Mécanisme attendu}: À préciser.
\item \textbf{Tendance attendue}: À préciser.
\item \textbf{Critère de décision}: À préciser.
\end{itemize}

\subsection{Résultats}
\textit{Section volontairement laissée vide avant exécution.}

\begin{table}[H]
\centering
\caption{Résultats quantitatifs --- Expérience R0\_VOX (à compléter)}
\begin{tabular}{lcccc}
\toprule
Seed & WER $\downarrow$ & Runtime inf. (s) $\downarrow$ & Débit (samples/s) $\uparrow$ & Mémoire GPU (MB) $\downarrow$ \\
\midrule
\multicolumn{5}{c}{\textit{À compléter après exécution des runs.}} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Discussion des résultats}
\textit{Section volontairement laissée vide avant interprétation finale.}


\section{Expérience R1\_VOX --- R1 VoxPopuli MAE rapide}
\subsection{Objectif scientifique local}
À préciser.

\subsection{Choix méthodologiques et justification}
\textbf{Phase}: final\_tp\_fast\\
\textbf{Seeds prévues}: 456, 789, 1024\\


\paragraph{Justification du plan}


\paragraph{Mécanisme scientifique visé}
À préciser.

\paragraph{Variables manipulées}
\begin{itemize}
\item \textit{Aucune surcharge explicite (configuration de base conservée).}
\end{itemize}

\subsection{Hypothèse causale et résultats attendus}
\begin{itemize}[leftmargin=1.2cm]
\item \textbf{Hypothèse}: 
\item \textbf{Mécanisme attendu}: À préciser.
\item \textbf{Tendance attendue}: À préciser.
\item \textbf{Critère de décision}: À préciser.
\end{itemize}

\subsection{Résultats}
\textit{Section volontairement laissée vide avant exécution.}

\begin{table}[H]
\centering
\caption{Résultats quantitatifs --- Expérience R1\_VOX (à compléter)}
\begin{tabular}{lcccc}
\toprule
Seed & WER $\downarrow$ & Runtime inf. (s) $\downarrow$ & Débit (samples/s) $\uparrow$ & Mémoire GPU (MB) $\downarrow$ \\
\midrule
\multicolumn{5}{c}{\textit{À compléter après exécution des runs.}} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Discussion des résultats}
\textit{Section volontairement laissée vide avant interprétation finale.}


\section{Expérience R2\_VOX --- R2 VoxPopuli MAE+Aug rapide}
\subsection{Objectif scientifique local}
À préciser.

\subsection{Choix méthodologiques et justification}
\textbf{Phase}: final\_tp\_fast\\
\textbf{Seeds prévues}: 456, 789, 1024\\


\paragraph{Justification du plan}


\paragraph{Mécanisme scientifique visé}
À préciser.

\paragraph{Variables manipulées}
\begin{itemize}
\item \texttt{datasets.asr\_train.augmentations.enabled=True}
\item \texttt{datasets.asr\_train.augmentations.gain\_prob=0.5}
\item \texttt{datasets.asr\_train.augmentations.gain\_db\_max=6.0}
\item \texttt{datasets.asr\_train.augmentations.noise\_prob=0.35}
\item \texttt{datasets.asr\_train.augmentations.noise\_snr\_db\_min=20.0}
\item \texttt{datasets.asr\_train.augmentations.noise\_snr\_db\_max=35.0}
\item \texttt{datasets.asr\_train.augmentations.specaugment\_prob=0.35}
\item \texttt{datasets.asr\_train.augmentations.num\_time\_masks=2}
\item \texttt{datasets.asr\_train.augmentations.max\_time\_mask\_frames=16}
\item \texttt{datasets.asr\_train.augmentations.num\_freq\_masks=1}
\item \texttt{datasets.asr\_train.augmentations.max\_freq\_mask\_bins=8}
\end{itemize}

\subsection{Hypothèse causale et résultats attendus}
\begin{itemize}[leftmargin=1.2cm]
\item \textbf{Hypothèse}: 
\item \textbf{Mécanisme attendu}: À préciser.
\item \textbf{Tendance attendue}: À préciser.
\item \textbf{Critère de décision}: À préciser.
\end{itemize}

\subsection{Résultats}
\textit{Section volontairement laissée vide avant exécution.}

\begin{table}[H]
\centering
\caption{Résultats quantitatifs --- Expérience R2\_VOX (à compléter)}
\begin{tabular}{lcccc}
\toprule
Seed & WER $\downarrow$ & Runtime inf. (s) $\downarrow$ & Débit (samples/s) $\uparrow$ & Mémoire GPU (MB) $\downarrow$ \\
\midrule
\multicolumn{5}{c}{\textit{À compléter après exécution des runs.}} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Discussion des résultats}
\textit{Section volontairement laissée vide avant interprétation finale.}


\section{Expérience R4\_VOX --- R4 VoxPopuli Patch-2 rapide}
\subsection{Objectif scientifique local}
À préciser.

\subsection{Choix méthodologiques et justification}
\textbf{Phase}: final\_tp\_fast\\
\textbf{Seeds prévues}: 456, 789, 1024\\


\paragraph{Justification du plan}


\paragraph{Mécanisme scientifique visé}
À préciser.

\paragraph{Variables manipulées}
\begin{itemize}
\item \texttt{model.patch\_time=2}
\item \texttt{model.max\_len=2048}
\item \texttt{datasets.asr\_train.max\_duration\_sec=4.0}
\item \texttt{datasets.asr\_valid.max\_duration\_sec=4.0}
\item \texttt{datasets.asr\_tests=[\{'name': 'facebook/voxpopuli', 'config': 'en', 'split': 'validation', 'max\_samples': 100, 'transcript\_key': 'raw\_text', 'max\_duration\_sec': 4.0, 'length\_policy': 'none', 'feature\_norm': 'utterance', 'max\_transcript\_chars': 80, 'max\_transcript\_words': 14, 'strict\_asr\_consistency': True\}]}
\end{itemize}

\subsection{Hypothèse causale et résultats attendus}
\begin{itemize}[leftmargin=1.2cm]
\item \textbf{Hypothèse}: 
\item \textbf{Mécanisme attendu}: À préciser.
\item \textbf{Tendance attendue}: À préciser.
\item \textbf{Critère de décision}: À préciser.
\end{itemize}

\subsection{Résultats}
\textit{Section volontairement laissée vide avant exécution.}

\begin{table}[H]
\centering
\caption{Résultats quantitatifs --- Expérience R4\_VOX (à compléter)}
\begin{tabular}{lcccc}
\toprule
Seed & WER $\downarrow$ & Runtime inf. (s) $\downarrow$ & Débit (samples/s) $\uparrow$ & Mémoire GPU (MB) $\downarrow$ \\
\midrule
\multicolumn{5}{c}{\textit{À compléter après exécution des runs.}} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Discussion des résultats}
\textit{Section volontairement laissée vide avant interprétation finale.}


\section{Consolidation Top-3 (E09--E11, après sélection Top-5)}
\subsection{Tableau final principal (mean $\pm$ std, 5 seeds)}
\textit{À compléter après exécution des consolidations.}

\begin{table}[H]
\centering
\caption{Synthèse comparative Top-3}
\begin{tabular}{lcccc}
\toprule
Modèle & WER $\downarrow$ & Runtime inf. (s) $\downarrow$ & Débit (samples/s) $\uparrow$ & Mémoire GPU (MB) $\downarrow$ \\
\midrule
Top-1 & --- & --- & --- & --- \\
Top-2 & --- & --- & --- & --- \\
Top-3 & --- & --- & --- & --- \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Visualisations prévues}
\begin{itemize}
\item Courbe WER par expérience (E01 $\rightarrow$ E11).
\item Frontière de Pareto WER vs runtime inférence.
\item Frontière de Pareto WER vs mémoire GPU.
\end{itemize}

\begin{figure}[H]
\centering
\fbox{\parbox[c][4cm][c]{0.92\linewidth}{\centering Placeholder figure: WER par expérience}}
\caption{Progression des performances WER (à compléter).}
\end{figure}

\begin{figure}[H]
\centering
\fbox{\parbox[c][4cm][c]{0.92\linewidth}{\centering Placeholder figure: Pareto WER vs runtime / mémoire}}
\caption{Analyse de compromis performance-frugalité (à compléter).}
\end{figure}


\section{Limites et menaces à la validité}
\subsection{Menaces internes}
\textit{À compléter: sensibilité aux hyperparamètres, stabilité d'optimisation, risque d'effet seed.}

\subsection{Menaces externes}
\textit{À compléter: généralisation inter-domaines, absence de VoxPopuli/TTS dans le cœur de l'étude, validité hors LibriSpeech.}

\subsection{Contraintes matérielles}
\textit{À compléter: limites vdigpu, budget disque/temps et impacts potentiels sur l'ampleur des ablations.}

\section{Conclusion}
\textit{À compléter: synthèse finale, réponse explicite à RQ1--RQ4, recommandations opérationnelles (architecture retenue, compromis retenu, extensions futures).}

\end{document}
