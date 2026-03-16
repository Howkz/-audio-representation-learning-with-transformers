from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final LaTeX scientific report template.")
    parser.add_argument("--suite-config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("YAML file must define a mapping.")
    return data


def _latex_escape(text: str) -> str:
    replace_map = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(replace_map.get(ch, ch))
    return "".join(out)


def _report_meta(suite_cfg: Dict[str, Any]) -> Dict[str, Any]:
    report_cfg = suite_cfg.get("suite", {}).get("report", {})
    return {
        "title": str(
            report_cfg.get(
                "title",
                "Rapport d'Experimentations sur l'Apprentissage de la représentation audio via transformateurs",
            )
        ),
        "subtitle": str(
            report_cfg.get(
                "subtitle",
                "Étude expérimentale sur l'apprentissage de représentations audio auto-supervisées",
            )
        ),
        "authors": list(
            report_cfg.get(
                "authors",
                ["CLAIR Maël", "SABINE Hugo", "CHANTELOUP William"],
            )
        ),
        "program": str(report_cfg.get("program", "Master 2 Algorithmiques et Systèmes Intelligents")),
        "course": str(report_cfg.get("course", "Projet DAAA")),
    }


def _authors_rows(authors: List[str]) -> str:
    if not authors:
        return r"\textbf{Étudiants} : & --- \\"
    rows = [rf"\textbf{{Étudiants}} : & {_latex_escape(str(authors[0]))} \\"]
    for name in authors[1:]:
        rows.append(rf" & {_latex_escape(str(name))} \\")
    return "\n".join(rows)


def _title_page(meta: Dict[str, Any]) -> str:
    return rf"""
\begin{{titlepage}}
\centering
{{\Large {_latex_escape(meta["program"])}\par}}
\vspace{{0.35cm}}
{{\large {_latex_escape(meta["course"])}\par}}
\vspace{{1.3cm}}
{{\LARGE\bfseries {_latex_escape(meta["title"])}\par}}
\vspace{{0.8cm}}
{{\large {_latex_escape(meta["subtitle"])}\par}}
\vfill
\begin{{tabular}}{{ll}}
{_authors_rows(meta["authors"])}
\textbf{{Date}} : & \today \\
\end{{tabular}}
\vfill
\end{{titlepage}}
"""


def _intro_text(suite_cfg: Dict[str, Any]) -> str:
    notes = suite_cfg.get("suite", {}).get("notes", [])
    notes_tex = "\n".join([rf"\item {_latex_escape(str(n))}" for n in notes]) if notes else r"\item ---"
    return rf"""
\section{{Introduction}}
Ce rapport présente une étude expérimentale progressive sur un encodeur Transformer audio pré-entraîné par MAE puis adapté en ASR via CTC.
L'objectif scientifique est triple: (i) vérifier l'apport du pré-entraînement MAE, (ii) identifier les compromis entre WER, temps d'inférence et mémoire GPU, (iii) produire un benchmark multi-variantes avec consolidation statistique finale.

Contraintes de conception: environnement vdigpu, budget stockage limité (environ 15 Go), exécution reproductible, et suivi frugal des ressources.

\paragraph{{Choix globaux assumés}}
\begin{{itemize}}
{notes_tex}
\end{{itemize}}
"""


def _research_questions_text() -> str:
    return r"""
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
"""


def _method_text(suite_cfg: Dict[str, Any]) -> str:
    policy = suite_cfg.get("suite", {}).get("dataset_policy", {})
    screening_policy = _latex_escape(str(policy.get("screening", "non spécifié")))
    final_policy = _latex_escape(str(policy.get("final", "non spécifié")))
    return rf"""
\section{{Méthodologie générale}}
\subsection{{Définitions opératoires}}
\begin{{itemize}}
\item \textbf{{ASR (Automatic Speech Recognition)}}: conversion d'un signal de parole en transcription texte.
\item \textbf{{CTC (Connectionist Temporal Classification)}}: fonction de perte permettant d'apprendre sans alignement frame-à-frame explicite.
\item \textbf{{WER (Word Error Rate)}}: métrique de transcription basée sur substitutions, insertions et suppressions.
\item \textbf{{MAE (Masked AutoEncoder)}}: pré-entraînement auto-supervisé par reconstruction de portions masquées du signal.
\item \textbf{{TTS (Text-To-Speech)}}: tâche inverse de l'ASR (texte vers audio), non incluse dans le cœur de cette suite.
\end{{itemize}}

\subsection{{Pipeline}}
Le protocole suit les étapes \texttt{{make data}}, \texttt{{make train}}, \texttt{{make test}}.
Chaque expérience est exécutée dans un namespace dédié, avec nettoyage des checkpoints intermédiaires après archivage des résultats, tout en conservant le cache de données.

\subsection{{Politique dataset et périmètre}}
\begin{{itemize}}
\item \textbf{{Screening}}: {screening_policy}.
\item \textbf{{Consolidation finale}}: {final_policy}.
\item VoxPopuli reste une extension possible, hors chemin critique de la présente étude (contraintes disque/temps).
\end{{itemize}}

\subsection{{Contrôle expérimental}}
\begin{{itemize}}
\item Même pipeline de preprocessing audio pour toutes les variantes.
\item Même protocole d'évaluation et mêmes splits pour la comparabilité.
\item Une seule variable manipulée à la fois durant le screening autant que possible.
\item Checkpointing complet (modèle, optimiseur, scheduler, scaler AMP, état RNG) pour reprise fiable.
\end{{itemize}}

\subsection{{Métriques}}
Les métriques principales sont:
\begin{{itemize}}
\item WER (Word Error Rate) pour la performance ASR.
\item Temps d'inférence et débit (samples/s) pour la frugalité temporelle.
\item Mémoire GPU pic pour la frugalité matérielle.
\item Taille modèle (\#paramètres) pour l'analyse capacité/coût.
\end{{itemize}}

\subsection{{Règles statistiques}}
Le screening initial est réalisé en 1 seed par variante (E01--E08).
La consolidation finale est réalisée sur 5 seeds pour les 3 meilleures variantes (E09--E11), avec reporting moyenne~$\pm$~écart-type.

\subsection{{Règle de décision Top-3}}
Le classement screening est effectué par tri lexicographique sur:
\begin{{enumerate}}
\item WER (ascendant),
\item runtime d'inférence (ascendant),
\item mémoire GPU pic (ascendant).
\end{{enumerate}}
"""


def _flatten_overrides(payload: Dict[str, Any], prefix: str = "") -> List[str]:
    rows: List[str] = []
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten_overrides(value, path))
        else:
            rows.append(f"{path}={value}")
    return rows


_EXPERIMENT_NOTES: Dict[str, Dict[str, str]] = {
    "E00": {
        "goal": "Vérifier la validité opérationnelle du pipeline complet (data, entraînement, évaluation) avant toute conclusion scientifique.",
        "rationale": "Un smoke test réduit le risque méthodologique: il détecte tôt les erreurs de données, de configuration ou de sérialisation des checkpoints.",
        "mechanism": "Si le pipeline est sain, toutes les étapes se terminent et produisent les artefacts attendus avec une faible consommation.",
        "expected": "Succès d'exécution, cohérence des fichiers de sortie, absence de crash ou de corruption d'état.",
        "decision": "Passage en screening uniquement si E00 est exécuté de bout en bout sans erreur bloquante.",
    },
    "E01": {
        "goal": "Établir une ligne de base frugale servant de référence commune pour toutes les comparaisons ultérieures.",
        "rationale": "Une baseline contrôlée est nécessaire pour attribuer les écarts de performance aux seuls changements expérimentaux.",
        "mechanism": "Le couple MAE+CTC sur un modèle compact doit offrir un compromis initial stable entre WER, mémoire et temps.",
        "expected": "WER de référence et coût de calcul modéré, utilisables comme point de comparaison direct.",
        "decision": "Toutes les variantes E02--E08 sont analysées relativement à E01.",
    },
    "E02": {
        "goal": "Mesurer explicitement l'effet causal du pré-entraînement MAE.",
        "rationale": "Comparer MAE vs NoMAE à architecture quasi identique isole l'effet représentationnel du prétexte auto-supervisé.",
        "mechanism": "Sans MAE, l'encodeur apprend uniquement via CTC supervisé; la représentation initiale est moins structurée acoustiquement.",
        "expected": "Tendance attendue: WER plus élevé qu'E01, avec coût d'entraînement possiblement plus faible.",
        "decision": "Confirmer H1 si E01 surpasse E02 sur WER à coût comparable.",
    },
    "E03": {
        "goal": "Évaluer l'impact d'une réduction forte de capacité (tiny) sur la frugalité et la qualité.",
        "rationale": "La contrainte disque/compute impose d'explorer le bas du spectre capacité/coût.",
        "mechanism": "Moins de dimensions/couches réduit la capacité d'approximation, mais diminue nettement mémoire et runtime.",
        "expected": "Mémoire et temps en baisse; possible hausse du WER (sous-capacité).",
        "decision": "Conserver la variante si le gain de frugalité compense une dégradation WER limitée.",
    },
    "E04": {
        "goal": "Tester une capacité plus élevée pour sonder le plafond de performance du design.",
        "rationale": "L'augmentation de capacité peut mieux modéliser la variabilité acoustique, au prix d'un coût de calcul supérieur.",
        "mechanism": "Plus de paramètres augmente l'expressivité, ce qui peut réduire le WER si la donnée est suffisante.",
        "expected": "WER potentiellement meilleur qu'E01; hausse attendue du runtime et de la mémoire.",
        "decision": "Retenue si amélioration WER substantielle pour un surcoût acceptable dans le budget.",
    },
    "E05": {
        "goal": "Comparer embeddings positionnels appris vs sinusoïdaux à architecture constante.",
        "rationale": "Le codage positionnel influence la modélisation temporelle des séquences audio patchifiées.",
        "mechanism": "Un positionnel appris peut mieux s'adapter au domaine, mais peut sur-apprendre si données limitées.",
        "expected": "Variation modérée du WER; coût similaire à E01.",
        "decision": "Choisir la variante la plus stable en WER sans pénalité de coût.",
    },
    "E06": {
        "goal": "Tester un patching temps-fréquence pour modifier le biais inductif de l'encodeur.",
        "rationale": "La structure spectrale locale peut être mieux capturée par un découpage mixte temps/fréquence.",
        "mechanism": "Un patching plus fin en fréquence peut enrichir la représentation, mais augmente la complexité de séquence.",
        "expected": "Possible gain WER sur signaux complexes; runtime/mémoire potentiellement en hausse.",
        "decision": "Retenir si gain de qualité non dominé par un coût excessif.",
    },
    "E07": {
        "goal": "Ablation du mask ratio MAE vers une difficulté plus faible (0.40).",
        "rationale": "Un prétexte trop simple peut limiter la richesse des représentations apprises.",
        "mechanism": "Moins de masquage facilite la reconstruction mais peut réduire la pression d'apprentissage contextuel.",
        "expected": "Entraînement plus stable/rapide; gain WER incertain, parfois inférieur à E01.",
        "decision": "Valider uniquement si la baisse de coût s'accompagne d'une qualité proche ou meilleure.",
    },
    "E08": {
        "goal": "Ablation du mask ratio MAE vers une difficulté élevée (0.75).",
        "rationale": "Un masquage fort peut forcer des représentations globales plus robustes, mais rend l'optimisation plus dure.",
        "mechanism": "Le modèle doit inférer davantage de contexte latent, ce qui peut améliorer ou détériorer le transfert ASR.",
        "expected": "Résultat dépendant de la capacité: soit robustesse accrue, soit dégradation si sous-capacité.",
        "decision": "Conserver si WER et stabilité restent compétitifs malgré la difficulté accrue.",
    },
    "E09": {
        "goal": "Consolider statistiquement la meilleure configuration issue du screening.",
        "rationale": "Une conclusion scientifique exige d'estimer la variabilité inter-seeds, pas seulement une seed favorable.",
        "mechanism": "Le réplica sur 5 seeds réduit le risque de sur-interprétation d'un résultat isolé.",
        "expected": "Moyenne WER la plus compétitive avec écart-type maîtrisé.",
        "decision": "Candidate principale pour la conclusion finale du rapport.",
    },
    "E10": {
        "goal": "Consolider la seconde meilleure configuration pour une comparaison robuste.",
        "rationale": "Un second candidat limite le risque de conclure sur une seule famille de réglages.",
        "mechanism": "La hiérarchie observée en screening est vérifiée sous variance aléatoire.",
        "expected": "Performance proche de E09 avec profil coût éventuellement différent.",
        "decision": "Comparer à E09 sur moyenne et dispersion pour qualifier le compromis.",
    },
    "E11": {
        "goal": "Consolider la troisième meilleure configuration pour compléter le benchmark multi-modèles.",
        "rationale": "Trois candidats finaux permettent une discussion Pareto plus crédible qu'un duel binaire.",
        "mechanism": "Analyse conjointe qualité/coût/stabilité sur trois points du front de compromis.",
        "expected": "Option potentiellement plus frugale avec qualité légèrement inférieure.",
        "decision": "Retenir comme alternative si le compromis coût/performance est préférable selon les contraintes.",
    },
}


def _experiment_section(experiment: Dict[str, Any]) -> str:
    exp_id = str(experiment["id"])
    exp_id_tex = _latex_escape(exp_id)
    title = _latex_escape(str(experiment.get("title", exp_id)))
    phase = _latex_escape(str(experiment.get("phase", "")))
    seeds_text = _latex_escape(", ".join(str(s) for s in experiment.get("seeds", [])))
    choice_manifest = _latex_escape(str(experiment.get("choice_justification", "")))
    expected_manifest = _latex_escape(str(experiment.get("expected_effect", "")))

    notes = _EXPERIMENT_NOTES.get(exp_id, {})
    goal = _latex_escape(str(notes.get("goal", "À préciser.")))
    rationale = _latex_escape(str(notes.get("rationale", "À préciser.")))
    mechanism = _latex_escape(str(notes.get("mechanism", "À préciser.")))
    expected = _latex_escape(str(notes.get("expected", "À préciser.")))
    decision = _latex_escape(str(notes.get("decision", "À préciser.")))

    overrides = experiment.get("overrides", {})
    override_rows = _flatten_overrides(overrides) if isinstance(overrides, dict) else []
    if override_rows:
        override_tex = "\n".join([rf"\item \texttt{{{_latex_escape(row)}}}" for row in override_rows])
    else:
        override_tex = r"\item \textit{Aucune surcharge explicite (configuration de base conservée).}"

    rank_text = ""
    if "auto_from_screening_rank" in experiment:
        rank_text = rf"\textbf{{Source auto}}: configuration héritée du rang screening {int(experiment['auto_from_screening_rank'])}."

    return rf"""
\section{{Expérience {exp_id_tex} --- {title}}}
\subsection{{Objectif scientifique local}}
{goal}

\subsection{{Choix méthodologiques et justification}}
\textbf{{Phase}}: {phase}\\
\textbf{{Seeds prévues}}: {seeds_text}\\
{rank_text}

\paragraph{{Justification du plan}}
{choice_manifest}

\paragraph{{Mécanisme scientifique visé}}
{rationale}

\paragraph{{Variables manipulées}}
\begin{{itemize}}
{override_tex}
\end{{itemize}}

\subsection{{Hypothèse causale et résultats attendus}}
\begin{{itemize}}[leftmargin=1.2cm]
\item \textbf{{Hypothèse}}: {expected_manifest}
\item \textbf{{Mécanisme attendu}}: {mechanism}
\item \textbf{{Tendance attendue}}: {expected}
\item \textbf{{Critère de décision}}: {decision}
\end{{itemize}}

\subsection{{Résultats}}
\textit{{Section volontairement laissée vide avant exécution.}}

\begin{{table}}[H]
\centering
\caption{{Résultats quantitatifs --- Expérience {exp_id_tex} (à compléter)}}
\begin{{tabular}}{{lcccc}}
\toprule
Seed & WER $\downarrow$ & Runtime inf. (s) $\downarrow$ & Débit (samples/s) $\uparrow$ & Mémoire GPU (MB) $\downarrow$ \\
\midrule
\multicolumn{{5}}{{c}}{{\textit{{À compléter après exécution des runs.}}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Discussion des résultats}}
\textit{{Section volontairement laissée vide avant interprétation finale.}}
"""


def _consolidation_text() -> str:
    return r"""
\section{Consolidation Top-3 (E09--E11)}
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
"""


def _limits_conclusion_text() -> str:
    return r"""
\section{Limites et menaces à la validité}
\subsection{Menaces internes}
\textit{À compléter: sensibilité aux hyperparamètres, stabilité d'optimisation, risque d'effet seed.}

\subsection{Menaces externes}
\textit{À compléter: généralisation inter-domaines, absence de VoxPopuli/TTS dans le cœur de l'étude, validité hors LibriSpeech.}

\subsection{Contraintes matérielles}
\textit{À compléter: limites vdigpu, budget disque/temps et impacts potentiels sur l'ampleur des ablations.}

\section{Conclusion}
\textit{À compléter: synthèse finale, réponse explicite à RQ1--RQ4, recommandations opérationnelles (architecture retenue, compromis retenu, extensions futures).}
"""


def build_latex_document(suite_cfg: Dict[str, Any]) -> str:
    meta = _report_meta(suite_cfg)
    experiments: List[Dict[str, Any]] = suite_cfg.get("experiments", [])
    enabled_experiments = [e for e in experiments if bool(e.get("enabled", True))]
    body_parts = [
        _title_page(meta),
        r"\tableofcontents",
        r"\newpage",
        _intro_text(suite_cfg),
        _research_questions_text(),
        _method_text(suite_cfg),
    ]
    body_parts.extend(_experiment_section(exp) for exp in enabled_experiments)
    body_parts.append(_consolidation_text())
    body_parts.append(_limits_conclusion_text())
    body = "\n".join(body_parts)
    title_tex = _latex_escape(meta["title"])
    authors_tex = _latex_escape(", ".join(meta["authors"]))
    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[french]{{babel}}
\usepackage{{lmodern}}
\usepackage{{geometry}}
\usepackage{{setspace}}
\usepackage{{booktabs}}
\usepackage{{microtype}}
\usepackage{{graphicx}}
\usepackage{{enumitem}}
\usepackage{{float}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\geometry{{margin=2.4cm}}
\onehalfspacing
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.35em}}
\hypersetup{{
  colorlinks=true,
  linkcolor=blue!40!black,
  urlcolor=blue!40!black,
  citecolor=blue!40!black
}}

\title{{{title_tex}}}
\author{{{authors_tex}}}
\date{{\today}}

\begin{{document}}
{body}
\end{{document}}
"""


def main() -> None:
    args = parse_args()
    suite_path = Path(args.suite_config)
    suite_cfg = _load_yaml(suite_path)
    output = args.output or suite_cfg.get("suite", {}).get("report_output", "results/suite/rapport_final.tex")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tex = build_latex_document(suite_cfg)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(tex)
    print(f"[REPORT] LaTeX template generated: {output_path}")


if __name__ == "__main__":
    main()

