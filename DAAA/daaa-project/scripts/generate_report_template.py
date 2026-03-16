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


def _intro_text(suite_cfg: Dict[str, Any]) -> str:
    notes = suite_cfg.get("suite", {}).get("notes", [])
    notes_tex = "\n".join([rf"\item {_latex_escape(str(n))}" for n in notes])
    return rf"""
\section{{Introduction}}
Ce rapport presente une etude experimentale progressive sur un encodeur Transformer audio pre-entraine par MAE puis adapte en ASR via CTC. 
L'objectif scientifique est triple: (i) verifier l'apport du pre-entrainement MAE, (ii) identifier les compromis entre WER, temps d'inference et memoire GPU, (iii) produire un benchmark multi-variantes avec consolidation statistique finale.

Contraintes de conception: environnement vdigpu, budget stockage limite (environ 15 Go), execution reproductible, et suivi frugal des ressources.

\paragraph{{Choix globaux assumes}}
\begin{{itemize}}
{notes_tex}
\end{{itemize}}
"""


def _method_text() -> str:
    return r"""
\section{Methodologie generale}
\subsection{Pipeline}
Le protocole suit les etapes \texttt{make data}, \texttt{make train}, \texttt{make test}. 
Chaque experience est executee dans un namespace dedie, avec nettoyage des checkpoints intermediaires apres archivage des resultats, tout en conservant le cache de donnees.

\subsection{Metriques}
Les metriques principales sont:
\begin{itemize}
\item WER (Word Error Rate) pour la performance ASR.
\item Temps d'inference et debit (samples/s) pour la frugalite temporelle.
\item Memoire GPU pic pour la frugalite materielle.
\item Taille modele (\#parametres) pour l'analyse capacite/cout.
\end{itemize}

\subsection{Regles statistiques}
Le screening initial est realise en 1 seed par variante (E01--E08). 
La consolidation finale est realisee sur 5 seeds pour les 3 meilleures variantes (E09--E11), avec reporting moyenne~$\pm$~ecart-type.
"""


def _experiment_section(experiment: Dict[str, Any]) -> str:
    exp_id = _latex_escape(str(experiment["id"]))
    title = _latex_escape(str(experiment.get("title", exp_id)))
    choice = _latex_escape(str(experiment.get("choice_justification", "")))
    expected = _latex_escape(str(experiment.get("expected_effect", "")))
    seeds = experiment.get("seeds", [])
    phase = _latex_escape(str(experiment.get("phase", "")))
    seeds_text = _latex_escape(", ".join(str(s) for s in seeds))
    return rf"""
\section{{Experience {exp_id} --- {title}}}
\subsection{{Choix et justification}}
\textbf{{Phase}}: {phase}\\
\textbf{{Seeds prevues}}: {seeds_text}

{choice}

\paragraph{{Hypothese attendue}}
{expected}

\subsection{{Resultats}}
\textit{{A completer apres execution des runs.}}

\subsection{{Discussion des resultats}}
\textit{{A completer apres analyse des resultats de cette experience.}}
"""


def _consolidation_text() -> str:
    return r"""
\section{Consolidation Top-3 (E09--E11)}
\subsection{Tableau final principal}
\textit{A completer: tableau mean $\pm$ std pour Top-1/Top-2/Top-3.}

\subsection{Visualisations prevues}
\begin{itemize}
\item Courbe WER par experience (E01 $\rightarrow$ E11).
\item Frontiere de Pareto WER vs runtime inference.
\item Frontiere de Pareto WER vs memoire GPU.
\end{itemize}
\textit{Inserer ici les figures et l'interpretation scientifique associee.}
"""


def _limits_conclusion_text() -> str:
    return r"""
\section{Limites et menaces a la validite}
\textit{A completer: limites compute/stockage, risques de variance, biais de selection des variantes, validite externe.}

\section{Conclusion}
\textit{A completer: synthese finale, reponse aux hypotheses, recommandations pour extension VoxPopuli et/ou TTS.}
"""


def build_latex_document(suite_cfg: Dict[str, Any]) -> str:
    experiments: List[Dict[str, Any]] = suite_cfg.get("experiments", [])
    enabled_experiments = [e for e in experiments if bool(e.get("enabled", True))]
    body_parts = [_intro_text(suite_cfg), _method_text()]
    body_parts.extend(_experiment_section(exp) for exp in enabled_experiments)
    body_parts.append(_consolidation_text())
    body_parts.append(_limits_conclusion_text())
    body = "\n".join(body_parts)
    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[french]{{babel}}
\usepackage{{lmodern}}
\usepackage{{geometry}}
\usepackage{{setspace}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\geometry{{margin=2.4cm}}
\onehalfspacing

\title{{Rapport Scientifique Final --- Suite E00$\rightarrow$E11}}
\author{{Equipe Projet DAAA}}
\date{{\today}}

\begin{{document}}
\maketitle
\tableofcontents
\newpage
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

