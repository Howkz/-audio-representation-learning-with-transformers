from __future__ import annotations

import argparse
import fnmatch
import sys
import zipfile
from pathlib import Path

# Ensure project root is importable when running `python scripts/package_submission.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final submission zip excluding heavy artifacts.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--zip-name",
        type=str,
        default="daaa_nomduprojet1_nom2.zip",
        help="Final archive name required by TP instructions.",
    )
    return parser.parse_args()


EXCLUDE_PREFIXES = (
    "data/cache/",
    "data/processed/",
    "outputs/",
)

EXCLUDE_PATTERNS = [
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    ".pytest_cache/*",
]


def _is_excluded(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    if any(normalized.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return True
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in EXCLUDE_PATTERNS)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project_root = Path(".").resolve()
    zip_path = project_root / args.zip_name

    include_roots = [
        project_root / "src",
        project_root / "scripts",
        project_root / "configs",
        project_root / "docs",
        project_root / "results",
        project_root / "Makefile",
        project_root / "requirements.txt",
        project_root / "README.md",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in include_roots:
            if item.is_file():
                archive.write(item, arcname=item.name)
                continue
            for path in item.rglob("*"):
                if path.is_dir():
                    continue
                rel = str(path.relative_to(project_root)).replace("\\", "/")
                if _is_excluded(rel):
                    continue
                archive.write(path, arcname=rel)

    print(f"[PACKAGE] Created archive: {zip_path}")
    print(f"[PACKAGE] Excluded cache, processed data and all outputs according to TP policy.")
    print(f"[PACKAGE] Experiment: {cfg['experiment']['name']}")


if __name__ == "__main__":
    main()
