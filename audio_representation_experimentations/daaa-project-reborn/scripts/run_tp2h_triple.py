from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    command = [
        sys.executable,
        "scripts/run_tp2h_stage.py",
        "--label",
        "triple",
        "--seeds",
        "456",
        "789",
        "1024",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(command, cwd=str(PROJECT_ROOT), check=False).returncode)


if __name__ == "__main__":
    main()
