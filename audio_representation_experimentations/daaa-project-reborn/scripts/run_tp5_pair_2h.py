from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_CONFIG = "configs/final_tp_fast_2h/suite_tp_5seeds_fast_2h.yaml"


def main() -> None:
    command = [
        sys.executable,
        "scripts/run_tp5_seed_shard.py",
        "--suite-config",
        SUITE_CONFIG,
        "--label",
        "pair_2h",
        "--seeds",
        "42",
        "123",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(command, cwd=str(PROJECT_ROOT), check=False).returncode)


if __name__ == "__main__":
    main()
