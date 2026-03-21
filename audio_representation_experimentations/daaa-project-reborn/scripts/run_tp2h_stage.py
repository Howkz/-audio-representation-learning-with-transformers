from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES_DIR = PROJECT_ROOT / "configs" / "final_tp_2h" / "stages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one compact TP 2h stage on a seed shard.")
    parser.add_argument("--stage", type=str, required=True, help="Stage id, e.g. R1_LIBRI or R2_VOX")
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disk-guard-gb", type=float, default=1.5)
    parser.add_argument("--storage-interval-sec", type=int, default=30)
    parser.add_argument("--set", dest="cli_overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_config = STAGES_DIR / f"{args.stage}.yaml"
    if not suite_config.exists():
        available = ", ".join(sorted(path.stem for path in STAGES_DIR.glob("*.yaml")))
        raise SystemExit(f"Unknown stage '{args.stage}'. Available: {available}")

    command = [
        sys.executable,
        "scripts/run_tp5_seed_shard.py",
        "--suite-config",
        str(suite_config),
        "--label",
        args.label,
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.resume:
        command.append("--resume")
    if args.verbose:
        command.append("--verbose")
    command.extend(["--disk-guard-gb", str(args.disk_guard_gb)])
    command.extend(["--storage-interval-sec", str(args.storage_interval_sec)])
    for item in args.cli_overrides:
        command.extend(["--set", item])

    print(f"[TP2H] stage={args.stage} label={args.label} seeds={args.seeds}")
    print(f"[TP2H] suite_config={suite_config}")
    print(f"[TP2H] command={' '.join(command)}")
    raise SystemExit(subprocess.run(command, cwd=str(PROJECT_ROOT), check=False).returncode)


if __name__ == "__main__":
    main()
