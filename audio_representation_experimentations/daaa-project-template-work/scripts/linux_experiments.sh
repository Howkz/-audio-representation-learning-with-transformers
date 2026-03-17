#!/usr/bin/env bash
set -euo pipefail

# One-command Linux launcher for DAAA audio experiments.
# It configures safe local temporary caches and runs the requested pipeline mode.
#
# Usage examples:
#   bash scripts/linux_experiments.sh smoke
#   bash scripts/linux_experiments.sh suite
#   bash scripts/linux_experiments.sh resume
#   bash scripts/linux_experiments.sh suite --clean
#   bash scripts/linux_experiments.sh dry-run

MODE="${1:-suite}"
shift || true

CLEAN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "/tmp/${USER}/hf_cache" "/tmp/${USER}/hf_tmp"
export HF_HOME="/tmp/${USER}/hf_cache"
export HF_DATASETS_CACHE="/tmp/${USER}/hf_cache/datasets"
export HF_HUB_CACHE="/tmp/${USER}/hf_cache/hub"
export TMPDIR="/tmp/${USER}/hf_tmp"
export PYTHONPATH="."

echo "[RUN] mode=${MODE} clean=${CLEAN}"
echo "[RUN] HF_HOME=${HF_HOME}"
echo "[RUN] TMPDIR=${TMPDIR}"

if [[ "$CLEAN" == "1" ]]; then
  echo "[CLEAN] Removing project caches and checkpoints..."
  rm -rf data/cache/* data/processed/* outputs/checkpoints/* results/experiments/*
  rm -rf results/suite/archive/*
  rm -f results/suite/leaderboard_screening.csv
  rm -f results/suite/leaderboard_selection.csv
  rm -f results/suite/suite_summary.csv
  rm -f results/suite/selection_manifest.json
  rm -f results/suite/runtime_configs/*.yaml
fi

case "$MODE" in
  smoke)
    python scripts/run_data.py --config configs/smoke.yaml
    python scripts/run_train.py --config configs/smoke.yaml
    python scripts/run_test.py --config configs/smoke.yaml
    ;;
  suite)
    python scripts/run_experiment_suite.py \
      --suite-config configs/suite_e00_e11.yaml \
      --verbose \
      --disk-guard-gb 1.5
    ;;
  resume)
    python scripts/run_experiment_suite.py \
      --suite-config configs/suite_e00_e11.yaml \
      --resume \
      --verbose \
      --disk-guard-gb 1.5
    ;;
  dry-run)
    python scripts/run_experiment_suite.py \
      --suite-config configs/suite_e00_e11.yaml \
      --dry-run \
      --verbose \
      --disk-guard-gb 1.5
    ;;
  clean)
    echo "[DONE] Clean complete."
    ;;
  *)
    echo "[ERROR] Unknown mode '${MODE}'. Use: smoke | suite | resume | dry-run | clean"
    exit 2
    ;;
esac

echo "[DONE] ${MODE}"
