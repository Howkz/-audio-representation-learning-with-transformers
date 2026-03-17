#!/usr/bin/env bash
set -euo pipefail

# One-command Linux launcher for DAAA audio experiments.
# It configures safe local temporary caches and runs the requested pipeline mode.
#
# Usage examples:
#   bash scripts/linux_experiments.sh smoke
#   bash scripts/linux_experiments.sh suite
#   bash scripts/linux_experiments.sh resume
#   bash scripts/linux_experiments.sh suite-18gb
#   bash scripts/linux_experiments.sh resume-18gb
#   bash scripts/linux_experiments.sh suite --clean
#   bash scripts/linux_experiments.sh dry-run
#   bash scripts/linux_experiments.sh resume --cache-root /mnt/bigdisk/$USER --clean-hf

MODE="${1:-suite}"
shift || true

CLEAN=0
CLEAN_HF=0
CACHE_ROOT="${EXPERIMENT_CACHE_ROOT:-${HOME}/.cache/daaa_audio}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --clean-hf)
      CLEAN_HF=1
      shift
      ;;
    --cache-root)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --cache-root requires a path argument."
        exit 2
      fi
      CACHE_ROOT="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/hf_tmp}"
export PYTHONPATH="."

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TMPDIR}"

echo "[RUN] mode=${MODE} clean=${CLEAN} clean_hf=${CLEAN_HF}"
echo "[RUN] cache_root=${CACHE_ROOT}"
echo "[RUN] HF_HOME=${HF_HOME}"
echo "[RUN] TMPDIR=${TMPDIR}"
echo "[RUN] disk usage:"
df -h "${HF_HOME}" "${TMPDIR}" 2>/dev/null || true

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

if [[ "$CLEAN_HF" == "1" ]]; then
  if [[ -z "${HF_HOME}" || "${HF_HOME}" == "/" || -z "${TMPDIR}" || "${TMPDIR}" == "/" ]]; then
    echo "[ERROR] Refusing to clean HF/TMP because path is unsafe."
    exit 2
  fi
  echo "[CLEAN] Removing HF/TMP caches..."
  rm -rf "${HF_HOME}" "${TMPDIR}"
  mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TMPDIR}"
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
  suite-18gb)
    python scripts/run_experiment_suite.py \
      --suite-config configs/suite_e00_e11_18gb.yaml \
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
  resume-18gb)
    python scripts/run_experiment_suite.py \
      --suite-config configs/suite_e00_e11_18gb.yaml \
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
    echo "[ERROR] Unknown mode '${MODE}'. Use: smoke | suite | resume | suite-18gb | resume-18gb | dry-run | clean"
    exit 2
    ;;
esac

echo "[DONE] ${MODE}"
