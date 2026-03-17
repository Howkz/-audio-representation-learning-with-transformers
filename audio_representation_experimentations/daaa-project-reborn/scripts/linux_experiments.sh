#!/usr/bin/env bash
set -euo pipefail

# One-command Linux launcher for the E00->E11 protocol.
#
# Public modes:
#   smoke | suite | resume | dry-run | clean
#
# Public env vars (defaults):
#   EXPERIMENT_CACHE_ROOT=~/.cache/daaa_audio
#
# Usage examples:
#   bash scripts/linux_experiments.sh dry-run
#   bash scripts/linux_experiments.sh suite
#   bash scripts/linux_experiments.sh resume --cache-root /fast-disk/daaa

MODE="${1:-suite}"
shift || true

CLEAN=0
CLEAN_HF=0
SUITE_CONFIG="${SUITE_CONFIG:-configs/suite_e00_e11.yaml}"
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
    --suite-config)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --suite-config requires a file path."
        exit 2
      fi
      SUITE_CONFIG="$2"
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

PYTHON_BIN=""
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[ERROR] Python introuvable. Activez un venv ou installez python3."
  exit 2
fi

if [[ ! -f "scripts/run_experiment_suite.py" && "$MODE" != "clean" ]]; then
  echo "[ERROR] Missing scripts/run_experiment_suite.py in ${ROOT_DIR}."
  echo "        The launcher interface is ready, but the Python backend is not present yet."
  echo "        Restore/migrate the training stack before running '${MODE}'."
  exit 2
fi

if [[ "$MODE" != "clean" && ! -f "$SUITE_CONFIG" ]]; then
  echo "[ERROR] Missing suite config: ${SUITE_CONFIG}"
  exit 2
fi

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/hf_tmp}"
export PYTHONPATH="."

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TMPDIR}"

echo "[RUN] mode=${MODE} clean=${CLEAN} clean_hf=${CLEAN_HF}"
echo "[RUN] suite_config=${SUITE_CONFIG}"
echo "[RUN] cache_root=${CACHE_ROOT}"
echo "[RUN] python=${PYTHON_BIN}"
echo "[RUN] Budget strategy: fixed hyperparameters and fixed phase quotas from suite config."
echo "[RUN] No global budget variables; edit configs to change steps/samples/seeds."
echo "[RUN] outputs: results/suite/leaderboard_screening.csv results/suite/leaderboard_selection.csv results/suite/suite_summary.csv"
echo "[RUN] disk usage:"
df -h "${HF_HOME}" "${TMPDIR}" 2>/dev/null || true

if [[ "$CLEAN" == "1" || "$MODE" == "clean" ]]; then
  echo "[CLEAN] Removing project caches/checkpoints/results..."
  rm -rf data/cache/* data/processed/*
  rm -rf outputs/checkpoints/* results/experiments/*
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

if [[ "$MODE" == "clean" ]]; then
  echo "[DONE] clean"
  exit 0
fi

case "$MODE" in
  smoke)
    "$PYTHON_BIN" scripts/run_experiment_suite.py \
      --suite-config "$SUITE_CONFIG" \
      --from-id E00 \
      --to-id E00 \
      --verbose \
      --disk-guard-gb 1.5
    ;;
  suite)
    "$PYTHON_BIN" scripts/run_experiment_suite.py \
      --suite-config "$SUITE_CONFIG" \
      --verbose \
      --disk-guard-gb 1.5
    ;;
  resume)
    "$PYTHON_BIN" scripts/run_experiment_suite.py \
      --suite-config "$SUITE_CONFIG" \
      --resume \
      --verbose \
      --disk-guard-gb 1.5
    ;;
  dry-run)
    "$PYTHON_BIN" scripts/run_experiment_suite.py \
      --suite-config "$SUITE_CONFIG" \
      --dry-run \
      --verbose \
      --disk-guard-gb 1.5
    ;;
  *)
    echo "[ERROR] Unknown mode '${MODE}'. Use: smoke | suite | resume | dry-run | clean"
    exit 2
    ;;
esac

echo "[DONE] ${MODE}"
