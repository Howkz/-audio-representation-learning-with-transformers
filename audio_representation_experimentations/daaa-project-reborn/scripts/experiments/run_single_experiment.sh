#!/usr/bin/env bash
set -euo pipefail

# Run one experiment id from the E00->E11 suite.
#
# Usage:
#   bash scripts/experiments/run_single_experiment.sh E00
#   bash scripts/experiments/run_single_experiment.sh E04 --resume
#   bash scripts/experiments/run_single_experiment.sh E09 --resume
#   bash scripts/experiments/run_single_experiment.sh E02 --dry-run
#
# Notes:
# - For E09/E10/E11, the suite runner will auto-run selection (SEL01-SEL05)
#   if needed, based on available screening results.

if [[ $# -lt 1 ]]; then
  echo "[ERROR] Missing experiment id. Expected one of E00..E11."
  exit 2
fi

EXP_ID="$1"
shift || true

if [[ ! "$EXP_ID" =~ ^E(0[0-9]|1[01])$ ]]; then
  echo "[ERROR] Invalid experiment id '${EXP_ID}'. Expected E00..E11."
  exit 2
fi

RESUME=0
DRY_RUN=0
BOOTSTRAP_PREREQS=0
SUITE_CONFIG="${SUITE_CONFIG:-configs/suite_e00_e11.yaml}"
DISK_GUARD_GB="${DISK_GUARD_GB:-1.5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --bootstrap-prereqs)
      BOOTSTRAP_PREREQS=1
      shift
      ;;
    --suite-config)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --suite-config requires a file path."
        exit 2
      fi
      SUITE_CONFIG="$2"
      shift 2
      ;;
    --disk-guard-gb)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --disk-guard-gb requires a numeric value."
        exit 2
      fi
      DISK_GUARD_GB="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "scripts/run_experiment_suite.py" ]]; then
  echo "[ERROR] Missing scripts/run_experiment_suite.py in ${ROOT_DIR}."
  exit 2
fi
if [[ ! -f "$SUITE_CONFIG" ]]; then
  echo "[ERROR] Missing suite config: ${SUITE_CONFIG}"
  exit 2
fi

PYTHON_BIN=""
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[ERROR] Python not found."
  exit 2
fi

CMD=(
  "$PYTHON_BIN" "scripts/run_experiment_suite.py"
  "--suite-config" "$SUITE_CONFIG"
  "--from-id" "$EXP_ID"
  "--to-id" "$EXP_ID"
  "--verbose"
  "--disk-guard-gb" "$DISK_GUARD_GB"
)

if [[ "$RESUME" == "1" ]]; then
  CMD+=("--resume")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=("--dry-run")
fi

echo "[RUN] single experiment=${EXP_ID} resume=${RESUME} dry_run=${DRY_RUN}"
echo "[RUN] suite_config=${SUITE_CONFIG}"
echo "[RUN] python=${PYTHON_BIN}"
echo "[RUN] bootstrap_prereqs=${BOOTSTRAP_PREREQS}"

if [[ "$EXP_ID" =~ ^E(09|10|11)$ ]]; then
  missing_prereq=0
  for req in E01 E02 E03 E04 E05; do
    if [[ ! -f "results/suite/runtime_configs/${req}.yaml" ]]; then
      missing_prereq=1
      break
    fi
  done
  if [[ "$missing_prereq" == "1" ]]; then
    if [[ "$BOOTSTRAP_PREREQS" == "1" ]]; then
      PRE_CMD=(
        "$PYTHON_BIN" "scripts/run_experiment_suite.py"
        "--suite-config" "$SUITE_CONFIG"
        "--from-id" "E01"
        "--to-id" "E08"
        "--verbose"
        "--disk-guard-gb" "$DISK_GUARD_GB"
      )
      if [[ "$RESUME" == "1" ]]; then
        PRE_CMD+=("--resume")
      fi
      if [[ "$DRY_RUN" == "1" ]]; then
        PRE_CMD+=("--dry-run")
      fi
      echo "[RUN] Missing prerequisites for ${EXP_ID}. Bootstrapping E01..E08."
      echo "[RUN] prereq_command=${PRE_CMD[*]}"
      "${PRE_CMD[@]}"
    else
      echo "[ERROR] ${EXP_ID} needs prior screening artifacts (E01..E08)."
      echo "        Run one of:"
      echo "        - bash scripts/experiments/run_E01.sh ... run_E08.sh"
      echo "        - bash scripts/experiments/run_single_experiment.sh ${EXP_ID} --bootstrap-prereqs"
      exit 2
    fi
  fi
fi

echo "[RUN] command=${CMD[*]}"
"${CMD[@]}"
