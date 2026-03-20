#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "[ERROR] Identifiant d'expérience manquant. Exemple: P6G01"
  exit 2
fi

EXP_ID="$1"
shift || true

if [[ ! "$EXP_ID" =~ ^P6G[0-9]{2}$ ]]; then
  echo "[ERROR] Identifiant invalide '${EXP_ID}'. Attendu: P6Gxx."
  exit 2
fi

RESUME=0
DRY_RUN=0
MATERIALIZE_DATA=0
SKIP_DATA=0
SKIP_TEST=0
CONFIG_PATH="${CONFIG_PATH:-configs/phase6/${EXP_ID}.yaml}"
CHECKPOINT_VARIANT="${CHECKPOINT_VARIANT:-all}"

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
    --materialize-data)
      MATERIALIZE_DATA=1
      shift
      ;;
    --skip-data)
      SKIP_DATA=1
      shift
      ;;
    --skip-test)
      SKIP_TEST=1
      shift
      ;;
    --checkpoint-variant)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --checkpoint-variant requiert une valeur."
        exit 2
      fi
      CHECKPOINT_VARIANT="$2"
      shift 2
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --config requiert un chemin."
        exit 2
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Argument inconnu: $1"
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config introuvable: ${CONFIG_PATH}"
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
  echo "[ERROR] Python introuvable."
  exit 2
fi

DATA_CMD=("$PYTHON_BIN" "scripts/run_data.py" "--config" "$CONFIG_PATH")
TRAIN_CMD=("$PYTHON_BIN" "scripts/run_train.py" "--config" "$CONFIG_PATH")
TEST_CMD=("$PYTHON_BIN" "scripts/run_test.py" "--config" "$CONFIG_PATH" "--checkpoint-variant" "$CHECKPOINT_VARIANT")

if [[ "$MATERIALIZE_DATA" == "1" ]]; then
  DATA_CMD+=("--materialize")
fi
if [[ "$RESUME" == "1" ]]; then
  TRAIN_CMD+=("--continue-completed")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  DATA_CMD+=("--dry-run")
  TRAIN_CMD+=("--dry-run")
  TEST_CMD+=("--dry-run")
fi

echo "[RUN-PHASE6] experiment=${EXP_ID} resume=${RESUME} dry_run=${DRY_RUN}"
echo "[RUN-PHASE6] checkpoint_variant=${CHECKPOINT_VARIANT}"
echo "[RUN-PHASE6] config=${CONFIG_PATH}"
echo "[RUN-PHASE6] python=${PYTHON_BIN}"

if [[ "$SKIP_DATA" != "1" ]]; then
  echo "[RUN-PHASE6] data_command=${DATA_CMD[*]}"
  "${DATA_CMD[@]}"
fi

echo "[RUN-PHASE6] train_command=${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

if [[ "$SKIP_TEST" != "1" ]]; then
  echo "[RUN-PHASE6] test_command=${TEST_CMD[*]}"
  "${TEST_CMD[@]}"
fi
