#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec bash "scripts/experiments_phase6/run_single_phase6_experiment.sh" "P6G01" "$@"
