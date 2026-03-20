#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec bash "scripts/experiments_phase4/run_single_phase4_experiment.sh" "P4D01" "$@"
