#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
exec bash "scripts/experiments_phase5/run_single_phase5_experiment.sh" "P5B01" "$@"
