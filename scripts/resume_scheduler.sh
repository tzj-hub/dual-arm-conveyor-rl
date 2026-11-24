#!/usr/bin/env bash
set -euo pipefail
export CONVEYOR_LIBM_COMPAT="${CONVEYOR_LIBM_COMPAT:-1}"
CONVEYOR_RESUME_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CONVEYOR_RESUME_ROOT/scripts/runtime_env.sh"
cd "$CONVEYOR_RESUME_ROOT"
export PYTHONPATH="$CONVEYOR_RESUME_ROOT/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "${CONVEYOR_PYTHON:-$CONVEYOR_RESUME_ROOT/.venv/bin/python}" \
  "$CONVEYOR_RESUME_ROOT/scripts/resume_scheduler.py" "$@"
