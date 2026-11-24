#!/usr/bin/env bash
set -euo pipefail
TRAINED_RUN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CONVEYOR_LIBM_COMPAT="${CONVEYOR_LIBM_COMPAT:-1}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$TRAINED_RUN_ROOT/src"
source "$TRAINED_RUN_ROOT/scripts/runtime_env.sh"
TRAINED_RUN_PYTHON="${CONVEYOR_PYTHON:-$TRAINED_RUN_ROOT/.venv/bin/python}"
if [[ ! -x "$TRAINED_RUN_PYTHON" ]] && ! command -v "$TRAINED_RUN_PYTHON" >/dev/null 2>&1; then
  echo "Python not found; run bash scripts/setup.sh or set CONVEYOR_PYTHON." >&2
  exit 2
fi
exec "$TRAINED_RUN_PYTHON" -B "$TRAINED_RUN_ROOT/scripts/run_trained.py" "$@"
