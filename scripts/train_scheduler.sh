#!/usr/bin/env bash
set -euo pipefail
export CONVEYOR_LIBM_COMPAT="${CONVEYOR_LIBM_COMPAT:-1}"
CONVEYOR_TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CONVEYOR_TRAIN_ROOT/scripts/runtime_env.sh"
cd "$CONVEYOR_TRAIN_ROOT"
export PYTHONPATH="$CONVEYOR_TRAIN_ROOT/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "${CONVEYOR_PYTHON:-$CONVEYOR_TRAIN_ROOT/.venv/bin/python}" \
  -m dual_arm_conveyor.hierarchical_train "$@"
