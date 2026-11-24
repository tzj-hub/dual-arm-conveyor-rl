#!/usr/bin/env bash
set -euo pipefail
SEQUENCE_TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CONVEYOR_LIBM_COMPAT="${CONVEYOR_LIBM_COMPAT:-1}"
export PYTHONPATH="$SEQUENCE_TRAIN_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SEQUENCE_TRAIN_ROOT"
exec "${CONVEYOR_PYTHON:-$SEQUENCE_TRAIN_ROOT/.venv/bin/python}" -B "$SEQUENCE_TRAIN_ROOT/scripts/train_sequence.py" "$@"
