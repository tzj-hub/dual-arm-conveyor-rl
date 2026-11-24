#!/usr/bin/env bash
set -euo pipefail
export CONVEYOR_LIBM_COMPAT="${CONVEYOR_LIBM_COMPAT:-1}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/runtime_env.sh"
if [[ -n "${CONVEYOR_PYTHON:-}" ]]; then
  : # Respect an explicitly selected environment.
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  CONVEYOR_PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  CONVEYOR_PYTHON="python3"
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# The CLI probes actual OpenGL contexts and selects hardware when available.
# An explicitly provided MUJOCO_GL still takes precedence over auto selection.
exec "$CONVEYOR_PYTHON" -m dual_arm_conveyor "$@"
