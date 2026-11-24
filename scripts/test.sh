#!/usr/bin/env bash
set -euo pipefail
export CONVEYOR_LIBM_COMPAT="${CONVEYOR_LIBM_COMPAT:-1}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/runtime_env.sh"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Tests create offscreen renderers even on a desktop with DISPLAY set. Select
# the backend before any test imports MuJoCo; changing it later is too late.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
exec "${CONVEYOR_PYTHON:-$PROJECT_ROOT/.venv/bin/python}" -m pytest "$@"
