#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  "${CONVEYOR_PYTHON:-python3}" -m venv .venv
fi
.venv/bin/python -m pip install --index-url "${CONVEYOR_PIP_INDEX:-https://pypi.org/simple}" -e '.[all]'
CONVEYOR_PYTHON="$PROJECT_ROOT/.venv/bin/python" bash scripts/run.sh check
