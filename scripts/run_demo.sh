#!/usr/bin/env bash
# End-to-end demo: create an isolated environment, install deps and run the
# full pipeline. Usage:  bash scripts/run_demo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

echo "==> creating virtual environment (.venv)"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "==> running the pipeline"
PYTHONPATH=src python -m roofsolar --config configs/default.yaml

echo
echo "==> done. Figures and data are in: $ROOT/outputs"
ls -1 outputs
