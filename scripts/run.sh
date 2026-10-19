#!/usr/bin/env bash
# Quick run against an already-provisioned interpreter (no venv creation).
# Usage:  bash scripts/run.sh [config]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/default.yaml}"
PYTHONPATH=src python3 -m roofsolar --config "$CONFIG"
