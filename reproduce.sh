#!/usr/bin/env bash
# Regenerate all current model checks, figure data and figures.
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p results figures
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" src/validate.py
"$python_bin" src/analyse.py
