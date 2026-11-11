#!/usr/bin/env bash
# Fresh-environment test: new venv, CPU torch, package install, notebook execution.
# Usage: tests/run_quickstart.sh <bundle-dir> [venv-dir]   (venv-dir must not exist; nothing is deleted)
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
bundle=$(cd "$1" && pwd)
venv=${2:-${TMPDIR:-/tmp}/methylgpt2_test_venv}
if [ -e "$venv" ]; then echo "the environment path $venv already exists; give a new path" >&2; exit 2; fi
python3 -m venv "$venv"
"$venv/bin/pip" install -q --upgrade pip
"$venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$venv/bin/pip" install -q -e "$here[examples]"
cd "$here/examples"
# isolate from any user-level Jupyter configuration
export JUPYTER_CONFIG_DIR="$venv/jupyter_config" JUPYTER_DATA_DIR="$venv/jupyter_data" IPYTHONDIR="$venv/ipython"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$IPYTHONDIR"
METHYLGPT_BUNDLE="$bundle" CUDA_VISIBLE_DEVICES="" "$venv/bin/jupyter" nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1800 quickstart.ipynb --output quickstart.executed.ipynb
echo "quickstart executed: $here/examples/quickstart.executed.ipynb"
# package ages against the analysis-code ages for the six tutorial profiles (tolerance 0.01 years)
CUDA_VISIBLE_DEVICES="" "$venv/bin/python" "$here/tests/check_tutorial_agreement.py" "$bundle" 0.01
