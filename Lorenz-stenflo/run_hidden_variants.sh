#!/usr/bin/env bash
# Run all hidden-dim variant trainings (20 and 32). Original 2th/4th/NeuralRK4 are not touched.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${ROOT}/src"
cd "$(dirname "$0")"

echo "PYTHONPATH includes: ${ROOT}/src"

run_py () {
  local dir="$1"
  local script="$2"
  echo "========== cd $dir && python $script =========="
  ( cd "$dir" && python "$script" )
}

# run_py 2th_h20 ANI2.py
# run_py 2th_h32 ANI2.py
# run_py 4th_h20 ANI4.py
# run_py 4th_h32 ANI4.py
# run_py NeuralRK4_h20 base.py
# run_py NeuralRK4_h32 base.py

run_py 2th ANI2.py
run_py 4th ANI4.py
run_py NeuralRK4 base.py

echo "All six runs finished."
