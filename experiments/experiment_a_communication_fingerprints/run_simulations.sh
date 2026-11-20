#!/usr/bin/env bash
set -euo pipefail

EXP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${EXP_DIR}/../.." && pwd)
BIN="${SIMAI_ANALYTICAL_BIN:-${ROOT_DIR}/bin/SimAI_analytical}"

if [[ ! -x "$BIN" ]]; then
  echo "Missing analytical binary: $BIN" >&2
  echo "Build it with: ./scripts/build.sh -c analytical" >&2
  exit 1
fi

python3 "${EXP_DIR}/generate_workloads.py" >/dev/null
mkdir -p "${EXP_DIR}/raw_outputs" "${EXP_DIR}/results"

for scenario in dp tp pp; do
  rm -rf "${EXP_DIR}/results/${scenario}"
  args=(
    -w "${EXP_DIR}/workloads/${scenario}.txt"
    -s "${EXP_DIR}/system.txt"
    -g 4 -g_p_s 4 -nv 360 -nic 48.5 -n_p_s 4 -g_type A100
    -r "../experiments/experiment_a_communication_fingerprints/results/${scenario}/"
  )
  if [[ "$scenario" == pp ]]; then
    # The backend's PP formula is undefined at zero overlap and zero at its
    # default full-overlap value. Use 0.5 solely to expose the aggregate term.
    args+=( -pp_o 0.5 )
  fi
  (
    cd "$ROOT_DIR"
    "$BIN" "${args[@]}"
  ) >"${EXP_DIR}/raw_outputs/${scenario}.stdout.txt" 2>"${EXP_DIR}/raw_outputs/${scenario}.stderr.txt"
done

python3 "${EXP_DIR}/parse_outputs.py"

echo "SimAI runs complete: DP, TP, and PP results were generated and parsed successfully."
