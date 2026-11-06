#!/usr/bin/env bash
# =============================================================================
# run_legacy_parity.sh — legacy parity benchmark (old subprocess.run SimAI vs new
# in-memory pysimai bridge).
#
# This is an OPTIONAL, MANUAL migration/parity tool, not a regression gate.
# It runs tests/pysimai/benchmark_legacy_parity.py --mode both, which:
#   1. spawns OLD mode  (pysimai NOT importable)  -> results_old.json
#   2. spawns NEW mode  (pysimai importable)      -> results_new.json
#   3. compares value parity and reports cache-miss speedup (informational).
#
# Requirements (see AGENTS.md — this project builds/runs on the REMOTE host):
#   conda env `vidur` active, e.g.
#     source /opt/miniforge3/etc/profile.d/conda.sh && conda activate vidur
#   <repo>/bin/pysimai*.so and <repo>/bin/SimAI_simulator built.
#
# Usage:
#   bash tests/pysimai/run_legacy_parity.sh [extra benchmark args...]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT"

if [[ ! -f bin/SimAI_simulator ]]; then
    echo "[ERROR] bin/SimAI_simulator not found. Build first (scripts/build.sh -c ns3)." >&2
    exit 1
fi
if ! ls bin/pysimai*.so >/dev/null 2>&1; then
    echo "[ERROR] bin/pysimai*.so not found. Build the pysimai target first." >&2
    exit 1
fi

python tests/pysimai/benchmark_legacy_parity.py --mode both "$@"
