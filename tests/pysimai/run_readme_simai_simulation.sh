#!/usr/bin/env bash
# =============================================================================
# run_readme_simai_simulation.sh
#
# Runs the "Run Llama-3-8B with SimAI Simulation" example from
# vidur-alibabacloud/README.md twice:
#
#   old : legacy stop-and-go path
#         TPTimePredictor -> subprocess.run(bin/SimAI_simulator) -> CSV parse
#   new : in-memory runtime path
#         TPTimePredictor -> pysimai.PySimAIBridge (pybind11)
#
# The two paths are selected at import time by communication_time_predictor.py:
#   `from pysimai import PySimAIBridge` succeeds -> new runtime
#   import fails                                  -> old subprocess path
# Therefore each mode MUST run in its own OS process with a different PYTHONPATH:
#   old: PYTHONPATH=<repo>/vidur-alibabacloud
#   new: PYTHONPATH=<repo>/vidur-alibabacloud:<repo>/bin
#
# Requirements (see AGENTS.md; this project builds/runs on the REMOTE host):
#   source /opt/miniforge3/etc/profile.d/conda.sh && conda activate vidur
#   <repo>/bin/SimAI_simulator          (old path)
#   <repo>/bin/pysimai*.so              (new path)
#   <repo>/Spectrum-X_128g_8gps_100Gbps_A100   (topology)
#   <repo>/astra-sim-alibabacloud/inputs/config/SimAI.conf
#
# Usage:
#   bash tests/pysimai/run_readme_simai_simulation.sh [options]
#
# Options:
#   --mode old|new|both   Which path(s) to run (default: both)
#   --build               Build the ns3 optimized profile first
#   --gen-topo            Generate the README topology if it is missing
#   --no-clean            Do not clear the old path's on-disk caches
#   -h, --help            Show this help
#
# Outputs (one directory per mode, plus logs):
#   readme_simai_simulation_results/<timestamp>/
#   ├── old/run.log, old/request_metrics.csv, old/config.json, ...
#   └── new/run.log, new/request_metrics.csv, new/config.json, ...
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VIDUR_DIR="$ROOT/vidur-alibabacloud"
BIN_DIR="$ROOT/bin"
TOPO="$ROOT/Spectrum-X_128g_8gps_100Gbps_A100"
CONF="$ROOT/astra-sim-alibabacloud/inputs/config/SimAI.conf"
TRACE="$VIDUR_DIR/data/processed_traces/splitwise_conv.csv"
GEN_TOPO="$ROOT/astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py"

MODE="both"
DO_BUILD=0
DO_GEN_TOPO=0
DO_CLEAN=1

usage() {
    sed -n '2,41p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="${2:?--mode requires old|new|both}"; shift 2 ;;
        --build) DO_BUILD=1; shift ;;
        --gen-topo) DO_GEN_TOPO=1; shift ;;
        --no-clean) DO_CLEAN=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

case "$MODE" in
    old|new|both) ;;
    *) echo "[ERROR] --mode must be old, new, or both (got: $MODE)" >&2; exit 2 ;;
esac

log()  { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Environment / artifact checks
# ---------------------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "vidur" ]]; then
    warn "conda env is '${CONDA_DEFAULT_ENV:-<none>}', expected 'vidur'."
    warn "Activate it first: source /opt/miniforge3/etc/profile.d/conda.sh && conda activate vidur"
fi

if [[ "$DO_BUILD" -eq 1 ]]; then
    log "Building ns3 optimized profile (this also builds pysimai when pybind11 is available)..."
    ( cd "$ROOT" && ./scripts/build.sh -c ns3 -d optimized )
fi

if [[ "$DO_GEN_TOPO" -eq 1 && ! -f "$TOPO" ]]; then
    log "Generating topology: $(basename "$TOPO")"
    ( cd "$ROOT" && python3 "$GEN_TOPO" \
        -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps )
fi

[[ -d "$VIDUR_DIR" ]] || die "vidur directory not found: $VIDUR_DIR"
[[ -f "$CONF" ]]      || die "SimAI config not found: $CONF"
[[ -f "$TRACE" ]]     || die "trace file not found: $TRACE"

if [[ "$MODE" == "old" || "$MODE" == "both" ]]; then
    [[ -x "$BIN_DIR/SimAI_simulator" ]] || \
        die "old path needs $BIN_DIR/SimAI_simulator (run with --build)"
fi
if [[ "$MODE" == "new" || "$MODE" == "both" ]]; then
    compgen -G "$BIN_DIR/pysimai*.so" >/dev/null || \
        die "new path needs $BIN_DIR/pysimai*.so (run with --build)"
fi
if [[ "$MODE" == "old" || "$MODE" == "both" ]]; then
    [[ -f "$TOPO" ]] || die "topology not found: $TOPO (run with --gen-topo)"
fi

# ---------------------------------------------------------------------------
# Old-path cache cleanup
#
# The legacy predictor caches workloads and per-command CSVs inside simai_dir.
# Clear them so the old run actually exercises subprocess.run + CSV parsing.
# ---------------------------------------------------------------------------
clean_old_caches() {
    rm -rf "$ROOT/tmp_simai_inference_workload"
    rm -f "$ROOT"/ncclFlowModel__EndToEnd.csv \
          "$ROOT"/ncclFlowModel_EndToEnd.csv \
          "$ROOT"/ncclFlowModel_EndToEnd_*.csv \
          "$ROOT"/ncclFlowModel__detailed_*.csv \
          "$ROOT"/ncclFlowModel__dimension_utilization_*.csv \
          "$ROOT"/ncclFlowModel_detailed_flows.csv
}

# ---------------------------------------------------------------------------
# The README example (Llama-3-8B, SimAI Simulation backend)
# ---------------------------------------------------------------------------
run_mode() {
    local mode="$1"
    local out_dir="$OUT_ROOT/$mode"
    local py_path="$VIDUR_DIR"
    mkdir -p "$out_dir"

    if [[ "$mode" == "new" ]]; then
        py_path="$VIDUR_DIR:$BIN_DIR"
    fi

    # Fail fast if the requested path is not actually selected.
    if [[ "$mode" == "old" ]]; then
        if PYTHONPATH="$py_path" python -c \
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pysimai') is None else 1)"; then
            :
        else
            die "old mode requested but 'pysimai' is importable. Remove it from site-packages or unset PYTHONPATH=$BIN_DIR."
        fi
        if [[ "$DO_CLEAN" -eq 1 ]]; then
            log "Clearing old-path disk caches under $ROOT"
            clean_old_caches
        fi
    else
        PYTHONPATH="$py_path" python -c "import pysimai" 2>/dev/null || \
            die "new mode requested but 'pysimai' is not importable from $BIN_DIR."
    fi

    log "Running mode=$mode"
    log "  PYTHONPATH=$py_path"
    log "  output=$out_dir"

    (
        cd "$VIDUR_DIR"
        AS_SEND_LAT=6 AS_NVLS_ENABLE=1 PYTHONPATH="$py_path" \
        python -m vidur.main \
            --replica_config_pd_p2p_comm_bandwidth 800 \
            --replica_config_nvlink_bandwidth 1600 \
            --replica_config_rdma_bandwidth 800 \
            --replica_config_pd_p2p_comm_dtype float32 \
            --poisson_request_interval_generator_config_qps 100 \
            --synthetic_request_generator_config_num_requests 10 \
            --length_generator_config_type trace \
            --trace_request_length_generator_config_max_tokens 2048 \
            --trace_request_length_generator_config_trace_file "$TRACE" \
            --interval_generator_config_type poisson \
            --cluster_config_num_replicas 4 \
            --replica_config_pd_node_ratio 0.5 \
            --global_scheduler_config_type split_wise \
            --replica_scheduler_config_type split_wise \
            --replica_config_model_name meta-llama/Meta-Llama-3-8B \
            --replica_config_tensor_parallel_size 4 \
            --replica_config_num_pipeline_stages 1 \
            --random_forrest_execution_time_predictor_config_backend simai_simulation \
            --random_forrest_execution_time_predictor_config_simai_dir "$ROOT" \
            --random_forrest_execution_time_predictor_config_simai_simulation_topo "$TOPO" \
            --random_forrest_execution_time_predictor_config_simai_simulation_config "$CONF" \
            --metrics_config_output_dir "$out_dir"
    ) 2>&1 | tee "$out_dir/run.log"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="$ROOT/readme_simai_simulation_results/$TIMESTAMP"
mkdir -p "$OUT_ROOT"

log "Repo root : $ROOT"
log "Output    : $OUT_ROOT"

if [[ "$MODE" == "old" || "$MODE" == "both" ]]; then
    run_mode old
fi
if [[ "$MODE" == "new" || "$MODE" == "both" ]]; then
    run_mode new
fi

log "Done. Results:"
for d in "$OUT_ROOT"/*/; do
    [[ -d "$d" ]] || continue
    if [[ -f "$d/request_metrics.csv" ]]; then
        log "  $(basename "$d"): $(find "$d" -maxdepth 1 -name request_metrics.csv) (OK)"
    else
        warn "  $(basename "$d"): request_metrics.csv not found (check run.log)"
    fi
done
