#!/bin/bash

# ASTRA-Sim Benchmark Runner (16-GPU single collectives, NCCL vs SimAI native)
# Lives in SimAIGitHubNetMeasurements/benchmark/; run from the repo root:
#   ./benchmark/run_single_collectives.sh
# All paths below are relative to the repo root.

set -e

export AS_LOG_LEVEL=2
export AS_PXN_ENABLE=1
export AS_SEND_LAT=3

# Configuration
BASE_DIR="astra-sim-alibabacloud"
SIM_BIN="./bin/SimAI_simulator"
RESULTS_DIR="benchmark/results"
THREADS=1 # threads per simulation

# Number of NUMA nodes to use
NUM_NUMA_NODES=2

# Parallel & Retry Configuration
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-10}
MAX_RETRIES=${MAX_RETRIES:-2}
RETRY_DELAY=${RETRY_DELAY:-5}

# Create results directory
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${RESULTS_DIR}/single_coll_${TIMESTAMP}/output"
OUTPUT_CSV="${RESULTS_DIR}//single_coll_${TIMESTAMP}/simulation_results.csv"
LOCK_FILE="${RESULTS_DIR}/single_coll_${TIMESTAMP}/.lock"

# Initialize CSV with header
echo "workload,failure_type,algorithm,num_gpus,simulation_time_ns,run_name,status" > "$OUTPUT_CSV"

# Fixed parameters for 16-GPU single-collective experiments
NUM_GPUS=16
WORKLOAD_DIR="$BASE_DIR/inputs/workloads/single_collective/${NUM_GPUS}_gpus"
WORKLOADS=("a2a-268435456_bytes" "a2a-1073741824_bytes" "ag-268435456_bytes" "ag-1073741824_bytes")
ALGORITHMS=("nccl" "simai_native")

# Topology for 16 GPUs (2 nodes x 8 GPUs).
# For failure types other than no_failure, "<TOPO_BASE>-<failure_type>" is used.
TOPO_DIM="1x2x8"
TOPO_BASE="$BASE_DIR/inputs/topo/single_collective/$TOPO_DIM"
FAILURE_TYPES=("no_failure" "failure1" "failure2" "failure3")

CONFIG_FILE="$BASE_DIR/inputs/config/SimAI.conf"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

TOTAL_RUNS=$(( ${#WORKLOADS[@]} * ${#ALGORITHMS[@]} * ${#FAILURE_TYPES[@]} ))

echo "========================================"
echo "ASTRA-Sim Single-Collective Suite (16 GPUs)"
echo "========================================"
echo -e "${CYAN}Configuration:${NC}"
echo "  Workloads: ${WORKLOADS[*]}"
echo "  Algorithms: ${ALGORITHMS[*]}"
echo "  Failure Types: ${FAILURE_TYPES[*]}"
echo "  GPU Count: $NUM_GPUS ($TOPO_DIM)"
echo "  Topology base: $TOPO_BASE"
echo "  NUMA Nodes: $NUM_NUMA_NODES"
echo "  Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "  Max Retries: $MAX_RETRIES (delay ${RETRY_DELAY}s)"
echo "========================================"
echo ""

select_simulator() {
    local numa_node=$1
    echo "numactl --cpunodebind=$numa_node --membind=$numa_node $SIM_BIN"
}

get_numa_node() {
    local job_id=$1
    echo $((job_id % NUM_NUMA_NODES))
}

log_locked() {
    # usage: log_locked "<console message>" ["<csv line>"]
    (
        flock -x 200
        echo -e "$1"
        [ -n "$2" ] && echo "$2" >> "$OUTPUT_CSV"
        true
    ) 200>"$LOCK_FILE"
}

run_simulation() {
    local workload="$1"
    local algorithm="$2"
    local failure_type="$3"
    local run_id="$4"

    local NUMA_NODE=$(get_numa_node "$run_id")
    local SIMULATOR=$(select_simulator "$NUMA_NODE")

    local RUN_NAME="${workload}_${failure_type}_${algorithm}_${NUM_GPUS}gpus"
    local WORKLOAD_FILE="$WORKLOAD_DIR/$workload"
    local SYS_CONFIG_FILE="$BASE_DIR/inputs/system/${algorithm}.txt"

    local TOPO_PATH
    if [ "$failure_type" == "no_failure" ]; then
        TOPO_PATH="$TOPO_BASE"
    else
        TOPO_PATH="$TOPO_BASE-$failure_type"
    fi

    local CSV_PREFIX="$workload,$failure_type,$algorithm,$NUM_GPUS"

    log_locked "${YELLOW}[$run_id/$TOTAL_RUNS]${NC} Running: $RUN_NAME ${CYAN}(NUMA $NUMA_NODE)${NC}"

    if [ ! -f "$WORKLOAD_FILE" ]; then
        log_locked "${RED}ERROR: Workload not found: $WORKLOAD_FILE${NC}" "$CSV_PREFIX,-1,$RUN_NAME,WORKLOAD_NOT_FOUND"
        return
    fi
    if [ ! -f "$TOPO_PATH" ]; then
        log_locked "${RED}ERROR: Topology not found: $TOPO_PATH${NC}" "$CSV_PREFIX,-1,$RUN_NAME,TOPO_NOT_FOUND"
        return
    fi
    if [ ! -f "$CONFIG_FILE" ]; then
        log_locked "${RED}ERROR: Config not found: $CONFIG_FILE${NC}" "$CSV_PREFIX,-1,$RUN_NAME,CONFIG_NOT_FOUND"
        return
    fi
    if [ ! -f "$SYS_CONFIG_FILE" ]; then
        log_locked "${RED}ERROR: Sys config not found: $SYS_CONFIG_FILE${NC}" "$CSV_PREFIX,-1,$RUN_NAME,SYS_CONFIG_NOT_FOUND"
        return
    fi

    local OUTPUT_FILE="${RESULTS_DIR}//single_coll_${TIMESTAMP}/output/${RUN_NAME}"
    local max_attempts=$((MAX_RETRIES + 1))
    local attempt=1
    local status=""
    local SIM_TIME=""

    while [ $attempt -le $max_attempts ]; do
        echo $SIMULATOR -t $THREADS -w "$WORKLOAD_FILE" -n "$TOPO_PATH" -c "$CONFIG_FILE" -s "$SYS_CONFIG_FILE" -r "$RUN_NAME" > "$OUTPUT_FILE"
        if $SIMULATOR -t $THREADS -w "$WORKLOAD_FILE" -n "$TOPO_PATH" -c "$CONFIG_FILE" -s "$SYS_CONFIG_FILE" -r "$RUN_NAME" >> "$OUTPUT_FILE" 2>&1; then
            SIM_TIME=$(grep "All messages finished. Stopping simulation at time" "$OUTPUT_FILE" | tail -n 1 | grep -oP 'time \K[0-9]+' || echo "")
            if [ -n "$SIM_TIME" ]; then
                status="SUCCESS"
                break
            fi
            status="PARSE_ERROR"
        else
            status="SIM_FAILED"
        fi

        mv "$OUTPUT_FILE" "${OUTPUT_FILE}.attempt${attempt}"
        if [ $attempt -lt $max_attempts ]; then
            log_locked "${YELLOW}[$run_id/$TOTAL_RUNS] RETRY: $RUN_NAME failed with $status (attempt $attempt/$max_attempts), retrying in ${RETRY_DELAY}s...${NC}"
            sleep "$RETRY_DELAY"
        else
            mv "${OUTPUT_FILE}.attempt${attempt}" "$OUTPUT_FILE"
        fi
        attempt=$((attempt + 1))
    done

    if [ "$status" == "SUCCESS" ]; then
        local retry_note=""
        if [ $attempt -gt 1 ]; then retry_note=" (after $attempt attempts)"; fi
        log_locked "${GREEN}[$run_id/$TOTAL_RUNS] SUCCESS: $RUN_NAME = $SIM_TIME ns${retry_note}${NC}" \
                   "$CSV_PREFIX,$SIM_TIME,$RUN_NAME,SUCCESS"
    else
        local msg
        if [ "$status" == "PARSE_ERROR" ]; then
            msg="${RED}[$run_id/$TOTAL_RUNS] ERROR: Could not parse simulation time for $RUN_NAME after $max_attempts attempts${NC}"
        else
            msg="${RED}[$run_id/$TOTAL_RUNS] ERROR: Simulation failed for $RUN_NAME after $max_attempts attempts${NC}"
        fi
        log_locked "$msg" "$CSV_PREFIX,-1,$RUN_NAME,$status"
    fi
}

echo "Simulations to execute: $TOTAL_RUNS"
echo "Parallel jobs: $MAX_PARALLEL_JOBS"
echo -e "Results will be saved to: ${BLUE}$OUTPUT_CSV${NC}"
echo "========================================"
echo ""

echo -e "${CYAN}Running $TOTAL_RUNS jobs (max $MAX_PARALLEL_JOBS in parallel)...${NC}"

JOB_ID=0
for workload in "${WORKLOADS[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
        for failure_type in "${FAILURE_TYPES[@]}"; do
            JOB_ID=$((JOB_ID + 1))
            while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL_JOBS" ]; do
                wait -n || true
            done
            run_simulation "$workload" "$algorithm" "$failure_type" "$JOB_ID" &
        done
    done
done
wait || true

rm -f "$LOCK_FILE"

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo -e "Results saved to: ${BLUE}$OUTPUT_CSV${NC}"
echo ""
echo "Summary:"
SUCCESSFUL=$(grep -c ",SUCCESS$" "$OUTPUT_CSV" || true)
TOTAL_IN_CSV=$(tail -n +2 "$OUTPUT_CSV" | wc -l)
FAILED=$((TOTAL_IN_CSV - SUCCESSFUL))
echo "  Successful: $SUCCESSFUL"
echo "  Failed: $FAILED"
echo "  Total rows: $TOTAL_IN_CSV"
echo "========================================"