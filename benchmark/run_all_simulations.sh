#!/bin/bash

# ASTRA-Sim Benchmark Runner (128-GPU NCCL Subset)

set -e

export AS_LOG_LEVEL=2
export AS_PXN_ENABLE=1
export AS_SEND_LAT=3000

# Configuration
BASE_DIR="astra-sim-alibabacloud"
RESULTS_DIR="benchmark/results"
THREADS=1 # threads per simulation
WORKLOAD_TYPE="partials"

# Number of NUMA nodes to use
NUM_NUMA_NODES=2

# Parallel & Retry Configuration
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-10}
MAX_RETRIES=${MAX_RETRIES:-2}
RETRY_DELAY=${RETRY_DELAY:-5}

# Create results directory
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "${RESULTS_DIR}/e2e_simulations_${TIMESTAMP}/output"
OUTPUT_CSV="${RESULTS_DIR}/e2e_simulations_${TIMESTAMP}/simulation_results.csv"

# Initialize CSV with header
echo "workload,failure_type,algorithm,num_gpus,simulation_time_ns,run_name,status" > "$OUTPUT_CSV"

# Fixed parameters for 128-GPU NCCL experiments
WORKLOADS=("DeepSeek671B-transformer" "DeepSeek671B-grad")
FAILURE_TYPES=("no_failure" "failure_1" "failure_2" "failure_3")
ALGORITHM="nccl"
NUM_GPUS=128
TOPO_DIM="1x16x8"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

TOTAL_RUNS=$(( ${#WORKLOADS[@]} * ${#FAILURE_TYPES[@]} ))

echo "========================================"
echo "ASTRA-Sim Benchmark Suite (128-GPU NCCL)"
echo "========================================"
echo -e "${CYAN}Configuration:${NC}"
echo "  Workloads: ${WORKLOADS[*]}"
echo "  Failure Types: ${FAILURE_TYPES[*]}"
echo "  Algorithm: $ALGORITHM"
echo "  GPU Count: $NUM_GPUS ($TOPO_DIM)"
echo "  NUMA Nodes: $NUM_NUMA_NODES"
echo "  Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "  Max Retries: $MAX_RETRIES (delay ${RETRY_DELAY}s)"
echo "========================================"
echo ""

select_simulator() {
    local numa_node=$1
    echo "numactl --cpunodebind=$numa_node --membind=$numa_node ./bin/SimAI_simulator"
}

get_numa_node() {
    local job_id=$1
    echo $((job_id % NUM_NUMA_NODES))
}

run_simulation() {
    local workload="$1"
    local failure_type="$2"
    local run_id="$3"

    local NUMA_NODE=$(get_numa_node "$run_id")
    local SIMULATOR=$(select_simulator "$NUMA_NODE")

    RUN_NAME="${workload}_${failure_type}_${ALGORITHM}_${NUM_GPUS}gpus"

    # Construct file paths
    WORKLOAD_DIR="$BASE_DIR/inputs/workloads/end_to_end/$WORKLOAD_TYPE/$workload"
    WORKLOAD_FILE=$(find "$WORKLOAD_DIR" -name "*world_size$((NUM_GPUS*16))*" -type f | head -n 1)

    TOPO_BASE="$BASE_DIR/inputs/topo/end_to_end/HPN/$TOPO_DIM"
    if [ "$failure_type" == "no_failure" ]; then
        TOPO_PATH="$TOPO_BASE"
    else
        TOPO_PATH="$TOPO_BASE-$failure_type"
    fi

    CONFIG_FILE="$BASE_DIR/inputs/config/SimAI.conf"
    SYS_CONFIG_FILE="$BASE_DIR/inputs/system/${ALGORITHM}.txt"

    (
        flock -x 200
        echo -e "${YELLOW}[$run_id/$TOTAL_RUNS]${NC} Running: $RUN_NAME ${CYAN}(NUMA $NUMA_NODE)${NC}"
    ) 200>/tmp/benchmark_lock

    if [ ! -f "$WORKLOAD_FILE" ]; then
        (
            flock -x 200
            echo -e "${RED}ERROR: Workload not found: $WORKLOAD_FILE${NC}"
            echo "$workload,$failure_type,$ALGORITHM,$NUM_GPUS,-1,$RUN_NAME,WORKLOAD_NOT_FOUND" >> "$OUTPUT_CSV"
        ) 200>/tmp/benchmark_lock
        return
    fi

    if [ ! -f "$TOPO_PATH" ]; then
        (
            flock -x 200
            echo -e "${RED}ERROR: Topology not found: $TOPO_PATH${NC}"
            echo "$workload,$failure_type,$ALGORITHM,$NUM_GPUS,-1,$RUN_NAME,TOPO_NOT_FOUND" >> "$OUTPUT_CSV"
        ) 200>/tmp/benchmark_lock
        return
    fi

    if [ ! -f "$CONFIG_FILE" ]; then
        (
            flock -x 200
            echo -e "${RED}ERROR: Config not found: $CONFIG_FILE${NC}"
            echo "$workload,$failure_type,$ALGORITHM,$NUM_GPUS,-1,$RUN_NAME,CONFIG_NOT_FOUND" >> "$OUTPUT_CSV"
        ) 200>/tmp/benchmark_lock
        return
    fi

    if [ ! -f "$SYS_CONFIG_FILE" ]; then
        (
            flock -x 200
            echo -e "${RED}ERROR: Sys config not found: $SYS_CONFIG_FILE${NC}"
            echo "$workload,$failure_type,$ALGORITHM,$NUM_GPUS,-1,$RUN_NAME,SYS_CONFIG_NOT_FOUND" >> "$OUTPUT_CSV"
        ) 200>/tmp/benchmark_lock
        return
    fi

    OUTPUT_FILE="${RESULTS_DIR}/e2e_simulations_${TIMESTAMP}/output/${RUN_NAME}"
    local max_attempts=$((MAX_RETRIES + 1))
    local attempt=1
    local status=""
    SIM_TIME=""

    while [ $attempt -le $max_attempts ]; do
        echo $SIMULATOR -t $THREADS -w "$WORKLOAD_FILE" -n "$TOPO_PATH" -c "$CONFIG_FILE" -s "$SYS_CONFIG_FILE" -r "$RUN_NAME" > "$OUTPUT_FILE"
        local rc=0
        $SIMULATOR -t $THREADS -w "$WORKLOAD_FILE" -n "$TOPO_PATH" -c "$CONFIG_FILE" -s "$SYS_CONFIG_FILE" -r "$RUN_NAME" >> "$OUTPUT_FILE" 2>&1 || rc=$?
        if [ $rc -eq 0 ]; then
            SIM_TIME=$(grep "All messages finished. Stopping simulation at time" "$OUTPUT_FILE" | tail -n 1 | grep -oP 'time \K[0-9]+' || echo "")
            if [ -n "$SIM_TIME" ]; then
                status="SUCCESS"
                break
            fi
            status="PARSE_ERROR"
        elif [ $rc -eq 2 ]; then
            # The simulator ran out of events with messages pending (it prints what is pending).
            # Deterministic: a retry would stall the same way.
            status="STALLED"
            break
        else
            status="SIM_FAILED"
        fi

        mv "$OUTPUT_FILE" "${OUTPUT_FILE}.attempt${attempt}"
        if [ $attempt -lt $max_attempts ]; then
            (
                flock -x 200
                echo -e "${YELLOW}[$run_id/$TOTAL_RUNS] RETRY: $RUN_NAME failed with $status (attempt $attempt/$max_attempts), retrying in ${RETRY_DELAY}s...${NC}"
            ) 200>/tmp/benchmark_lock
            sleep "$RETRY_DELAY"
        else
            mv "${OUTPUT_FILE}.attempt${attempt}" "$OUTPUT_FILE"
        fi
        attempt=$((attempt + 1))
    done

    if [ "$status" == "SUCCESS" ]; then
        local retry_note=""
        [ $attempt -gt 1 ] && retry_note=" (after $attempt attempts)"
        (
            flock -x 200
            echo -e "${GREEN}[$run_id/$TOTAL_RUNS] SUCCESS: $RUN_NAME = $SIM_TIME ns${retry_note}${NC}"
            echo "$workload,$failure_type,$ALGORITHM,$NUM_GPUS,$SIM_TIME,$RUN_NAME,SUCCESS" >> "$OUTPUT_CSV"
        ) 200>/tmp/benchmark_lock
    else
        (
            flock -x 200
            if [ "$status" == "PARSE_ERROR" ]; then
                echo -e "${RED}[$run_id/$TOTAL_RUNS] ERROR: Could not parse simulation time for $RUN_NAME after $max_attempts attempts${NC}"
            elif [ "$status" == "STALLED" ]; then
                echo -e "${RED}[$run_id/$TOTAL_RUNS] ERROR: Simulation stalled for $RUN_NAME (pending messages listed in $OUTPUT_FILE)${NC}"
            else
                echo -e "${RED}[$run_id/$TOTAL_RUNS] ERROR: Simulation failed for $RUN_NAME after $max_attempts attempts${NC}"
            fi
            echo "$workload,$failure_type,$ALGORITHM,$NUM_GPUS,-1,$RUN_NAME,$status" >> "$OUTPUT_CSV"
        ) 200>/tmp/benchmark_lock
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
    for failure_type in "${FAILURE_TYPES[@]}"; do
        JOB_ID=$((JOB_ID + 1))
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL_JOBS" ]; do
            wait -n || true
        done
        run_simulation "$workload" "$failure_type" "$JOB_ID" &
    done
done
wait || true

rm -f /tmp/benchmark_lock

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