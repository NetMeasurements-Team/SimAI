#!/bin/bash
# MSCCL vs NCCL equivalence test (16 GPUs, single collectives).
#
# Runs every single-collective workload twice, with the same topology and network settings:
#   nccl          -> all-gather / all-to-all through mscclCustomFlow (NcclCustom), using the
#                    NCCL-equivalent XML schedules in astra-sim-alibabacloud/inputs/msccl/nccl/
#   simai_native  -> the same collectives through ncclFlowModel (NcclFlowModel)
# and checks that the simulated end times are identical.
#
# MSCCL pipelining (512KB chunks) is disabled with SIMAI_PROTO_AWARE=0 (NCCL_PROTO_UNDEF):
# NcclFlowModel sends every step as one flow, so with chunking the MSCCL path legitimately
# starts the bottleneck links earlier (pipeline fill) and the times differ.
#
# Run from anywhere:   tests/ns3/msccl-nccl-equivalence/run.sh [options]
#   -b PATH   simulator binary            (default: bin/SimAI_simulator)
#   -o DIR    output directory            (default: simai-test-results-msccl-nccl-<timestamp>)
#   -j N      parallel simulations        (default: number of CPUs)
#   -f LIST   failure types, comma list   (default: no_failure,failure1,failure2,failure3)
#   -w LIST   workloads, comma list       (default: ag/a2a at 256MB and 1GB)
#   -q        quick: no_failure, 256MB workloads only
#   -h        help
# Exit status: 0 if every pair matches, 1 otherwise.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="$ROOT/astra-sim-alibabacloud"
BINARY="$ROOT/bin/SimAI_simulator"
OUT=""
JOBS="$(nproc 2>/dev/null || echo 2)"
FAILURES="no_failure,failure1,failure2,failure3"
WORKLOADS="ag-268435456_bytes,a2a-268435456_bytes,ag-1073741824_bytes,a2a-1073741824_bytes"

while getopts "b:o:j:f:w:qh" opt; do
  case "$opt" in
    b) BINARY="$(realpath "$OPTARG")" ;;
    o) OUT="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    f) FAILURES="$OPTARG" ;;
    w) WORKLOADS="$OPTARG" ;;
    q) FAILURES="no_failure"; WORKLOADS="ag-268435456_bytes,a2a-268435456_bytes" ;;
    h) sed -n '2,24p' "$0"; exit 0 ;;
    *) sed -n '2,24p' "$0" >&2; exit 2 ;;
  esac
done

# Fixed environment: the topology is rail-only (PXN required), 3 us send latency (value in ns),
# no MSCCL pipelining. Logs are kept at WARNING to keep the runs fast.
export AS_PXN_ENABLE=1
export AS_SEND_LAT=3000
export SIMAI_PROTO_AWARE=0
export AS_LOG_LEVEL=2
export SIMAI_DUMP_DETAILED_FLOWS=0

WORKLOAD_DIR="$BASE/inputs/workloads/single_collective/16_gpus"
TOPO_BASE="$BASE/inputs/topo/single_collective/1x2x8"
NET_CONF="$BASE/inputs/config/SimAI.conf"
ALGORITHMS=(nccl simai_native)

[ -x "$BINARY" ] || { echo "Simulator not found: $BINARY (build with scripts/build.sh -c ns3)" >&2; exit 2; }
for f in "$NET_CONF" "$BASE/inputs/system/nccl.txt" "$BASE/inputs/system/simai_native.txt"; do
  [ -f "$f" ] || { echo "Missing input: $f" >&2; exit 2; }
done
[ -d "$BASE/inputs/msccl/nccl" ] || { echo "Missing MSCCL schedules: $BASE/inputs/msccl/nccl" >&2; exit 2; }

[ -n "$OUT" ] || OUT="$ROOT/simai-test-results-msccl-nccl-$(date +%Y%m%d_%H%M%S)"
if [ -e "$OUT" ]; then echo "Output directory already exists: $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
TAG="mncl$$"

IFS=',' read -r -a FT_LIST <<< "$FAILURES"
IFS=',' read -r -a WL_LIST <<< "$WORKLOADS"

# One simulation in its own directory. msccl-path in nccl.txt is relative to the repo root,
# so each run directory gets an astra-sim-alibabacloud symlink; ./csv_files is the result path.
run_one() {
  local wl="$1" ft="$2" alg="$3"
  local topo="$TOPO_BASE"; [ "$ft" != no_failure ] && topo="$TOPO_BASE-$ft"
  local dir="$OUT/$wl/$ft/$alg"
  mkdir -p "$dir/csv_files"
  ln -sfn "$BASE" "$dir/astra-sim-alibabacloud"
  if [ ! -f "$WORKLOAD_DIR/$wl" ] || [ ! -f "$topo" ]; then
    echo "missing input: $WORKLOAD_DIR/$wl or $topo" > "$dir/run.log"
    return
  fi
  (cd "$dir" && "$BINARY" -t 1 -w "$WORKLOAD_DIR/$wl" -n "$topo" -c "$NET_CONF" \
      -s "$BASE/inputs/system/$alg.txt" -r "${TAG}_${wl}_${ft}_${alg}" > run.log 2>&1)
  echo $? > "$dir/exit_code"
}

echo "MSCCL vs NCCL equivalence test"
echo "  binary:    $BINARY"
echo "  output:    $OUT"
echo "  workloads: ${WL_LIST[*]}"
echo "  failures:  ${FT_LIST[*]}"
echo "  env:       AS_PXN_ENABLE=1 AS_SEND_LAT=3000 SIMAI_PROTO_AWARE=0 (no MSCCL chunking)"
echo "  running $(( ${#WL_LIST[@]} * ${#FT_LIST[@]} * 2 )) simulations, $JOBS in parallel..."

for wl in "${WL_LIST[@]}"; do
  for ft in "${FT_LIST[@]}"; do
    for alg in "${ALGORITHMS[@]}"; do
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
      run_one "$wl" "$ft" "$alg" &
    done
  done
done
wait

end_time() { grep -oP 'All messages finished. Stopping simulation at time \K[0-9]+' "$1" 2>/dev/null | tail -n 1; }

CSV="$OUT/results.csv"
echo "workload,failure_type,nccl_msccl_ns,simai_native_ns,status" > "$CSV"
printf "\n%-22s %-11s %15s %15s  %s\n" workload failure nccl_msccl_ns simai_native_ns status
failed=0
for wl in "${WL_LIST[@]}"; do
  for ft in "${FT_LIST[@]}"; do
    m_log="$OUT/$wl/$ft/nccl/run.log"
    n_log="$OUT/$wl/$ft/simai_native/run.log"
    m=$(end_time "$m_log"); n=$(end_time "$n_log")
    if [ -z "$m" ] || [ -z "$n" ]; then
      status="FAIL (simulation did not finish, see run.log)"
    elif ! grep -q "Found matching MSCCL XML algorithm" "$m_log" || grep -q "No matching MSCCL XML algorithm" "$m_log"; then
      status="FAIL (MSCCL schedule not used: fell back to NCCL)"
    elif grep -q "Using MSCCL Custom Flow" "$n_log"; then
      status="FAIL (simai_native used MSCCL)"
    elif [ "$m" != "$n" ]; then
      status="FAIL (mismatch: $(( m - n )) ns)"
    else
      status="PASS"
    fi
    [ "$status" = PASS ] || failed=$((failed + 1))
    printf "%-22s %-11s %15s %15s  %s\n" "$wl" "$ft" "${m:--}" "${n:--}" "$status"
    echo "$wl,$ft,${m:-},${n:-},$status" >> "$CSV"
  done
done

total=$(( ${#WL_LIST[@]} * ${#FT_LIST[@]} ))
echo
if [ "$failed" -eq 0 ]; then
  echo "PASS: $total/$total pairs identical. Results: $CSV"
  exit 0
fi
echo "FAIL: $failed/$total pairs differ or failed. Results: $CSV"
exit 1
