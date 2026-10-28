#!/bin/bash
# MCCL artifact runner (CoNEXT 2026 AE).
#
# Usage: artifact/run.sh <experiment> [options]
#
# Experiments
#   smoke   E1-quick: MSCCL-vs-native equivalence, no failure, 256 MB AG + A2A   (4 sims,  ~15 s)
#   equiv   E1:       MSCCL-vs-native equivalence, 4 workloads x 4 topologies    (32 sims, ~4 min)
#   single  E2:       16-GPU single collectives (AG/A2A, 256 MB and 1 GB) on the
#                     2x8 rail-optimized topology, no failure + Fail1/2/3,
#                     through the custom-schedule path (nccl) and SimAI native  (32 sims, ~4 min)
#   all     equiv + single
#   (times measured with 2 CPUs)
#
# Options
#   -b PATH   simulator binary            (default: bin/SimAI_simulator)
#   -o DIR    output directory            (default: artifact/results/<experiment>-<timestamp>)
#   -j N      parallel simulations        (default: number of CPUs)
#   -f LIST   failure types (comma list)  (default: all four for the experiment)
#   -h        help
#
# Every simulation runs in its own directory (run.log, exit_code, csv_files/), so runs never
# overwrite each other. A results.csv is written per experiment; summarize it with
#   python3 artifact/summarize.py <output dir>

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/astra-sim-alibabacloud"
BINARY="$ROOT/bin/SimAI_simulator"
OUT=""
JOBS="$(nproc 2>/dev/null || echo 2)"
FAILURE_OVERRIDE=""

usage() { sed -n '2,24p' "$0"; }

EXP="${1:-}"
[ -n "$EXP" ] || { usage >&2; exit 2; }
[ "$EXP" = "-h" ] || [ "$EXP" = "--help" ] && { usage; exit 0; }
shift

while getopts "b:o:j:f:h" opt; do
  case "$opt" in
    b) BINARY="$(realpath "$OPTARG")" ;;
    o) OUT="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    f) FAILURE_OVERRIDE="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

case "$EXP" in smoke|equiv|single|all) ;; *) echo "Unknown experiment: $EXP" >&2; usage >&2; exit 2 ;; esac
[ -x "$BINARY" ] || { echo "Simulator not found: $BINARY (build with: bash scripts/build.sh -c ns3)" >&2; exit 2; }

[ -n "$OUT" ] || OUT="$ROOT/artifact/results/${EXP}-$(date +%Y%m%d_%H%M%S)"
if [ -e "$OUT" ]; then echo "Output directory already exists: $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

NET_CONF="$BASE/inputs/config/SimAI.conf"

# Record provenance of this run.
{
  echo "date:        $(date -Iseconds)"
  echo "experiment:  $EXP"
  echo "host:        $(uname -a)"
  echo "cpus:        $(nproc 2>/dev/null)"
  echo "jobs:        $JOBS"
  echo "binary:      $BINARY"
  echo "binary_sha:  $(sha256sum "$BINARY" | cut -d' ' -f1)"
  echo "git:         $(git -C "$ROOT" rev-parse HEAD 2>/dev/null) $(git -C "$ROOT" status --porcelain 2>/dev/null | grep -q . && echo '(dirty)')"
  git -C "$ROOT" submodule status 2>/dev/null | sed 's/^/submodule:   /'
} > "$OUT/provenance.txt"

end_time() { grep -oP 'All messages finished. Stopping simulation at time \K[0-9]+' "$1" 2>/dev/null | tail -n 1; }

# run_sim <dir> <workload> <topology> <system.txt> <run-name>
# msccl-path in the sys files is relative to the repo root, so each run directory gets an
# astra-sim-alibabacloud symlink; ./csv_files is the simulator's result path.
run_sim() {
  local dir="$1" wl="$2" topo="$3" sys="$4" name="$5"
  mkdir -p "$dir/csv_files"
  ln -sfn "$BASE" "$dir/astra-sim-alibabacloud"
  for f in "$wl" "$topo" "$sys"; do
    [ -f "$f" ] || { echo "missing input: $f" > "$dir/run.log"; echo 2 > "$dir/exit_code"; return; }
  done
  echo "$BINARY -t 1 -w $wl -n $topo -c $NET_CONF -s $sys -r $name" > "$dir/command.txt"
  local t0=$SECONDS
  (cd "$dir" && "$BINARY" -t 1 -w "$wl" -n "$topo" -c "$NET_CONF" -s "$sys" -r "$name" > run.log 2>&1)
  echo $? > "$dir/exit_code"
  echo $((SECONDS - t0)) > "$dir/wall_seconds"
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done; }

# collect <csv> <dir>... : one line per run directory, from its run.log
collect_row() {
  local dir="$1"; shift
  local log="$dir/run.log" t status schedule
  t="$(end_time "$log")"
  if [ -z "$t" ]; then status="FAILED"; else status="SUCCESS"; fi
  if grep -q "Found matching MSCCL XML algorithm" "$log" 2>/dev/null; then schedule="msccl"; else schedule="native"; fi
  if grep -q "No matching MSCCL XML algorithm" "$log" 2>/dev/null; then schedule="${schedule}+fallback"; fi
  echo "$*,${t:-},$(cat "$dir/wall_seconds" 2>/dev/null),$schedule,$status"
}

# ---------------------------------------------------------------- E1 equivalence
run_equiv() {
  local quick="$1" out="$2"
  local args=(-b "$BINARY" -o "$out" -j "$JOBS")
  if [ "$quick" = 1 ]; then args+=(-q); fi
  [ -n "$FAILURE_OVERRIDE" ] && [ "$quick" = 0 ] && args+=(-f "$FAILURE_OVERRIDE")
  echo "== E1: MSCCL-vs-native equivalence ($([ "$quick" = 1 ] && echo quick || echo full))"
  "$ROOT/tests/ns3/msccl-nccl-equivalence/run.sh" "${args[@]}"
}

# ---------------------------------------------------------------- E2 single collectives
run_single() {
  local out="$1"
  # Paper configuration: protocol-aware MSCCL path (512 KB chunking), PXN, 3 us send latency.
  export AS_LOG_LEVEL=2 AS_PXN_ENABLE=1 AS_SEND_LAT=3000 SIMAI_DUMP_DETAILED_FLOWS=0
  unset SIMAI_PROTO_AWARE
  local wl_dir="$BASE/inputs/workloads/single_collective/16_gpus"
  local topo_base="$BASE/inputs/topo/single_collective/1x2x8"
  local workloads=(ag-268435456_bytes ag-1073741824_bytes a2a-268435456_bytes a2a-1073741824_bytes)
  local algs=(nccl simai_native)
  local failures; IFS=',' read -r -a failures <<< "${FAILURE_OVERRIDE:-no_failure,failure1,failure2,failure3}"
  echo "== E2: single collectives, 16 GPUs (1x2x8): $(( ${#workloads[@]} * ${#algs[@]} * ${#failures[@]} )) simulations, $JOBS in parallel"
  for wl in "${workloads[@]}"; do for ft in "${failures[@]}"; do for alg in "${algs[@]}"; do
    local topo="$topo_base"; [ "$ft" != no_failure ] && topo="$topo_base-$ft"
    throttle
    run_sim "$out/$wl/$ft/$alg" "$wl_dir/$wl" "$topo" "$BASE/inputs/system/$alg.txt" "e2_${wl}_${ft}_${alg}" &
  done; done; done
  wait
  local csv="$out/results.csv"
  echo "workload,failure_type,algorithm,sim_time_ns,wall_seconds,schedule,status" > "$csv"
  for wl in "${workloads[@]}"; do for ft in "${failures[@]}"; do for alg in "${algs[@]}"; do
    collect_row "$out/$wl/$ft/$alg" "$wl,$ft,$alg" >> "$csv"
  done; done; done
  (command -v column >/dev/null && column -s, -t < "$csv") || cat "$csv"
}

rc=0
case "$EXP" in
  smoke)  run_equiv 1 "$OUT/E1-equivalence" || rc=1 ;;
  equiv)  run_equiv 0 "$OUT/E1-equivalence" || rc=1 ;;
  single) run_single "$OUT/E2-single" ;;
  all)
    run_equiv 0 "$OUT/E1-equivalence" || rc=1
    run_single "$OUT/E2-single"
    ;;
esac
if [ -f "$OUT/E2-single/results.csv" ] && grep -q ",FAILED$" "$OUT/E2-single/results.csv"; then rc=1; fi

echo
echo "Results in: $OUT"
echo "Summarize:  python3 artifact/summarize.py $OUT"
exit $rc
