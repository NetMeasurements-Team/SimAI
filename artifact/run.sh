#!/bin/bash
# MCCL artifact runner (CoNEXT 2026 AE).
#
# Usage: artifact/run.sh <experiment> [options]
#
# Experiments
#   smoke   E1-quick: MSCCL-vs-native equivalence, no failure, 256 MB AG + A2A   (4 sims)
#   equiv   E1:       MSCCL-vs-native equivalence, 4 workloads x 4 topologies    (32 sims)
#   single  E2:       16-GPU single collectives (AG/A2A, 256 MB and 1 GB) on the
#                     2x8 rail-optimized topology, no failure + Fail1/2/3, with the
#                     NCCL-equivalent schedules through the custom-schedule path (16 sims)
#   mccl    E3:       MCCL schedules vs NCCL, 2 GB AllGather and AllToAll (128 MB per source)
#                     on 1x2x8: NCCL, MCCL no-reopt (no-failure schedule) and
#                     MCCL reopt (schedule synthesized for that failure),
#                     no failure + Fail1/2/3                                     (22 sims)
#   e2e     E4:       DeepSeek-671B training, one pipeline stage on 128 GPUs
#                     (1x16x8 HPN), transformer + gradient phases,
#                     no failure + Fail1/2/3                                     (8 sims)
#   all     equiv + single + mccl + e2e
#   (see ARTIFACT.md for run times)
#
# Options
#   -b PATH   simulator binary            (default: bin/SimAI_simulator)
#   -t TAG    run label                   (default: "docker" inside a container, else "native")
#   -o DIR    output directory            (default: artifact/results/<TAG>-<experiment>-<timestamp>)
#   -j N      parallel simulations        (default: number of CPUs)
#   -f LIST   failure types (comma list)  (default: no_failure,failure_1,failure_2,failure_3)
#   -h        help
#
# Every simulation runs in its own directory (command.txt, run.log, exit_code, wall_seconds,
# csv_files/), and its simulator run name (which names the ns-3 and log files the simulator
# writes under /etc/astra-sim) is unique to this invocation (<label><pid>_...). The default output
# directory is prefixed with the run label, so a native and a Docker run can go at the same time
# on the same checkout (Docker writes into artifact/results through the volume).
# A results.csv is written per experiment; summarize with:
#   python3 artifact/summarize.py <output dir>

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/astra-sim-alibabacloud"
BINARY="$ROOT/bin/SimAI_simulator"
OUT=""
TAG=""
JOBS="$(nproc 2>/dev/null || echo 2)"
FAILURES="no_failure,failure_1,failure_2,failure_3"

usage() { sed -n '2,36p' "$0"; }

EXP="${1:-}"
[ -n "$EXP" ] || { usage >&2; exit 2; }
if [ "$EXP" = "-h" ] || [ "$EXP" = "--help" ]; then usage; exit 0; fi
shift

while getopts "b:o:t:j:f:h" opt; do
  case "$opt" in
    b) BINARY="$(realpath "$OPTARG")" ;;
    o) OUT="$OPTARG" ;;
    t) TAG="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    f) FAILURES="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

case "$EXP" in smoke|equiv|single|mccl|e2e|all) ;; *) echo "Unknown experiment: $EXP" >&2; usage >&2; exit 2 ;; esac
[ -x "$BINARY" ] || { echo "Simulator not found: $BINARY (build with: bash scripts/build.sh -c ns3)" >&2; exit 2; }

if [ -z "$TAG" ]; then
  if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then TAG=docker; else TAG=native; fi
fi
[ -n "$OUT" ] || OUT="$ROOT/artifact/results/${TAG}-${EXP}-$(date +%Y%m%d_%H%M%S)"
if [ -e "$OUT" ]; then echo "Output directory already exists: $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

NET_CONF="$BASE/inputs/config/SimAI.conf"
RUN_PREFIX="${TAG}$$"   # unique per invocation: simulator file names under /etc/astra-sim never clash
IFS=',' read -r -a FAILURE_LIST <<< "$FAILURES"

# Record provenance of this run.
{
  echo "date:        $(date -Iseconds)"
  echo "experiment:  $EXP"
  echo "label:       $TAG"
  echo "host:        $(uname -a)"
  echo "cpus:        $(nproc 2>/dev/null)"
  echo "jobs:        $JOBS"
  echo "binary:      $BINARY"
  echo "binary_sha:  $(sha256sum "$BINARY" | cut -d' ' -f1)"
  echo "git:         $(git -C "$ROOT" rev-parse HEAD 2>/dev/null) $(git -C "$ROOT" status --porcelain 2>/dev/null | grep -q . && echo '(dirty)')"
  git -C "$ROOT" submodule status 2>/dev/null | sed 's/^/submodule:   /'
  [ -f "$ROOT/ARCHIVE_PROVENANCE.txt" ] && sed 's/^/archive:     /' "$ROOT/ARCHIVE_PROVENANCE.txt"
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
    [ -f "$f" ] || { echo "missing input: $f" > "$dir/run.log"; echo 127 > "$dir/exit_code"; return; }
  done
  name="${RUN_PREFIX}_$name"
  echo "$BINARY -t 1 -w $wl -n $topo -c $NET_CONF -s $sys -r $name" > "$dir/command.txt"
  local t0=$SECONDS
  (cd "$dir" && "$BINARY" -t 1 -w "$wl" -n "$topo" -c "$NET_CONF" -s "$sys" -r "$name" > run.log 2>&1)
  echo $? > "$dir/exit_code"
  echo $((SECONDS - t0)) > "$dir/wall_seconds"
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done; }

# collect_row <run dir> <leading csv fields>: one CSV line from the run's log and exit code.
# Exit code 2 = the simulator ran out of events with messages still pending (it lists them).
collect_row() {
  local dir="$1"; shift
  local log="$dir/run.log" t status schedule rc
  t="$(end_time "$log")"
  rc="$(cat "$dir/exit_code" 2>/dev/null || echo -)"
  if [ "$rc" = 0 ] && [ -n "$t" ]; then status="SUCCESS"
  elif [ "$rc" = 2 ]; then status="STALLED"
  elif [ "$rc" = 127 ]; then status="MISSING_INPUT"
  else status="FAILED"; fi
  if grep -q "Found matching MSCCL XML algorithm" "$log" 2>/dev/null; then schedule="msccl"; else schedule="native"; fi
  if grep -q "No matching MSCCL XML algorithm" "$log" 2>/dev/null; then schedule="${schedule}+fallback"; fi
  echo "$*,${t:-},$(cat "$dir/wall_seconds" 2>/dev/null),$schedule,$status"
}

show_csv() { (command -v column >/dev/null && column -s, -t < "$1") || cat "$1"; }

# Paper configuration for E2-E4: MSCCL steps pipelined in 512 KiB chunks as in NCCL/MSCCL (Simple
# protocol: 4 MiB NCCL_BUFFSIZE / 8 NCCL_STEPS), PXN, 3 us send latency.
paper_env() {
  export AS_LOG_LEVEL=2 AS_PXN_ENABLE=1 AS_SEND_LAT=3000 SIMAI_DUMP_DETAILED_FLOWS=0
  unset SIMAI_PROTO_AWARE
}

# ---------------------------------------------------------------- E1 equivalence
run_equiv() {
  local quick="$1" out="$2"
  local args=(-b "$BINARY" -o "$out" -j "$JOBS")
  if [ "$quick" = 1 ]; then args+=(-q); else args+=(-f "$FAILURES"); fi
  echo "== E1: MSCCL-vs-native equivalence ($([ "$quick" = 1 ] && echo quick || echo full))"
  "$ROOT/tests/ns3/msccl-nccl-equivalence/run.sh" "${args[@]}"
}

# ---------------------------------------------------------------- E2 single collectives
run_single() {
  local out="$1"
  paper_env
  local wl_dir="$BASE/inputs/workloads/single_collective/16_gpus"
  local topo_base="$BASE/inputs/topo/single_collective/1x2x8"
  local workloads=(ag-268435456_bytes ag-1073741824_bytes a2a-268435456_bytes a2a-1073741824_bytes)
  local algs=(nccl)
  echo "== E2: single collectives, 16 GPUs (1x2x8): $(( ${#workloads[@]} * ${#algs[@]} * ${#FAILURE_LIST[@]} )) simulations, $JOBS in parallel"
  for wl in "${workloads[@]}"; do for ft in "${FAILURE_LIST[@]}"; do for alg in "${algs[@]}"; do
    local topo="$topo_base"; [ "$ft" != no_failure ] && topo="$topo_base-$ft"
    throttle
    run_sim "$out/$wl/$ft/$alg" "$wl_dir/$wl" "$topo" "$BASE/inputs/system/$alg.txt" "e2_${wl}_${ft}_${alg}" &
  done; done; done
  wait
  local csv="$out/results.csv"
  echo "workload,failure_type,algorithm,sim_time_ns,wall_seconds,schedule,status" > "$csv"
  for wl in "${workloads[@]}"; do for ft in "${FAILURE_LIST[@]}"; do for alg in "${algs[@]}"; do
    collect_row "$out/$wl/$ft/$alg" "$wl,$ft,$alg" >> "$csv"
  done; done; done
  show_csv "$csv"
}

# ---------------------------------------------------------------- E3 MCCL re-optimization
# For each topology T (no failure, Fail1-3) and collective (2 GB AllGather / AllToAll, i.e.
# 128 MB per source on 16 GPUs):
#   nccl          NCCL-equivalent schedules                       (inputs/system/nccl.txt)
#   mccl_noreopt  MCCL schedule synthesized for the healthy fabric (mccl-no_failure.txt)
#   mccl_reopt    MCCL schedule re-synthesized for failure T       (mccl-<T>.txt; T != no_failure)
run_mccl() {
  local out="$1"
  paper_env
  local wl_dir="$BASE/inputs/workloads/single_collective/16_gpus"
  local topo_base="$BASE/inputs/topo/single_collective/1x2x8"
  local workloads=(ag-2147483648_bytes a2a-2147483648_bytes)
  local jobs_list=()
  for wl in "${workloads[@]}"; do for ft in "${FAILURE_LIST[@]}"; do
    jobs_list+=("$wl $ft nccl nccl.txt" "$wl $ft mccl_noreopt mccl-no_failure.txt")
    [ "$ft" != no_failure ] && jobs_list+=("$wl $ft mccl_reopt mccl-$ft.txt")
  done; done
  echo "== E3: MCCL vs NCCL, 2 GB AG/A2A on 16 GPUs (1x2x8): ${#jobs_list[@]} simulations, $JOBS in parallel"
  local j wl ft alg sys topo
  for j in "${jobs_list[@]}"; do
    read -r wl ft alg sys <<< "$j"
    topo="$topo_base"; [ "$ft" != no_failure ] && topo="$topo_base-$ft"
    throttle
    run_sim "$out/$wl/$ft/$alg" "$wl_dir/$wl" "$topo" "$BASE/inputs/system/$sys" "e3_${wl}_${ft}_${alg}" &
  done
  wait
  local csv="$out/results.csv"
  echo "workload,failure_type,algorithm,sim_time_ns,wall_seconds,schedule,status,schedule_file" > "$csv"
  for j in "${jobs_list[@]}"; do
    read -r wl ft alg sys <<< "$j"
    local dir="$out/$wl/$ft/$alg" file
    file="$(grep -oP 'Found matching MSCCL XML algorithm.*?: \K\S+' "$dir/run.log" 2>/dev/null | head -n 1 | sed 's#.*inputs/msccl/##')"
    echo "$(collect_row "$dir" "$wl,$ft,$alg"),${file:-none}" >> "$csv"
  done
  show_csv "$csv"
}

# ---------------------------------------------------------------- E4 full training
run_e2e() {
  local out="$1"
  paper_env
  local wl_root="$BASE/inputs/workloads/end_to_end/partials"
  local topo_base="$BASE/inputs/topo/end_to_end/HPN/1x16x8"
  # 128 simulated GPUs = one pipeline stage (PP=16) of a 2048-GPU DeepSeek-671B job.
  # The iteration is split into its transformer (fwd/bwd) and gradient-synchronization phases.
  local parts=(DeepSeek671B-transformer DeepSeek671B-grad)
  local alg=nccl
  echo "== E4: DeepSeek-671B training, 128 GPUs (1x16x8): $(( ${#parts[@]} * ${#FAILURE_LIST[@]} )) simulations, $JOBS in parallel"
  for p in "${parts[@]}"; do for ft in "${FAILURE_LIST[@]}"; do
    local wl; wl="$(find "$wl_root/$p" -name '*world_size2048*' -type f 2>/dev/null | head -n 1)"
    local topo="$topo_base"; [ "$ft" != no_failure ] && topo="$topo_base-$ft"
    throttle
    run_sim "$out/$p/$ft/$alg" "${wl:-$wl_root/$p/missing}" "$topo" "$BASE/inputs/system/$alg.txt" "e4_${p}_${ft}_${alg}" &
  done; done
  wait
  local csv="$out/results.csv"
  echo "workload,failure_type,algorithm,sim_time_ns,wall_seconds,schedule,status" > "$csv"
  for p in "${parts[@]}"; do for ft in "${FAILURE_LIST[@]}"; do
    collect_row "$out/$p/$ft/$alg" "$p,$ft,$alg" >> "$csv"
  done; done
  show_csv "$csv"
}

rc=0
case "$EXP" in
  smoke)  run_equiv 1 "$OUT/E1-equivalence" || rc=1 ;;
  equiv)  run_equiv 0 "$OUT/E1-equivalence" || rc=1 ;;
  single) run_single "$OUT/E2-single" ;;
  mccl)   run_mccl "$OUT/E3-mccl" ;;
  e2e)    run_e2e "$OUT/E4-e2e" ;;
  all)
    run_equiv 0 "$OUT/E1-equivalence" || rc=1
    run_single "$OUT/E2-single"
    run_mccl "$OUT/E3-mccl"
    run_e2e "$OUT/E4-e2e"
    ;;
esac
for csv in "$OUT/E2-single/results.csv" "$OUT/E3-mccl/results.csv" "$OUT/E4-e2e/results.csv"; do
  # every row's "status" column must be SUCCESS (the column position differs between experiments)
  if [ -f "$csv" ] && ! awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="status") c=i; next} $c!="SUCCESS"{bad=1} END{exit bad}' "$csv"; then rc=1; fi
done

echo
echo "Results in: $OUT"
echo "Summarize:  python3 artifact/summarize.py $OUT"
exit $rc
