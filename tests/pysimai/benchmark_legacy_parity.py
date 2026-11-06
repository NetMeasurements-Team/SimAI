#!/usr/bin/env python3
"""Legacy parity benchmark: vidur TPTimePredictor old subprocess.run vs new pysimai bridge.

Stage 1 of unified-vidur-simai-frontend-plan.md replaced vidur's stop-and-go
subprocess.run(SimAI_simulator) + CSV-parse path in
vidur-alibabacloud/vidur/execution_time_predictor/communication_time_predictor.py
with an in-memory pybind11 bridge (pysimai). This is an OPTIONAL, MANUAL
migration/parity tool, not a regression gate:

  * Value parity : for identical (model, tokens) inputs the old and new
                   TPTimePredictor should return the same simulated allreduce
                   latency. The 1e-4 tolerance only holds cold-vs-cold
                   (--compute-gap-us 0); with the default steady-state gap the
                   two regimes differ by design.
  * Speedup      : cache-miss wall-clock latency of the new in-memory path
                   vs the old subprocess path is reported for information
                   only; it does NOT gate pass/fail.

Because communication_time_predictor picks the code path at import time
(guarded `from pysimai import ...`), old and new MUST run in separate OS
processes. `--mode both` spawns them with the correct PYTHONPATH:

  old: PYTHONPATH=<repo>/vidur-alibabacloud             (pysimai NOT importable)
  new: PYTHONPATH=<repo>/vidur-alibabacloud:<repo>/bin  (pysimai importable)

Both processes run with AS_SEND_LAT=6 AS_NVLS_ENABLE=1 to match the old
subprocess command line exactly (see communication_time_predictor.py).

Usage:
  # single mode (one process):
  PYTHONPATH=<repo>/vidur-alibabacloud python tests/pysimai/benchmark_legacy_parity.py --mode old
  PYTHONPATH=<repo>/vidur-alibabacloud:<repo>/bin python tests/pysimai/benchmark_legacy_parity.py --mode new

  # orchestrated comparison (spawns old+new and prints the comparison table):
  python tests/pysimai/benchmark_legacy_parity.py --mode both

Results land in --output-dir (default <repo>/m1_parity_results/<timestamp>/):
  results_old.json, results_new.json, summary.json
"""

import argparse
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
VIDUR_ROOT = ROOT / "vidur-alibabacloud"
BIN_DIR = ROOT / "bin"
DEFAULT_TOPO = TEST_DIR / "topology.txt"
DEFAULT_CONF = ROOT / "astra-sim-alibabacloud" / "inputs" / "config" / "SimAI.conf"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B"
# NOTE: the OLD subprocess path's ns-3 binary only produces an EndToEnd CSV for
# allreduce sizes <= 1MB (verified on the Stage-1 build); 2MB+ comes back empty and
# the old predictor returns -1. The NEW pysimai path has no such ceiling (verified
# up to 32MB). compare() therefore only gates parity on sizes BOTH paths
# support, and reports the rest as OLD_UNSUPPORTED (new still returns values).
DEFAULT_NUM_TOKENS = [8, 32, 128, 256, 512, 1024, 2048, 4096]


# The OLD subprocess path caps at allreduce <= 1MB.
OLD_MAX_ALLREDUCE_BYTES = 1048576


def pysimai_importable() -> bool:
    return importlib.util.find_spec("pysimai") is not None


def check_mode_consistency(mode: str) -> None:
    has = pysimai_importable()
    if mode == "old" and has:
        sys.exit(
            "ERROR: --mode old but pysimai IS importable. Run with PYTHONPATH that "
            "does NOT include <repo>/bin (and pysimai not in site-packages)."
        )
    if mode == "new" and not has:
        sys.exit(
            "ERROR: --mode new but pysimai is NOT importable. Run with "
            "PYTHONPATH=<repo>/bin added so the compiled pysimai .so is on sys.path."
        )


def clean_old_caches(simai_dir: Path) -> None:
    """Remove old-path disk caches so every old-mode call is a true cache-miss.

    The old predictor caches by workload file (tmp_simai_inference_workload/) and
    by command hash (ncclFlowModel_EndToEnd_<hash>.csv). Without cleaning, a
    repeated call would hit the file cache and not measure the subprocess cost.
    """
    wk = Path(simai_dir) / "tmp_simai_inference_workload"
    if wk.is_dir():
        shutil.rmtree(wk)
    patterns = [
        "ncclFlowModel__EndToEnd.csv",
        "ncclFlowModel_EndToEnd.csv",
        "ncclFlowModel_EndToEnd_*.csv",
        "ncclFlowModel__detailed_*.csv",
        "ncclFlowModel__dimension_utilization_*.csv",
        "ncclFlowModel_detailed_flows.csv",
    ]
    for pat in patterns:
        for f in glob.glob(str(Path(simai_dir) / pat)):
            try:
                os.remove(f)
            except OSError:
                pass


def make_batch(num_tokens: int):
    from vidur.entities import Batch

    return Batch(replica_id=0, requests=[], num_tokens=[num_tokens])


def build_predictor(tp: int, pp: int, simai_dir: Path, topo: Path, conf: Path, compute_gap_us: int = 0):
    from vidur.config import RandomForrestExecutionTimePredictorConfig, ReplicaConfig

    replica_config = ReplicaConfig(
        model_name=DEFAULT_MODEL,
        num_pipeline_stages=pp,
        tensor_parallel_size=tp,
    )
    predictor_config = RandomForrestExecutionTimePredictorConfig(
        backend="simai_simulation",
        simai_dir=str(simai_dir),
        simai_simulation_topo=str(topo),
        simai_simulation_config=str(conf),
        simai_compute_gap_us=compute_gap_us,
        nccl_cpu_launch_overhead_ms=0.0,
        nccl_cpu_skew_overhead_per_device_ms=0.0,
    )
    from vidur.execution_time_predictor.communication_time_predictor import (
        TPTimePredictor,
    )

    predictor = TPTimePredictor(
        replica_config.model_config, replica_config, predictor_config
    )
    return predictor, replica_config, predictor_config


def run_mode(mode: str, args: argparse.Namespace) -> dict:
    check_mode_consistency(mode)
    simai_dir = Path(args.simai_dir).resolve()
    topo = Path(args.topo).resolve()
    conf = Path(args.conf).resolve()

    if mode == "old":
        clean_old_caches(simai_dir)

    t0 = time.perf_counter()
    predictor, replica_config, _ = build_predictor(args.tp, args.pp, simai_dir, topo, conf, args.compute_gap_us)
    init_wall_s = time.perf_counter() - t0

    if mode == "old":
        assert predictor.bridge is None, "old mode should use the subprocess fallback"
    else:
        assert predictor.bridge is not None, "new mode should have an active pysimai bridge"

    hidden_size = replica_config.model_config.embedding_dim
    results = []
    for nt in args.num_tokens:
        batch = make_batch(nt)
        t0 = time.perf_counter()
        latency_ms = predictor.get_execution_time(batch)
        wall_s = time.perf_counter() - t0
        all_reduce_bytes = hidden_size * nt * 2  # tensor_size = 2 (fp16)
        results.append(
            {
                "num_tokens": nt,
                "all_reduce_bytes": all_reduce_bytes,
                "latency_ms": latency_ms,
                "wall_s": wall_s,
            }
        )
        status = "OK" if latency_ms >= 0 else "FAIL(returned -1)"
        print(
            f"[{mode}] tokens={nt:5d} bytes={all_reduce_bytes:>10d} "
            f"latency={latency_ms * 1000:9.3f} us wall={wall_s * 1000:9.1f} ms  {status}",
            flush=True,
        )

    out = {
        "mode": mode,
        "backend": "subprocess" if mode == "old" else "pysimai",
        "model": DEFAULT_MODEL,
        "tp": args.tp,
        "pp": args.pp,
        "num_threads": 16,
        "compute_gap_us": args.compute_gap_us,
        "simai_dir": str(simai_dir),
        "topo": str(topo),
        "conf": str(conf),
        "env": {
            "AS_SEND_LAT": os.environ.get("AS_SEND_LAT"),
            "AS_NVLS_ENABLE": os.environ.get("AS_NVLS_ENABLE"),
        },
        "init_wall_s": init_wall_s,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{mode}] init_wall_s={init_wall_s:.3f} wrote {out_path}", flush=True)
    return out


def compare(old: dict, new: dict, args: argparse.Namespace) -> int:
    old_by_nt = {r["num_tokens"]: r for r in old["results"]}
    new_by_nt = {r["num_tokens"]: r for r in new["results"]}

    rows = []
    max_rel = 0.0
    min_speedup = float("inf")
    n_cmp = 0
    n_old_unsupported = 0
    new_fail = False

    for nt in args.num_tokens:
        o, n = old_by_nt[nt], new_by_nt[nt]
        row = {
            "num_tokens": nt,
            "all_reduce_bytes": o["all_reduce_bytes"],
            "old_latency_us": o["latency_ms"] * 1000 if o["latency_ms"] >= 0 else None,
            "new_latency_us": n["latency_ms"] * 1000 if n["latency_ms"] >= 0 else None,
            "old_wall_ms": o["wall_s"] * 1000,
            "new_wall_ms": n["wall_s"] * 1000,
        }
        if o["latency_ms"] < 0:
            row["status"] = "OLD_UNSUPPORTED"
            row["reason"] = (
                f"old subprocess produced no result (old binary allreduce ceiling "
                f"<= {OLD_MAX_ALLREDUCE_BYTES} bytes)"
            )
            n_old_unsupported += 1
            rows.append(row)
            continue
        if n["latency_ms"] < 0:
            row["status"] = "FAIL"
            row["reason"] = "new pysimai returned -1"
            new_fail = True
            rows.append(row)
            continue
        n_cmp += 1
        rel = abs(n["latency_ms"] - o["latency_ms"]) / o["latency_ms"]
        speedup = o["wall_s"] / n["wall_s"] if n["wall_s"] > 0 else float("inf")
        max_rel = max(max_rel, rel)
        min_speedup = min(min_speedup, speedup)
        row["rel_diff"] = rel
        row["speedup"] = speedup
        row["status"] = "OK"
        rows.append(row)

    parity_ok = (not new_fail) and n_cmp > 0 and max_rel < args.parity_tol

    print()
    print("=" * 104)
    print(
        f"M1 parity: old(subprocess) vs new(pysimai)  |  model={old['model']} "
        f"tp={old['tp']} world={old['tp'] * old['pp']} threads={old['num_threads']}"
    )
    print("=" * 104)
    header = (
        f"{'tokens':>6} {'bytes':>10} {'old_lat_us':>11} {'new_lat_us':>11} "
        f"{'rel_diff':>9} {'old_wall_ms':>11} {'new_wall_ms':>11} {'speedup':>9}  status"
    )
    print(header)
    print("-" * 104)
    for r in rows:
        if r["status"] == "OK":
            print(
                f"{r['num_tokens']:>6} {r['all_reduce_bytes']:>10} "
                f"{r['old_latency_us']:>11.3f} {r['new_latency_us']:>11.3f} "
                f"{r['rel_diff']:>9.2e} {r['old_wall_ms']:>11.1f} "
                f"{r['new_wall_ms']:>11.1f} {r['speedup']:>9.1f}x  OK"
            )
        else:
            ol = f"{r['old_latency_us']:.3f}" if r["old_latency_us"] is not None else "    -"
            nl = f"{r['new_latency_us']:.3f}" if r["new_latency_us"] is not None else "    -"
            print(
                f"{r['num_tokens']:>6} {r['all_reduce_bytes']:>10} "
                f"{ol:>11} {nl:>11} "
                f"{'--':>9} {r['old_wall_ms']:>11.1f} {r['new_wall_ms']:>11.1f} "
                f"{'--':>9}  {r['status']} ({r['reason']})"
            )
    print("-" * 104)
    if n_old_unsupported:
        print(f"note: {n_old_unsupported} size(s) excluded from pass/fail: old binary allreduce ceiling <= {OLD_MAX_ALLREDUCE_BYTES} bytes")
    print(f"max relative latency diff : {max_rel:.3e}  (accept < {args.parity_tol:.0e}, {n_cmp} comparable) -> {'PASS' if parity_ok else 'FAIL'}")
    print(f"min cache-miss speedup    : {min_speedup:.1f}x  ({n_cmp} comparable, informational - not a gate)")
    print(f"old init wall             : {old['init_wall_s']:.3f} s   new init wall: {new['init_wall_s']:.3f} s")

    summary = {
        "model": old["model"],
        "tp": old["tp"],
        "pp": old["pp"],
        "num_threads": old["num_threads"],
        "compute_gap_us": new.get("compute_gap_us", 0),
        "parity_tol": args.parity_tol,
        "n_cmp": n_cmp,
        "n_old_unsupported": n_old_unsupported,
        "max_rel_diff": max_rel,
        "min_speedup": min_speedup,
        "parity_pass": parity_ok,
        "overall_pass": parity_ok,
        "rows": rows,
    }
    out_path = Path(args.output_dir) / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary -> {out_path}")
    return 0 if summary["overall_pass"] else 1


def run_both(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env.update({"AS_SEND_LAT": "6", "AS_NVLS_ENABLE": "1"})

    env_old = dict(base_env)
    env_old["PYTHONPATH"] = str(VIDUR_ROOT)
    env_new = dict(base_env)
    env_new["PYTHONPATH"] = str(VIDUR_ROOT) + os.pathsep + str(BIN_DIR)

    script = str(Path(__file__).resolve())
    common = [
        "--tp", str(args.tp),
        "--pp", str(args.pp),
        "--simai-dir", str(args.simai_dir),
        "--topo", str(args.topo),
        "--conf", str(args.conf),
        "--compute-gap-us", str(args.compute_gap_us),
        "--num-tokens", *[str(t) for t in args.num_tokens],
    ]

    cmd_old = [sys.executable, script, "--mode", "old", "--output", str(out_dir / "results_old.json")] + common
    cmd_new = [sys.executable, script, "--mode", "new", "--output", str(out_dir / "results_new.json")] + common

    print(f"[both] running OLD (subprocess) ...  PYTHONPATH={env_old['PYTHONPATH']}")
    subprocess.run(cmd_old, env=env_old, cwd=str(ROOT), check=True)
    print(f"[both] running NEW (pysimai) ...     PYTHONPATH={env_new['PYTHONPATH']}")
    subprocess.run(cmd_new, env=env_new, cwd=str(ROOT), check=True)

    old = json.loads((out_dir / "results_old.json").read_text())
    new = json.loads((out_dir / "results_new.json").read_text())
    return compare(old, new, args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["old", "new", "both"], default="both")
    parser.add_argument("--output", default=None, help="JSON output file (single mode only)")
    parser.add_argument("--output-dir", default=None, help="Directory for results_old/new/summary.json")
    parser.add_argument("--tp", type=int, default=16, help="tensor parallel size (= world size, must fit topo)")
    parser.add_argument("--pp", type=int, default=1, help="pipeline stages")
    parser.add_argument("--simai-dir", default=str(ROOT))
    parser.add_argument("--topo", default=str(DEFAULT_TOPO))
    parser.add_argument("--conf", default=str(DEFAULT_CONF))
    parser.add_argument("--compute-gap-us", type=int, default=1000,
                        help="simulated compute gap (us) between collectives in the new path; "
                             "0 = stop sim immediately when idle (raw warm)")
    parser.add_argument("--num-tokens", type=int, nargs="+", default=DEFAULT_NUM_TOKENS,
                        help="batch token counts -> allreduce sizes (hidden*2*nt bytes)")
    parser.add_argument("--parity-tol", type=float, default=1e-4, help="max rel diff for value parity (0.01%%)")
    args = parser.parse_args()

    if args.mode == "both":
        if args.output_dir is None:
            ts = time.strftime("%Y%m%d-%H%M%S")
            args.output_dir = str(ROOT / "m1_parity_results" / ts)
        return run_both(args)

    if args.output is None:
        parser.error("--output is required for --mode old|new")
    if args.output_dir is None:
        args.output_dir = str(Path(args.output).parent)
    out = run_mode(args.mode, args)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
