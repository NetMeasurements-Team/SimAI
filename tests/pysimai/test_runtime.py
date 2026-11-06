#!/usr/bin/env python3
"""Runtime regression test for the in-memory `pysimai` bridge.

Checks:
  * the compiled `pysimai` module is importable from `<repo>/bin`;
  * `init_network()` initializes the resident topology;
  * `predict_collective()` returns a positive latency for several ALLREDUCE sizes;
  * `current_time_ns()` advances monotonically across collectives;
  * `post_collective()` invokes the completion callback and advances the clock.

Exit code 0 = PASS, 1 = FAIL.

Prerequisites (see `AGENTS.md`; run on the remote host in the `vidur` env):
  * `<repo>/bin/pysimai*.so` built.
"""

import argparse
import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
DEFAULT_TOPO = TEST_DIR / "topology.txt"
DEFAULT_CONF = ROOT / "astra-sim-alibabacloud" / "inputs" / "config" / "SimAI.conf"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def find_pysimai_dir():
    bin_dir = ROOT / "bin"
    if glob.glob(str(bin_dir / "pysimai*")):
        return bin_dir
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topo", default=str(DEFAULT_TOPO))
    parser.add_argument("--conf", default=str(DEFAULT_CONF))
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--tp", type=int, default=16, help="tensor-parallel size (= world size)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[3276800, 6553600])
    parser.add_argument("--timeout", type=int, default=300, help="seconds to wait for the async callback")
    args = parser.parse_args()

    bin_dir = find_pysimai_dir()
    if bin_dir is None:
        print(f"FAIL: cannot locate compiled pysimai module in {ROOT / 'bin'}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(bin_dir))

    try:
        from pysimai import PySimAIBridge
    except ImportError as exc:
        print(f"FAIL: cannot import pysimai: {exc}", file=sys.stderr)
        return 1

    try:
        bridge = PySimAIBridge.get_instance()
        bridge.init_network(str(args.topo), str(args.conf), num_threads=args.num_threads)
        check(bridge.is_initialized(), "init_network did not initialize the bridge")
        total_gpus = bridge.get_total_gpus()
        check(total_gpus >= args.tp, f"topology has {total_gpus} GPUs, need tp={args.tp}")
        print(f"initialized: {total_gpus} GPUs, t0={bridge.current_time_ns()} ns")

        ranks = list(range(args.tp))
        last_time = bridge.current_time_ns()
        for size in args.sizes:
            latency_us = bridge.predict_collective("ALLREDUCE", ranks, size)
            check(latency_us > 0, f"ALLREDUCE {size}B returned {latency_us}")
            now = bridge.current_time_ns()
            check(now >= last_time, f"clock went backwards: {now} < {last_time}")
            last_time = now
            print(f"ALLREDUCE {size:>10}B -> {latency_us:10.1f} us")

        completed = []

        def on_complete(duration_us):
            completed.append(duration_us)

        bridge.post_collective(
            1, "ALLREDUCE", ranks, args.sizes[-1], completion_callback=on_complete
        )
        deadline = time.monotonic() + args.timeout
        while not completed and time.monotonic() < deadline:
            time.sleep(0.01)
        check(completed, "async post_collective callback did not fire")
        check(completed[0] > 0, f"async callback got {completed[0]}")
        check(bridge.current_time_ns() >= last_time, "clock went backwards after async collective")
        print(f"async ALLREDUCE -> {completed[0]:.1f} us")
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
