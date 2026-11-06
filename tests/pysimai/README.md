# pysimai runtime tests

Tests for the in-memory `pysimai` bridge and its Vidur integration. These are
**not** ns-3 backend tests: they import `vidur` / `pysimai` and need the `vidur`
conda environment, so they live here rather than under `tests/ns3/`.

| Script | Kind | What it checks |
|---|---|---|
| `test_runtime.py` | regression | `init_network`, sync `predict_collective`, async `post_collective` + callback, monotonic clock, positive latency. |
| `test_compute_gap.py` | wiring test | Vidur-derived compute-gap math, `get_execution_time` plumbing, fixed-gap fallback, optional real bridge call. |
| `benchmark_legacy_parity.py` | optional manual tool | Old `subprocess.run(SimAI_simulator)` vs new `pysimai` latency parity and cache-miss speedup. |
| `run_legacy_parity.sh` | wrapper | Runs `benchmark_legacy_parity.py --mode both`. |

## Prerequisites

These tests assume you already have `vidur-alibabacloud` working and can run it
with the `simai_simulation` backend (see `vidur-alibabacloud/README.md` →
Environment Setup). On top of that setup you need:

- Install the pip requirements and the build-time pybind11:

  ```bash
  conda env create -p ./env -f vidur-alibabacloud/environment.yml
  conda activate vidur
  pip install -r vidur-alibabacloud/requirements.txt
  pip install pybind11
  ```

- A built ns-3 SimAI backend plus the `pysimai` extension in `<repo>/bin/`:
  `SimAI_simulator` and `pysimai.cpython-310-*.so`. Build both from the repo
  root with `./scripts/build.sh -c ns3 -d optimized`. Without `pybind11` the
  `pysimai` target is silently skipped.
- `pysimai` importable by the Python running the tests. The build drops it in
  `<repo>/bin/`; add that directory to `PYTHONPATH` if your Vidur setup does not
  already.
- The topology and config files referenced by Vidur's `simai_simulation_topo`
  and `simai_simulation_config` (defaults under
  `astra-sim-alibabacloud/inputs/`).

## Run

```bash
# runtime regression
python tests/pysimai/test_runtime.py

# compute-gap wiring
python tests/pysimai/test_compute_gap.py

# optional legacy parity benchmark (manual, not a gate)
bash tests/pysimai/run_legacy_parity.sh
```

Exit code 0 = PASS, 1 = FAIL.

## About the parity benchmark

`benchmark_legacy_parity.py` is a **migration/parity tool**, not a regression gate. It
compares the new in-memory path against the legacy subprocess path, which  only 
supports allreduce `<= 1MB`. The value-parity gate (`1e-4`) only holds 
cold-vs-cold (`--compute-gap-us 0`); with the default
steady-state gap the two regimes differ by design. 

The old and new paths are selected at import time (`pysimai` importable or not),
so `--mode both` spawns them in separate OS processes with different
`PYTHONPATH`s.
