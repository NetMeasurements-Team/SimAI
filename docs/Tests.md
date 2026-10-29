# Tests

Simulator-level tests run workloads through the full simulator. Native backend
tests exercise components directly. Run commands from the repository root in
the build environment (Bash and Python 3; runners use only the standard library).

| Test | What it checks |
| --- | --- |
| [Collective/PFC](#collectivepfc) | Workload completion without PFC in a baseline and with pause/resume under congestion. |
| [Collective matrix](#collective-completion-and-selection-matrix) | Collective completion and algorithm/protocol selection across versions and topologies. |
| [Native MMU/PFC](#native-backend-tests) | Packet delivery, buffer accounting, pause/resume and drainage inside ns-3. |

## Simulator-level tests: ns3

These tests exercise collective scheduling, RDMA transport and the switch
backend using `bin/SimAI_simulator`. The build option `--ns3-tests` is not required.
Each test has its own runner and validation logic; shared configuration lives
in `common/`.

```text
tests/ns3/
|-- common/
|   |-- network.conf.in
|   '-- system.txt
|-- collective-pfc/
|   |-- README.md
|   |-- run.py
|   |-- cases.json
|   |-- topology.txt.in
|   '-- workload.txt.in
'-- collective-matrix/
    |-- README.md
    |-- run.py
    '-- test_validation.py
```

PFC expands its local templates; the matrix generates workloads and topologies
in Python. Both use `common/system.txt` and expand `common/network.conf.in`:
PFC selects the case's buffer size and enables queue monitoring; the matrix
uses a 32 MiB buffer and disables monitoring.

Both runners accept:

| Option | Purpose |
| --- | --- |
| `--binary PATH` | Use another simulator executable. |
| `--output PATH` | Choose a results directory; it must not exist or contain whitespace. |
| `--timeout SECONDS` | Set the wall-clock limit per case. |
| `--prepare-only` | Generate inputs without a simulator; this does not check simulation behavior. |
| `--help` | Show all options. |

Results contain per-case inputs, `invocation.json` (command, environment
overrides and input hashes), `run.log` and network traces. A top-level
`summary.json` records validation results. Keep the case directory to investigate
a failure. These tests check communication behavior, not numerical tensor
correctness or reference completion times.

### Collective/PFC

#### What runs and what passes

Four ranks on two switches issue a forward All-to-All and a weight-gradient
All-Reduce. The All-Reduce crosses the inter-switch trunk. PFC is enabled in
both cases:

| Case | Collective size | Trunk | Buffer | Expected result |
| --- | --- | --- | --- | --- |
| `baseline` | 64 KiB | 200 Gb/s | 16 MiB | Both collectives complete without PFC events. |
| `congested` | 16 MiB | 25 Gb/s | 1 MiB | Both complete, with PFC pause **and** resume events. |

The congested case combines larger transfers, a slower trunk and a smaller
buffer to trigger backpressure. Both cases require completed messages covering
all four ranks, matching nonzero stream counts, queue samples and no MMU packet
loss. Exact pause counts and queue peaks are not fixed.

#### Run

```bash
# Build and run both cases.
bash scripts/build.sh -c ns3 --ns3-asserts --sys-asserts &&
python3 tests/ns3/collective-pfc/run.py

# Run either case using the existing build.
python3 tests/ns3/collective-pfc/run.py --case baseline
python3 tests/ns3/collective-pfc/run.py --case congested

# Generate inputs only.
python3 tests/ns3/collective-pfc/run.py --prepare-only
```

Expect `PASS baseline` and `PASS congested`, followed by the results location
under `simai-test-results-*`. The runner stops at the first failure and exits
nonzero. The default timeout is **300 seconds per case**.

See the [test README](../tests/ns3/collective-pfc/README.md) for topology, congestion settings and validation.

### Collective completion and selection matrix

#### What runs and what passes

Each case issues one AllReduce, AllGather, ReduceScatter or AllToAll. Cases vary
rank count, server placement, size and settings to exercise Ring/PTP, PXN and
NVLS. There is no congestion requirement.

| Build | Cases | Expected result |
| --- | --- | --- |
| v2.20 | 15 | Completion with expected collective selection and network messages. |
| v2.30 | 55 | Completion plus algorithm/protocol IDs, PAT, automatic thresholds and expected fallbacks. |

Every case requires matching nonzero stream counts and completed network
messages matching the generated flows.

#### Run

Build and run versions sequentially. Switching versions requires a clean build
because shared ns-3 export paths can retain stale objects. `--nccl-version`
selects validation expectations; it does not rebuild or switch the executable.

```bash
bash scripts/build.sh -lc ns3 -d debug --nccl-version 2.20 &&
python3 tests/ns3/collective-matrix/run.py --nccl-version 2.20

bash scripts/build.sh -lc ns3 -d debug --nccl-version 2.30 &&
python3 tests/ns3/collective-matrix/run.py --nccl-version 2.30

# List cases or generate inputs without running a simulation.
python3 tests/ns3/collective-matrix/run.py --nccl-version 2.30 --list
python3 tests/ns3/collective-matrix/run.py --nccl-version 2.30 --prepare-only

# Run a selected case with the matching build.
python3 tests/ns3/collective-matrix/run.py --nccl-version 2.30 --case pat3-allgather

# Check the validator without a simulator.
python3 tests/ns3/collective-matrix/test_validation.py
```

Expect `PASS` for every selected case and a results location under
`simai-test-results-matrix-*`. The runner continues after failures and exits
nonzero if any case fails. The default timeout is **120 seconds per case**
(use `--timeout 300` on a slower machine). Ctrl-C records `INTERRUPTED` and
exits with status 130.

Repeat `--case NAME` to select multiple cases. PXN ReduceScatter selections
also include their matching cross-server reference case.

See the [test README](../tests/ns3/collective-matrix/README.md) for coverage and validation.

### MSCCL vs NCCL equivalence

`tests/ns3/msccl-nccl-equivalence/run.sh` checks that the custom-collective path
(`mscclCustomFlow`, `NcclCustom`) reproduces the NCCL flow model (`ncclFlowModel`,
`NcclFlowModel`) when it is given an MSCCL schedule that encodes the NCCL algorithm.
Each single-collective workload runs twice on the 16-GPU (2 x 8) topology and its
failure variants:

- `inputs/system/nccl.txt`: all-gather and all-to-all through `mscclCustomFlow`, with
  the schedules in `inputs/msccl/nccl/` (ring AllGather on the SimCCL rings, P2P
  AllToAll on the SimCCL PXN channel map).
- `inputs/system/simai_native.txt`: the same collectives through `ncclFlowModel`.

The test passes when every pair finishes with the same simulated end time, the
`nccl` run used the MSCCL schedules (no fallback to NCCL), and the `simai_native`
run did not use MSCCL.

```bash
bash scripts/build.sh -c ns3
tests/ns3/msccl-nccl-equivalence/run.sh        # 4 workloads x 4 topologies
tests/ns3/msccl-nccl-equivalence/run.sh -q     # no_failure, 256MB only (~30 s)
tests/ns3/msccl-nccl-equivalence/run.sh -h     # options (binary, output, jobs, subsets)
```

The script fixes `AS_PXN_ENABLE=1` (the topology is rail-only), `AS_SEND_LAT=3000`
(ns) and `SIMAI_PROTO_AWARE=0`. The last one selects `NCCL_PROTO_UNDEF`, which
disables MSCCL pipelining: by default the MSCCL path splits every step into 512KB
chunks as NCCL does, while `NcclFlowModel` sends each step as one flow. Both models then keep the
bottleneck links equally busy, but with chunking the first inter-node transfer
starts earlier (after one 512KB chunk has crossed NVLink to the PXN proxy instead of
a whole step), so the MSCCL times are lower by that pipeline fill. Without
`SIMAI_PROTO_AWARE=0` the test is expected to fail. For the paper MCCL, we used always the ncclCustom backend in all the configurations.

Each simulation runs in its own directory under the output folder
(`simai-test-results-msccl-nccl-<timestamp>` by default) with its `run.log`;
`results.csv` lists both end times and the verdict per pair. The script exits with
status 1 if any pair differs or fails.

### Adding simulator tests

Give each test a dedicated directory and `run.py` with its own preparation and
validation logic. Use `common/` for configuration shared by multiple tests.
Document the purpose, pass criteria and commands here, and keep detailed checks
in a README.md inside the test directory. Other modes can use `tests/analytical/` or `tests/phy/`.

## Native backend tests

The ns-3 backend includes native test suites in module-specific `test/`
directories, such as `ns-3-alibabacloud/simulation/src/point-to-point/test/`.
Enable them through the normal build script:

```bash
bash scripts/build.sh -c ns3 --ns3-tests --ns3-asserts --sys-asserts
```

`--ns3-tests` sets the standard CMake option `NS3_TESTS=ON`, building the native
suites and `test-runner`. Native tests are disabled in normal simulator builds.
Use `--suite` to select a suite rather than running every backend test.

For example, run the MMU/PFC suite:

```bash
ns-3-alibabacloud/simulation/build/utils/ns3.36.1-test-runner-default \
  --suite=mmu-pfc --verbose \
  --tempdir="$PWD/mmu-pfc-test-output"
```

This suite checks uncongested traffic, congestion with PFC, and congestion
without PFC. It validates packet delivery, buffer accounting, pause/resume
activity and queue drainage. The runner reports case results and exits nonzero
on failure. Per-case occupancy and PFC traces are written under `--tempdir`;
choose a fresh directory to retain results from earlier runs.

For a Debug build, add `-d debug` to the build command and use the executable
ending in `-debug`. The version component of the executable name follows the
backend's ns-3 version. See the backend's
[MMU/PFC test documentation](../ns-3-alibabacloud/simulation/src/point-to-point/test/README.mmu-pfc.md)
for topology, checks and trace formats.
