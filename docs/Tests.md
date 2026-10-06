# Tests

SimAI provides two levels of testing:

- **Simulator-level tests** run workloads through the full simulator, including
  collective scheduling, transport and network behavior. Runners and inputs are
  grouped by mode under `tests/`.
- **Native backend tests** exercise components directly through the backend's own
  test framework. Their sources and detailed documentation live in the backend
  submodule.

The commands below run from the repository root.

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

## Simulator-level tests: ns3

The ns3 workload tests exercise the SimAI frontend, workload scheduler,
collective flow generator, RDMA transport and switch backend. They use the
normal simulator executable; `--ns3-tests` is not required.

### Build and run

```bash
bash scripts/build.sh -c ns3 --ns3-asserts --sys-asserts
python3 tests/ns3/run.py
```

The default binary is `bin/SimAI_simulator`. Select a different build with
`--binary /path/to/SimAI_simulator`. Python 3 and the standard library suffice.
Run only one case using `--case baseline` or `--case congested`.

To inspect generated inputs without running a simulation:

```bash
python3 tests/ns3/run.py --prepare-only
```

Each invocation creates a fresh `simai-test-results-*` directory. Each case
contains expanded inputs, `invocation.json` (command, environment overrides,
input hashes and parameters), `run.log`, and network traces. `summary.json`
records validation results. Failures return a nonzero status. A 300-second
wall-clock timeout per case bounds stalled simulations; override with `--timeout`.
The simulator's additional collective CSV reports use its compiled result path
and a unique run-name prefix. Existing result directories are never overwritten.

### Directory layout

```text
tests/
└── ns3/
    ├── run.py
    └── collective-pfc/
        ├── cases.json
        ├── network.conf.in
        ├── system.txt
        ├── topology.txt.in
        └── workload.txt.in
```

The runner and its fixtures live together. Additional ns3 scenarios can be added
alongside `collective-pfc/`; other modes can use `tests/analytical/` or `tests/phy/`
when needed. Test documentation lives here in `docs/`.

### Adding scenarios

`tests/ns3/collective-pfc/` contains the collective/PFC fixture. Its files provide
a starting point for additional four-rank scenarios:

- `workload.txt.in`: real SimAI layer workload; `@BYTES@` sets collective size.
- `topology.txt.in`: topology with `@TRUNK@` for the inter-switch link rate.
- `network.conf.in`: snapshot of network settings; `@BUFFER_MB@` sets capacity.
- `system.txt`: explicit system/collective configuration.
- `cases.json`: case parameters and PFC expectations (`present` or `absent`).

Clone a fixture, edit its inputs and case parameters, and pass `--fixture PATH`.
The current runner's completion checker expects four participating ranks and
SimAI's current stdout/FCT/PFC/queue trace formats. Extend that checker when
adding a different rank count, layer/collective sequence or validation contract. Completion markers and trace checks are required in addition to a successful
process exit.

The runner fixes one simulation thread and disables NVLS/PXN and send latency
through explicit environment overrides. It snapshots generated inputs so that
changes to normal production defaults do not silently alter a saved test run.
The collective/PFC scenario below describes the topology and expected behavior.

### Collective/PFC scenario

```text
rank 0 --100G--+                 +--100G-- rank 2
              S0 --- trunk --- S1
rank 1 --100G--+                 +--100G-- rank 3
```

Four ranks represent four single-GPU servers. The topology has two ordinary
switches and no NVSwitch. All links have 1 us propagation; switch forwarding
delay is 1.5 us. A single layer issues forward All-to-All and weight-gradient
All-Reduce, with small compute delays and no input-gradient collective.
TP=2 forms groups {0,1} and {2,3} for the forward All-to-All. DP=2 forms
groups {0,2} and {1,3} for the weight-gradient All-Reduce. Both collectives
therefore have participating peers; the All-Reduce crosses the trunk. A100
selects the existing ring All-Reduce implementation; NVLS and PXN are disabled.
This is a timing/communication workload simulation, not a numerical tensor test.

| Case | Size of each collective | Trunk | Switch buffer | Required PFC |
| --- | --- | --- | --- | --- |
| baseline | 64 KiB | 200 Gb/s | 16 MiB | None |
| congested | 16 MiB | 25 Gb/s | 1 MiB | Both pause and resume |

Sizes are the values supplied to SimAI's collective implementation; they are not
claims about exact per-link traffic volume. The cases intentionally vary message
size, trunk capacity and buffer size together. They are functional tests, not a
controlled comparison of completion times.

#### Congestion and PFC

A collective can run without triggering PFC. Queue growth requires arrivals to
exceed service on a shared output, and PFC requires that growth to reach the MMU
threshold before the burst ends. The two data-parallel All-Reduce groups
exchange data across the partition concurrently. Each side has two 100 Gb/s host links feeding a
25 Gb/s trunk. Sustained offered cross-partition traffic can therefore outrun
the trunk by a large margin. A 1 MiB buffer gives low enough dynamic thresholds
for multi-MiB transfers to reach them. Backpressure should pause adjacent senders
until the bottleneck drains sufficiently to resume them.

`ENABLE_QCN=0` disables switch ECN marking, `HAS_WIN=0` removes the sender window
limit, and `CC_MODE=1` keeps a supported DCQCN implementation configured. Without
ECN feedback it should not throttle away the intended congestion. The test uses
1,000-byte payloads, error-free links and PFC enabled in both cases. Headroom is
initialized through the normal frontend configuration in `common.h`.

Exact queue peaks, pause counts and collective completion times depend on flow
scheduling and are not fixed reference values. The congested case requires both
pause and resume events. If it completes without PFC, inspect the generated
topology, window/ECN settings and traffic traces to determine why the workload
did not exercise the expected backpressure.

#### Pass conditions and interpretation

Both cases must exit successfully within the timeout, print the workload and
all-messages completion markers, report matching nonzero rank-0 injected/finished
stream counts, issue both configured collectives, and contain positive
completed-message FCT records covering all four ranks. MMU loss diagnostics fail either case. The baseline requires an empty
PFC trace; the stress case requires at least one XOFF and one XON.

The checker reports completed-message count, pause/resume counts and the maximum
sampled egress-port occupancy. The fixture sets `ENABLE_MONITOR 1` to enable
queue sampling every 1 us (`QLEN_MON_INTERVAL`), starting at `MON_START`.
Queue monitoring is disabled by default; specifying `QLEN_MON_FILE` alone does
not enable it. A missing or empty queue trace fails validation. Rebuild the
simulator after updating the frontend to use this option. These are
observations, not independent verification of every internal MMU invariant.
The test does not require equal PFC counts or drained queues at application exit:
the frontend stops when messages finish and may leave final control events
pending. The isolated backend test checks drainage separately.

A PFC-disabled negative control is deliberately omitted here: the RDMA recovery
behavior can retransmit or stall after loss, making a timeout ambiguous. The
isolated backend test already provides that negative control without RDMA.

### Collective completion and selection matrix

`tests/ns3/collective-matrix/run.py` checks AllReduce, AllGather, ReduceScatter
and AllToAll across Ring/PTP, PXN and NVLS. The v2.30 matrix also covers PAT,
LL/LL128/Simple selection, automatic selection thresholds and fallback cases.
There are 15 cases for v2.20 and 55 for v2.30, including non-power-of-two ranks.

Build and run each version sequentially. Currently, switching versions requires
a clean build: shared ns-3 export paths can otherwise retain stale objects.
The runner's `--nccl-version` validates expectations; it does not select or
rebuild the executable.

```bash
bash scripts/build.sh -lc ns3 -d debug --nccl-version 2.20 &&
python3 tests/ns3/collective-matrix/run.py --nccl-version 2.20

bash scripts/build.sh -lc ns3 -d debug --nccl-version 2.30 &&
python3 tests/ns3/collective-matrix/run.py --nccl-version 2.30
```

Use `--list` to show cases, repeat `--case NAME` to select cases, or use
`--prepare-only` to generate inputs without running the simulator. Each case has
its own directory with inputs, invocation metadata, binary/input hashes, logs
and traces. The runner continues after failures and updates `summary.json`;
any failure gives a nonzero exit status. The default wall-clock timeout is
120 seconds per case (`--timeout` overrides it). Ctrl-C records `INTERRUPTED`.

A pass requires collective and workload completion, matching stream counts,
nonempty generated flows, and completed network messages matching the generated
source/destination/size multiset. Selection and PXN forwarding are checked;
v2.30 additionally exposes algorithm/protocol IDs in its detailed-flow CSV.
These are completion tests, not numerical tensor correctness or timing
benchmarks. Tree, CollNet, NVLS-tree and Broadcast are not covered through the
current selector/workload frontend.

The runner clears inherited `AS_*`, `SIMAI_*` and `NCCL_*` settings before
applying recorded case overrides. It leaves send latency enabled. Its validator
can be checked without a simulator:

```bash
python3 tests/ns3/collective-matrix/test_validation.py
```

#### Coverage

| Cases                                                                     | v2.20          | v2.30                                |
|---------------------------------------------------------------------------|----------------|--------------------------------------|
| AllReduce, AllGather, ReduceScatter, AllToAll; 4 ranks, same/cross server | UNDEF protocol | LL, LL128, Simple, explicitly forced |
| All four collectives with 3 ranks on 3 servers                            | Yes            | Yes                                  |
| AllToAll and ReduceScatter with PXN, 2 servers x 2 ranks                  | Yes            | All three protocols                  |
| NVLS AllReduce, H100, 8 ranks and one NVSwitch, 4 MiB                     | Yes            | Simple                               |
| PAT AllGather/ReduceScatter, 2/3/4 servers, one rank each                 | Unavailable    | Simple                               |
| PAT automatic selection inside/outside size window; ineligible topology   | Unavailable    | Yes                                  |
| PAT keeps Simple even when LL/LL128 is forced                             | Unavailable    | Yes                                  |
| Automatic LL/LL128/Simple size selection; protocol-aware mode disabled    | Unavailable    | Yes                                  |

There are 15 v2.20 cases and 55 v2.30 cases. Sizes are modest (mostly 768 KiB),
except explicit NVLS/protocol-threshold cases. Most tests use a minimal Ethernet
star with an explicit GPUs-per-server value, no NVSwitch, and no congestion
requirement. Thus "same server" tests the selector's locality handling, not a
calibrated NVLink topology. The NVLS case also adds GPU-to-NVSwitch links.
All cases issue exactly one forward-pass collective in a TP group containing all
ranks. DP/EP grouping, pipelining, and concurrency are outside this matrix.

#### Results and reruns

The runner creates a fresh `simai-test-results-matrix-...` directory, prints its
location, continues after failures, and returns nonzero if any case fails. Each
case has generated workload/topology/configuration inputs, `invocation.json`,
`run.log`, the detailed-flow CSV and simulator output files. `summary.json` is
updated after every case, including its failure reason and elapsed wall time. 
Ctrl-C records the current case as `INTERRUPTED` and exits with status 130. A 
timeout defaults to 120 wall-clock seconds **per case**; use `--timeout 300` on a
slower server. An absent CSV, a fallback, a no-op, or an incomplete message count
is a failure, not an automatic skip. Keep the case directory when investigating a
failure.

```bash
# List names and expectations without building or running anything.
python3 tests/ns3/collective-matrix/run.py --nccl-version 2.30 --list

# Generate all input files without executing the simulator.
python3 tests/ns3/collective-matrix/run.py --nccl-version 2.30 --prepare-only

# Rerun an exact case after building the corresponding version.
python3 tests/ns3/collective-matrix/run.py --nccl-version 2.30 \
  --case pat3-allgather --timeout 300

# Optional binary/output overrides; output must not already exist.
python3 tests/ns3/collective-matrix/run.py --nccl-version 2.20 \
  --binary /absolute/path/to/SimAI_simulator --output /tmp/my-matrix-results

# Check the validator itself, without a simulator.
python3 tests/ns3/collective-matrix/test_validation.py
```

#### Current build limitation

Use a clean build when switching SimCCL versions. The ns-3 build exports
versioned mock sources through shared symlink paths, and an incremental build
can retain stale objects after a version change.
