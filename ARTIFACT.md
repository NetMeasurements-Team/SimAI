# MCCL artifact: SimAI with custom collective schedules

Artifact for **"MCCL: Measurement-based Collective Communication Library"**, ACM CoNEXT 2026
(Proc. ACM Netw. 4, CoNEXT4, Article 55, [10.1145/3856969](https://doi.org/10.1145/3856969)).

Badges requested: **Artifact Available** and **Artifact Evaluated – Functional**.
(*Results Reproduced* is not requested: it would also need the MCCL schedule synthesizers and
the H200 hardware prototype, which are not part of this artifact.)

| | |
|---|---|
| Archived artifact (DOI) | **ZENODO-DOI-TODO** |
| Code | https://github.com/NetMeasurements-Team/SimAI, branch/tag `conext26-ae` |
| Hardware | any x86-64 Linux machine, no GPU; ≥ 2 cores, ≥ 8 GB RAM (≈ 0.7 GB per parallel E1–E3 simulation, up to ≈ 3 GB per E4 simulation), ≈ 3 GB disk |
| Software | Ubuntu 22.04/24.04 (or Docker), GCC ≥ 11, CMake ≥ 3.13, libxml2-dev, Python 3 (matplotlib optional; pandas for the paper-style E4 plots) |
| Proprietary software / hardware | none |
| Time | build ≈ 30 min on 2 cores; smoke test 15 s; E1 + E2 ≈ 4 min; E3 ≈ 8 min; E4 ≈ 1 h (2 cores) |

---

## 1. What this artifact is

MCCL re-synthesizes AllGather and AllToAll schedules at runtime when gray failures degrade
GPU-to-GPU paths. To evaluate those schedules at cluster scale, the paper uses
[SimAI](https://github.com/aliyun/SimAI) (NSDI'25), a simulator validated against production
clusters, **extended to execute arbitrary collective schedules written in the MSCCL XML
format** (paper Sec. 2.3 and 6.1, footnotes 2 and 7). This repository is that extended simulator.

What we added on top of upstream SimAI (ns-3 backend):

| Component | Where | What it does |
|---|---|---|
| `NcclCustom` collective | `astra-sim-alibabacloud/astra-sim/system/collective/NcclCustom.{hh,cc}` | Executes an MSCCL schedule (thread blocks, steps, dependencies, channels) as ns-3 RDMA flows, using the same point-to-point primitives, MemBus injection model and send-latency model as SimAI's native NCCL algorithms. |
| `mscclCustomFlow` implementation | `astra-sim/system/Sys.{hh,cc}` | New value for `all-gather-implementation`, `all-to-all-implementation`, `reduce-scatter-implementation`, `all-reduce-implementation` in the system file; new keys `msccl-path`, `msccl-channels`, `collective-graph*`. |
| MSCCL schedule lookup | `SimCCL/src/mock/v2.{20,30}/MockNcclGroup.cc`, `XmlReader.cc` | Finds the schedule matching collective type, group size, data size, group type and first rank; falls back to SimAI's NCCL model if none matches. |
| Collective graph | `astra-sim/system/collective/CollectiveGraph.{hh,cc}` | Optional dump of the executed dependency graph (debugging). |
| Inputs used in the paper | `astra-sim-alibabacloud/inputs/` | NCCL-equivalent MSCCL schedules (`msccl/nccl/`), the MCCL schedules synthesized for the 16-GPU topology with and without each failure (`msccl/mccl/`), rail-optimized HPN topologies with the three failure profiles (`topo/`), single-collective and DeepSeek-671B training workloads (`workloads/`), system files (`system/`). |
| Tests and runners | `tests/ns3/msccl-nccl-equivalence/`, `artifact/` | Equivalence test, experiment runner, summarizer, Docker image, archive script. |

### Claims supported by this artifact

| # | Claim (paper) | Experiment | Expected result |
|---|---|---|---|
| C1 | The custom-schedule path preserves SimAI's accuracy: an NCCL ring/P2P schedule fed through it reproduces SimAI's native NCCL completion times (Sec. 2.3). | **E1** | Identical simulated end time (to the ns) for every workload × topology pair; `PASS 16/16`. |
| C2 | Under the three gray-failure profiles, static schedules collapse to the speed of the degraded link (Sec. 2.3, 6.1; "Orig"/"No reopt" in Fig. 1 and 8). | **E2** | Static schedules slow down to the degraded link: AllGather ≈ 3.4× (Fail1), ≈ 7× (Fail2), ≈ 55× (Fail3); AllToAll unchanged (Fail1, degraded NVLink not on its bottleneck), ≈ 8× (Fail2), ≈ 14× (Fail3). |
| C3 | Under gray failures, keeping the schedule synthesized for the healthy fabric ("no reopt") performs about like NCCL, while re-synthesizing it for the failure ("reopt") gives a multiple of NCCL's bus bandwidth (Sec. 6.1, Fig. 8, 1x2x8 points; MCCL schedules included, synthesizers not). | **E3** | All 22 simulations `SUCCESS`, each with its intended schedule. BusBW ratio vs NCCL, no reopt → reopt: AllGather 1.14× (no failure), 1.14 → 1.47× (Fail1), 1.04 → 7.38× (Fail2), 1.13 → 3.01× (Fail3); AllToAll 1.00× (no failure), 0.77 → 1.00× (Fail1), 1.00 → 7.11× (Fail2), 0.67 → 2.79× (Fail3). |
| C4 | Full LLM training (DeepSeek-671B, one of 16 pipeline stages on 128 GPUs, i.e. a 2048-GPU job) runs through the custom-schedule path under every failure profile, and the training time on one batch is reconstructed from the simulated phases as in the paper (Sec. 6.3; NCCL bars of Fig. 11c,d and Fig. 17). | **E4** | All 8 simulations `SUCCESS` with the MSCCL schedules in use; reconstructed training time on one batch 18.66 s with no failure; unchanged under Fail1 (1.00×), ≈ 5.5× under Fail2 and ≈ 5.3× under Fail3. |

### Not in this artifact (out of scope)

* The MCCL **schedule synthesizers** (ForestColl-warm, column-generation AllToAll): solving
  times (Fig. 7) and rounding gap (Table 5). The schedules they produced for the 16-GPU topology
  are included and evaluated in E3; schedules for larger topologies (other points of Fig. 8) and
  for the training workloads (MCCL bars of Fig. 11) are not.
* The **hardware prototype** (measurement module, hot-swap; Fig. 9, 10, Tables 2–3), which needs
  H200 GPUs, ConnectX-7 NICs and a modified NCCL/MSCCL.
* Llama-405B workloads and cluster sizes other than 128 GPUs for the training simulations
  (Fig. 11a,b and the 256–2048-GPU points); they run through the same path and can be generated
  with AICB (§6).

Any MSCCL-XML schedule (e.g. from MSCCL, MSCCLang, TACCL, TE-CCL or ForestColl exporters) can be
dropped in and evaluated, see §6.

---

## 2. Getting the artifact

**Option A – Zenodo archive (self-contained, includes all submodules):**
```bash
# download SimAI-MCCL-artifact-<rev>.tar.gz from **ZENODO-DOI-TODO**
tar xzf SimAI-MCCL-artifact-*.tar.gz && cd SimAI-MCCL-artifact-*
```

**Option B – git:**
```bash
git clone --recurse-submodules --branch conext26-ae https://github.com/NetMeasurements-Team/SimAI
cd SimAI
```
(The submodules `SimCCL` and `ns-3-alibabacloud` are pinned to commits on their
`custom-collectives` / `RDMA-Push` branches; `--recurse-submodules` fetches the right ones.
The Zenodo archive is preferred: it cannot change after publication.)

## 3. Building (≈ 30 min on 2 cores)

**Native (Ubuntu 22.04/24.04):**
```bash
sudo apt-get install -y build-essential cmake libxml2-dev python3 python3-matplotlib python3-pandas
bash scripts/build.sh -c ns3            # add -j N to set build parallelism
ls -l bin/SimAI_simulator               # the simulator binary
```
The build script creates `/etc/astra-sim/` (SimAI's log/trace directory), so run it as a user
that can write there (or as root / in Docker).

**Docker:**
```bash
docker build -f artifact/Dockerfile -t mccl-simai .
docker run --rm -it -v "$PWD/artifact/results:/workspace/SimAI/artifact/results" mccl-simai
# all commands below work unchanged inside the container
```
Behind an HTTP proxy, or if Docker's bridge network does not work on the host, add
`--network host --build-arg http_proxy=$http_proxy` to `docker build` and `--network host` to
`docker run`.

**Native and Docker side by side.** Both routes can run at the same time on the same checkout.
Result directories are named `native-<experiment>-<timestamp>` and
`docker-<experiment>-<timestamp>` (the label is detected automatically, `-t LABEL` overrides it),
every invocation uses its own simulator run names, and the container has its own `/etc/astra-sim`.
The image is built from the sources only (`.dockerignore` excludes the native build trees), so a
native build in the same checkout does not leak into it.

## 4. Smoke test (≈ 15 s, "kick the tires")

```bash
artifact/run.sh smoke
```
Runs the 256 MB AllGather and AllToAll on 16 GPUs with no failure, once through the
custom-schedule path and once through SimAI's native NCCL model. Expected tail of the output:
```
workload               failure       nccl_msccl_ns simai_native_ns  status
ag-268435456_bytes     no_failure           735808          735808  PASS
a2a-268435456_bytes    no_failure          3035739         3035739  PASS

PASS: 2/2 pairs identical. Results: .../native-smoke-<timestamp>/E1-equivalence/results.csv
```

## 5. Experiments

Each command writes to a fresh `artifact/results/<label>-<experiment>-<timestamp>/` directory
(`<label>` = `native` or `docker`; or `-o DIR`) containing
`provenance.txt` (git commits, binary hash, host), one sub-directory per simulation
(`command.txt`, `run.log`, `exit_code`, `wall_seconds`) and a `results.csv`.
`-j N` sets the number of parallel simulations (default: all CPUs);
`-f no_failure,failure_2` restricts the failure profiles.

```bash
artifact/run.sh equiv     # E1  32 sims   ≈ 2 min
artifact/run.sh single    # E2  16 sims   ≈ 2 min
artifact/run.sh mccl      # E3  22 sims   ≈ 8 min
artifact/run.sh e2e       # E4   8 sims   ≈ 1 h (up to ≈ 3 GB RAM per simulation)
artifact/run.sh all       # E1 + E2 + E3 + E4 into one directory
python3 artifact/summarize.py artifact/results/<dir> [--plots]   # tables -> summary.md
```
A run exits non-zero if any simulation does not finish. Status `STALLED` means the simulator ran
out of events with messages still pending (it lists them in `run.log`); it is deterministic, so
retrying does not help.
Times were measured on a 2-vCPU / 8 GB cloud VM (Ubuntu 24.04, GCC 13.3) with `-j 2`; they
shrink roughly linearly with more cores.

### E1 – Custom-schedule path vs native NCCL model (claim C1)

`tests/ns3/msccl-nccl-equivalence/run.sh`, wrapped by `artifact/run.sh equiv`. AllGather and
AllToAll, 256 MB and 1 GB, on the 2×8 topology with no failure and Fail1–3. The `nccl` system
file routes them through `mscclCustomFlow` with the NCCL-equivalent schedules in
`inputs/msccl/nccl/` (ring AllGather on SimCCL's rings, P2P AllToAll on SimCCL's PXN channel
map); `simai_native` uses `ncclFlowModel`. The test sets `SIMAI_PROTO_AWARE=0` so neither side
pipelines steps into 512 KB chunks (NCCL's native flow model sends each step as one flow).
**Pass:** every pair has the same end time, the `nccl` run actually used the MSCCL schedule
(`Found matching MSCCL XML algorithm` in its log) and the native run did not.

### E2 – Single collectives under gray failures (claim C2)

Same workloads and topologies, with the configuration used for all MCCL experiments in the paper
(protocol-aware MSCCL path with 512 KB chunking, PXN on, 3 µs send latency), through the
custom-schedule path with the NCCL-equivalent schedules. Expected slowdown vs no failure
(reference values):

| Collective (256 MB and 1 GB) | no failure | Fail1 | Fail2 | Fail3 |
|---|---|---|---|---|
| AllGather | 0.76 / 2.92 ms | ≈ 3.4× slower | ≈ 7× slower | ≈ 54–56× slower |
| AllToAll | 2.72 / 10.86 ms | unchanged (1.00×) | ≈ 8× slower | ≈ 14× slower |

AllToAll is unaffected by Fail1 because, with 512 KB chunking, its NVLink hop to the PXN proxy
overlaps with the NIC transfer, which hides the degraded NVLink of GPU 0. (E1 runs without
chunking to compare against SimAI's native model, which does not model NCCL's pipelining; there
the hops are sequential and Fail1 does cost AllToAll ≈ 1.56×.)

All runs must be `SUCCESS` and report `schedule used = msccl`.

`summarize.py` compares every simulated time against our reference run in
[`artifact/expected/`](artifact/expected/) and reports `N/N runs identical`; the simulator is
deterministic, so with the archived commits and the default build profile the times match to
the nanosecond (a different compiler could introduce small differences).

Failure profiles as encoded in the topology files `*-failure_{1,2,3}` (each GPU has an 18-lane
NVSwitch link of 2880 Gbps and one 400 Gbps NIC to its rail ToR; the same changes are applied
to the 2×8 topology of E1–E3 and the 1×16×8 topology of E4):

| Profile | Paper definition (Sec. 6.1) | Topology file change |
|---|---|---|
| Fail1 | 13 of 18 NVSwitch lanes of GPU 0 fail | GPU 0 ↔ NVSwitch 2880 → 800 Gbps |
| Fail2 | GPU 0's link to its ToR degraded | GPU 0 ↔ ToR 400 → 50 Gbps |
| Fail3 | GPU 0 replaced by a backup in another rack (+1 µs ToR latency) | GPU 0 ↔ NVSwitch 50 Gbps, GPU 0 ↔ ToR 100 Gbps, latency 0.5 → 1.5 µs |

### E3 – MCCL schedules vs NCCL under gray failures (claim C3)

2 GB AllGather and AllToAll (128 MB per source) on the 2×8 topology, with the paper configuration
of E2, for each topology (no failure and Fail1–3) and three schedule sets:

| Run | System file | Schedules (`inputs/msccl/`) |
|---|---|---|
| `nccl` | `nccl.txt` | NCCL-equivalent (`nccl/`) |
| `mccl_noreopt` | `mccl-no_failure.txt` | MCCL schedules synthesized for the healthy fabric (`mccl/no_failure/`), kept under the failure |
| `mccl_reopt` | `mccl-failure_N.txt` | MCCL schedules re-synthesized for that failure (`mccl/failure_N/`); not run without failure |

That is 22 simulations. `results.csv` records the schedule file each run actually loaded and
`summarize.py` fails if it is not the intended one. It reports, per topology, the three times and
the BusBW ratio against NCCL (= T_NCCL / T, the paper's metric in Fig. 1 and 8) and, with
`--plots`, draws them like Fig. 8 (hatched: no reopt; solid: reopt; dashed line: NCCL).
Expected (reference values, `artifact/expected/E3-mccl.csv`):

| | | NCCL | MCCL no reopt | MCCL reopt |
|---|---|---|---|---|
| AllGather | no failure | 5.77 ms | 5.06 ms (1.14×) | – |
| | Fail1 | 20.15 ms | 17.75 ms (1.14×) | 13.72 ms (1.47×) |
| | Fail2 | 40.77 ms | 39.14 ms (1.04×) | 5.53 ms (7.38×) |
| | Fail3 | 326.02 ms | 287.35 ms (1.13×) | 108.35 ms (3.01×) |
| AllToAll | no failure | 21.72 ms | 21.73 ms (1.00×) | – |
| | Fail1 | 21.72 ms | 28.24 ms (0.77×) | 21.72 ms (1.00×) |
| | Fail2 | 173.86 ms | 173.86 ms (1.00×) | 24.45 ms (7.11×) |
| | Fail3 | 304.53 ms | 456.51 ms (0.67×) | 109.32 ms (2.79×) |

Without re-optimization MCCL's schedules stay close to NCCL under a failure (or fall below it for
AllToAll under Fail1/Fail3); re-optimized for the failure they reach up to 7.38× (AllGather) and
7.11× (AllToAll) NCCL's bus bandwidth under Fail2. The AllGather ratios correspond to the 1x2x8
group of Fig. 8(a).

### E4 – DeepSeek-671B training under gray failures (claim C4)

DeepSeek-v3 671B (256 experts; EP = 64, PP = 16), 16K-token batch, forward and backward pass of
one pipeline stage simulated on 128 GPUs (1 segment × 16 hosts × 8 GPUs, HPN rail-optimized,
`inputs/topo/end_to_end/HPN/1x16x8*`), which corresponds to a 2048-GPU training job (Sec. 6.3).
The iteration is simulated as two runs per failure profile, the transformer phase (MoE AllToAll
and AllGather on EP groups) and the gradient-synchronization phase (DP and DP_EP
AllGather/ReduceScatter). AllGather, ReduceScatter and AllToAll go through the custom-schedule
path with the NCCL-equivalent schedules (any collective without a matching schedule falls back
to SimAI's NCCL model and is reported as `msccl+fallback`), i.e. this is the NCCL baseline of
Fig. 11(c,d) and Fig. 17 at 128 GPUs.

**Reconstruction of the full training.** The transformer workload simulates one transformer
block for the per-GPU share of the batch, so the training time on one 16K-token batch of the
stage is reconstructed as in the paper:

```
T_batch = T_transformer × 3 (transformer blocks per stage) × 16384 / (128 GPUs / TP 1) + T_grad
```

`summarize.py` prints this per failure profile together with the slowdown vs no failure.
`benchmark/analyze_results.py` applies the same reconstruction and draws the training time per
failure profile as a bar plot in the style of Fig. 16–17 (`benchmark/plot_results.py`):

```bash
python3 benchmark/analyze_results.py artifact/results/<dir>/E4-e2e/results.csv
# -> <dir>/E4-e2e/combined_results.csv and <dir>/E4-e2e/plots/DeepSeek671B_training_time.{png,pdf}
```

Expected (reference values; all 8 runs `SUCCESS` with `schedule used = msccl`):

| Failure | transformer pass (ms) | gradient phase (ms) | training time on one batch (s) | vs no failure |
|---|---|---|---|---|
| no failure | 48.45 | 59.44 | 18.66 | 1.00× |
| Fail1 | 48.45 | 86.94 | 18.69 | 1.00× |
| Fail2 | 266.87 | 339.68 | 102.82 | 5.51× |
| Fail3 | 257.02 | 756.40 | 99.45 | 5.33× |

Fail1 barely matters for the training: the gradient phase slows down (59 → 87 ms) but the
transformer pass, which dominates the batch time, does not. As for E1–E3, `summarize.py` checks
every simulated time against `artifact/expected/E4-e2e.csv`.

E4 is much longer than E1–E3 (≈ 1 h on 2 cores with `-j 2`; the gradient-phase simulations
are the longest). The 8 simulations are independent, so with ≥ 8 cores and `-j 8` it takes about
as long as the slowest one. To check a subset, e.g. only no failure and Fail2:
`artifact/run.sh e2e -f no_failure,failure_2`.

## 6. Reuse: evaluating your own schedules

1. Put MSCCL XML files (MSCCL/MSCCLang format: `<algo coll="allgather|alltoall|..." ngpus=...
   nchannels=... nchunksperloop=... minBytes=... maxBytes=...>` with `<gpu>/<tb>/<step>`) in a
   directory, e.g. `my-schedules/`.
2. Copy `astra-sim-alibabacloud/inputs/system/nccl.txt` and set
   `msccl-path: my-schedules/` (relative to the directory the simulator runs in) and the
   `*-implementation: mscclCustomFlow` lines for the collectives you want to replace.
3. Run `bin/SimAI_simulator -t 1 -w <workload> -n <topology> -c astra-sim-alibabacloud/inputs/config/SimAI.conf -s <your sys file>`
   and check the log for `Found matching MSCCL XML algorithm ... <file>`.

Matching rules: a schedule is used for a collective when `coll`, `ngpus` (= communicator size)
and the data size (`minBytes ≤ size ≤ maxBytes`, `maxBytes=0` = unbounded) match; optional
`groupType` (TP, DP, EP, DP_EP, …) and `firstRankId` restrict it to one communicator. Files in
deeper sub-directories take priority, then lexicographic order (hence `priority-0/`,
`priority-1/`); directories named `disabled/` are skipped. Channel ids must be below
`msccl-channels` (default 128). Unmatched collectives fall back to SimAI's NCCL model and the log
says so. Workloads and topologies for other models and cluster sizes are produced with AICB
(`aicb/`) and `astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py`, as in upstream SimAI
(see `docs/Tutorial.md`).

## 7. Contact

Michele Ferrero, EURECOM – ferrerom@eurecom.fr (contact author during the evaluation; please use
HotCRP for anonymous questions).
