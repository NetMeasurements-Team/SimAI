# MCCL artifact: SimAI with custom collective schedules

Artifact for **"MCCL: Measurement-based Collective Communication Library"**, ACM CoNEXT 2026
(Proc. ACM Netw. 4, CoNEXT4, Article 55, [10.1145/3856969](https://doi.org/10.1145/3856969)).

Badges requested: **Artifact Available** and **Artifact Evaluated – Functional**.
(*Results Reproduced* is not requested: it would also need the MCCL schedule synthesizers and
the H200 hardware prototype, which are not part of this artifact.)

| | |
|---|---|
| Archived artifact (DOI) | **ZENODO-DOI-TODO** |
| Code | https://github.com/NetMeasurements-Team/SimAI, tag `conext26-ae` (branch `custom-collectives`) |
| Hardware | any x86-64 Linux machine, no GPU; ≥ 2 cores, ≥ 4 GB RAM (≈ 0.7 GB per parallel simulation), ≈ 3 GB disk |
| Software | Ubuntu 22.04/24.04 (or Docker), GCC ≥ 11, CMake ≥ 3.13, libxml2-dev, Python 3 (matplotlib optional) |
| Proprietary software / hardware | none |
| Time | build ≈ 30 min on 2 cores; smoke test 15 s; all experiments ≈ 8 min on 2 cores |

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
| Inputs used in the paper | `astra-sim-alibabacloud/inputs/` | NCCL-equivalent MSCCL schedules (`msccl/nccl/`), rail-optimized HPN topologies with the three failure profiles (`topo/`), single-collective workloads (`workloads/single_collective/`), system files (`system/`). |
| Tests and runners | `tests/ns3/msccl-nccl-equivalence/`, `artifact/` | Equivalence test, experiment runner, summarizer, Docker image, archive script. |

### Claims supported by this artifact

| # | Claim (paper) | Experiment | Expected result |
|---|---|---|---|
| C1 | The custom-schedule path preserves SimAI's accuracy: an NCCL ring/P2P schedule fed through it reproduces SimAI's native NCCL completion times (Sec. 2.3). | **E1** | Identical simulated end time (to the ns) for every workload × topology pair; `PASS 16/16`. |
| C2 | Under the three gray-failure profiles, static schedules collapse to the speed of the degraded link (Sec. 2.3, 6.1; "Orig"/"No reopt" in Fig. 1 and 8). | **E2** | Static schedules slow down to the degraded link: AllGather ≈2× (Fail1), ≈10× (Fail2), ≈32× (Fail3); AllToAll ≈10× (Fail2), ≈8× (Fail3), unchanged under Fail1 (the degraded NVLink is not on its bottleneck cut). |

### Not in this artifact (out of scope)

* The MCCL **schedule synthesizers** (ForestColl-warm, column-generation AllToAll) and the
  re-optimized schedules they produce: solving times (Fig. 7), rounding gap (Table 5) and the
  MCCL bars of Fig. 1, 8 and 11.
* The **hardware prototype** (measurement module, hot-swap; Fig. 9, 10, Tables 2–3), which needs
  H200 GPUs, ConnectX-7 NICs and a modified NCCL/MSCCL.
* The **end-to-end training simulations** (Llama-405B and DeepSeek-671B, 128–2048 GPUs;
  Sec. 6.3, Fig. 11, 16, 17). The work-in-progress workload and topology files under
  `inputs/workloads/end_to_end/` and `inputs/topo/end_to_end/` are not part of the evaluated
  artifact. The same custom-schedule path runs them; workloads can be generated with AICB (§6).

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
`custom-collectives` / `RDMA-Push` branches; `--recurse-submodules` fetches the right ones.)

## 3. Building (≈ 30 min on 2 cores)

**Native (Ubuntu 22.04/24.04):**
```bash
sudo apt-get install -y build-essential cmake libxml2-dev python3 python3-matplotlib
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

## 4. Smoke test (≈ 15 s, "kick the tires")

```bash
artifact/run.sh smoke
```
Runs the 256 MB AllGather and AllToAll on 16 GPUs with no failure, once through the
custom-schedule path and once through SimAI's native NCCL model. Expected tail of the output:
```
workload               failure       nccl_msccl_ns simai_native_ns  status
ag-268435456_bytes     no_failure          1327384         1327384  PASS
a2a-268435456_bytes    no_failure          5388473         5388473  PASS

PASS: 2/2 pairs identical. Results: .../smoke-<timestamp>/E1-equivalence/results.csv
```

## 5. Experiments

Each command writes to a fresh `artifact/results/<experiment>-<timestamp>/` directory containing
`provenance.txt` (git commits, binary hash, host), one sub-directory per simulation
(`command.txt`, `run.log`, `exit_code`, `wall_seconds`) and a `results.csv`.
`-j N` sets the number of parallel simulations (default: all CPUs);
`-f no_failure,failure2` restricts the failure profiles.

```bash
artifact/run.sh equiv     # E1  32 sims   ≈ 4 min
artifact/run.sh single    # E2  32 sims   ≈ 4 min
artifact/run.sh all       # E1 + E2 into one directory
python3 artifact/summarize.py artifact/results/<dir> [--plots]   # tables -> summary.md
```
Times were measured on a 2-vCPU / 8 GB cloud VM (Ubuntu 24.04, GCC 13.3), all with `-j 2`; they scale down roughly linearly with more cores.

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
(protocol-aware MSCCL path with 512 KB chunking, PXN on, 3 µs send latency). Both the
custom-schedule path (`nccl`) and SimAI native are run. Expected (reference values,
`nccl` path):

| Collective | Fail1 | Fail2 | Fail3 |
|---|---|---|---|
| AllGather (256 MB, 1 GB) | ≈ 2.0× slower | ≈ 10× slower | ≈ 32× slower |
| AllToAll (256 MB, 1 GB) | unchanged (1.00×) | ≈ 10× slower | ≈ 8× slower |

AllToAll is unaffected by Fail1 because the degraded 800 Gbps NVLink of GPU 0 is still faster
than the NIC links that bound the P2P schedule. All runs must be `SUCCESS` and the `nccl` runs
must report `schedule used = msccl`. With the protocol-aware chunking the custom path is slightly
faster than SimAI native (NCCL pipelining is modelled), which is why E1 disables it.

`summarize.py` compares every simulated time against our reference run in
[`artifact/expected/`](artifact/expected/) and reports `N/N runs identical`; the simulator is
deterministic, so with the archived commits and the default build profile the times match to
the nanosecond (a different compiler could introduce small differences).

Failure profiles as encoded in `inputs/topo/single_collective/1x2x8-failure{1,2,3}` (each GPU
has an 18-lane NVSwitch link of 2880 Gbps and two 200 Gbps rail ports):

| Profile | Paper definition | Topology file change |
|---|---|---|
| Fail1 | 13 of 18 NVSwitch lanes of GPU 0 fail | GPU 0 ↔ NVSwitch 2880 → 800 Gbps |
| Fail2 | NIC link of GPU 0 degraded | one GPU 0 rail port 200 → 20 Gbps |
| Fail3 | GPU 0 replaced by a backup in another rack (+1 µs) | GPU 0 ↔ NVSwitch and both rail ports → 50 Gbps, 1.5 µs |

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
