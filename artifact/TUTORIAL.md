# MCCL tutorial — student guide

This guide walks through the MCCL story from the original NCCL algorithms to
static optimization, gray failures, re-optimization, and the training reference
results. Run every command from the repository root.

The tutorial uses only the **16-GPU 1x2x8 topology** for drawings and packet
animations. The 128-GPU 1x16x8 topology is used only by the training experiment,
and that experiment is shown from reference results.

---

## 0. Setup

```bash
sudo apt-get install -y build-essential cmake libxml2-dev python3 \
  python3-numpy python3-scipy python3-pil python3-matplotlib python3-pandas
bash scripts/build.sh -c ns3
ls -l bin/SimAI_simulator
```

If a previous build is broken, or you changed compiler/toolchain, clean and
rebuild:

```bash
bash scripts/build.sh -lc ns3
```

Docker alternative:

```bash
docker build -f artifact/Dockerfile -t mccl-simai .
docker run --rm -it -v "$PWD/artifact/results:/workspace/SimAI/artifact/results" mccl-simai
```

### Smoke test

Run this first. It catches most build, path and simulator problems before the
longer experiments.

```bash
rm -rf artifact/results/00_setup
artifact/run.sh smoke -o artifact/results/00_setup
```

Expected:

```text
PASS: 2/2 pairs identical.
```

If it fails, try:

```bash
bash scripts/build.sh -lc ns3
artifact/run.sh smoke
```

---

## 1. How SimAI runs a collective

SimAI is an ns-3-based network simulator. For one collective it takes four
inputs:

- a **workload** file: which collective, which message size, how many GPUs;
- a **topology** file: nodes and links, each with bandwidth and latency;
- an **ns-3 config** (`inputs/config/SimAI.conf`): packet, congestion-control and
  PFC settings;
- a **system config**: which implementation to use and where the schedule XMLs
  live.

The simulator is a single binary:

```bash
bin/SimAI_simulator \
  -t 1 \
  -w astra-sim-alibabacloud/inputs/workloads/single_collective/16_gpus/ag-2147483648_bytes \
  -n astra-sim-alibabacloud/inputs/topo/single_collective/1x2x8 \
  -c astra-sim-alibabacloud/inputs/config/SimAI.conf \
  -s astra-sim-alibabacloud/inputs/system/nccl.txt \
  -r demo_ag
```

Parameters used throughout the tutorial:

| Flag | Meaning |
|---|---|
| `-t` | number of simulator threads (1 here) |
| `-w` | workload file (collective, size, GPU group) |
| `-n` | topology file |
| `-c` | ns-3 network config |
| `-s` | system config (implementation + schedule directory) |
| `-r` | run name (used for logs and output files) |

`artifact/run.sh` and `artifact/visualize_flows.py` build exactly this command
for you; the runner executes it inside the run's output directory.

### The workload file

The workload is a short text description. Its header sets the GPU group
(`model_parallel_NPU_group: 16`); the collective line for the 2 GB AllGather is:

```text
allgather	-1	1	ALLGATHER	2147483648	1		NONE	0	1	NONE	0	1
```

The fields that matter here are the collective type (`ALLGATHER`) and the message
size (`2147483648` bytes = 2 GB).

### The system config: which schedule is simulated

The system config is where the schedule is selected. The relevant fields are:

```text
all-gather-implementation: mscclCustomFlow
all-to-all-implementation: mscclCustomFlow
reduce-scatter-implementation: mscclCustomFlow
msccl-path: astra-sim-alibabacloud/inputs/msccl/nccl/
collective-optimization: baseline
collective-graph: 0
collective-graph-ranks: 0
```

- `*-implementation: mscclCustomFlow` tells SimAI to execute an XML schedule
  instead of the analytic flow model.
- `msccl-path` is the directory SimAI searches for the XML. **Changing this one
  line changes the simulated schedule.**
- `collective-graph: 1` dumps the realized per-flow graph to CSV (the packet
  animations reuse it); `collective-graph-ranks` optionally restricts the dump to
  groups containing the listed rank ids, and `0` keeps the group that contains
  rank 0.

The tutorial switches schedules only by changing `msccl-path`:

| System config | `msccl-path` | Schedule simulated |
|---|---|---|
| `inputs/system/nccl.txt` | `inputs/msccl/nccl/` | NCCL ring / P2P baseline |
| `inputs/system/mccl-no_failure.txt` | `inputs/msccl/mccl/no_failure/` | static MCCL, healthy fabric |
| `inputs/system/mccl-failure_2.txt` | `inputs/msccl/mccl/failure_2/` | MCCL re-synthesized for Fail2 |

SimAI picks the XML in that directory that matches the collective, the message
size and the topology (for example `1x2x8_MCCL_AG_Fail2.xml`) and logs
`Found matching MSCCL XML algorithm: ...`. For a single visualization,
`artifact/visualize_flows.py --schedule <xml>` writes a temporary system file
with `msccl-path` pointing at that XML's directory.

---

## 2. Original NCCL algorithms: Ring/Tree and PAT

NCCL 2.20 chooses between **Ring** and **Tree** depending on the collective, the
message size and the topology:

- **Ring** sends chunks around a logical ring. It is bandwidth-optimal for large
  messages, but the number of steps grows with the number of GPUs.
- **Tree** has logarithmic depth and lower latency for small messages. NCCL uses
  it mainly for reduction-heavy collectives such as AllReduce.
- In 2.20, AllGather and AllToAll have no PAT.

NCCL 2.30 adds **PAT** (Parallel Aggregated Trees) for AllGather and ReduceScatter:

- eligible when there is more than one node and one GPU per node;
- selected only inside a size window (default **512 KiB to 1 MiB**);
- uses the Simple protocol;
- outside the window, or on an ineligible topology, NCCL falls back to Ring.

**For reference, do not run as already compiled**
Compile each version with a clean build:

```bash
bash scripts/build.sh -lc ns3 --nccl-version 2.20
bash scripts/build.sh -lc ns3 --nccl-version 2.30
```

The XML schedules are version-independent. Use the NCCL-style XML schedules to see
the ring traffic pattern on the 16-GPU topology. **All visualizations in this
tutorial use the same 2 GB collective**, so the ring, static and re-optimized
schedules can be compared directly:

The two animations are independent, so run them in parallel:

```bash
python3 artifact/visualize_flows.py \
  --topology astra-sim-alibabacloud/inputs/topo/single_collective/1x2x8 \
  --schedule astra-sim-alibabacloud/inputs/msccl/nccl/priority-1/NCCL_AllGather_16gpus_2nodes.xml \
  --size 2147483648 \
  --workload astra-sim-alibabacloud/inputs/workloads/single_collective/16_gpus/ag-2147483648_bytes \
  --run-sim --out artifact/results/01_nccl_algorithms/nccl_ring_ag.gif &

python3 artifact/visualize_flows.py \
  --topology astra-sim-alibabacloud/inputs/topo/single_collective/1x2x8 \
  --schedule astra-sim-alibabacloud/inputs/msccl/nccl/priority-1/NCCL_AllToAll_P2P_16gpus.xml \
  --size 2147483648 \
  --workload astra-sim-alibabacloud/inputs/workloads/single_collective/16_gpus/a2a-2147483648_bytes \
  --run-sim --out artifact/results/01_nccl_algorithms/nccl_ring_a2a.gif &

wait
```

At 2 GB the ring baseline is **5.772 ms** for AllGather (372.0 GB/s) and
**21.720 ms** for AllToAll (98.9 GB/s). These are the reference points for the
static and re-optimized schedules in the next sections.

In the 16-GPU 2x8 topology there are eight GPUs per node, so PAT is not eligible;
NCCL 2.30 uses Ring here. PAT matters for multi-node topologies with one GPU per
node.

---

## 3. Static optimization on the healthy fabric

The first optimization step is still open-loop: synthesize a schedule once for the
healthy topology and keep it. The MCCL schedules for the healthy fabric are in:

```text
astra-sim-alibabacloud/inputs/msccl/mccl/no_failure/priority-9/1x2x8_MCCL_AG.xml
astra-sim-alibabacloud/inputs/msccl/mccl/no_failure/priority-9/MCCL_1x2x8.xml
```

On the healthy 16-GPU topology, this static schedule is slightly better than NCCL
for AllGather: **5.062 ms vs 5.772 ms**, or **1.14x** algorithm bandwidth. AllToAll is
equal to NCCL: **21.733 ms vs 21.720 ms**, or **1.00x**. The bottleneck cut is the
ingress of a single GPU for AllGather (every chunk has to reach that last GPU) and
the eight inter-node ToR links for AllToAll (the two servers exchange all their
data).

Show the static AllGather and AllToAll schedules with the simulator's original
timestamps. They are independent, so run them in parallel:

```bash
python3 artifact/visualize_flows.py \
  --topology astra-sim-alibabacloud/inputs/topo/single_collective/1x2x8 \
  --schedule astra-sim-alibabacloud/inputs/msccl/mccl/no_failure/priority-9/1x2x8_MCCL_AG.xml \
  --size 2147483648 \
  --workload astra-sim-alibabacloud/inputs/workloads/single_collective/16_gpus/ag-2147483648_bytes \
  --run-sim --out artifact/results/02_static_optimization/mccl_static_ag.gif &

python3 artifact/visualize_flows.py \
  --topology astra-sim-alibabacloud/inputs/topo/single_collective/1x2x8 \
  --schedule astra-sim-alibabacloud/inputs/msccl/mccl/no_failure/priority-9/MCCL_1x2x8.xml \
  --size 2147483648 \
  --workload astra-sim-alibabacloud/inputs/workloads/single_collective/16_gpus/a2a-2147483648_bytes \
  --run-sim --out artifact/results/02_static_optimization/mccl_static_a2a.gif &

wait
```

The AllToAll schedule is shown as well, even though it is 1.00x NCCL.

These are the schedules that will be kept when the first failure appears.

---

## 4. Draw the topologies and its failures

The main topology is **1 segment x 2 hosts x 8 GPUs = 16 GPUs**:

- 8 GPUs per host, connected by an **NVSwitch** at **2880 Gbps**, 25 ns delay;
- each GPU has one **400 Gbps** NIC connected to a **ToR / leaf switch**, 0.5 µs delay;
- it is **rail-optimized**: GPU `i` on every host connects to the same rail. ToR 18
  connects GPU 0 and GPU 8, ToR 19 connects GPU 1 and GPU 9, and so on.

```text
Host 0:  GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
            \    |     |     |     |     |     |    /
             +---+-----+-----+-----+-----+-----+---+
                       NVSwitch 16
                    (2880 Gbps, 25 ns)

Host 1:  GPU8  GPU9  GPU10 GPU11 GPU12 GPU13 GPU14 GPU15
            \    |     |     |     |     |     |    /
             +---+-----+-----+-----+-----+-----+---+
                       NVSwitch 17
                    (2880 Gbps, 25 ns)

Rail 0:  GPU0 + GPU8   -> ToR18      Rail 4:  GPU4 + GPU12  -> ToR22
Rail 1:  GPU1 + GPU9   -> ToR19      Rail 5:  GPU5 + GPU13  -> ToR23
Rail 2:  GPU2 + GPU10  -> ToR20      Rail 6:  GPU6 + GPU14  -> ToR24
Rail 3:  GPU3 + GPU11  -> ToR21      Rail 7:  GPU7 + GPU15  -> ToR25
                    each ToR link: 400 Gbps, 0.5 us
```

Topology files:

```text
astra-sim-alibabacloud/inputs/topo/single_collective/1x2x8
astra-sim-alibabacloud/inputs/topo/single_collective/1x2x8-failure_1
astra-sim-alibabacloud/inputs/topo/single_collective/1x2x8-failure_2
astra-sim-alibabacloud/inputs/topo/single_collective/1x2x8-failure_3
```

### Draw the four topologies

Each profile is written as its own large PNG so it can be shown full-screen:

```bash
python3 artifact/draw_topologies.py --outdir artifact/results/03_topologies
```

Output:

```text
artifact/results/03_topologies/topology_1x2x8_no_failure.png
artifact/results/03_topologies/topology_1x2x8_failure_1.png
artifact/results/03_topologies/topology_1x2x8_failure_2.png
artifact/results/03_topologies/topology_1x2x8_failure_3.png
```

Green lines are intra-node NVLink, dark lines are inter-node ToR links, and
degraded links are magenta dashed with a `FAIL <rate>` badge.

### Failure profiles

| Profile | What changes |
|---|---|
| **Fail1** | GPU 0 ↔ NVSwitch: **2880 → 800 Gbps** (13 of 18 NVSwitch lanes fail) |
| **Fail2** | GPU 0 ↔ ToR: **400 → 50 Gbps** |
| **Fail3** | GPU 0 ↔ NVSwitch: **50 Gbps**; GPU 0 ↔ ToR: **100 Gbps**; ToR latency **0.5 → 1.5 µs** |

All other links stay at peak capacity. The failure is local, so a re-optimized
schedule can route around GPU 0.

---

## 5. Experiment 1 — static schedules under gray failures

Keep the schedule synthesized for the healthy fabric and run it under each
failure. The schedule does not know that GPU 0 is degraded, so it keeps sending
traffic through it.

```bash
rm -rf artifact/results/04_static_failures
artifact/run.sh single -o artifact/results/04_static_failures
```

This runs the NCCL-equivalent schedules with 512 KiB pipelining, PXN enabled, and
3 µs send latency.

Expected slowdowns versus no failure (exact, deterministic):

| Collective | Size | Fail1 | Fail2 | Fail3 |
|---|---|---:|---:|---:|
| AllGather | 256 MB | 3.35x | 6.76x | 53.92x |
| AllGather | 1 GB | 3.46x | 6.99x | 55.89x |
| AllToAll | 256 MB | 1.00x | 7.99x | 14.09x |
| AllToAll | 1 GB | 1.00x | 8.00x | 14.03x |

The two rows per collective are the two workload sizes, not run-to-run variation.
The simulator is deterministic.

AllToAll Fail1 is unchanged because the NVLink hop to the PXN proxy overlaps with
the NIC transfer. With chunking off, Fail1 does cost AllToAll about 1.56x.

Summarize and plot E2 immediately:

```bash
python3 artifact/summarize.py artifact/results/04_static_failures --plots
```

---

## 6. Experiment 2 — MCCL no-reopt vs reopt

Run this experiment into a fixed directory. The MCCL run enables collective graphs,
so the visualization commands can reuse them instead of running the simulator
again.

```bash
rm -rf artifact/results/05_reoptimization
artifact/run.sh mccl -o artifact/results/05_reoptimization
```

This runs 2 GB AllGather and AllToAll (128 MB per source) on the 2x8 topology, for
each failure, with three schedule sets:

| Run | Meaning |
|---|---|
| `nccl` | NCCL-equivalent baseline |
| `mccl_noreopt` | MCCL schedule for the healthy fabric, kept under the failure |
| `mccl_reopt` | MCCL schedule re-synthesized for that failure |

Expected results (time and AlgBW ratio against NCCL):

| | | NCCL | MCCL no reopt | MCCL reopt |
|---|---|---|---|---|
| AllGather | no failure | 5.772 ms | 5.062 ms (1.14x) | — |
| | Fail1 | 20.152 ms | 17.747 ms (1.14x) | 13.720 ms (1.47x) |
| | Fail2 | 40.768 ms | 39.145 ms (1.04x) | 5.525 ms (7.38x) |
| | Fail3 | 326.025 ms | 287.346 ms (1.13x) | 108.352 ms (3.01x) |
| AllToAll | no failure | 21.720 ms | 21.733 ms (1.00x) | — |
| | Fail1 | 21.720 ms | 28.240 ms (0.77x) | 21.720 ms (1.00x) |
| | Fail2 | 173.859 ms | 173.861 ms (1.00x) | 24.445 ms (7.11x) |
| | Fail3 | 304.527 ms | 456.513 ms (0.67x) | 109.318 ms (2.79x) |

Headline: AllGather Fail2 goes from **39.145 ms** to **5.525 ms**, a **7.08x**
speedup over no-reopt and **7.38x** NCCL algorithm bandwidth. AllToAll Fail2 goes from
**173.861 ms** to **24.445 ms**, a **7.11x** improvement.

Without re-optimization, MCCL's schedule stays close to NCCL, or falls below it
for AllToAll under Fail1/Fail3. The gain comes from re-optimizing.

Summarize and plot E3 immediately:

```bash
python3 artifact/summarize.py artifact/results/05_reoptimization --plots
```

---

## 7. Watch the packets flow

These commands reuse the collective graphs from the MCCL run. The simulator is not
executed again, so all six animations can run in parallel:

```bash
python3 artifact/visualize_flows.py --demo ag-failure_1  --results-dir artifact/results/05_reoptimization/E3-mccl --out artifact/results/06_packet_animations/ag_failure_1.gif &
python3 artifact/visualize_flows.py --demo ag-failure_2  --results-dir artifact/results/05_reoptimization/E3-mccl --out artifact/results/06_packet_animations/ag_failure_2.gif &
python3 artifact/visualize_flows.py --demo ag-failure_3  --results-dir artifact/results/05_reoptimization/E3-mccl --out artifact/results/06_packet_animations/ag_failure_3.gif &
python3 artifact/visualize_flows.py --demo a2a-failure_1 --results-dir artifact/results/05_reoptimization/E3-mccl --out artifact/results/06_packet_animations/a2a_failure_1.gif &
python3 artifact/visualize_flows.py --demo a2a-failure_2 --results-dir artifact/results/05_reoptimization/E3-mccl --out artifact/results/06_packet_animations/a2a_failure_2.gif &
python3 artifact/visualize_flows.py --demo a2a-failure_3 --results-dir artifact/results/05_reoptimization/E3-mccl --out artifact/results/06_packet_animations/a2a_failure_3.gif &

wait
```

What to look for:

- left panel: healthy-fabric schedule kept under the failure (no reopt);
- right panel: schedule re-synthesized for the failure (reopt);
- green lines are intra-node NVLink, dark lines are inter-node ToR;
- magenta dashed line with a `FAIL <rate>` label is the degraded link;
- orange inner line marks a congested link (blue → amber → orange as load grows);
- red cross and star mark the bottleneck cut. The cut is fixed by the topology,
  failure and collective (the minimum-ratio cut of the collective's traffic), so
  **both panels show the same links**; each commodity crosses it once. Red is used
  only for this marker, never for a link;
- packets are coloured by source GPU and carry a short trail plus an arrowhead
  showing their direction;
- the bottom bar shows progress through the collective;
- the clock stops at the last ACK for each schedule;
- each panel freezes at its own completion, so the reoptimized panel stops at
  **5.525 ms** while no-reopt continues to **39.145 ms**;
- the GIF holds the final frame for 1.5 s before looping (`--hold-ms`).

Each run writes:

```text
<name>.gif
<name>_summary.md
<name>_summary.csv
```

The Markdown summary reports completion time, AlgBW (GB/s), the bandwidth each
schedule actually pushes through the cut and the speedup. The reoptimized schedule
reaches about **99% of the cut capacity**; the healthy-fabric schedule uses much
less of the same cut, which is exactly why it takes longer. Add `--timeline` if you
also want a bottleneck timeline PNG.

---

## 8. Experiment 3 — DeepSeek-671B training (reference only)

Do not run this in the 30-minute tutorial. It takes about **37.5 minutes** in our
environment. The reference results are in `artifact/expected/E4-e2e.csv`; show this
table instead.

| Failure | transformer (ms) | gradient (ms) | batch (s) | vs no failure |
|---|---:|---:|---:|---:|
| no failure | 48.448 | 59.443 | 18.66 | 1.00x |
| Fail1 | 48.454 | 86.942 | 18.69 | 1.00x |
| Fail2 | 266.874 | 339.680 | 102.82 | 5.51x |
| Fail3 | 257.022 | 756.403 | 99.45 | 5.33x |

Fail1 barely matters for training: the gradient phase slows down, but the
transformer phase dominates the batch time and is unchanged. Fail2 and Fail3 are
severe: the batch time grows by about 5.5x and 5.3x.

---

## 9. Useful options

```text
-j N                         parallel simulations
-f no_failure,failure_2      restrict failure profiles
-o DIR                       choose the output directory
-t LABEL                     run label (native / docker)
-b PATH                      simulator binary
```

Typical timings on the VM machine with the build already done:

| Step | Time |
|---|---:|
| Draw 1x2x8 topologies | 3.4 s |
| Smoke test | 15.6 s |
| Ring animations (2 GB, in parallel) | 142 s |
| Static animations (2 GB, in parallel) | 142 s |
| E2 single collectives | 73.6 s |
| E3 MCCL + graphs | 270.7 s |
| Six failure animations (in parallel) | 15 s |
| Summarize E2 + E3 | 2.8 s |
| **Total without E4** | **≈ 11 min** |
