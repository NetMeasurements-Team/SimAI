# ns-3-ub collective completion and selection matrix

Standalone smoke matrix for the Unified Bus (`ns3-ub`) frontend. It reuses the
shared system and network configs from `tests/ns3/common/` but generates UB
topologies (no NVSwitch column) and runs the `bin/SimAI_simulator` built with
`SIMAI_MODE=ns3-ub`.

## Build

The runner does not build. Build the matching SimCCL version first:

```bash
./scripts/build.sh -lc ns3-ub -d optimized --nccl-version v2.20
# or
./scripts/build.sh -lc ns3-ub -d optimized --nccl-version v2.30
```

## Run

```bash
python3 tests/ns3-ub/run.py --nccl-version v2.20
python3 tests/ns3-ub/run.py --nccl-version v2.30
python3 tests/ns3-ub/run.py --nccl-version v2.30 --list
python3 tests/ns3-ub/run.py --nccl-version v2.30 --case cross-allreduce-LL
```

`--list` prints the case matrix and coverage gaps. Each case runs in its own
directory with `run.log`, `invocation.json`, the detailed-flow CSV, and the FCT
file. `summary.json` is updated after every case.

## Checks

The validator checks:

- collective/workload completion and issue markers,
- exactly one collective per layer,
- matching rank-0 stream counts,
- nonempty generated flows,
- completed FCT messages matching the generated source/destination/size multiset,
- v2.30 algorithm/protocol selection and PXN forwarding labels,
- duplicate `(collective, channel_id, flow_id)` detection,
- no `Size mismatch in sentHash!`, `no UbApp`, or packet loss.

The `nvls-unsupported` case is a negative test: it must exit non-zero and print
`[UB] ERROR: NVLS is not supported`.

## Coverage

| Cases | v2.20 | v2.30 |
|---|---|---|
| AllReduce, AllGather, ReduceScatter, AllToAll; local/cross server | UNDEF protocol | LL, LL128, Simple, explicitly forced |
| All four collectives with 3 ranks on 3 servers | Yes | Yes |
| AllToAll and ReduceScatter with PXN | Yes | All three protocols |
| PAT AllGather/ReduceScatter, 2/3/4 servers, one rank each | Unavailable | Simple |
| PAT automatic selection inside/outside size window; ineligible topology | Unavailable | Yes |
| PAT keeps Simple even when LL/LL128 is forced | Unavailable | Yes |
| Automatic LL/LL128/Simple size selection; protocol-aware mode disabled | Unavailable | Yes |
| MTP (`-t 8`) local AllReduce and cross AllGather | Yes | Yes |
| NVLS AllReduce | Must fail fast | Must fail fast |

## Gaps

- NVLS is unsupported by the UB backend; the frontend fails fast.
- Tree, CollNet Direct/Chain and NVLS-tree are not selectable by the current mock selector.
- Broadcast has no workload issue path in this frontend.
- PAT requires one GPU per server, so it does not exercise UB scale-up topologies.
- PXN is topology-dependent: it is only meaningful on rail-optimized fabrics without a spine.
- PAT and PXN cannot both trigger (PAT requires one local rank).
- v2.20 has no algorithm/protocol selection columns; flow labels are weaker evidence.
- The harness generates one switch per server so the frontend's virtual NVSwitch IDs map to real nodes. Topologies with fewer switches than servers can corrupt memory in the current UB frontend.
- Completion does not validate reduced tensor values, schedule dependency correctness, or timing accuracy.
