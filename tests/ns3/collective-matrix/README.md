# Collective completion and selection matrix

See [Tests: Collective completion and selection matrix](../../../docs/Tests.md#collective-completion-and-selection-matrix) for the overview,
expected results and run commands.

## Exact checks and results

The validator checks collective/workload completion, matching stream counts,
nonempty generated flows, and completed messages matching the generated
source/destination/size multiset. It checks selection and PXN forwarding.
Both versions supply algorithm/protocol IDs in their detailed-flow CSVs. Missing CSVs,
unexpected fallbacks, no-ops and incomplete messages fail rather than skip.

Each case retains its detailed-flow CSV and binary/input hashes.
`summary.json` is updated after each case with results, failure reasons and
elapsed wall time. The runner clears inherited `AS_*`, `SIMAI_*` and `NCCL_*`
settings before applying recorded overrides; send latency remains enabled.

## Coverage

| Cases                                                                     | v2.20          | v2.30                                |
|---------------------------------------------------------------------------|----------------|--------------------------------------|
| AllReduce, AllGather, ReduceScatter, AllToAll; 4 ranks, same/cross server | LL, LL128, Simple, explicitly forced | LL, LL128, Simple, explicitly forced |
| All four collectives with 3 ranks on 3 servers                            | Yes            | Yes                                  |
| AllToAll and ReduceScatter with PXN, 2 servers x 2 ranks                  | All three protocols | All three protocols                  |
| NVLS AllReduce, H100, 8 ranks and one NVSwitch, 4 MiB                     | Simple         | Simple                               |
| PAT AllGather/ReduceScatter, 2/3/4 servers, one rank each                 | Unavailable    | Simple                               |
| PAT automatic selection inside/outside size window; ineligible topology   | Unavailable    | Yes                                  |
| PAT keeps Simple even when LL/LL128 is forced                             | Unavailable    | Yes                                  |
| Automatic LL/LL128/Simple size selection; protocol-aware mode disabled    | Yes            | Yes                                  |

There are 39 v2.20 cases and 55 v2.30 cases. Sizes are modest (mostly 768 KiB),
except explicit NVLS/protocol-threshold cases. Most tests use a minimal Ethernet
star with an explicit GPUs-per-server value, no NVSwitch, and no congestion
requirement. Thus "same server" tests the selector's locality handling, not a
calibrated NVLink topology. The NVLS case also adds GPU-to-NVSwitch links.
All cases issue exactly one forward-pass collective in a TP group containing all
ranks. DP/EP grouping, pipelining, and concurrency are outside this matrix.

Tree, CollNet, NVLS-tree and Broadcast are not covered through the
selector/workload frontend.
