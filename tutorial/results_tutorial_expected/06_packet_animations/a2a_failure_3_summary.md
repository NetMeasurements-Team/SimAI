# MCCL schedule visualisation summary

| scenario | completion (ms) | AlgBW (GB/s) | cut (Gbps) | cut util | speedup |
|---|---:|---:|---:|---:|---:|
| No reopt (healthy-fabric schedule) | 456.513 | 4.7 | 68 | 45% | 1.00x |
| Reopt (failure-aware schedule) | 109.318 | 19.6 | 147 | 98% | 4.18x |

Bottleneck cut (fixed by topology, failure and collective, so both schedules share it): GPU0 <-> ToR18 ; GPU0 <-> NVSwitch16.
Cut capacity: 150 Gbps.

Speedup of **Reopt (failure-aware schedule)** over **No reopt (healthy-fabric schedule)**: **4.18x** (456.513 ms -> 109.318 ms).
Cut bandwidth: 68 -> 147 Gbps (45% -> 98% of the cut).
