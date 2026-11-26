# MCCL schedule visualisation summary

| scenario | completion (ms) | AlgBW (GB/s) | cut (Gbps) | cut util | speedup |
|---|---:|---:|---:|---:|---:|
| No reopt (healthy-fabric schedule) | 287.346 | 7.5 | 56 | 37% | 1.00x |
| Reopt (failure-aware schedule) | 108.352 | 19.8 | 149 | 99% | 2.65x |

Bottleneck cut (fixed by topology, failure and collective, so both schedules share it): GPU0 <-> ToR18 ; GPU0 <-> NVSwitch16.
Cut capacity: 150 Gbps.

Speedup of **Reopt (failure-aware schedule)** over **No reopt (healthy-fabric schedule)**: **2.65x** (287.346 ms -> 108.352 ms).
Cut bandwidth: 56 -> 149 Gbps (37% -> 99% of the cut).
