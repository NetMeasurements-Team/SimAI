# MCCL schedule visualisation summary

| scenario | completion (ms) | AlgBW (GB/s) | cut (Gbps) | cut util | speedup |
|---|---:|---:|---:|---:|---:|
| No reopt (healthy-fabric schedule) | 17.747 | 121.0 | 908 | 76% | 1.00x |
| Reopt (failure-aware schedule) | 13.720 | 156.5 | 1174 | 98% | 1.29x |

Bottleneck cut (fixed by topology, failure and collective, so both schedules share it): GPU0 <-> NVSwitch16 ; GPU0 <-> ToR18.
Cut capacity: 1200 Gbps.

Speedup of **Reopt (failure-aware schedule)** over **No reopt (healthy-fabric schedule)**: **1.29x** (17.747 ms -> 13.720 ms).
Cut bandwidth: 908 -> 1174 Gbps (76% -> 98% of the cut).
