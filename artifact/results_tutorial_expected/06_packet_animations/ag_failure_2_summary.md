# MCCL schedule visualisation summary

| scenario | completion (ms) | AlgBW (GB/s) | cut (Gbps) | cut util | speedup |
|---|---:|---:|---:|---:|---:|
| No reopt (healthy-fabric schedule) | 39.145 | 54.9 | 411 | 14% | 1.00x |
| Reopt (failure-aware schedule) | 5.525 | 388.7 | 2915 | 99% | 7.08x |

Bottleneck cut (fixed by topology, failure and collective, so both schedules share it): GPU0 <-> NVSwitch16 ; GPU0 <-> ToR18.
Cut capacity: 2930 Gbps.

Speedup of **Reopt (failure-aware schedule)** over **No reopt (healthy-fabric schedule)**: **7.08x** (39.145 ms -> 5.525 ms).
Cut bandwidth: 411 -> 2915 Gbps (14% -> 99% of the cut).
