# MCCL schedule visualisation summary

| scenario | completion (ms) | AlgBW (GB/s) | cut (Gbps) | cut util | speedup |
|---|---:|---:|---:|---:|---:|
| No reopt (healthy-fabric schedule) | 28.240 | 76.0 | 2433 | 76% | 1.00x |
| Reopt (failure-aware schedule) | 21.720 | 98.9 | 3164 | 99% | 1.30x |

Bottleneck cut (fixed by topology, failure and collective, so both schedules share it): GPU8 <-> ToR18 ; GPU9 <-> ToR19 ; GPU10 <-> ToR20 ; GPU11 <-> ToR21 ; GPU12 <-> ToR22 ; GPU13 <-> ToR23 ; GPU14 <-> ToR24 ; GPU15 <-> ToR25.
Cut capacity: 3200 Gbps.

Speedup of **Reopt (failure-aware schedule)** over **No reopt (healthy-fabric schedule)**: **1.30x** (28.240 ms -> 21.720 ms).
Cut bandwidth: 2433 -> 3164 Gbps (76% -> 99% of the cut).
