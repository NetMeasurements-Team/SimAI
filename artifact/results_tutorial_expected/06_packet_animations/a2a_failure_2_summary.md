# MCCL schedule visualisation summary

| scenario | completion (ms) | AlgBW (GB/s) | cut (Gbps) | cut util | speedup |
|---|---:|---:|---:|---:|---:|
| No reopt (healthy-fabric schedule) | 173.861 | 12.4 | 395 | 14% | 1.00x |
| Reopt (failure-aware schedule) | 24.445 | 87.8 | 2811 | 99% | 7.11x |

Bottleneck cut (fixed by topology, failure and collective, so both schedules share it): GPU1 <-> ToR19 ; GPU2 <-> ToR20 ; GPU3 <-> ToR21 ; GPU4 <-> ToR22 ; GPU5 <-> ToR23 ; GPU6 <-> ToR24 ; GPU7 <-> ToR25 ; GPU0 <-> ToR18.
Cut capacity: 2850 Gbps.

Speedup of **Reopt (failure-aware schedule)** over **No reopt (healthy-fabric schedule)**: **7.11x** (173.861 ms -> 24.445 ms).
Cut bandwidth: 395 -> 2811 Gbps (14% -> 99% of the cut).
