# MCCL schedule visualisation summary

| scenario | completion (ms) | AlgBW (GB/s) | cut (Gbps) | cut util | speedup |
|---|---:|---:|---:|---:|---:|
| MCCL_1x2x8 | 21.733 | 98.8 | 3162 | 99% | - |

Bottleneck cut (fixed by topology, failure and collective, so both schedules share it): GPU0 <-> ToR18 ; GPU1 <-> ToR19 ; GPU2 <-> ToR20 ; GPU3 <-> ToR21 ; GPU4 <-> ToR22 ; GPU5 <-> ToR23 ; GPU6 <-> ToR24 ; GPU7 <-> ToR25.
Cut capacity: 3200 Gbps.
