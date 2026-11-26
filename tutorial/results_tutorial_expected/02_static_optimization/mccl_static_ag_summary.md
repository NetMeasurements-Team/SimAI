# MCCL schedule visualisation summary

| scenario | completion (ms) | AlgBW (GB/s) | cut (Gbps) | cut util | speedup |
|---|---:|---:|---:|---:|---:|
| 1x2x8_MCCL_AG | 5.062 | 424.2 | 3182 | 97% | - |

Bottleneck cut (fixed by topology, failure and collective, so both schedules share it): GPU3 <-> NVSwitch16 ; GPU3 <-> ToR21.
Cut capacity: 3280 Gbps.
