# MCCL schedule visualisation summary

| scenario | completion (ms) | AlgBW (GB/s) | cut (Gbps) | cut util | speedup |
|---|---:|---:|---:|---:|---:|
| NCCL_AllGather_16gpus_2nodes | 5.772 | 372.0 | 3139 | 96% | - |

Bottleneck cut (fixed by topology, failure and collective, so both schedules share it): GPU3 <-> NVSwitch16 ; GPU3 <-> ToR21.
Cut capacity: 3280 Gbps.
