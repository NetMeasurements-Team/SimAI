This branch introduces a feature that allows users to define **custom collective schedules** and execute them on top of the SimAI simulator. 

Currently, this feature supports only **All-Gather** operations.  The custom collective schedule need to be stored in msccl-algorithms/ folder () The first compatible schedule is used; both [text](msccl-algorithms/1x2x8-allgather.xml), and [text](msccl-algorithms/NCCL-Like-2x8.xml) should work. 

---
## Quick Start (Example)

To run a simulation using a custom All-Gather schedule (256MB) on a 1x2x8 topology, execute the following command from the root directory:

```bash
AS_PXN_ENABLE=1 ./bin/SimAI_simulator -t 10  -w example/ag-268435456_bytes -n example/1x2x8 -c astra-sim-alibabacloud/inputs/config/SimAI.conf
```