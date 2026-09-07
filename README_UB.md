# SimAI Unified Bus (UB) Network Simulation Backend

This document describes the implementation of the **Unified Bus (UB)** network simulation frontend in SimAI, the porting of the backend from `ns-3-ub`, the routing modifications, and provides a quick test example.

---

## Quick Start

Follow these steps to compile and run a quick test simulation with the UB backend.

### Step 1: Compile the Simulator with the UB Backend
Run the following build command from the root `SimAI` directory:
```bash
bash scripts/build.sh -lc ns3-ub -d optimized
```
This builds the `ns-3-ub` packet simulator and creates the `bin/SimAI_simulator` executable symlink.

### Step 2: Generate the UB Topology
Generate a small 16-GPU Rail Optimized topology for UB:
```bash
python3 astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo AlibabaHPN-st -g 16 -gt H100 -bw 400Gbps -asn 8 -psn 8 --ub
```
This outputs a file named `Rail_Opti_SingleToR_16g_8gps_400Gbps_H100_UB` in your current directory.

### Step 3: Run the Simulation
Run the simulator using the generated UB topology and a sample AllReduce workload with PXN enabled:
```bash
 AS_PXN_ENABLE=1 \
  ./bin/SimAI_simulator \
  -t 1 \
  -w astra-sim-alibabacloud/inputs/workloads/single_collective/16_gpus/ag-268435456_bytes \
  -s astra-sim-alibabacloud/inputs/system/nccl_flow_model.txt \
  -n Rail_Opti_SingleToR_16g_8gps_400Gbps_H100_UB \
  -c astra-sim-alibabacloud/inputs/config/SimAI.conf
```

### Step 4: Verify the output
When the simulation finishes, you should see the statistics output printed to the terminal:
```
All data sent from node 0 is 283115520
All data received by node 0 is 283115520
...
[UB] All messages finished. Stopping simulation.
```

---

## 1. Overview and Core Philosophy

In traditional RDMA/NVIDIA setups, the network stack is split:
* **Scale-up** (intra-server) communication uses proprietary **NVLink** protocols routed through **NVSwitches**.
* **Scale-out** (inter-server) communication uses **RoCEv2/InfiniBand** through NICs and traditional switches.

**Unified Bus (UB)** is a converged interconnect protocol designed to unify scale-up (intra-server HCCS) and scale-out (inter-server) fabrics:
* Both local and remote transfers utilize the **same UB protocol stack** (under URMA or LDST options).
* Intra-server connections are represented directly in the topology as point-to-point high-speed UB links with massive bandwidth (e.g., `2880Gbps` or `800Gbps`) and low latency (e.g., `10ns` or `25ns`).
* Inter-server connections go out of the NPU's network port to Leaf/Spine switches at lower bandwidth (e.g., `400Gbps`).

---

## 2. Architecture & Implementation Details

### 2.1 Virtual NVSwitch Mapping
Astra-Sim's mock NCCL layer (`MockNcclGroup`) expects to see NVSwitch IDs to assign ring/tree group memberships. Since physical NVSwitches do not exist in the UB specification, the UB frontend in [AstraSimNetwork.cc](astra-sim-alibabacloud/astra-sim/network_frontend/ns3-ub/AstraSimNetwork.cc) virtualizes NVSwitches. It assigns each GPU server a dummy switch ID (`gpu_num + server_idx`) so Astra-Sim can build collective groups without crashing. In the underlying `ns-3-ub` simulation, the traffic is routed directly over point-to-point links.

### 2.2 Transit Routing and Local Node Protection (HCCS Isolation)
In the `ns3-ub` frontend, after UB setup, routes are automatically installed using a Breadth-First Search (BFS) algorithm. 

In a rail-optimized network without spine switches (isolated planes), cross-rail traffic must be routed at the application layer via **PXN (Proxy-based Cross-Node)**. To prevent the simulator's network routing calculation from routing non-local traffic over HCCS links by transparently using NPUs as transit routers (which is physically unrealistic in production clusters), the BFS router in [common.h](astra-sim-alibabacloud/astra-sim/network_frontend/ns3-ub/common.h) has been patched. 

If PXN is disabled (`AS_PXN_ENABLE=0`), cross-rail transfers will now realistically fail to route. If PXN is enabled (`AS_PXN_ENABLE=1`), the application layer splits the transfer into two hops (`Src NPU -> Sibling NPU` locally, and `Sibling NPU -> Dst NPU` remotely), both of which are valid under the new rule.

### 2.3 Communication Primitives

For now, UB applications communicate using these task parameters: 
```
task.opType       = "URMA_WRITE";
task.priority     = 1;
task.delay        = "1ns";
```
Due to UB limitations in `TrafficRecord::dataSize` (which only accepts a 32-bit integer), larger transmissions are naturally split across different Work Queue Entries (i.e., using multiple QP - Jetty in UB terminology). 

Finally, because the receiving application in the underlying `ns-3-ub` backend lacks a direct callback to notify the frontend when a message has fully arrived, a workaround is used to trigger receiver completion. Specifically, the receiver is triggered with a "receive finished" event once the sender receives the last ACK for the transmission signaled a the sender's WQE completion trace, preserving the original design of the SimAI (then corrected in our fork).


---

## 3. How to Generate UB Topologies

The topology generator [gen_Topo_Template.py](astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py) has been updated with a `--ub` flag. When enabled:
1. It writes `0` to the NVSwitch column in the file header (since UB has no physical NVSwitches).
2. It lists all switches (including local server-level switches) under the general SWITCH count column.
3. It appends `_UB` to the generated output filename.

To generate a 128-GPU Spectrum-X UB topology:
```bash
python3 astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo Spectrum-X -g 128 -gt H100 -bw 400Gbps --ub
```
This produces a topology file named `Spectrum-X_128g_8gps_400Gbps_H100_UB`.

---
