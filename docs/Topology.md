# Topology generator script

A Python [script](../astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py) for generating network topology files compatible with **SimAI**, supporting several real-world and reference datacenter network architectures used in large-scale GPU training clusters.

## Overview

The script models multi-tier GPU cluster topologies composed of servers, NVSwitches, Access Switches (ASW), and Pod Switches (PSW). It outputs a topology file encoding nodes, switch assignments, and links (with bandwidth, latency, and error rate) that can be consumed directly by SimAI.

## Supported Topologies

| Topology     | Description                                | Rail-Optimized | Dual ToR          | Dual Plane        |
|--------------|--------------------------------------------|----------------|-------------------|-------------------|
| `AlibabaHPN` | Alibaba HPN-style 3-tier GPU fabric        | ✅ (forced)     | ✅ (default)       | Optional (`--dp`) |
| `Spectrum-X` | NVIDIA Spectrum-X rail-optimized topology  | ✅ (forced)     | ❌                 | ❌                 |
| `DCN+`       | Meta DCN+ style non-rail-optimized fabric  | ❌ (forced)     | Optional (`--dt`) | ❌                 |
| `SimpleTree` | Generic 3-tier tree (NVSwitch → ASW → CSW) | ❌              | ❌                 | ❌                 |

## Usage

```shell
python gen_Topo_Template.py [-h] [-topo TOPOLOGY] [--ro] [--dt] [--dp]
                            [-g GPU] [-er ERROR_RATE] [-gps GPU_PER_SERVER]
                            [-gt GPU_TYPE] [-nsps NV_SWITCH_PER_SERVER]
                            [-nvbw NVLINK_BW] [-nl NV_LATENCY] [-l LATENCY]
                            [-bw BANDWIDTH] [-asn ASW_SWITCH_NUM]
                            [-npa NICS_PER_ASWITCH] [-psn PSW_SWITCH_NUM]
                            [-apbw AP_BANDWIDTH] [-app ASW_PER_PSW]
```

## Arguments

### Topology Structure

| Argument     | Short   | Default      | Description                                                                     |
|--------------|---------|--------------|---------------------------------------------------------------------------------|
| `--topology` | `-topo` | `SimpleTree` | Topology template: `AlibabaHPN`, `Spectrum-X`, `DCN+`, or `SimpleTree`          |
| `--ro`       |         | `False`      | Enable rail-optimized structure                                                 |
| `--dt`       |         | `False`      | Enable Dual ToR, i.e., two NICs per GPU (DCN+ only)                             |
| `--dp`       |         | `False`      | Enable Dual Plane, i.e., two separated groups of Pod switches (AlibabaHPN only) |

### Intra-Host (NVLink domain)

| Argument                 | Short   | Default      | Description                              |
|--------------------------|---------|--------------|------------------------------------------|
| `--gpu`                  | `-g`    | `32`         | Total number of GPUs                     |
| `--gpu_per_server`       | `-gps`  | `8`          | GPUs per server node                     |
| `--gpu_type`             | `-gt`   | `H100`       | GPU type label (used in output filename) |
| `--nv_switch_per_server` | `-nsps` | `1`          | NVSwitches per server                    |
| `--nvlink_bw`            | `-nvbw` | `2880Gbps`   | NVLink bandwidth (GPU ↔ NVSwitch)        |
| `--nv_latency`           | `-nl`   | `0.000025ms` | NVSwitch link latency                    |
| `--latency`              | `-l`    | `0.0005ms`   | NIC link latency (GPU ↔ ASW)             |
| `--error_rate`           | `-er`   | `0`          | Link error rate                          |

### Intra-Segment (ASW domain)

| Argument             | Short  | Default   | Description               |
|----------------------|--------|-----------|---------------------------|
| `--bandwidth`        | `-bw`  | `400Gbps` | NIC-to-ASW bandwidth      |
| `--asw_switch_num`   | `-asn` | `8`       | Number of Access Switches |
| `--nics_per_aswitch` | `-npa` | `64`      | GPUs per Access Switch    |

### Intra-Pod (PSW domain)

| Argument           | Short   | Default   | Description            |
|--------------------|---------|-----------|------------------------|
| `--psw_switch_num` | `-psn`  | `64`      | Number of Pod Switches |
| `--ap_bandwidth`   | `-apbw` | `400Gbps` | ASW-to-PSW bandwidth   |
| `--asw_per_psw`    | `-app`  | `64`      | Number of ASWs per PSW |

## Topology-Specific Defaults

When a named topology is selected, the script automatically overrides certain defaults to reflect the reference architecture:

| Topology     | GPU   | Bandwidth               | ASW | NPA                | PSW | ASW/PSW             |
|--------------|-------|-------------------------|-----|--------------------|-----|---------------------|
| `AlibabaHPN` | 15360 | 200Gbps                 | 240 | 128                | 120 | 240 (120 w/ `--dp`) |
| `Spectrum-X` | 4096  | 400Gbps                 | 8   | 64                 | 64  | 64                  |
| `DCN+`       | 512   | 400Gbps (200 w/ `--dt`) | 8   | 64 (128 w/ `--dt`) | 8   | 8                   |
| `SimpleTree` | 16    | 400Gbps                 | 2   | 8                  | 0   | 0                   |

## Topology Variants

The rail-optimized and ToR flags select between internal wiring strategies:

- **Rail-Optimized + Single ToR**: Each GPU connects to the ASW indexed by its rail position within the server — all GPU 0s share one ASW, all GPU 1s share another, etc.
- **Rail-Optimized + Dual ToR**: As above, but each GPU connects to two mirrored ASW planes for redundancy.
- **Rail-Optimized + Dual ToR + Dual Plane** (`AlibabaHPN` only): Two independent PSW planes, each connected to one ASW set.
- **Non-Rail-Optimized + Single ToR**: All GPUs in a server connect to the same ASW.
- **Non-Rail-Optimized + Dual ToR** (`DCN+`): Each server connects to two ASWs for redundancy without rail alignment.
- **SimpleTree**: Flat 3-tier tree; uses a single Core Switch when there are more than 2 ASWs, or a direct link between 2 ASWs otherwise.

## Examples

### AlibabaHPN 4×4×8

```shell
python gen_Topo_Template.py -topo AlibabaHPN -gt H100 -g 128 -gps 8 -asn 64 -npa 4 -psn 8 -app 32 --dp --ro --dt
asw_switch_num: 64
psw_switch_num: 8
Creating Topology of totally 4 segment(s), totally 1 pod(s).
AlibabaHPN_128g_8gps_DualToR_DualPlane_200Gbps_H100
```

### AlibabaHPN 2×8×8

```shell
python gen_Topo_Template.py -topo AlibabaHPN -gt H100 -g 128 -gps 8 -asn 32 -npa 8 -psn 8 -app 16 --dp --ro --dt
asw_switch_num: 32
psw_switch_num: 8
Creating Topology of totally 2 segment(s), totally 1 pod(s).
AlibabaHPN_128g_8gps_DualToR_DualPlane_200Gbps_H100
```

### DCN+ 4×4×8

```shell
python gen_Topo_Template.py -topo DCN+ -gt H100 -g 128 -gps 8 -asn 8 -psn 8 -npa 32 -app 8 --dt
asw_switch_num: 8
psw_switch_num: 8
Creating Topology of totally 4 segment(s), totally 1 pod(s).
DCN+DualToR_128g_8gps_200Gbps_H100
```

### DCN+ 2×8×8

```shell
python gen_Topo_Template.py -topo DCN+ -gt H100 -g 128 -gps 8 -asn 8 -psn 8 -npa 32 -app 8 --dt
asw_switch_num: 8
psw_switch_num: 8
Creating Topology of totally 4 segment(s), totally 1 pod(s).
DCN+DualToR_128g_8gps_200Gbps_H100
```

### Spectrum-X (default scale)

```shell
python gen_Topo_Template.py -topo Spectrum-X -gt H100 -asn 64 
asw_switch_num: 64
psw_switch_num: 64
Creating Topology of totally 8 segment(s), totally 1 pod(s).
Spectrum-X_4096g_8gps_400Gbps_H100
```

### SimpleTree (small testbed)

```shell
python gen_Topo_Template.py -topo SimpleTree -g 16 -gps 8 -asn 2 -npa 8
Creating a Simple Tree Topology with 2 servers, 2 access switches, and 0 core switches.
Simple_3_Tier_16g_8gps_400Gbps_H100

```

## Constraint Notes

- Total GPU count must be divisible by `--gpu_per_server`.
- If the provided `--asw_switch_num` is inconsistent with the GPU count and NPA, the script autocorrects it and emits a warning.
- `--dp` (Dual Plane) is only valid with `AlibabaHPN` and rail-optimized mode; combining it with non-rail-optimized mode raises an error.
- `--dt` (Dual ToR) without `--ro` is the DCN+ mode; with `--ro` it activates the HPN dual-ToR fabric.
- `SimpleTree` with more than 2 ASWs automatically adds a single Core Switch; with 1–2 ASWs the ASWs are directly interconnected.
