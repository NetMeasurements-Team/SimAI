# Collective/PFC

See [Tests: Collective/PFC](../../../docs/Tests.md#collectivepfc) for the overview,
expected results and run commands.

## Workload and topology

```text
rank 0 --100G--+                 +--100G-- rank 2
              S0 --- trunk --- S1
rank 1 --100G--+                 +--100G-- rank 3
```

Each rank represents a single-GPU server; there is no NVSwitch. Links have
1 us propagation and switches have 1.5 us forwarding delay. One layer issues
the two collectives with small compute delays and no input-gradient collective.
TP=2 forms All-to-All groups {0,1} and {2,3}; DP=2 forms All-Reduce groups
{0,2} and {1,3}. A100 selects ring All-Reduce. The runner fixes one simulation
thread and disables NVLS, PXN and send latency.

Sizes are inputs to the collective implementation, not per-link traffic volumes.
Because size, trunk rate and buffer capacity vary together, the two cases are
not a controlled timing comparison.

## Congestion settings

In the congested case, concurrent All-Reduce groups feed each 25 Gb/s trunk
direction through two 100 Gb/s host links. Sustained arrivals exceed service,
and the 1 MiB buffer gives thresholds low enough to trigger PFC. Adjacent
senders pause until the bottleneck drains sufficiently to resume.

`ENABLE_QCN=0` disables ECN marking and `HAS_WIN=0` removes the sender window
limit, preventing these mechanisms from suppressing the intended congestion.
`CC_MODE=1` configures DCQCN, with no ECN feedback. Payloads are 1,000 bytes,
links are error-free, and headroom uses the frontend configuration in `common.h`.
If the congested workload completes without PFC, inspect the generated topology,
window/ECN settings and traffic traces.

## Exact checks and limitations

The validator requires:

- Successful exit within the timeout and both workload/all-messages completion markers.
- Both configured collectives issued, with exactly two collectives for the layer.
- Matching nonzero rank-0 injected/finished stream counts.
- Positive completed-message FCT records covering all four ranks and no MMU loss diagnostics.
- An empty baseline PFC trace; at least one XOFF and one XON in the congested case.
- A nonempty queue trace with valid samples.

Queue sampling uses `ENABLE_MONITOR 1` and a 1 us `QLEN_MON_INTERVAL`, starting
at `MON_START`. The simulator must support queue monitoring; `QLEN_MON_FILE`
alone does not enable it. The report includes completed-message and pause/resume
counts plus maximum sampled egress-port occupancy.

These traces do not verify every internal MMU invariant. Equal pause/resume
counts and drained queues are not required because application exit may leave
control events pending. The native backend test checks drainage and provides
a PFC-disabled negative control; RDMA recovery would make loss or timeout
ambiguous in this workload test.

Additional collective CSV reports use the simulator's compiled result path and
a unique run-name prefix.

## Extending the test

Edit `cases.json` for sizes, trunk rates, buffers and PFC expectations.
`workload.txt.in` expands `@BYTES@`; `topology.txt.in` expands `@TRUNK@`.
Shared network settings expand `@BUFFER_MB@` and `@ENABLE_MONITOR@`.
The validator assumes four ranks and this two-collective workload; changing
those assumptions requires updating its checks.
