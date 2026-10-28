# Collective Scheduling and Stalls

How each rank chooses which collective to run next, and how a simulation that cannot finish is reported.

## Sys file options

| Option | Values | Default | Description |
|---|---|---|---|
| `scheduling-policy:` | `LIFO`, `FIFO` | — (required) | Order in which a rank starts its waiting collectives: `LIFO` = most recently issued first, `FIFO` = issue order. |
| `active-chunks-per-dimension:` | integer | `1` | Collectives (streams) a rank runs at the same time per queue. |

## Why several collectives wait in the queue

Forward-pass and input-gradient collectives block the workload: the next layer waits for them. Weight-gradient
collectives do not: backpropagation issues a layer's weight-gradient collective and moves on to the previous layer,
and the collective only has to finish before the layer's next forward pass (like gradient buckets overlapping with
backpropagation). With one active collective per queue, several weight-gradient collectives can therefore be waiting
when the running one finishes, and the scheduling policy decides which one starts. With `LIFO`, a blocking collective
issued while gradients are queued (the most recent one) also goes before them.

## LIFO can stall

Each rank makes that choice on its own, when its running collective finishes. Ranks of a group usually finish at the
same time, but not always: with link failures some ranks are slower. A slow rank may then see a collective that was
issued while it was busy and, with `LIFO`, start it first, while the faster ranks of the same group already started
the older one. Every rank then waits for the other half of its group and the simulation stalls (reported as below).
NCCL forbids this: the collectives of a communicator must run in the same order on every rank.

`FIFO` cannot stall this way: every rank starts the collectives in issue order, which is the same on every rank (what
NCCL does for the collectives of a communicator). `LIFO` and `FIFO` only differ when a rank finds several collectives
waiting, i.e. weight-gradient collectives queued up while a newer one is issued: on healthy clusters (all ranks in
step) they give the same results.

## Scheduling trace

`AS_SCHEDULING_TRACE=<file.csv>` writes one line per stream event on every rank:

```
time_ns,rank,event,stream,phase,group_type,group,group_seq,chunk,layer_num,layer,pass,op,bytes
```

`event` is `issue` (the workload issued the collective), `start` (the scheduler started the stream) or `finish`.
A collective is identified on every rank by `(group, group_seq)`, so two runs (e.g. `FIFO` and `LIFO`) can be compared
collective by collective: per-rank start orders show where the policies chose differently, and the issue times show
which collectives were waiting at each choice.

## Retransmission timeout (`SimAI.conf`)

| Key | Default | Description |
|---|---|---|
| `RTO` | `0` | `1` arms a go-back-N retransmission timer on the QPs that cross the network (NIC and switches). QPs over NVLink (same server, through the NVSwitch) never use it: NVLink is lossless and the NVSwitch is given enough headroom never to drop. |
| `RTO_TIMEOUT` | `4294967.296` | Timeout in µs. The default is 4.096 µs × 2^20 (about 4.3 s): the timeout of a RoCE NIC with NCCL's default `NCCL_IB_TIMEOUT=20`. `0` restores the original ns-3 value, 30 times the base RTT of the QP (a few µs). |

With `RTO 0` a lost packet is recovered only by the receiver's NACK when a later packet arrives. The original
ns-3 timeout (`RTO_TIMEOUT 0`) is shorter than the queueing delay of a congested lossless link (PFC, a slow NIC or a
degraded NVLink): the timer then fires on packets that are only queued, and each firing re-sends the whole window.
The default timeout, like a real RoCE NIC (4.096 µs × 2^`NCCL_IB_TIMEOUT`), never fires on queueing delay.

## Stall report

If the simulation ends without every message finishing (ranks waiting for each other, a receive that no send
matches, ...), `SimAI_simulator` prints what is still pending and exits with status `2` instead of `0`.

A deadlock is detected as soon as no message is left in the network and no system-layer event (compute, workload
step) is scheduled while the simulation is not finished: nothing can make progress any more. The simulator then
prints `Deadlock detected at time ... ns` and stops. Without this, the run would never end when a QP had received
congestion notifications (CNPs): QPs are only closed at the end of the simulation, and their DCQCN timers keep the
event queue busy forever.

```
ERROR: simulation stalled: it ended at time 175879263 ns without every message finishing (...)
  pending sends (sentHash): 0, expected receives (expeRecvHash): 692564, arrived without receive (recvHash): 1024
    recv expected tag 8, 7 -> 0, 524288 B left
    recv orphan   tag 655368, 7 -> 0, 524288 B arrived
```

`recv expected` lists receives that are still waiting; `recv orphan` lists data that arrived but that no receive
was posted for. A receive expected with one tag while data with another tag arrived on the same link usually means
that the two ranks are running different collectives. The benchmark scripts record status `2` as `STALLED` and do
not retry it (it is deterministic).
