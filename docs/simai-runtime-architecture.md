# SimAIRuntime & NS3_MTP Co-Simulation Architecture

## 1. Overview

`SimAIRuntime` (`astra-sim-alibabacloud/astra-sim/runtime/SimAIRuntime.{h,cc}`) is a
resident, in-memory coordinator that lets Python frontends (Vidur, Frontier, via
`pysimai`) drive Astra-Sim collectives over the ns-3 packet-level network.

Unlike the legacy CLI (`SimAI_simulator`), which reads a static workload, runs
`Simulator::Run()` once, and exits, `SimAIRuntime` keeps the topology, switch
buffers, RDMA QPs, and `AstraSim::Sys` instances alive across a single monotonic
timeline. It offers two entry points:

- `predict_collective(...)` — synchronous; blocks until the collective completes
  and returns its simulated duration in microseconds.
- `post_collective(..., callback)` — asynchronous; returns immediately and invokes
  the callback on completion.

## 2. Threading Model

```mermaid
flowchart LR
    subgraph Python["Caller Threads (Python / Vidur / Frontier)"]
        P1["predict_collective()"]
        P2["post_collective()"]
    end
    subgraph Runtime["SimAIRuntime"]
        Q["pending_collectives_ + busy_ranks_ + active_collectives_\n[guarded by MtpInterface::CriticalSection]"]
        SimThread["sim_thread_\nSimulator::Run() driver"]
        CbThread["cb_thread_\ncallback dispatcher"]
    end
    subgraph MTP["NS3_MTP"]
        LP0["Public LP: g_systems[0]"]
        LPW["Worker LPs: g_systems[1..K]"]
    end
    P1 & P2 -->|submit_collective| Q
    P1 & P2 -->|ScheduleGlobal| LP0
    P1 & P2 -->|sim_cv_.notify_one| SimThread
    SimThread -->|Simulator::Run()| LP0
    LP0 <-->|barrier stages| LPW
    LPW -->|last rank done| CbThread
    CbThread -->|callback(duration_us)| Python
```

- **Caller threads** never touch ns-3 event heaps or `Sys` state directly. They
  enqueue a `PendingCollective` and post a wakeup to the public LP via
  `MtpInterface::ScheduleGlobal()`.
- **`sim_thread_`** sleeps on `sim_cv_` while idle and drives `Simulator::Run()`
  when collectives arrive.
- **`cb_thread_`** runs completion callbacks off the MTP thread pool, so a slow
  Python callback never blocks a worker at the barrier.

`NS3_MTP` partitions the topology into worker LPs plus one public LP (`g_systems[0]`)
for context-free global events. Each round runs three barrier-synchronized stages:
parallel worker events, serial public-LP events, then parallel mailbox drain.

## 3. Collective Lifecycle

1. **Submit** — `submit_collective()` pushes the request onto `pending_collectives_`
   under `CriticalSection`, posts `drain_and_start_ready_collectives` to the public
   LP, and wakes `sim_thread_`.
2. **Dispatch** — on the public LP, `drain_and_start_ready_collectives()` starts every
   request whose ranks are all free, marks those ranks busy, and calls
   `start_collective_on_sim_thread()`. Requests on overlapping ranks stay queued so
   per-rank `NcclFlowModel` state is not corrupted.
3. **Simulate** — worker LPs run RDMA transmission, switch queueing, PFC/DCQCN, and
   packet delivery. Collectives on disjoint rank sets share the fabric concurrently.
4. **Complete** — when the last rank finishes, `on_collective_finished()` captures
   `duration_us`, enqueues the callback, frees the ranks, and either chains the next
   waiting collective or schedules a stop (optionally delayed by the compute gap).

## 4. Sync vs. Async API

| | `predict_collective` | `post_collective` |
| --- | --- | --- |
| Blocking | Blocks on `std::future::get()` (GIL released) | Returns immediately |
| Delivery | `std::promise` set on completion | User callback on `cb_thread_` |
| Ranks | Empty list = all GPUs | Empty list = all GPUs |
| Ordering | FIFO; starts when ranks free | FIFO; starts at prior completion |

Both accept `compute_gap_us`, stored as `compute_gap_ns_`.

## 5. Compute Gap

When the simulator goes idle, `schedule_gap_stop()` decides whether to stop now or
keep running for `compute_gap_ns_`:

- `> 0` — keep the simulator alive through the gap so DCQCN recovery timers fire and
  the network converges to steady state before the next collective.
- `0` — `Simulator::Stop()` immediately (raw warm state).

The gap never affects measured latency: `duration_us` is captured at completion,
before the gap runs. In Vidur the gap is derived from per-layer compute predictions
by default (`simai_compute_gap_from_vidur`); `simai_compute_gap_us` is the fixed
fallback.

