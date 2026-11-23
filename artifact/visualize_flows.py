#!/usr/bin/env python3
"""Animate how an MSCCL/MCCL schedule moves packets over a SimAI topology.

The script is a tutorial: it draws the GPU/NVSwitch/ToR graph and moves one dot
per scheduled chunk along its physical path, with a trail and an arrowhead so the
direction is clear.  Link type is colour-coded (green intra-node, dark inter-node),
the degraded link is magenta dashed, congestion is an inner heat line (blue ->
amber -> orange), and the bottleneck cut is a red cross/star marker.  Red is used
only for the cut, so a congested link is never confused with a failure.  A
side-by-side mode compares a schedule synthesized for the healthy fabric
("no reopt") with the schedule re-synthesized for the failure ("reopt"), using
the exact MCCL artifact files.

Examples
--------
Exact flow timings from the simulator:

    python3 artifact/visualize_flows.py --demo ag-failure_2 --out ag_f2.gif

One topology + one schedule:

    python3 artifact/visualize_flows.py \
        --topology astra-sim-alibabacloud/inputs/topo/single_collective/1x2x8-failure_2 \
        --schedule astra-sim-alibabacloud/inputs/msccl/mccl/failure_2/priority-9/1x2x8_MCCL_AG_Fail2.xml \
        --size 2147483648 \
        --workload astra-sim-alibabacloud/inputs/workloads/single_collective/16_gpus/ag-2147483648_bytes \
        --run-sim --out ag_f2_reopt.gif

The demo names are ag-failure_1, ag-failure_2, ag-failure_3, a2a-failure_1,
a2a-failure_2 and a2a-failure_3.  They use the 2 GB E3 workloads, the 1x2x8
topologies and the MCCL schedules shipped in this artifact.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import heapq
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "Pillow is required for the animation.\n"
        "Install it with: python3 -m pip install pillow\n"
        f"({exc})\n"
    )
    raise SystemExit(2)


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "astra-sim-alibabacloud"
INPUTS = BASE / "inputs"
DEFAULT_BINARY = ROOT / "bin" / "SimAI_simulator"
DEFAULT_CONFIG = INPUTS / "config" / "SimAI.conf"
EXPECTED_E3 = ROOT / "artifact" / "expected" / "E3-mccl.csv"

FAILURES = ("failure_1", "failure_2", "failure_3")
COLLECTIVES = {
    "ag": {
        "name": "AllGather",
        "workload": INPUTS / "workloads" / "single_collective" / "16_gpus" / "ag-2147483648_bytes",
        "baseline": INPUTS / "msccl" / "mccl" / "no_failure" / "priority-9" / "1x2x8_MCCL_AG.xml",
        "reopt": "1x2x8_MCCL_AG_{failure_short}.xml",
        "system": INPUTS / "system" / "mccl-no_failure.txt",
    },
    "a2a": {
        "name": "AllToAll",
        "workload": INPUTS / "workloads" / "single_collective" / "16_gpus" / "a2a-2147483648_bytes",
        "baseline": INPUTS / "msccl" / "mccl" / "no_failure" / "priority-9" / "MCCL_1x2x8.xml",
        "reopt": "MCCL_1x2x8_{failure_short}.xml",
        "system": INPUTS / "system" / "mccl-no_failure.txt",
    },
}

FLOW_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

# Distinct semantic colours: link type, failure, congestion and bottleneck cut
# must never share a colour, otherwise a congested link looks like a failure.
COLOR_INTRA = "#2e7d32"        # intra-node NVLink
COLOR_INTER = "#37474f"        # inter-node ToR
COLOR_FAIL = "#c2185b"         # degraded/failed link (magenta)
COLOR_BOTTLENECK = "#e53935"   # bottleneck cut marker (red, only here)
COLOR_TEXT = "#111111"
COLOR_MUTED = "#555555"


def parse_rate(text: str) -> float:
    """Return a link rate in Gbps from strings such as '2880Gbps'."""
    match = re.match(r"\s*([0-9.]+)\s*([GMK]?)bps\s*$", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse link rate: {text!r}")
    value = float(match.group(1))
    unit = match.group(2).upper()
    return value * {"": 1e-9, "K": 1e-6, "M": 1e-3, "G": 1.0}[unit]


def parse_delay_ms(text: str) -> float:
    """Return a link delay in milliseconds from strings such as '0.0005ms'."""
    match = re.match(r"\s*([0-9.]+)\s*ms\s*$", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse link delay: {text!r}")
    return float(match.group(1))


def link_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


@dataclass
class Link:
    u: int
    v: int
    bw_gbps: float
    delay_ms: float
    error_rate: float = 0.0

    @property
    def key(self) -> Tuple[int, int]:
        return link_key(self.u, self.v)

    @property
    def delay_ns(self) -> float:
        return self.delay_ms * 1e6


@dataclass
class Topology:
    path: Path
    node_num: int
    gpus_per_server: int
    nvswitch_ids: List[int]
    switch_ids: List[int]
    gpu_ids: List[int]
    links: Dict[Tuple[int, int], Link] = field(default_factory=dict)
    adj: Dict[int, Dict[int, Link]] = field(default_factory=dict)

    @classmethod
    def parse(cls, path: Path) -> "Topology":
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError(f"topology file is too short: {path}")
        first = lines[0].split()
        node_num = int(first[0])
        gpus_per_server = int(first[1])
        nvswitch_num = int(first[2])
        switch_num = int(first[3])
        link_num = int(first[4])
        second = [int(x) for x in lines[1].split()]
        if len(second) < nvswitch_num + switch_num:
            raise ValueError(f"topology {path} has a truncated switch list")
        nvswitch_ids = second[:nvswitch_num]
        switch_ids = second[nvswitch_num:nvswitch_num + switch_num]
        gpu_ids = [i for i in range(node_num) if i not in set(nvswitch_ids) | set(switch_ids)]
        topo = cls(path, node_num, gpus_per_server, nvswitch_ids, switch_ids, gpu_ids)
        for line in lines[2:2 + link_num]:
            parts = line.split()
            if len(parts) < 4:
                continue
            u, v = int(parts[0]), int(parts[1])
            rate = parse_rate(parts[2])
            delay = parse_delay_ms(parts[3])
            err = float(parts[4]) if len(parts) > 4 else 0.0
            link = Link(u, v, rate, delay, err)
            topo.links[link.key] = link
            topo.adj.setdefault(u, {})[v] = link
            topo.adj.setdefault(v, {})[u] = link
        return topo

    def route(self, src: int, dst: int) -> Optional[List[int]]:
        """Shortest-hop path, preferring the path with the largest bottleneck.

        SimAI's ns-3 frontend uses hop-count shortest paths.  Among equal-hop
        paths we prefer the one with the highest minimum link rate, which is
        what makes a failure-aware schedule visibly avoid the degraded link.
        """
        if src == dst:
            return [src]
        if src not in self.adj or dst not in self.adj:
            return None
        inf = float("inf")
        best: Dict[int, Tuple[int, float]] = {src: (0, -inf)}
        prev: Dict[int, int] = {}
        heap: List[Tuple[int, float, int]] = [(0, -inf, src)]
        while heap:
            hops, neg_bw, u = heapq.heappop(heap)
            if best.get(u) != (hops, neg_bw):
                continue
            if u == dst:
                break
            cur_bw = inf if neg_bw == -inf else -neg_bw
            for v, link in self.adj[u].items():
                new_bw = min(cur_bw, link.bw_gbps)
                cand = (hops + 1, -new_bw)
                if v not in best or cand < best[v]:
                    best[v] = cand
                    prev[v] = u
                    heapq.heappush(heap, (hops + 1, -new_bw, v))
        if dst != src and dst not in prev:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def path_links(self, path: Sequence[int]) -> List[Tuple[int, int]]:
        return [link_key(a, b) for a, b in zip(path, path[1:])]

    def path_delay_ns(self, path: Sequence[int]) -> float:
        total = 0.0
        for a, b in zip(path, path[1:]):
            total += self.links[link_key(a, b)].delay_ns
        return total

    def path_bw_gbps(self, path: Sequence[int]) -> float:
        rates = [self.links[link_key(a, b)].bw_gbps for a, b in zip(path, path[1:])]
        return min(rates) if rates else float("inf")

    def layout(self) -> Dict[int, Tuple[float, float]]:
        gps = self.gpus_per_server
        nservers = len(self.gpu_ids) // gps if gps else 0
        pos: Dict[int, Tuple[float, float]] = {}

        if nservers == 2 and len(self.nvswitch_ids) == 2 and len(self.switch_ids) == gps:
            for server in range(2):
                y = 0.0 if server == 0 else -2.2
                for rail in range(gps):
                    pos[server * gps + rail] = (float(rail), y)
            for server, nv in enumerate(self.nvswitch_ids):
                pos[nv] = (gps / 2.0 - 0.5, 0.9 if server == 0 else -3.1)
            for rail, tor in enumerate(self.switch_ids):
                pos[tor] = (float(rail), -1.1)
            return pos

        if nservers and len(self.nvswitch_ids) == nservers and len(self.switch_ids) == gps:
            for server in range(nservers):
                y = -server * 1.7
                for rail in range(gps):
                    pos[server * gps + rail] = (float(rail), y)
                pos[self.nvswitch_ids[server]] = (-1.4, y)
            for rail, tor in enumerate(self.switch_ids):
                pos[tor] = (float(rail), 1.8)
            return pos

        for i, gid in enumerate(self.gpu_ids):
            pos[gid] = (float(i % 8), -float(i // 8))
        for i, sid in enumerate(self.nvswitch_ids + self.switch_ids):
            pos[sid] = (-2.0 - float(i), 1.0)
        return pos


@dataclass
class Flow:
    src: int
    dst: int
    size: int
    start: float
    end: float
    path: Optional[List[int]] = None
    link_keys: List[Tuple[int, int]] = field(default_factory=list)
    dir_keys: List[Tuple[int, int]] = field(default_factory=list)
    color: str = "#1f77b4"
    _points: List[Tuple[float, float]] = field(default_factory=list, repr=False)
    _cum: List[float] = field(default_factory=list, repr=False)
    _total_len: float = 0.0

    @property
    def duration(self) -> float:
        return max(1.0, self.end - self.start)

    @property
    def rate_gbps(self) -> float:
        return self.size * 8.0 / self.duration

    def point_at(self, frac: float, pos: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
        if not self.path:
            return pos.get(self.src, (0.0, 0.0))
        if not self._points:
            self._points = [pos[n] for n in self.path]
            self._cum = [0.0]
            for a, b in zip(self._points, self._points[1:]):
                self._cum.append(self._cum[-1] + math.dist(a, b))
            self._total_len = self._cum[-1]
        if self._total_len <= 0:
            return self._points[0]
        distance = max(0.0, min(1.0, frac)) * self._total_len
        idx = bisect.bisect_right(self._cum, distance) - 1
        idx = max(0, min(idx, len(self._points) - 2))
        (x1, y1), (x2, y2) = self._points[idx], self._points[idx + 1]
        seg = self._cum[idx + 1] - self._cum[idx]
        t = (distance - self._cum[idx]) / seg if seg else 0.0
        return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)


@dataclass
class Scenario:
    name: str
    topology: Topology
    flows: List[Flow]
    pos: Dict[int, Tuple[float, float]]
    size: int = 0
    frame_times: List[float] = field(default_factory=list)
    loads: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)
    bottleneck: Optional[Tuple[int, int]] = None
    bottlenecks: List[Tuple[int, int]] = field(default_factory=list)
    cut_directed: List[Tuple[int, int]] = field(default_factory=list)
    summary: Dict[str, object] = field(default_factory=dict)
    _completion: Optional[float] = None

    @property
    def completion(self) -> float:
        if self._completion is None:
            self._completion = max((f.end for f in self.flows), default=0.0)
        return self._completion

    def node_label(self, node: int) -> str:
        if node in self.topology.gpu_ids:
            return f"GPU{node}"
        if node in self.topology.nvswitch_ids:
            return f"NVSwitch{node}"
        if node in self.topology.switch_ids:
            return f"ToR{node}"
        return f"node{node}"

    def link_label(self, key: Tuple[int, int]) -> str:
        return f"{self.node_label(key[0])} <-> {self.node_label(key[1])}"

    def link_capacity(self, key: Tuple[int, int]) -> float:
        return self.topology.links[key].bw_gbps

    def load_at(self, key: Tuple[int, int], frame_index: int) -> float:
        """Load on an undirected link = the busier of its two directions.

        A physical link is full duplex: each direction has its own capacity, so
        the two directions must not be summed (that would double the load)."""
        a, b = key
        best = 0.0
        for directed in ((a, b), (b, a)):
            series = self.loads.get(directed)
            if series:
                best = max(best, series[min(frame_index, len(series) - 1)])
        return best

    def utilization_at(self, key: Tuple[int, int], frame_index: int) -> float:
        cap = self.link_capacity(key)
        return self.load_at(key, frame_index) / cap if cap else 0.0

    def directed_load_at(self, directed: Tuple[int, int], frame_index: int) -> float:
        series = self.loads.get(directed)
        if not series:
            return 0.0
        return series[min(frame_index, len(series) - 1)]

    def active_flows(self, t: float) -> List[Flow]:
        return [f for f in self.flows if f.start <= t < f.end]


def parse_graph_csv(path: Path) -> List[Flow]:
    flows: List[Flow] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return flows
        idx = {name: i for i, name in enumerate(header)}
        i_src = idx.get("src")
        i_dst = idx.get("dest")
        i_size = idx.get("flow_size")
        if i_src is None or i_dst is None or i_size is None:
            return flows
        i_start = idx.get("started", idx.get("released", idx.get("posted")))
        # Stop at the last ACK (the simulator's completion), then fall back to
        # arrival / send time for graph files that do not carry ACK timestamps.
        i_end = idx.get("acked", idx.get("arrived", idx.get("sent", i_start)))
        for row in reader:
            try:
                src = int(row[i_src])
                dst = int(row[i_dst])
                size = int(row[i_size])
                start = float(row[i_start]) if i_start is not None and row[i_start] else 0.0
                end = float(row[i_end]) if i_end is not None and row[i_end] else start
            except (IndexError, TypeError, ValueError):
                continue
            if size <= 0:
                continue
            if end <= start:
                end = start + 1.0
            flows.append(Flow(src, dst, size, start, end))
    return flows


@dataclass
class Step:
    gpu: int
    tb: int
    s: int
    kind: str
    peer: int
    chan: int
    depid: int
    deps: int
    cnt: int

    @property
    def key(self) -> Tuple[int, int, int]:
        return (self.gpu, self.tb, self.s)


def parse_schedule_xml(path: Path, size: int) -> List[Flow]:
    """Turn an MSCCL XML schedule into logical GPU-to-GPU chunk flows.

    This is the offline model used when --run-sim is not given.  It follows the
    XML dependencies (thread-block order, explicit depid/deps, and the matching
    send/recv step of every chunk) and estimates each transfer from the physical
    path's bottleneck bandwidth.
    """
    root = ET.parse(path).getroot()
    nchunks = int(root.attrib.get("nchunksperloop", "1") or "1")
    chunk_bytes = max(1, size // max(1, nchunks))
    steps: Dict[Tuple[int, int, int], Step] = {}
    by_tb: Dict[Tuple[int, int], List[Step]] = defaultdict(list)
    recv_by_chan_step: Dict[Tuple[int, int, int], List[Step]] = defaultdict(list)
    recv_by_chan_off: Dict[Tuple[int, int, int], List[Step]] = defaultdict(list)

    for gpu_el in root.findall("gpu"):
        gid = int(gpu_el.attrib["id"])
        for tb_el in gpu_el.findall("tb"):
            tid = int(tb_el.attrib["id"])
            send_peer = int(tb_el.attrib.get("send", "-1"))
            recv_peer = int(tb_el.attrib.get("recv", "-1"))
            chan = int(tb_el.attrib.get("chan", "0"))
            for step_el in tb_el.findall("step"):
                sid = int(step_el.attrib["s"])
                kind = step_el.attrib.get("type", "?")
                peer = send_peer if kind == "s" else recv_peer if kind == "r" else -1
                step = Step(
                    gpu=gid,
                    tb=tid,
                    s=sid,
                    kind=kind,
                    peer=peer,
                    chan=chan,
                    depid=int(step_el.attrib.get("depid", "-1")),
                    deps=int(step_el.attrib.get("deps", "-1")),
                    cnt=int(step_el.attrib.get("cnt", "1")),
                )
                steps[step.key] = step
                by_tb[(gid, tid)].append(step)
                if kind == "r":
                    recv_by_chan_step[(gid, chan, sid)].append(step)
                    off = int(step_el.attrib.get("dstoff", step_el.attrib.get("srcoff", "-1")))
                    recv_by_chan_off[(gid, chan, off)].append(step)

    if not steps:
        return []

    base_successors: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = defaultdict(list)
    base_indegree: Dict[Tuple[int, int, int], int] = {key: 0 for key in steps}

    def add_base_edge(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> None:
        if a in steps and b in steps:
            base_successors[a].append(b)
            base_indegree[b] += 1

    for tb_steps in by_tb.values():
        tb_steps.sort(key=lambda st: st.s)
        for prev, cur in zip(tb_steps, tb_steps[1:]):
            add_base_edge(prev.key, cur.key)

    for step in steps.values():
        if step.depid >= 0 and step.deps >= 0:
            add_base_edge((step.gpu, step.depid, step.deps), step.key)

    def topo(successors, indegree):
        queue = [key for key, deg in indegree.items() if deg == 0]
        result: List[Tuple[int, int, int]] = []
        while queue:
            key = queue.pop()
            result.append(key)
            for nxt in successors.get(key, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        return result

    order = topo({k: list(v) for k, v in base_successors.items()}, dict(base_indegree))
    successors = {k: list(v) for k, v in base_successors.items()}
    indegree = dict(base_indegree)
    used_recv: set = set()
    for step in steps.values():
        if step.kind != "s" or step.peer < 0:
            continue
        candidates = recv_by_chan_step.get((step.peer, step.chan, step.s), [])
        if not candidates:
            candidates = recv_by_chan_off.get((step.peer, step.chan, step.s), [])
        for recv in candidates:
            if recv.key not in used_recv:
                used_recv.add(recv.key)
                successors.setdefault(step.key, []).append(recv.key)
                indegree[recv.key] += 1
                break
    with_recv = topo({k: list(v) for k, v in successors.items()}, dict(indegree))
    if len(with_recv) == len(steps):
        order = with_recv
    elif len(order) != len(steps):
        order = sorted(steps, key=lambda k: (k[0], k[1], k[2]))

    predecessors: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = defaultdict(list)
    for a, targets in successors.items():
        for b in targets:
            predecessors[b].append(a)

    start: Dict[Tuple[int, int, int], float] = {}
    end: Dict[Tuple[int, int, int], float] = {}
    flows: List[Flow] = []

    for key in order:
        step = steps[key]
        st = max((end.get(p, 0.0) for p in predecessors[key]), default=0.0)
        dur = 0.0
        route = None
        if step.kind == "s" and step.peer >= 0:
            route = _ROUTE_CACHE.get((step.gpu, step.peer))
            if route is None:
                route = _ACTIVE_TOPOLOGY.route(step.gpu, step.peer) if _ACTIVE_TOPOLOGY else None
                _ROUTE_CACHE[(step.gpu, step.peer)] = route
            if route:
                bw = _ACTIVE_TOPOLOGY.path_bw_gbps(route) if _ACTIVE_TOPOLOGY else 0.0
                if bw > 0:
                    dur = step.cnt * chunk_bytes * 8.0 / bw
                    dur += _ACTIVE_TOPOLOGY.path_delay_ns(route) if _ACTIVE_TOPOLOGY else 0.0
                else:
                    route = None
        start[key] = st
        end[key] = st + dur
        if route:
            flows.append(Flow(step.gpu, step.peer, step.cnt * chunk_bytes, st, st + dur, route))

    return flows


_ACTIVE_TOPOLOGY: Optional[Topology] = None
_ROUTE_CACHE: Dict[Tuple[int, int], Optional[List[int]]] = {}


def build_link_loads(
    flows: Iterable[Flow],
    topology: Topology,
    frame_times: Sequence[float],
) -> Dict[Tuple[int, int], List[float]]:
    """Achieved throughput per link (Gbps) over a sliding window.

    A flow transfers ``size`` bytes across every link on its path between its
    start and ACK.  We accumulate the bytes that have crossed each link and take
    the derivative over a short window.  This is the physical rate the link
    actually carries, so it is bounded by the link capacity by construction --
    unlike summing ``size*8/duration`` of every overlapping flow, which counts
    queued chunks as if they were all transmitting at once.
    """
    starts: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    ends: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    sizes: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for flow in flows:
        directed = flow.dir_keys or flow.link_keys
        if not directed:
            continue
        for key in directed:
            starts[key].append(flow.start)
            ends[key].append(flow.end)
            sizes[key].append(float(flow.size))

    times = np.asarray(frame_times, dtype=float)
    span = float(times[-1] - times[0]) if len(times) >= 2 else 1.0
    window = max(1.0, span / 20.0)
    loads: Dict[Tuple[int, int], List[float]] = {}
    for key in starts:
        st = np.asarray(starts[key], dtype=float)
        en = np.asarray(ends[key], dtype=float)
        sz = np.asarray(sizes[key], dtype=float)
        dur = np.maximum(1.0, en - st)

        def cumulative(tt: "np.ndarray") -> "np.ndarray":
            frac = np.clip((tt[:, None] - st[None, :]) / dur[None, :], 0.0, 1.0)
            return (frac * sz[None, :]).sum(axis=1)

        rate = (cumulative(times) - cumulative(times - window)) * 8.0 / window
        loads[key] = np.maximum(rate, 0.0).tolist()
    return loads

def assign_paths(flows: Iterable[Flow], topology: Topology) -> None:
    cache: Dict[Tuple[int, int], Tuple[Optional[List[int]], List[Tuple[int, int]],
                                       List[Tuple[int, int]]]] = {}
    colors = {gid: FLOW_COLORS[gid % len(FLOW_COLORS)] for gid in topology.gpu_ids}
    for flow in flows:
        key = (flow.src, flow.dst)
        if key not in cache:
            route = topology.route(flow.src, flow.dst)
            links = topology.path_links(route) if route else []
            directed = list(zip(route, route[1:])) if route else []
            cache[key] = (route, links, directed)
        flow.path, flow.link_keys, flow.dir_keys = cache[key]
        flow.color = colors.get(flow.src, "#1f77b4")
        flow._points = []
        flow._cum = []
        flow._total_len = 0.0


_CUT_CACHE: Dict[Tuple[str, str], Tuple[List[Tuple[int, int]], float, int, float, frozenset]] = {}


def optimal_bottleneck_cut(
    topology: Topology, collective: str, size: int
) -> Tuple[List[Tuple[int, int]], float, int, float, frozenset]:
    """Bottleneck cut of a collective on a topology (ForestColl-style).

    The cut is a partition (S, T) of the graph chosen to maximise the data the
    collective must push across it per unit of cut bandwidth:

        AllGather:  max_S |S∩Vc| / B+(S)
        AllToAll:   max_S |S∩Vc|·|T∩Vc| / B+(S)

    where B+(S) is the total capacity of the links leaving S.  The result
    depends only on the topology (including failures) and the collective, never
    on the schedule, so the no-reopt and reopt panels share exactly the same
    cut.  The partition is found with a small MILP (scipy/HiGHS).

    Returns (cut_links, capacity_gbps, k, optimal_ms, source_side) where
    source_side is the set of nodes on the S side of the cut.  A schedule moves
    each commodity from S to T exactly once, so the required crossing data is
    fixed and the optimal schedule achieves it at the cut capacity.
    """
    cache_key = (str(topology.path), collective)
    cached = _CUT_CACHE.get(cache_key)
    if cached is not None:
        cut, capacity, k, optimal_ms, source_side = cached
        return list(cut), capacity, k, optimal_ms, source_side

    try:
        import numpy as _np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required to compute the bottleneck cut") from exc

    nodes = sorted(topology.adj)
    index = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    arcs: List[Tuple[int, int, float]] = []
    for (u, v), link in topology.links.items():
        arcs.append((u, v, link.bw_gbps))
        arcs.append((v, u, link.bw_gbps))
    n_arcs = len(arcs)
    n_var = n_nodes + n_arcs

    cost = _np.zeros(n_var)
    for j, (_, _, cap) in enumerate(arcs):
        cost[n_nodes + j] = cap
    # Tiny tie-breaker: among cuts of equal capacity, keep the ToR switches on
    # the S side.  This yields the canonical "ingress of the excluded GPU" cut
    # for AllGather and a clean one-sided ToR cut for AllToAll, instead of an
    # arbitrary mix of the two sides.
    for tor in topology.switch_ids:
        cost[index[tor]] = -1e-3

    matrix = lil_matrix((n_arcs, n_var))
    for j, (u, v, _) in enumerate(arcs):
        matrix[j, index[u]] = 1.0
        matrix[j, index[v]] = -1.0
        matrix[j, n_nodes + j] = -1.0
    arc_constraints = LinearConstraint(
        matrix.tocsr(), _np.full(n_arcs, -_np.inf), _np.zeros(n_arcs)
    )

    gpus = list(topology.gpu_ids)
    n_gpus = len(gpus)
    integrality = _np.zeros(n_var)
    integrality[:n_nodes] = 1
    bounds = Bounds(_np.zeros(n_var), _np.ones(n_var))

    best: Optional[Tuple[float, int, float, "_np.ndarray"]] = None
    for k in range(1, n_gpus):
        row = lil_matrix((1, n_var))
        for gpu in gpus:
            row[0, index[gpu]] = 1.0
        constraints = [arc_constraints, LinearConstraint(row.tocsr(), k, k)]
        result = milp(cost, constraints=constraints, integrality=integrality,
                      bounds=bounds, options={"time_limit": 30, "mip_rel_gap": 0.0})
        if not result.success:
            continue
        cut_bw = float(result.fun)
        if cut_bw <= 0:
            continue
        demand = float(k) if collective == "ag" else float(k * (n_gpus - k))
        ratio = demand / cut_bw
        if best is None or ratio > best[0]:
            best = (ratio, k, cut_bw, _np.round(result.x[:n_nodes]).astype(int))
    if best is None:
        raise RuntimeError(f"could not compute a bottleneck cut for {topology.path}")

    _, k, cut_bw, side = best
    in_s = {nodes[i] for i in range(n_nodes) if side[i] == 1}
    cut_links: List[Tuple[int, int]] = []
    for (u, v), link in topology.links.items():
        if (u in in_s) != (v in in_s):
            cut_links.append(link.key)
    cut_links.sort(key=lambda key: (-topology.links[key].bw_gbps, key))
    capacity = sum(topology.links[key].bw_gbps for key in cut_links)

    side_set = frozenset(in_s)
    demand = float(k) if collective == "ag" else float(k * (n_gpus - k))
    crossing_bits = demand * (size / n_gpus) * 8.0
    optimal_ms = crossing_bits / capacity / 1e6 if capacity else 0.0
    _CUT_CACHE[cache_key] = (tuple(cut_links), capacity, k, optimal_ms, side_set)
    return list(cut_links), capacity, k, optimal_ms, side_set


def summarize_scenario(
    scenario: Scenario,
    cut_edges: Sequence[Tuple[int, int]],
    collective: str,
    size: int,
    source_side: frozenset,
) -> Dict[str, object]:
    """Summarise a schedule against the shared topology/collective cut.

    The cut is fixed by the topology (including failures) and the collective, so
    both schedules are measured on the same links.  The optimal schedule moves
    every commodity from S to T exactly once; we therefore count the required
    S->T data once per commodity (not both directions, and not repeated
    crossings, which only a suboptimal schedule would add).  The achieved
    bandwidth is that required data divided by the completion time, so it is
    bounded by the cut capacity by construction and never by clamping.
    """
    cut_set = set(cut_edges)
    scenario.bottlenecks = list(cut_edges)
    scenario.bottleneck = cut_edges[0] if cut_edges else None
    scenario.cut_directed = [
        (a, b) if a in source_side else (b, a) for a, b in cut_edges
    ]

    capacity = sum(scenario.link_capacity(key) for key in cut_edges)
    crossing_bytes = 0
    for flow in scenario.flows:
        if flow.src in source_side and flow.dst not in source_side:
            crossing_bytes += flow.size

    completion = scenario.completion
    achieved = crossing_bytes * 8.0 / completion if completion > 0 else 0.0
    utilization = achieved / capacity if capacity else 0.0
    scenario.summary = {
        "name": scenario.name,
        "completion_ms": completion / 1e6,
        "algbw_gbps": size / completion if completion > 0 else 0.0,
        "bottleneck_cut": " ; ".join(scenario.link_label(k) for k in cut_edges) or "-",
        "bottleneck_count": len(cut_edges),
        "cut_capacity_gbps": capacity,
        "cut_achieved_gbps": achieved,
        "cut_utilization": utilization,
        "crossing_bytes": crossing_bytes,
        "flows": len(scenario.flows),
    }
    return scenario.summary

def _heat_color(util: float) -> Optional[str]:
    """Congestion heat, deliberately avoiding red (reserved for the cut)."""
    if util < 0.25:
        return None
    if util < 0.50:
        return "#90caf9"   # light blue
    if util < 0.75:
        return "#ffd54f"   # amber
    if util < 1.00:
        return "#fb8c00"   # orange
    return "#ef6c00"       # dark orange: oversubscribed


def _blend(color: str, target: str, alpha: float) -> Tuple[int, int, int]:
    c = _rgb(color)
    t = _rgb(target)
    return tuple(int(c[i] + (t[i] - c[i]) * alpha) for i in range(3))


def _link_style(
    scenario: Scenario,
    key: Tuple[int, int],
    frame_index: int,
    max_bw_by_kind: Dict[str, float],
) -> Tuple[str, float, float, Optional[str]]:
    """Return (base colour, base width, utilization, heat colour).

    The base colour encodes the link type (or the failure); the heat colour is a
    thinner inner line that encodes load. Red is never used here."""
    cap = scenario.link_capacity(key)
    util = scenario.utilization_at(key, frame_index)
    kind = _link_kind(scenario.topology, key)
    max_bw = max_bw_by_kind.get(kind, cap) or cap
    degraded = cap < 0.5 * max_bw
    if degraded:
        return COLOR_FAIL, 6.0, util, None
    if kind == "nvlink":
        return COLOR_INTRA, 6.0, util, _heat_color(util)
    return COLOR_INTER, 5.0, util, _heat_color(util)


def _link_kind(topology: Topology, key: Tuple[int, int]) -> str:
    a, b = key
    if a in topology.nvswitch_ids or b in topology.nvswitch_ids:
        return "nvlink"
    if a in topology.switch_ids or b in topology.switch_ids:
        return "nic"
    return "switch"


def _max_bw_by_kind(topology: Topology) -> Dict[str, float]:
    result: Dict[str, float] = defaultdict(float)
    for key, link in topology.links.items():
        result[_link_kind(topology, key)] = max(result[_link_kind(topology, key)], link.bw_gbps)
    return dict(result)


_FONT_CACHE: Dict[int, "ImageFont.ImageFont"] = {}


def _font(size: int):
    size = max(6, int(size))
    if size not in _FONT_CACHE:
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ):
            try:
                _FONT_CACHE[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                pass
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(c * 2 for c in color)
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _panel_transform(pos: Dict[int, Tuple[float, float]], width: int, height: int,
                     top_pad: int, bottom_pad: int, margin: int,
                     margin_x: Optional[int] = None):
    margin_x = margin if margin_x is None else margin_x
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    data_w = max(1e-6, max_x - min_x)
    data_h = max(1e-6, max_y - min_y)
    usable_h = height - top_pad - bottom_pad - 2 * margin
    scale = min((width - 2 * margin_x) / data_w, usable_h / data_h)
    ox = (width - data_w * scale) / 2.0
    oy = top_pad + margin + (usable_h - data_h * scale) / 2.0

    def to_px(x: float, y: float) -> Tuple[float, float]:
        return ox + (x - min_x) * scale, oy + (max_y - y) * scale

    return to_px, scale


def _draw_dashed(draw, p1, p2, color, width: int, dash: int = 10, gap: int = 6) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    d = 0.0
    while d < length:
        e = min(length, d + dash)
        draw.line([(x1 + ux * d, y1 + uy * d), (x1 + ux * e, y1 + uy * e)],
                  fill=color, width=width)
        d += dash + gap


def _draw_legend(draw, x: float, y: float, font_scale: float) -> None:
    small = _font(int(12 * font_scale))
    rows = [
        ("line", COLOR_INTRA, "intra-node"),
        ("line", COLOR_INTER, "inter-node"),
        ("dash", COLOR_FAIL, "degraded"),
        ("line", "#fb8c00", "congested"),
        ("cross", COLOR_BOTTLENECK, "bottleneck cut"),
        ("dot", "#4c72b0", "packet (src GPU)"),
    ]
    per_col = 3
    row_h = int(19 * font_scale)
    col_w = int(210 * font_scale)
    width = col_w * 2
    height = row_h * per_col + int(12 * font_scale)
    draw.rectangle([x, y, x + width, y + height], fill="#ffffff", outline="#b0bec5")
    for i, (kind, color, label) in enumerate(rows):
        col, row = divmod(i, per_col)
        cx0 = x + col * col_w + int(10 * font_scale)
        cy = y + int(8 * font_scale) + row * row_h + row_h // 2
        sx0, sx1 = cx0, cx0 + int(30 * font_scale)
        if kind == "line":
            draw.line([(sx0, cy), (sx1, cy)], fill=_rgb(color), width=max(2, int(3 * font_scale)))
        elif kind == "dash":
            _draw_dashed(draw, (sx0, cy), (sx1, cy), _rgb(color), max(2, int(3 * font_scale)),
                         dash=max(4, int(6 * font_scale)), gap=max(3, int(4 * font_scale)))
        elif kind == "cross":
            mx = (sx0 + sx1) / 2.0
            draw.line([(mx, cy - 6 * font_scale), (mx, cy + 6 * font_scale)],
                      fill=_rgb(color), width=max(2, int(3 * font_scale)))
            draw.ellipse([mx - 4 * font_scale, cy - 4 * font_scale,
                          mx + 4 * font_scale, cy + 4 * font_scale], fill=_rgb(color))
        elif kind == "dot":
            r = 4 * font_scale
            draw.ellipse([sx1 - r, cy - r, sx1 + r, cy + r], fill=_rgb(color), outline="white")
        draw.text((sx1 + int(8 * font_scale), cy - int(7 * font_scale)), label,
                  fill=_rgb(COLOR_TEXT), font=small)


def _text_badge(draw, cx: float, cy: float, text: str, font, fg: str,
                anchor: str = "center", border: Optional[str] = None, pad: int = 4):
    """Draw text on a white badge so it stays readable over a link."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if anchor == "left":
        x0 = cx
    elif anchor == "right":
        x0 = cx - tw
    else:
        x0 = cx - tw / 2.0
    y0 = cy - th / 2.0
    draw.rectangle([x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad],
                   fill="white", outline=border or "#b0bec5")
    draw.text((x0, y0 - 1), text, fill=_rgb(fg), font=font)
    return tw, th


def _render_base_panel(scenario: Scenario, width: int, height: int, dpi: int, *,
                       draw_legend: bool = True, draw_progress: bool = True,
                       subtitle: Optional[str] = None,
                       top_pad: int = 140, bottom_pad: int = 250,
                       margin: int = 40, margin_x: Optional[int] = None,
                       clamp_badges: bool = False):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    topo = scenario.topology
    pos = scenario.pos
    to_px, scale = _panel_transform(pos, width, height, top_pad, bottom_pad,
                                    margin, margin_x)
    max_bw = _max_bw_by_kind(topo)
    base_scale = dpi / 100.0
    font_scale = max(1.0, base_scale) * 1.7

    for key, link in topo.links.items():
        x1, y1 = to_px(*pos[link.u])
        x2, y2 = to_px(*pos[link.v])
        kind = _link_kind(topo, key)
        cap = link.bw_gbps
        degraded = cap < 0.5 * max_bw.get(kind, cap)
        if degraded:
            lw = max(2, int(6.0 * base_scale))
            _draw_dashed(draw, (x1, y1), (x2, y2), _rgb(COLOR_FAIL), lw,
                         dash=max(6, int(10 * base_scale)), gap=max(4, int(6 * base_scale)))
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            text = f"FAIL {cap:g} Gbps"
            font = _font(int(12 * font_scale))
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length
            off = tw / 2.0 + int(20 * base_scale)
            if clamp_badges:
                candidates = [(mx + nx * off, my + ny * off),
                              (mx - nx * off, my - ny * off)]

                def room(point: Tuple[float, float]) -> float:
                    return min(point[0], width - point[0],
                               point[1], height - point[1])

                lx, ly = max(candidates, key=room)
                pad = tw / 2.0 + int(8 * base_scale)
                lx = min(max(lx, pad), width - pad)
                ly = min(max(ly, int(24 * base_scale)), height - int(24 * base_scale))
            else:
                # Original behaviour: put the badge on the side of the link that
                # faces away from the graph.
                if (mx - width / 2.0) * nx + (my - height / 2.0) * ny < 0:
                    nx, ny = -nx, -ny
                lx, ly = mx + nx * off, my + ny * off
            draw.line([(mx, my), (lx, ly)], fill=_rgb(COLOR_FAIL),
                      width=max(1, int(2 * base_scale)))
            _text_badge(draw, lx, ly, text, font, COLOR_FAIL, border=_rgb(COLOR_FAIL))
        else:
            color = COLOR_INTRA if kind == "nvlink" else COLOR_INTER
            lw = max(2, int((6.0 if kind == "nvlink" else 5.0) * base_scale))
            draw.line([(x1, y1), (x2, y2)], fill=_rgb(color), width=lw)

    gpu_r = max(10, int(16 * base_scale))
    for gid in topo.gpu_ids:
        x, y = to_px(*pos[gid])
        draw.ellipse([x - gpu_r, y - gpu_r, x + gpu_r, y + gpu_r], fill=_rgb("#4c72b0"),
                     outline="white", width=max(1, int(1.5 * base_scale)))
        font = _font(int(11 * font_scale))
        bbox = draw.textbbox((0, 0), str(gid), font=font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2 - 1),
                  str(gid), fill="white", font=font)

    nv_r = max(9, int(13 * base_scale))
    nv_font = _font(int(11 * font_scale))
    for sid in topo.nvswitch_ids:
        x, y = to_px(*pos[sid])
        draw.rectangle([x - nv_r, y - nv_r, x + nv_r, y + nv_r], fill=_rgb("#55a868"),
                       outline="white", width=max(1, int(1.5 * base_scale)))
        _text_badge(draw, x - nv_r - int(6 * base_scale), y, f"NVSwitch{sid}",
                    nv_font, "#2e6b3a", anchor="right")

    tor_r = max(9, int(12 * base_scale))
    tor_font = _font(int(11 * font_scale))
    for sid in topo.switch_ids:
        x, y = to_px(*pos[sid])
        draw.polygon([(x, y - tor_r), (x + tor_r, y), (x, y + tor_r), (x - tor_r, y)],
                     fill=_rgb("#c44e52"), outline="white")
        _text_badge(draw, x + tor_r + int(6 * base_scale), y, f"ToR{sid}",
                    tor_font, "#7b2d2f", anchor="left")

    title_font = _font(int(18 * font_scale))
    bbox = draw.textbbox((0, 0), scenario.name, font=title_font)
    title_y = 12
    draw.text(((width - (bbox[2] - bbox[0])) / 2, title_y), scenario.name,
              fill=_rgb(COLOR_TEXT), font=title_font)
    if subtitle:
        sub_font = _font(int(13 * font_scale))
        sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sub_y = title_y + (bbox[3] - bbox[1]) + int(10 * base_scale)
        draw.text(((width - (sub_bbox[2] - sub_bbox[0])) / 2, sub_y), subtitle,
                  fill=_rgb(COLOR_MUTED), font=sub_font)

    if draw_legend:
        _draw_legend(draw, margin, height - bottom_pad + int(14 * base_scale), font_scale)

    if draw_progress:
        bar_x0, bar_x1 = int(width * 0.48), width - margin
        bar_y = height - int(34 * base_scale)
        draw.rectangle([bar_x0, bar_y, bar_x1, bar_y + int(12 * base_scale)],
                       fill="#eceff1", outline="#90a4ae")
    draw.rectangle([0, 0, width - 1, height - 1], outline="#cfd8dc", width=2)
    return img, to_px, scale


def _render_frame_panel(base_img, scenario: Scenario, to_px, scale: float, frame_index: int,
                        frame_times: Sequence[float], max_bw: Dict[str, float],
                        max_dots: int, rng: random.Random, dpi: int):
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    topo = scenario.topology
    pos = scenario.pos
    width, height = img.size
    global_t = frame_times[frame_index]
    t = min(global_t, scenario.completion)
    done = global_t > scenario.completion + 1e-9
    base_scale = dpi / 100.0
    font_scale = max(1.0, base_scale) * 1.7

    # Congestion is an inner heat line on top of the static link, so the link
    # type (green/black) and the failure (magenta dashed) stay readable.
    for key, link in topo.links.items():
        _, base_w, util, heat = _link_style(scenario, key, frame_index, max_bw)
        if heat is None:
            continue
        x1, y1 = to_px(*pos[link.u])
        x2, y2 = to_px(*pos[link.v])
        lw = max(1, int(min(base_w * 0.7, 1.5 + 2.5 * min(util, 1.5)) * base_scale))
        draw.line([(x1, y1), (x2, y2)], fill=_rgb(heat), width=lw)

    all_active = scenario.active_flows(t)
    active = all_active
    active_count = len(all_active)
    if active_count > max_dots:
        bn_set = set(scenario.bottlenecks)
        on_bottleneck = [f for f in active if any(k in bn_set for k in f.link_keys)]
        bn_ids = {id(f) for f in on_bottleneck}
        rest = [f for f in active if id(f) not in bn_ids]
        keep = on_bottleneck[: max_dots // 2]
        remaining = max_dots - len(keep)
        if remaining > 0 and rest:
            keep += rng.sample(rest, min(remaining, len(rest)))
        active = keep

    dot_r = max(3, int(5.5 * base_scale))
    for flow in active:
        frac = (t - flow.start) / flow.duration
        if frac < 0:
            continue
        for k, offset in enumerate((0.20, 0.14, 0.08, 0.03)):
            f = frac - offset
            if f < 0:
                continue
            x, y = flow.point_at(f, pos)
            px, py = to_px(x, y)
            r = max(1, int(dot_r * (0.40 + 0.11 * k)))
            col = _blend(flow.color, "#ffffff", 0.62 - 0.10 * k)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=col)
        x, y = flow.point_at(frac, pos)
        px, py = to_px(x, y)
        draw.ellipse([px - dot_r, py - dot_r, px + dot_r, py + dot_r],
                     fill=_rgb(flow.color), outline="white",
                     width=max(1, int(1.3 * base_scale)))
        f2 = min(1.0, frac + 0.02)
        if f2 > frac:
            x2, y2 = flow.point_at(f2, pos)
            qx, qy = to_px(x2, y2)
            dx, dy = qx - px, qy - py
            length = math.hypot(dx, dy)
            if length > 0.5:
                ux, uy = dx / length, dy / length
                nx, ny = -uy, ux
                ah = dot_r * 2.0
                tip = (px + ux * ah, py + uy * ah)
                b1 = (px + ux * ah * 0.35 + nx * ah * 0.75, py + uy * ah * 0.35 + ny * ah * 0.75)
                b2 = (px + ux * ah * 0.35 - nx * ah * 0.75, py + uy * ah * 0.35 - ny * ah * 0.75)
                draw.polygon([tip, b1, b2], fill=_rgb(flow.color), outline="white")

    bn_positions = []
    if scenario.bottlenecks:
        phase = (frame_index % 18) / 18.0
        for key in scenario.bottlenecks:
            link = topo.links[key]
            x1, y1 = to_px(*pos[link.u])
            x2, y2 = to_px(*pos[link.v])
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            util = scenario.utilization_at(key, frame_index)
            if util <= 0.45:
                continue
            bn_positions.append((mx, my))
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length
            cut = 26 * base_scale
            _draw_dashed(draw, (mx - nx * cut, my - ny * cut), (mx + nx * cut, my + ny * cut),
                         _rgb(COLOR_BOTTLENECK), max(2, int(3.5 * base_scale)),
                         dash=max(4, int(6 * base_scale)), gap=max(2, int(4 * base_scale)))
            for k in range(3):
                radius = (0.10 + 0.30 * ((phase + k / 3.0) % 1.0)) * scale
                alpha = 0.65 * (1.0 - ((phase + k / 3.0) % 1.0))
                if radius > 1:
                    col = _blend(COLOR_BOTTLENECK, "#ffffff", 1.0 - alpha)
                    draw.ellipse([mx - radius, my - radius, mx + radius, my + radius],
                                 outline=col, width=max(1, int(2 * base_scale)))
            star_r = (14 + 10 * min(util, 1.5)) * base_scale
            points = []
            for i in range(10):
                ang = -math.pi / 2 + i * math.pi / 5
                rr = star_r if i % 2 == 0 else star_r * 0.45
                points.append((mx + math.cos(ang) * rr, my + math.sin(ang) * rr))
            draw.polygon(points, fill=_rgb(COLOR_BOTTLENECK), outline="white")

    stats = [
        f"t = {t / 1e6:7.2f} / {scenario.completion / 1e6:.2f} ms  "
        f"({'DONE' if done else 'running'}, {active_count} packets)",
    ]
    cut_line = None
    if scenario.bottlenecks:
        cap = float(scenario.summary.get("cut_capacity_gbps", 0.0))
        load = sum(scenario.directed_load_at(d, frame_index)
                   for d in scenario.cut_directed)
        util = load / cap if cap else 0.0
        cut_line = (
            f"cut: {len(scenario.bottlenecks)} links, "
            f"{load:.0f}/{cap:.0f} Gbps ({util * 100:.0f}%)"
        )
        stats.append(cut_line)
    font = _font(int(13 * font_scale))
    line_h = int(19 * font_scale)
    box_w = max(draw.textbbox((0, 0), s, font=font)[2] for s in stats) + int(20 * base_scale)
    box_h = line_h * len(stats) + int(12 * base_scale)
    box_y = int(50 * base_scale)
    draw.rectangle([12, box_y, 12 + box_w, box_y + box_h], fill="#fafafa", outline="#b0bec5")
    for i, text in enumerate(stats):
        color = COLOR_BOTTLENECK if text is cut_line else COLOR_TEXT
        draw.text((20, box_y + 6 + i * line_h), text, fill=_rgb(color), font=font)

    bar_x0, bar_x1 = int(width * 0.48), width - int(40 * base_scale)
    bar_y = height - int(34 * base_scale)
    bar_h = max(8, int(12 * base_scale))
    frac = min(1.0, t / scenario.completion) if scenario.completion > 0 else 0.0
    draw.rectangle([bar_x0, bar_y, bar_x1, bar_y + bar_h], fill="#eceff1", outline="#90a4ae")
    if frac > 0:
        draw.rectangle([bar_x0, bar_y, bar_x0 + (bar_x1 - bar_x0) * frac, bar_y + bar_h],
                       fill="#1e88e5")
    small = _font(int(11 * font_scale))
    draw.text((bar_x0, bar_y - int(16 * font_scale)), f"progress {frac * 100:5.1f}%",
              fill=_rgb(COLOR_MUTED), font=small)
    return img


def _animate_pillow(scenarios: Sequence[Scenario], out_path: Path, fps: int, frames: int,
                    dpi: int, max_dots: int, hold_ms: int = 1500,
                    cut_edges: Sequence[Tuple[int, int]] = (),
                    collective: str = "ag",
                    source_side: frozenset = frozenset()) -> None:
    duration = max((s.completion for s in scenarios), default=1.0)
    frame_times = [duration * i / max(1, frames - 1) for i in range(frames)]
    for scenario in scenarios:
        scenario.frame_times = frame_times
        scenario.loads = build_link_loads(scenario.flows, scenario.topology, frame_times)
        summarize_scenario(scenario, cut_edges, collective, scenario.size, source_side)

    scale = dpi / 100.0
    panel_w = int(1000 * scale)
    panel_h = int(850 * scale)
    total_w = panel_w * len(scenarios)
    total_h = panel_h
    max_bw = [_max_bw_by_kind(s.topology) for s in scenarios]
    rngs = [random.Random(1234 + i) for i in range(len(scenarios))]
    bases = [_render_base_panel(s, panel_w, panel_h, dpi) for s in scenarios]

    def render(frame_index: int):
        frame = Image.new("RGB", (total_w, total_h), "white")
        for i, (scenario, base, bw, rng) in enumerate(zip(scenarios, bases, max_bw, rngs)):
            base_img, to_px, sc = base
            panel = _render_frame_panel(base_img, scenario, to_px, sc, frame_index,
                                        frame_times, bw, max_dots, rng, dpi)
            frame.paste(panel, (i * panel_w, 0))
        return frame

    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = render(0)
    palette = first.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    first_p = first.quantize(palette=palette, dither=Image.Dither.NONE)

    def rest():
        for fi in range(1, frames):
            yield render(fi).quantize(palette=palette, dither=Image.Dither.NONE)

    frame_ms = max(20, int(1000 / max(1, fps)))
    durations = [frame_ms] * max(1, frames)
    durations[-1] += max(0, hold_ms)
    first_p.save(out_path, save_all=True, append_images=rest(), duration=durations,
                 loop=0, optimize=False, disposal=2)


def animate(scenarios: Sequence[Scenario], out_path: Path, fps: int, frames: int, dpi: int,
            max_dots: int = 110, hold_ms: int = 1500,
            cut_edges: Sequence[Tuple[int, int]] = (),
            collective: str = "ag",
            source_side: frozenset = frozenset()) -> None:
    _animate_pillow(scenarios, out_path, fps, frames, dpi, max_dots, hold_ms,
                    cut_edges, collective, source_side)

def write_summary(scenarios: Sequence[Scenario], out_path: Path) -> None:
    rows = []
    for scenario in scenarios:
        s = scenario.summary
        rows.append(s)
    csv_path = out_path.with_name(out_path.stem + "_summary.csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    base_ms = rows[0]["completion_ms"] if len(rows) >= 2 else None
    lines = [
        "# MCCL schedule visualisation summary",
        "",
        "| scenario | completion (ms) | AlgBW (GB/s) | cut (Gbps) | cut util | speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row, scenario in zip(rows, scenarios):
        completion_ms = float(row["completion_ms"])
        algbw = scenario.size / scenario.completion if scenario.completion > 0 else 0.0
        speedup = f"{base_ms / completion_ms:.2f}x" if base_ms else "-"
        lines.append(
            f"| {row['name']} | {completion_ms:.3f} | {algbw:.1f} | "
            f"{float(row['cut_achieved_gbps']):.0f} | "
            f"{float(row['cut_utilization']) * 100:.0f}% | {speedup} |"
        )
    if rows:
        first = rows[0]
        lines += [
            "",
            f"Bottleneck cut (fixed by topology, failure and collective, so both "
            f"schedules share it): {first['bottleneck_cut']}.",
            f"Cut capacity: {float(first['cut_capacity_gbps']):.0f} Gbps.",
        ]
    if len(rows) >= 2 and base_ms:
        base, reopt = rows[0], rows[1]
        lines += [
            "",
            f"Speedup of **{reopt['name']}** over **{base['name']}**: "
            f"**{base_ms / reopt['completion_ms']:.2f}x** "
            f"({base_ms:.3f} ms -> {reopt['completion_ms']:.3f} ms).",
            f"Cut bandwidth: {float(base['cut_achieved_gbps']):.0f} -> "
            f"{float(reopt['cut_achieved_gbps']):.0f} Gbps "
            f"({float(base['cut_utilization']) * 100:.0f}% -> "
            f"{float(reopt['cut_utilization']) * 100:.0f}% of the cut).",
        ]
    md_path = out_path.with_name(out_path.stem + "_summary.md")
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {csv_path} and {md_path}")


def write_timeline(scenarios: Sequence[Scenario], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [2.2, 1.0]})
    for scenario in scenarios:
        if not scenario.bottlenecks:
            continue
        series = [sum(scenario.load_at(k, i) for k in scenario.bottlenecks)
                  for i in range(len(scenario.frame_times))]
        times = [t / 1e6 for t in scenario.frame_times]
        label = f"{scenario.name}: bottleneck cut ({len(scenario.bottlenecks)} links)"
        ax1.plot(times, series, lw=2.0, label=label)
        ax1.axhline(sum(scenario.link_capacity(k) for k in scenario.bottlenecks),
                    color=ax1.lines[-1].get_color(), ls=":", lw=1.0)
    ax1.set_ylabel("aggregate load on bottleneck cut (Gbps)")
    ax1.set_xlabel("time (ms)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    names = [s.name for s in scenarios]
    times = [s.completion / 1e6 for s in scenarios]
    bars = ax2.bar(range(len(names)), times, color=["#cc2222", "#2ca02c"][: len(names)])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("completion (ms)")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Bottleneck load over time and collective completion", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png_path = out_path.with_name(out_path.stem + "_timeline.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {png_path}")


def run_simulator_graph(
    topology_path: Path,
    schedule_path: Path,
    workload_path: Path,
    size: int,
    binary: Path,
    base_system: Path,
    label: str,
) -> List[Flow]:
    if not binary.exists():
        raise FileNotFoundError(f"simulator binary not found: {binary}")
    with tempfile.TemporaryDirectory(prefix="mccl_viz_") as tmp:
        work = Path(tmp)
        (work / "astra-sim-alibabacloud").symlink_to(BASE, target_is_directory=True)
        system_lines = []
        for line in base_system.read_text().splitlines():
            if line.startswith(("msccl-path:", "collective-graph:", "collective-graph-ranks:")):
                continue
            system_lines.append(line)
        system_lines.append(f"msccl-path: {schedule_path.parent}/")
        system_lines.append("collective-graph: 1")
        system_lines.append("collective-graph-ranks: 0")
        (work / "sys.txt").write_text("\n".join(system_lines) + "\n")
        env = os.environ.copy()
        env.update({
            "AS_LOG_LEVEL": "2",
            "AS_PXN_ENABLE": "1",
            "AS_SEND_LAT": "3000",
            "SIMAI_DUMP_DETAILED_FLOWS": "0",
        })
        cmd = [
            str(binary), "-t", "1",
            "-w", str(workload_path),
            "-n", str(topology_path),
            "-c", str(DEFAULT_CONFIG),
            "-s", "sys.txt",
            "-r", f"viz_{label}",
        ]
        print(f"[viz] running simulator for {label} ...", flush=True)
        result = subprocess.run(cmd, cwd=work, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            tail = "\n".join(result.stdout.splitlines()[-25:])
            raise RuntimeError(f"simulator failed for {label} (rc={result.returncode}):\n{tail}")
        graphs = sorted(work.glob("collective_graph_*.csv"))
        if not graphs:
            raise RuntimeError(f"simulator did not produce a collective_graph CSV for {label}")
        try:
            coll = ET.parse(schedule_path).getroot().attrib.get("coll", "").lower()
        except ET.ParseError:
            coll = ""
        if coll:
            matching = [p for p in graphs if coll in p.name.lower()]
            if matching:
                graphs = matching
        graph = max(graphs, key=lambda p: p.stat().st_size)
        print(f"[viz] using {graph.name} ({graph.stat().st_size / 1e6:.1f} MB)", flush=True)
        return parse_graph_csv(graph)


def build_scenario(
    name: str,
    topology_path: Path,
    schedule_path: Path,
    size: int,
    flows: Optional[List[Flow]] = None,
) -> Scenario:
    global _ACTIVE_TOPOLOGY, _ROUTE_CACHE
    topology = Topology.parse(topology_path)
    _ACTIVE_TOPOLOGY = topology
    _ROUTE_CACHE = {}
    if flows is None:
        flows = parse_schedule_xml(schedule_path, size)
    assign_paths(flows, topology)
    scenario = Scenario(name=name, topology=topology, flows=flows, pos=topology.layout(), size=size)
    return scenario


def demo_paths(demo: str) -> Tuple[str, Path, Path, Path, Path, int]:
    collective, failure = demo.split("-", 1)
    if collective not in COLLECTIVES:
        raise SystemExit(f"unknown demo collective {collective!r}")
    if failure not in FAILURES:
        raise SystemExit(f"unknown demo failure {failure!r}")
    info = COLLECTIVES[collective]
    topology = INPUTS / "topo" / "single_collective" / f"1x2x8-{failure}"
    baseline = info["baseline"]
    reopt_name = info["reopt"].format(failure=failure, failure_short=failure.replace("failure_", "Fail"))
    reopt = INPUTS / "msccl" / "mccl" / failure / "priority-9" / reopt_name
    workload = info["workload"]
    system = info["system"]
    return collective, topology, baseline, reopt, workload, system


def _collective_kind(args) -> str:
    """Infer AllGather vs AllToAll from the workload/schedule names."""
    texts = []
    if args.workload:
        texts.append(args.workload.name)
    if args.schedule:
        texts.append(args.schedule.name)
    if args.baseline:
        texts.append(args.baseline.name)
    for text in texts:
        low = text.lower()
        if "alltoall" in low or "a2a" in low:
            return "a2a"
        if "allgather" in low or "ag-" in low or low.startswith("ag"):
            return "ag"
    return "ag"


def _collective_kind(args) -> str:
    """Infer AllGather vs AllToAll from the workload/schedule names."""
    texts = []
    if args.workload:
        texts.append(args.workload.name)
    if args.schedule:
        texts.append(args.schedule.name)
    if args.baseline:
        texts.append(args.baseline.name)
    for text in texts:
        low = text.lower()
        if "alltoall" in low or "a2a" in low:
            return "a2a"
        if "allgather" in low or "ag-" in low or low.startswith("ag"):
            return "ag"
    return "ag"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Animate MSCCL/MCCL schedules over a SimAI topology and expose the bottleneck.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--demo", choices=[f"{c}-{f}" for c in COLLECTIVES for f in FAILURES],
                        help="use one of the exact E3 artifact scenarios (AG/A2A x Fail1/2/3)")
    parser.add_argument("--topology", type=Path, help="SimAI topology file")
    parser.add_argument("--schedule", type=Path, help="MSCCL XML schedule to animate")
    parser.add_argument("--baseline", type=Path, help="optional second schedule to compare (no-reopt baseline)")
    parser.add_argument("--size", type=int, default=2147483648, help="collective size in bytes (default: 2 GB)")
    parser.add_argument("--workload", type=Path, help="workload file (needed with --run-sim)")
    parser.add_argument("--system", type=Path, default=INPUTS / "system" / "mccl-no_failure.txt",
                        help="base system file (needed with --run-sim)")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY, help="SimAI simulator binary")
    parser.add_argument("--run-sim", action="store_true",
                        help="run the simulator with collective-graph:1 for exact flow timings")
    parser.add_argument("--graph-csv", type=Path, help="use an existing collective_graph_*.csv")
    parser.add_argument("--baseline-graph-csv", type=Path, help="graph CSV for the baseline schedule")
    parser.add_argument("--results-dir", type=Path,
                        help="E3-mccl results directory; reuse its graph CSVs instead of running the simulator")
    parser.add_argument("--out", type=Path, default=ROOT / "artifact" / "results" / "schedule_flow.gif",
                        help="output GIF (default: artifact/results/schedule_flow.gif)")
    parser.add_argument("--fps", type=int, default=12, help="animation frames per second (default: 12)")
    parser.add_argument("--frames", type=int, default=50, help="number of animation frames (default: 50)")
    parser.add_argument("--dpi", type=int, default=100, help="output resolution (default: 100)")
    parser.add_argument("--max-dots", type=int, default=80, help="max packet dots per panel and frame")
    parser.add_argument("--timeline", action="store_true",
                        help="also write a bottleneck timeline PNG (imports matplotlib)")
    parser.add_argument("--hold-ms", type=int, default=1500,
                        help="pause on the final frame before the GIF loops (default: 1500)")
    parser.add_argument("--show", action="store_true", help="also open an interactive window (needs a display)")
    args = parser.parse_args(argv)

    if args.demo:
        collective, topology, baseline, reopt, workload, system = demo_paths(args.demo)
        collective_kind = collective
        args.topology = topology
        args.schedule = reopt
        args.baseline = baseline
        args.workload = workload
        args.system = system
        args.size = 2147483648
        if not args.out.name or args.out.name == "schedule_flow.gif":
            args.out = ROOT / "artifact" / "results" / f"{args.demo}_flows.gif"
        if args.results_dir:
            collective, failure = args.demo.split("-", 1)
            wl = "ag-2147483648_bytes" if collective == "ag" else "a2a-2147483648_bytes"
            base_dir = args.results_dir / wl / failure / "mccl_noreopt"
            reopt_dir = args.results_dir / wl / failure / "mccl_reopt"

            def _find_graph(directory: Path) -> Path:
                files = sorted(directory.glob("collective_graph_*.csv"))
                if not files:
                    raise SystemExit(f"no collective_graph_*.csv under {directory}")
                return max(files, key=lambda p: p.stat().st_size)

            args.baseline_graph_csv = _find_graph(base_dir)
            args.graph_csv = _find_graph(reopt_dir)
            args.run_sim = False
        elif not args.graph_csv and args.binary.exists():
            args.run_sim = True
    else:
        collective_kind = _collective_kind(args)
        if not args.topology or not args.schedule:
            parser.error("give --topology and --schedule, or use --demo")
        if args.run_sim and not args.workload:
            parser.error("--run-sim needs --workload")
        if args.results_dir:
            parser.error("--results-dir is used together with --demo")

    for path in (args.topology, args.schedule):
        if not path.exists():
            parser.error(f"file not found: {path}")
    if args.baseline and not args.baseline.exists():
        parser.error(f"baseline schedule not found: {args.baseline}")

    scenarios: List[Scenario] = []
    try:
        if args.run_sim:
            baseline_flows = None
            if args.baseline:
                baseline_flows = run_simulator_graph(
                    args.topology, args.baseline, args.workload, args.size,
                    args.binary, Path(args.system), "baseline",
                )
            reopt_flows = run_simulator_graph(
                args.topology, args.schedule, args.workload, args.size,
                args.binary, Path(args.system), "reopt",
            )
            if args.baseline:
                scenarios.append(build_scenario("No reopt (healthy-fabric schedule)", args.topology,
                                                args.baseline, args.size, baseline_flows))
            reopt_name = "Reopt (failure-aware schedule)" if args.baseline else args.schedule.stem
            scenarios.append(build_scenario(reopt_name, args.topology,
                                            args.schedule, args.size, reopt_flows))
        elif args.graph_csv:
            flows = parse_graph_csv(args.graph_csv)
            reopt_name = "Reopt (failure-aware schedule)" if args.baseline else args.schedule.stem
            scenarios.append(build_scenario(reopt_name, args.topology, args.schedule, args.size, flows))
            if args.baseline and args.baseline_graph_csv:
                base_flows = parse_graph_csv(args.baseline_graph_csv)
                scenarios.insert(0, build_scenario("No reopt (healthy-fabric schedule)", args.topology,
                                                   args.baseline, args.size, base_flows))
        else:
            if args.baseline:
                scenarios.append(build_scenario("No reopt (healthy-fabric schedule)", args.topology,
                                                args.baseline, args.size))
            scenarios.append(build_scenario(
                "Reopt (failure-aware schedule)" if args.baseline else args.schedule.stem,
                args.topology, args.schedule, args.size,
            ))
    except Exception as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    if not scenarios:
        sys.stderr.write("no scenarios to animate\n")
        return 1

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cut_edges, cut_capacity, cut_k, optimal_ms, source_side = optimal_bottleneck_cut(
        scenarios[0].topology, collective_kind, scenarios[0].size)
    print(f"[viz] bottleneck cut: {len(cut_edges)} links, {cut_capacity:.0f} Gbps "
          f"(k={cut_k}, optimal {optimal_ms:.2f} ms)")
    animate(scenarios, out, fps=args.fps, frames=args.frames, dpi=args.dpi,
            max_dots=args.max_dots, hold_ms=args.hold_ms,
            cut_edges=cut_edges, collective=collective_kind,
            source_side=source_side)
    write_summary(scenarios, out)
    if args.timeline:
        write_timeline(scenarios, out)
    print(f"Wrote {out}")
    if args.show:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(mpimg.imread(out))
        plt.axis("off")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
