#!/usr/bin/env python3
"""Animate how an MSCCL/MCCL schedule moves packets over a SimAI topology.

The script is a tutorial: it draws the GPU/NVSwitch/ToR graph, moves one dot per
scheduled chunk along its physical path, colours every link by its current load,
and makes the bottleneck link pulse/explode.  A side-by-side mode compares a
schedule synthesized for the healthy fabric ("no reopt") with the schedule
re-synthesized for the failure ("reopt"), using the exact MCCL artifact files.

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
        series = self.loads.get(key)
        if not series:
            return 0.0
        return series[min(frame_index, len(series) - 1)]

    def utilization_at(self, key: Tuple[int, int], frame_index: int) -> float:
        cap = self.link_capacity(key)
        return self.load_at(key, frame_index) / cap if cap else 0.0

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
    starts: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    ends: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    rates: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for flow in flows:
        if not flow.link_keys:
            continue
        rate = flow.rate_gbps
        for key in flow.link_keys:
            starts[key].append(flow.start)
            ends[key].append(flow.end)
            rates[key].append(rate)

    times = np.asarray(frame_times, dtype=float)
    loads: Dict[Tuple[int, int], List[float]] = {}
    for key in starts:
        st = np.asarray(starts[key], dtype=float)
        en = np.asarray(ends[key], dtype=float)
        rt = np.asarray(rates[key], dtype=float)
        ev = np.concatenate([st, en])
        de = np.concatenate([rt, -rt])
        order = np.argsort(ev, kind="stable")
        ev = ev[order]
        prefix = np.cumsum(de[order])
        idx = np.searchsorted(ev, times, side="right") - 1
        series = np.where(idx >= 0, prefix[np.clip(idx, 0, len(prefix) - 1)], 0.0)
        loads[key] = np.maximum(series, 0.0).tolist()
    return loads

def assign_paths(flows: Iterable[Flow], topology: Topology) -> None:
    cache: Dict[Tuple[int, int], Tuple[Optional[List[int]], List[Tuple[int, int]]]] = {}
    colors = {gid: FLOW_COLORS[gid % len(FLOW_COLORS)] for gid in topology.gpu_ids}
    for flow in flows:
        key = (flow.src, flow.dst)
        if key not in cache:
            route = topology.route(flow.src, flow.dst)
            cache[key] = (route, topology.path_links(route) if route else [])
        flow.path, flow.link_keys = cache[key]
        flow.color = colors.get(flow.src, "#1f77b4")
        flow._points = []
        flow._cum = []
        flow._total_len = 0.0


def summarize_scenario(scenario: Scenario) -> Dict[str, object]:
    bytes_by_link: Dict[Tuple[int, int], int] = defaultdict(int)
    events: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    for flow in scenario.flows:
        rate = flow.rate_gbps
        for key in flow.link_keys:
            bytes_by_link[key] += flow.size
            events[key].append((flow.start, rate))
            events[key].append((flow.end, -rate))

    drain_by_link: Dict[Tuple[int, int], float] = {}
    peak_load_by_link: Dict[Tuple[int, int], float] = {}
    for key, evs in events.items():
        cap = scenario.link_capacity(key)
        if cap <= 0:
            continue
        evs.sort(key=lambda item: (item[0], item[1]))
        running = 0.0
        peak = 0.0
        for _, delta in evs:
            running += delta
            peak = max(peak, running)
        peak_load_by_link[key] = max(0.0, peak)
        drain_by_link[key] = bytes_by_link.get(key, 0) * 8.0 / cap / 1e6

    max_util = max((peak_load_by_link[k] / scenario.link_capacity(k)
                    for k in peak_load_by_link), default=0.0)
    # The bottleneck is the set of links with the highest fraction of their
    # capacity in use. 1% keeps only true ties at the top.
    bottlenecks = [key for key in peak_load_by_link
                   if max_util > 0 and peak_load_by_link[key] / scenario.link_capacity(key) >= 0.99 * max_util]
    bottlenecks.sort(key=lambda k: (-peak_load_by_link[k] / scenario.link_capacity(k),
                                    -drain_by_link[k]))
    scenario.bottlenecks = bottlenecks
    scenario.bottleneck = bottlenecks[0] if bottlenecks else None

    total_cap = sum(scenario.link_capacity(k) for k in bottlenecks)
    total_bytes = sum(bytes_by_link.get(k, 0) for k in bottlenecks)
    peak_load = sum(peak_load_by_link[k] for k in bottlenecks)
    selected_drain = max((drain_by_link[k] for k in bottlenecks), default=0.0)
    scenario.summary = {
        "name": scenario.name,
        "completion_ms": scenario.completion / 1e6,
        "bottleneck_cut": " ; ".join(scenario.link_label(k) for k in bottlenecks) or "-",
        "bottleneck_count": len(bottlenecks),
        "capacity_gbps": total_cap,
        "drain_ms": selected_drain,
        "peak_load_gbps": peak_load,
        "peak_offered_util": peak_load / total_cap if total_cap else 0.0,
        "bytes_on_bottleneck": total_bytes,
        "flows": len(scenario.flows),
    }
    return scenario.summary

def _link_style(
    scenario: Scenario,
    key: Tuple[int, int],
    frame_index: int,
    max_bw_by_kind: Dict[str, float],
) -> Tuple[str, float, float]:
    cap = scenario.link_capacity(key)
    util = min(1.0, scenario.utilization_at(key, frame_index))
    kind = _link_kind(scenario.topology, key)
    max_bw = max_bw_by_kind.get(kind, cap) or cap
    degraded = cap < 0.5 * max_bw
    if key in scenario.bottlenecks and util > 0.45:
        color = "#d62728"
        width = 3.0 + 5.5 * min(1.0, util)
    elif util > 0.85:
        color = "#ff7f0e"
        width = 2.6 + 4.0 * min(1.0, util)
    elif util > 0.55:
        color = "#f1c40f"
        width = 2.2 + 3.0 * min(1.0, util)
    elif util > 0.15:
        color = "#7fb3d5"
        width = 2.0 + 2.0 * min(1.0, util)
    else:
        if degraded:
            color = "#d62728"
        elif kind == "nvlink":
            color = "#2e7d32"
        else:
            color = "#111111"
        width = 4.0 if degraded else 2.8 if kind == "nvlink" else 2.4
    return color, width, util


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
                     top_pad: int, margin: int):
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    data_w = max(1e-6, max_x - min_x)
    data_h = max(1e-6, max_y - min_y)
    scale = min((width - 2 * margin) / data_w, (height - top_pad - 2 * margin) / data_h)
    ox = (width - data_w * scale) / 2.0
    oy = top_pad + (height - top_pad - data_h * scale) / 2.0

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


def _render_base_panel(scenario: Scenario, width: int, height: int, dpi: int):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    topo = scenario.topology
    pos = scenario.pos
    to_px, scale = _panel_transform(pos, width, height, top_pad=42, margin=34)
    max_bw = _max_bw_by_kind(topo)
    base_scale = dpi / 100.0
    font_scale = max(1.0, base_scale)

    for key, link in topo.links.items():
        x1, y1 = to_px(*pos[link.u])
        x2, y2 = to_px(*pos[link.v])
        kind = _link_kind(topo, key)
        cap = link.bw_gbps
        degraded = cap < 0.5 * max_bw.get(kind, cap)
        if degraded:
            color = "#d62728"
            lw = max(2, int(4.0 * base_scale))
        elif kind == "nvlink":
            color = "#2e7d32"
            lw = max(2, int(2.8 * base_scale))
        else:
            color = "#111111"
            lw = max(1, int(2.4 * base_scale))
        draw.line([(x1, y1), (x2, y2)], fill=_rgb(color), width=lw)
        if degraded:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            text = f"{cap:g}G"
            font = _font(8 * font_scale)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([mx - tw / 2 - 3, my - th / 2 - 2, mx + tw / 2 + 3, my + th / 2 + 2],
                           fill="white")
            draw.text((mx - tw / 2, my - th / 2 - 1), text, fill=_rgb(color), font=font)

    gpu_r = max(6, int(9 * base_scale))
    for gid in topo.gpu_ids:
        x, y = to_px(*pos[gid])
        draw.ellipse([x - gpu_r, y - gpu_r, x + gpu_r, y + gpu_r], fill=_rgb("#4c72b0"),
                     outline="black", width=1)
        font = _font(8 * font_scale)
        bbox = draw.textbbox((0, 0), str(gid), font=font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2 - 1),
                  str(gid), fill="white", font=font)

    nv_r = max(5, int(8 * base_scale))
    for sid in topo.nvswitch_ids:
        x, y = to_px(*pos[sid])
        draw.rectangle([x - nv_r, y - nv_r, x + nv_r, y + nv_r], fill=_rgb("#55a868"),
                       outline="black", width=1)
        font = _font(7 * font_scale)
        draw.text((x - nv_r, y + nv_r + 1), f"NVSwitch{sid}", fill=_rgb("#2e6b3a"), font=font)

    tor_r = max(5, int(7 * base_scale))
    for sid in topo.switch_ids:
        x, y = to_px(*pos[sid])
        draw.polygon([(x, y - tor_r), (x + tor_r, y), (x, y + tor_r), (x - tor_r, y)],
                     fill=_rgb("#c44e52"), outline="black")
        font = _font(7 * font_scale)
        draw.text((x - tor_r, y + tor_r + 1), f"ToR{sid}", fill=_rgb("#7b2d2f"), font=font)

    title_font = _font(12 * font_scale)
    bbox = draw.textbbox((0, 0), scenario.name, font=title_font)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, 10), scenario.name,
              fill="black", font=title_font)
    return img, to_px, scale


def _render_frame_panel(base_img, scenario: Scenario, to_px, scale: float, frame_index: int,
                        frame_times: Sequence[float], max_bw: Dict[str, float],
                        max_dots: int, rng: random.Random, dpi: int):
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    topo = scenario.topology
    pos = scenario.pos
    global_t = frame_times[frame_index]
    t = min(global_t, scenario.completion)
    done = global_t > scenario.completion + 1e-9
    base_scale = dpi / 100.0
    font_scale = max(1.0, base_scale)

    for key, link in topo.links.items():
        color, width, util = _link_style(scenario, key, frame_index, max_bw)
        x1, y1 = to_px(*pos[link.u])
        x2, y2 = to_px(*pos[link.v])
        lw = max(1, int(width * base_scale))
        draw.line([(x1, y1), (x2, y2)], fill=_rgb(color), width=lw)

    active = scenario.active_flows(t)
    active_count = len(active)
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

    dot_r = max(1, int(2.5 * base_scale))
    for flow in active:
        frac = (t - flow.start) / flow.duration
        for offset in (0.0, 0.10, 0.20):
            f = frac - offset
            if f < 0:
                continue
            x, y = flow.point_at(f, pos)
            px, py = to_px(x, y)
            draw.ellipse([px - dot_r, py - dot_r, px + dot_r, py + dot_r], fill=_rgb(flow.color))

    show_bn = False
    bn_positions = []
    if scenario.bottlenecks:
        phase = (frame_index % 18) / 18.0
        for key in scenario.bottlenecks:
            link = topo.links[key]
            x1, y1 = to_px(*pos[link.u])
            x2, y2 = to_px(*pos[link.v])
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            util = min(1.0, scenario.utilization_at(key, frame_index))
            if util <= 0.45:
                continue
            show_bn = True
            bn_positions.append((mx, my))
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length
            cut = 22 * base_scale
            _draw_dashed(draw, (mx - nx * cut, my - ny * cut), (mx + nx * cut, my + ny * cut),
                         _rgb("#d62728"), max(2, int(3 * base_scale)))
            for k in range(3):
                radius = (0.12 + 0.30 * ((phase + k / 3.0) % 1.0)) * scale
                alpha = 0.65 * (1.0 - ((phase + k / 3.0) % 1.0))
                if radius > 1:
                    col = tuple(int(255 - (255 - c) * alpha) for c in _rgb("#d62728"))
                    draw.ellipse([mx - radius, my - radius, mx + radius, my + radius],
                                 outline=col, width=max(1, int(2 * base_scale)))
            star_r = (12 + 10 * util) * base_scale
            points = []
            for i in range(10):
                ang = -math.pi / 2 + i * math.pi / 5
                rr = star_r if i % 2 == 0 else star_r * 0.45
                points.append((mx + math.cos(ang) * rr, my + math.sin(ang) * rr))
            draw.polygon(points, fill=_rgb("#d62728"), outline="white")

    if show_bn and bn_positions:
        cx = sum(p[0] for p in bn_positions) / len(bn_positions)
        cy = sum(p[1] for p in bn_positions) / len(bn_positions)
        font = _font(9 * font_scale)
        text = f"BOTTLENECK CUT ({len(bn_positions)} links)"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = cy - 22 * base_scale - th
        draw.rectangle([cx - tw / 2 - 4, ty - 3, cx + tw / 2 + 4, ty + th + 3],
                       fill="white", outline=_rgb("#d62728"))
        draw.text((cx - tw / 2, ty), text, fill=_rgb("#a00000"), font=font)

    stats = (
        f"t = {t / 1e6:7.2f} ms\n"
        f"active chunks: {active_count}\n"
        f"completion: {scenario.completion / 1e6:.2f} ms\n"
        f"status: {'DONE' if done else 'running'}"
    )
    if scenario.bottlenecks:
        load = sum(scenario.load_at(k, frame_index) for k in scenario.bottlenecks)
        cap = sum(scenario.link_capacity(k) for k in scenario.bottlenecks)
        offered = load / cap * 100.0 if cap else 0.0
        state = "SATURATED" if load > cap else "flowing"
        stats += (
            f"\nbottleneck cut: {len(scenario.bottlenecks)} links\n"
            f"aggregate: {load:6.1f} / {cap:.0f} Gbps ({state}, {offered:5.1f}% offered)"
        )
    font = _font(9 * font_scale)
    bbox = draw.multiline_textbbox((0, 0), stats, font=font, spacing=2)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([8, 8, 8 + tw + 10, 8 + th + 10], fill="white", outline=_rgb("#999999"))
    draw.multiline_text((13, 11), stats, fill="black", font=font, spacing=2)
    return img


def _animate_pillow(scenarios: Sequence[Scenario], out_path: Path, fps: int, frames: int,
                    dpi: int, max_dots: int, hold_ms: int = 1500) -> None:
    duration = max((s.completion for s in scenarios), default=1.0)
    frame_times = [duration * i / max(1, frames - 1) for i in range(frames)]
    for scenario in scenarios:
        scenario.frame_times = frame_times
        scenario.loads = build_link_loads(scenario.flows, scenario.topology, frame_times)
        summarize_scenario(scenario)

    scale = dpi / 100.0
    panel_w = int(1100 * scale)
    panel_h = int(800 * scale)
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
            max_dots: int = 110, hold_ms: int = 1500) -> None:
    _animate_pillow(scenarios, out_path, fps, frames, dpi, max_dots, hold_ms)

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
        "| scenario | completion (ms) | AlgBW (GB/s) | speedup |",
        "|---|---:|---:|---:|",
    ]
    for row, scenario in zip(rows, scenarios):
        completion_ms = float(row["completion_ms"])
        algbw = scenario.size / scenario.completion if scenario.completion > 0 else 0.0
        speedup = f"{base_ms / completion_ms:.2f}x" if base_ms else "-"
        lines.append(
            f"| {row['name']} | {completion_ms:.3f} | {algbw:.1f} | {speedup} |"
        )
    if len(rows) >= 2 and base_ms:
        base, reopt = rows[0], rows[1]
        lines += [
            "",
            f"Speedup of **{reopt['name']}** over **{base['name']}**: "
            f"**{base_ms / reopt['completion_ms']:.2f}x** "
            f"({base_ms:.3f} ms -> {reopt['completion_ms']:.3f} ms).",
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
    parser.add_argument("--fps", type=int, default=10, help="animation frames per second (default: 10)")
    parser.add_argument("--frames", type=int, default=40, help="number of animation frames (default: 40)")
    parser.add_argument("--dpi", type=int, default=80, help="output resolution (default: 80)")
    parser.add_argument("--max-dots", type=int, default=60, help="max packet dots per panel and frame")
    parser.add_argument("--timeline", action="store_true",
                        help="also write a bottleneck timeline PNG (imports matplotlib)")
    parser.add_argument("--hold-ms", type=int, default=1500,
                        help="pause on the final frame before the GIF loops (default: 1500)")
    parser.add_argument("--show", action="store_true", help="also open an interactive window (needs a display)")
    args = parser.parse_args(argv)

    if args.demo:
        collective, topology, baseline, reopt, workload, system = demo_paths(args.demo)
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
    animate(scenarios, out, fps=args.fps, frames=args.frames, dpi=args.dpi,
            max_dots=args.max_dots, hold_ms=args.hold_ms)
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
