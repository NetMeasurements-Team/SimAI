#!/usr/bin/env python3
"""Draw the MCCL artifact topologies, including the three gray-failure profiles.

The figures are meant to be used at the start of the tutorial: they show the
healthy rail-optimized fabric and then the same fabric after Fail1, Fail2 and
Fail3, with the degraded links highlighted and labelled.

Usage:
    python3 artifact/draw_topologies.py
    python3 artifact/draw_topologies.py --outdir artifact/results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import visualize_flows as vf


ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "astra-sim-alibabacloud" / "inputs"

PROFILES = [
    ("no_failure", "Healthy fabric", "all links at peak capacity"),
    ("failure_1", "Fail1: NVSwitch lanes", "GPU0 <-> NVSwitch 2880 -> 800 Gbps"),
    ("failure_2", "Fail2: NIC / ToR link", "GPU0 <-> ToR 400 -> 50 Gbps"),
    ("failure_3", "Fail3: GPU replacement", "GPU0 <-> NVSwitch 50 Gbps; GPU0 <-> ToR 100 Gbps; +1 us ToR latency"),
]


def topology_paths(family: str) -> List[Tuple[str, str, str, Path]]:
    if family == "1x2x8":
        base = INPUTS / "topo" / "single_collective"
        return [(key, title, subtitle, base / f"1x2x8{'' if key == 'no_failure' else '-' + key}")
                for key, title, subtitle in PROFILES]
    base = INPUTS / "topo" / "end_to_end" / "HPN"
    return [(key, title, subtitle, base / f"1x16x8{'' if key == 'no_failure' else '-' + key}")
            for key, title, subtitle in PROFILES]


def draw_panel(ax, topo: vf.Topology, title: str, subtitle: str) -> None:
    pos = topo.layout()
    max_bw = vf._max_bw_by_kind(topo)
    degraded: List[Tuple[Tuple[int, int], float, float]] = []

    for key, link in topo.links.items():
        x1, y1 = pos[link.u]
        x2, y2 = pos[link.v]
        kind = vf._link_kind(topo, key)
        cap = link.bw_gbps
        max_kind = max_bw.get(kind, cap) or cap
        is_degraded = cap < 0.5 * max_kind
        if is_degraded:
            color = "#d62728"
            width = 4.0
            degraded.append((key, cap, max_kind))
        elif kind == "nvlink":
            color = "#2e7d32"
            width = 2.8
        else:
            color = "#111111"
            width = 2.4
        ax.plot([x1, x2], [y1, y2], color=color, lw=width, alpha=0.92 if is_degraded else 0.75, zorder=1)

    for key, cap, max_kind in degraded:
        link = topo.links[key]
        x1, y1 = pos[link.u]
        x2, y2 = pos[link.v]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax.text(mx, my, f"{cap:g} Gbps\n(was {max_kind:g})", ha="center", va="center",
                fontsize=6.0, color="#8b0000", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#c0392b", alpha=0.95), zorder=8)

    for gid in topo.gpu_ids:
        x, y = pos[gid]
        ax.scatter([x], [y], s=300, c="#4c72b0", edgecolors="black", linewidths=0.6, zorder=4)
        ax.text(x, y, str(gid), ha="center", va="center", fontsize=7, color="white", zorder=5)
    for sid in topo.nvswitch_ids:
        x, y = pos[sid]
        ax.scatter([x], [y], s=210, marker="s", c="#55a868", edgecolors="black", linewidths=0.6, zorder=4)
        ax.text(x, y - 0.25, f"NVSwitch{sid}", ha="center", va="top", fontsize=6, color="#2e6b3a")
    for sid in topo.switch_ids:
        x, y = pos[sid]
        ax.scatter([x], [y], s=170, marker="D", c="#c44e52", edgecolors="black", linewidths=0.6, zorder=4)
        ax.text(x, y - 0.25, f"ToR{sid}", ha="center", va="top", fontsize=6, color="#7b2d2f")

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.set_title(f"{title}\n{subtitle}", fontsize=9, fontweight="bold")


def draw_family(family: str, outdir: Path) -> Path:
    rows = topology_paths(family)
    fig, axes = plt.subplots(2, 2, figsize=(18, 13) if family == "1x2x8" else (22, 22))
    axes = axes.ravel()
    for ax, (_, title, subtitle, path) in zip(axes, rows):
        if not path.exists():
            ax.axis("off")
            ax.set_title(f"missing: {path.name}")
            continue
        draw_panel(ax, vf.Topology.parse(path), title, subtitle)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c72b0", markeredgecolor="black",
               markersize=10, label="GPU"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#55a868", markeredgecolor="black",
               markersize=10, label="NVSwitch"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#c44e52", markeredgecolor="black",
               markersize=9, label="ToR / leaf switch"),
        Line2D([0], [0], color="#2e7d32", lw=3.0, label="intra-node / NVLink (2880 Gbps)"),
        Line2D([0], [0], color="#111111", lw=3.0, label="inter-node / Ethernet to ToR (400 Gbps)"),
        Line2D([0], [0], color="#d62728", lw=3.5, label="degraded link (Fail1 / Fail2 / Fail3)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(
        f"MCCL rail-optimized topology: {family}  (healthy fabric and the three gray-failure profiles)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"topology_{family}.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifact" / "results",
                        help="directory for the PNG files (default: artifact/results)")
    parser.add_argument("--family", choices=["1x2x8", "1x16x8", "all"], default="1x2x8",
                        help="which topology family to draw (default: 1x2x8)")
    args = parser.parse_args()

    families = ["1x2x8", "1x16x8"] if args.family == "all" else [args.family]
    for family in families:
        out = draw_family(family, args.outdir)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
