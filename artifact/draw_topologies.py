#!/usr/bin/env python3
"""Draw the MCCL artifact topologies, including the three gray-failure profiles.

One large PNG is written per profile (healthy, Fail1, Fail2, Fail3) so each can
be shown full-screen.  The panels use the same visual
language as the packet animations: green intra-node NVLink, dark inter-node ToR
links, magenta dashed degraded links with a FAIL badge, and readable white
labels.

Usage:
    python3 artifact/draw_topologies.py
    python3 artifact/draw_topologies.py --outdir artifact/results
    python3 artifact/draw_topologies.py --family all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

import visualize_flows as vf


ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "astra-sim-alibabacloud" / "inputs"

PROFILES = [
    ("no_failure", "Healthy fabric", "all links at peak capacity"),
    ("failure_1", "Fail1: NVSwitch lanes", "GPU0 <-> NVSwitch 2880 -> 800 Gbps"),
    ("failure_2", "Fail2: NIC / ToR link", "GPU0 <-> ToR 400 -> 50 Gbps"),
    ("failure_3", "Fail3: two degraded networks",
     "GPU0: dedicated intra-node network 50 Gbps; shared inter-node network 100 Gbps (+1 us)"),
]

PANEL_W = 1600
PANEL_H = 1150
# Larger than the animation panels so the same relative font size reads bigger
# on a full-screen topology slide.
DPI = 150


def topology_paths(family: str) -> List[Tuple[str, str, str, Path]]:
    if family == "1x2x8":
        base = INPUTS / "topo" / "single_collective"
        return [(key, title, subtitle, base / f"1x2x8{'' if key == 'no_failure' else '-' + key}")
                for key, title, subtitle in PROFILES]
    base = INPUTS / "topo" / "end_to_end" / "HPN"
    return [(key, title, subtitle, base / f"1x16x8{'' if key == 'no_failure' else '-' + key}")
            for key, title, subtitle in PROFILES]


def draw_legend(draw: ImageDraw.ImageDraw, x: float, y: float, font_scale: float,
                two_network: bool = False) -> None:
    # Same disposition as the animation legend: link rows first, then the
    # node/marker rows, so the shared labels sit in the same place.
    if two_network:
        rows = [
            ("line", vf.COLOR_INTRA, "intra-node / NVLink"),
            ("line", vf.COLOR_INTER, "inter-node / to ToR"),
            ("hex", vf.COLOR_INTRA_NET, "intra-node network switch"),
            ("line", vf.COLOR_INTRA_NET, "dedicated intra-node fabric"),
            ("dash", vf.COLOR_FAIL, "inter-node network"),
            ("dot", "#4c72b0", "GPU"),
            ("square", "#55a868", "NVSwitch"),
            ("diamond", "#c44e52", "ToR / leaf switch"),
        ]
        per_col = 4
    else:
        rows = [
            ("line", vf.COLOR_INTRA, "intra-node / NVLink"),
            ("line", vf.COLOR_INTER, "inter-node / to ToR"),
            ("dash", vf.COLOR_FAIL, "degraded link"),
            ("dot", "#4c72b0", "GPU"),
            ("square", "#55a868", "NVSwitch"),
            ("diamond", "#c44e52", "ToR / leaf switch"),
        ]
        per_col = 3
    vf._draw_legend_rows(draw, x, y, font_scale, rows, per_col)


def render_panel(topo: vf.Topology, title: str, subtitle: str,
                 width: int = PANEL_W, height: int = PANEL_H, dpi: int = DPI) -> Image.Image:
    scenario = vf.Scenario(name=title, topology=topo, flows=[], pos=topo.layout(), size=0)
    vf._apply_remote_layout(scenario)
    img, _, _ = vf._render_base_panel(
        scenario, width, height, dpi,
        draw_legend=False, draw_progress=False, subtitle=subtitle,
        top_pad=120, bottom_pad=340, margin=80, margin_x=340,
        clamp_badges=True, switch_labels_outside=True,
        failure_labels_outside=True,
    )
    draw = ImageDraw.Draw(img)
    base_scale = dpi / 100.0
    font_scale = max(1.0, base_scale) * 1.7
    max_bw = vf._max_bw_by_kind(topo)
    degraded_kinds = {
        vf._link_kind(topo, key) for key, link in topo.links.items()
        if link.bw_gbps < 0.5 * max_bw.get(vf._link_kind(topo, key), link.bw_gbps)
    }
    two_network = "nvlink" in degraded_kinds and "nic" in degraded_kinds
    # The topology slide is viewed on its own, so give its legend the same
    # disposition as the animations but a larger relative font.
    legend_font_scale = font_scale * 1.3
    per_col = 4 if two_network else 3
    _, legend_h = vf._legend_size(legend_font_scale, per_col)
    draw_legend(draw, 50, height - legend_h - int(30 * base_scale),
                legend_font_scale, two_network=two_network)
    return img


def draw_family(family: str, outdir: Path) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for key, title, subtitle, path in topology_paths(family):
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        img = render_panel(vf.Topology.parse(path), title, subtitle)
        out = outdir / f"topology_{family}_{key}.png"
        img.save(out)
        written.append(out)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifact" / "results",
                        help="directory for the PNG files (default: artifact/results)")
    parser.add_argument("--family", choices=["1x2x8", "1x16x8", "all"], default="1x2x8",
                        help="which topology family to draw (default: 1x2x8)")
    args = parser.parse_args()

    families = ["1x2x8", "1x16x8"] if args.family == "all" else [args.family]
    for family in families:
        for out in draw_family(family, args.outdir):
            print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
