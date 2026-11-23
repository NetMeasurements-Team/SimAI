#!/usr/bin/env python3
"""Draw the MCCL artifact topologies, including the three gray-failure profiles.

One large PNG is written per profile (healthy, Fail1, Fail2, Fail3) so each can
be shown full-screen during the tutorial.  The panels use the same visual
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
    ("failure_3", "Fail3: GPU replacement",
     "GPU0 <-> NVSwitch 50 Gbps; GPU0 <-> ToR 100 Gbps; +1 us ToR latency"),
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


def draw_legend(draw: ImageDraw.ImageDraw, x: float, y: float, font_scale: float) -> None:
    small = vf._font(int(12 * font_scale))
    rows = [
        ("dot", "#4c72b0", "GPU"),
        ("square", "#55a868", "NVSwitch"),
        ("diamond", "#c44e52", "ToR / leaf switch"),
        ("line", vf.COLOR_INTRA, "intra-node / NVLink"),
        ("line", vf.COLOR_INTER, "inter-node / to ToR"),
        ("dash", vf.COLOR_FAIL, "degraded link"),
    ]
    per_col = 3
    row_h = int(20 * font_scale)
    col_w = int(250 * font_scale)
    width = col_w * 2
    height = row_h * per_col + int(12 * font_scale)
    draw.rectangle([x, y, x + width, y + height], fill="#ffffff", outline="#b0bec5")
    for i, (kind, color, label) in enumerate(rows):
        col, row = divmod(i, per_col)
        cx0 = x + col * col_w + int(12 * font_scale)
        cy = y + int(10 * font_scale) + row * row_h + row_h // 2
        sx0, sx1 = cx0, cx0 + int(34 * font_scale)
        if kind == "line":
            draw.line([(sx0, cy), (sx1, cy)], fill=vf._rgb(color), width=max(2, int(4 * font_scale)))
        elif kind == "dash":
            vf._draw_dashed(draw, (sx0, cy), (sx1, cy), vf._rgb(color),
                            max(2, int(4 * font_scale)),
                            dash=max(5, int(8 * font_scale)), gap=max(3, int(5 * font_scale)))
        elif kind == "dot":
            r = int(7 * font_scale)
            draw.ellipse([sx0, cy - r, sx0 + 2 * r, cy + r], fill=vf._rgb(color),
                         outline="white", width=2)
        elif kind == "square":
            r = int(6 * font_scale)
            draw.rectangle([sx0, cy - r, sx0 + 2 * r, cy + r], fill=vf._rgb(color),
                           outline="white", width=2)
        elif kind == "diamond":
            r = int(7 * font_scale)
            mx = sx0 + r
            draw.polygon([(mx, cy - r), (mx + r, cy), (mx, cy + r), (mx - r, cy)],
                         fill=vf._rgb(color), outline="white")
        draw.text((sx1 + int(10 * font_scale), cy - int(7 * font_scale)), label,
                  fill=vf._rgb(vf.COLOR_TEXT), font=small)


def render_panel(topo: vf.Topology, title: str, subtitle: str,
                 width: int = PANEL_W, height: int = PANEL_H, dpi: int = DPI) -> Image.Image:
    scenario = vf.Scenario(name=title, topology=topo, flows=[], pos=topo.layout(), size=0)
    img, _, _ = vf._render_base_panel(
        scenario, width, height, dpi,
        draw_legend=False, draw_progress=False, subtitle=subtitle,
        top_pad=120, bottom_pad=240, margin=80, margin_x=340,
        clamp_badges=True, switch_labels_outside=True,
        failure_labels_outside=True,
    )
    draw = ImageDraw.Draw(img)
    base_scale = dpi / 100.0
    font_scale = max(1.0, base_scale) * 1.7
    legend_h = int(20 * font_scale) * 3 + int(12 * font_scale)
    draw_legend(draw, 50, height - legend_h - int(30 * base_scale), font_scale)
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
