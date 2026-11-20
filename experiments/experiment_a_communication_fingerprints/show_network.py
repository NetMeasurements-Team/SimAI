#!/usr/bin/env python3
"""Print the active SimAI network configuration as a concise table."""

from __future__ import annotations

import re
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent


def option(script: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}\s+(\S+)", script, re.MULTILINE)
    if not match:
        raise SystemExit(f"Could not find {name} in run_simulations.sh")
    return match.group(1)


def system_value(system: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.+)$", system, re.MULTILINE)
    if not match:
        raise SystemExit(f"Could not find {name} in system.txt")
    return match.group(1).strip()


def print_table(rows: list[tuple[str, str]]) -> None:
    left_width = max(len("Setting"), *(len(label) for label, _ in rows))
    right_width = max(len("Configured value"), *(len(value) for _, value in rows))
    print(f'{"Setting".ljust(left_width)} | {"Configured value"}')
    print(f'{"-" * left_width}-+-{"-" * right_width}')
    for label, value in rows:
        print(f"{label.ljust(left_width)} | {value}")


def main() -> None:
    run_script = (EXP_DIR / "run_simulations.sh").read_text(encoding="utf-8")
    system = (EXP_DIR / "system.txt").read_text(encoding="utf-8")

    rows = [
        ("GPUs", option(run_script, "-g")),
        ("GPUs per server", option(run_script, "-g_p_s")),
        ("Intra-server bandwidth", f'{option(run_script, "-nv")} GB/s'),
        ("NIC bandwidth", f'{option(run_script, "-nic")} GB/s'),
        ("GPU model", option(run_script, "-g_type")),
        ("AllReduce model", system_value(system, "all-reduce-implementation")),
        ("Endpoint delay", system_value(system, "endpoint-delay")),
    ]

    print("\nSIMAI NETWORK CONFIGURATION\n")
    print_table(rows)
    print("\nSources: run_simulations.sh and system.txt")


if __name__ == "__main__":
    main()
