#!/usr/bin/env python3
"""Parse SimAI workload records and analytical EndToEnd CSV outputs."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHASES = (
    ("forward", 3, 4),
    ("backward_input_gradient", 6, 7),
    ("optimizer_weight_gradient", 9, 10),
)


def parse_header(line: str) -> dict[str, int]:
    return {k: int(v) for k, v in re.findall(
        r"(model_parallel_NPU_group|ep|pp|vpp|ga|all_gpus|pp_comm):\s*(\d+)", line
    )}


def parse_workload(path: Path, scenario: str) -> tuple[dict, list[dict]]:
    lines = path.read_text().splitlines()
    header = parse_header(lines[0])
    count = int(lines[1])
    records = [line.split() for line in lines[2:2 + count]]
    events = []

    # Forward traversal is ascending; the two backward phases traverse layers
    # in reverse. This is event order inferred from Workload.cc, not timestamps.
    order = 0
    for phase, op_i, bytes_i in PHASES:
        phase_records = records if phase == "forward" else list(reversed(records))
        for record_index, record in enumerate(phase_records):
            operation = record[op_i].upper()
            payload = int(record[bytes_i])
            if operation == "NONE" or payload == 0:
                continue
            events.append({
                "scenario": scenario, "event_order": order,
                "layer": record[0], "phase": phase,
                "operation": operation, "payload_bytes": payload,
                "source": "validated SimAI workload record",
                "has_simulator_timestamp": False,
            })
            order += 1
    return header, events


def parse_end_to_end(path: Path) -> tuple[dict, dict[tuple[str, str], float]]:
    if not path.exists():
        return {"available": False}, {}
    lines = path.read_text().splitlines()
    result = {"available": True, "path": str(path.relative_to(ROOT))}
    if len(lines) >= 2 and lines[0].startswith("File name"):
        keys = [x.strip() for x in lines[0].split(",")]
        values = [x.strip() for x in lines[1].split(",")]
        result["aggregate_report"] = dict(zip(keys, values))
    phase_times = {}
    header_i = next((i for i, line in enumerate(lines) if line.startswith("layer_name,")), None)
    if header_i is not None:
        reader = csv.DictReader(lines[header_i:])
        for row in reader:
            layer = row.get("layer_name", "")
            if not layer or layer in {"SUM", "total exposed comm"}:
                continue
            for phase, column in (
                ("forward", "fwd total comm"),
                ("optimizer_weight_gradient", "wg total comm"),
                ("backward_input_gradient", "ig total comm"),
            ):
                try:
                    phase_times[(layer, phase)] = float(row[column])
                except (KeyError, TypeError, ValueError):
                    pass
    return result, phase_times


def main() -> None:
    parsed_dir = ROOT / "parsed"
    parsed_dir.mkdir(exist_ok=True)
    all_events = []
    validation = {}
    for scenario in ("dp", "tp", "pp"):
        header, events = parse_workload(ROOT / "workloads" / f"{scenario}.txt", scenario.upper())
        e2e, phase_times = parse_end_to_end(
            ROOT / "results" / scenario / "Analytical_test_EndToEnd.csv"
        )
        for event in events:
            event["estimated_comm_time_us"] = phase_times.get((event["layer"], event["phase"]))
        all_events.extend(events)
        aggregate = e2e.get("aggregate_report", {})
        pp_match = re.match(r"[-+0-9.]+", aggregate.get("Expose_PP_comm", ""))
        validation[scenario.upper()] = {
            "header": header,
            "event_count": len(events),
            "operation_counts": dict(Counter(e["operation"] for e in events)),
            "logical_payload_bytes": sum(e["payload_bytes"] for e in events),
            "event_estimated_comm_time_us": sum(
                e["estimated_comm_time_us"] or 0 for e in events
            ),
            "pp_aggregate_estimate_us": float(pp_match.group()) if pp_match else 0.0,
            "ep_operations": sum("EP" in e["operation"] for e in events),
            "analytical_output": e2e,
            "pp_event_support": False if scenario == "pp" else None,
        }

    fields = [
        "scenario", "event_order", "layer", "phase", "operation",
        "payload_bytes", "source", "has_simulator_timestamp",
        "estimated_comm_time_us",
    ]
    with (parsed_dir / "events.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_events)

    # SimAI-Analytical does not emit PP send/receive events.  Export a separate,
    # transparent reference fingerprint derived only from the workload header:
    # (PP - 1) boundaries x GA microbatches x forward/backward directions.
    pp_header = validation["PP"]["header"]
    pp_reference = []
    reference_index = 0
    for microbatch in range(1, pp_header["ga"] + 1):
        for source_stage in range(pp_header["pp"] - 1):
            pp_reference.append({
                "reference_index": reference_index,
                "microbatch": microbatch,
                "phase": "forward_activation",
                "source_stage": source_stage + 1,
                "destination_stage": source_stage + 2,
                "payload_bytes": pp_header["pp_comm"],
                "source": "analytical reference derived from PP header; not a backend event",
            })
            reference_index += 1
        for source_stage in range(pp_header["pp"] - 1, 0, -1):
            pp_reference.append({
                "reference_index": reference_index,
                "microbatch": microbatch,
                "phase": "backward_gradient",
                "source_stage": source_stage + 1,
                "destination_stage": source_stage,
                "payload_bytes": pp_header["pp_comm"],
                "source": "analytical reference derived from PP header; not a backend event",
            })
            reference_index += 1
    reference_fields = [
        "reference_index", "microbatch", "phase", "source_stage",
        "destination_stage", "payload_bytes", "source",
    ]
    with (parsed_dir / "pp_analytical_reference.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reference_fields)
        writer.writeheader()
        writer.writerows(pp_reference)
    validation["PP"]["analytical_reference"] = {
        "event_count": len(pp_reference),
        "logical_payload_bytes": sum(row["payload_bytes"] for row in pp_reference),
        "definition": "(PP - 1) boundaries x GA microbatches x 2 directions",
        "is_backend_output": False,
    }
    (parsed_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    main()
