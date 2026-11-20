#!/usr/bin/env python3
"""Print the Experiment A metadata as concise, presentation-friendly tables."""

from __future__ import annotations

import json
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def line(parts: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(parts))

    print(line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(line(row))


def mib(byte_count: int) -> str:
    return f"{byte_count / (1024**2):.2f} MiB"


def main() -> None:
    metadata_path = EXP_DIR / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(
            "metadata.json is missing. Run generate_workloads.py first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = metadata["model"]
    scenarios = metadata["scenarios"]
    payloads = metadata["reference_bytes"]

    print("\nMODEL PROFILE — GPT-2 SMALL\n")
    print_table(
        ["Parameter", "Value"],
        [
            ["Transformer layers", f'{model["layers"]:,}'],
            ["Hidden size", f'{model["hidden_size"]:,}'],
            ["Attention heads", f'{model["attention_heads"]:,}'],
            ["Sequence length", f'{model["sequence_length"]:,} tokens'],
            ["Vocabulary size", f'{model["vocabulary_size"]:,}'],
            ["Parameters", f'{model["parameter_count"]:,} (124.44M)'],
            ["Communicated value", f'{model["bytes_per_value"]} bytes (BF16 assumption)'],
            ["Global batch", f'{metadata["global_batch_sequences"]} sequences'],
        ],
    )

    print("\nPARALLEL CONFIGURATIONS\n")
    scenario_rows = []
    for name in ("DP", "TP", "PP"):
        scenario = scenarios[name]
        computed_global_batch = (
            scenario["micro_batch"]
            * scenario["gradient_accumulation"]
            * scenario["dp"]
        )
        scenario_rows.append(
            [
                name,
                str(scenario["dp"]),
                str(scenario["tp"]),
                str(scenario["pp"]),
                str(scenario["micro_batch"]),
                str(scenario["gradient_accumulation"]),
                str(computed_global_batch),
            ]
        )
    print_table(
        ["Case", "DP", "TP", "PP", "Microbatch", "Accumulation", "Global batch"],
        scenario_rows,
    )

    print("\nCOMMUNICATION PAYLOADS\n")
    print_table(
        ["Case", "Logical payload", "Frequency per iteration"],
        [
            ["DP", f'{payloads["full_model_gradient"]:,} B ({mib(payloads["full_model_gradient"])})', "1 AllReduce"],
            ["TP", f'{payloads["activation_batch_32"]:,} B ({mib(payloads["activation_batch_32"])})', "48 AllReduces"],
            ["PP", f'{payloads["pp_activation_microbatch_8"]:,} B ({mib(payloads["pp_activation_microbatch_8"])})', "24 P2P transfers"],
        ],
    )

    print(
        "\nNote: DP communicates BF16 gradients; PP uses four microbatches and a 1F1B schedule."
    )


if __name__ == "__main__":
    main()
