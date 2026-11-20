#!/usr/bin/env python3
"""Generate the three small, explicit workloads used by Experiment A.

The pinned AICB generator was audited but is not used directly here because its
current non-AIOB Megatron path emits TP collectives even when TP=1 and emits
distributed-optimizer collectives for the nominal DP case.  These workloads
use the same documented SimAI text format while keeping the introductory
parallelism cases pure and inspectable.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKLOADS = ROOT / "workloads"

LAYERS = 12
HIDDEN = 768
HEADS = 12
SEQ = 1024
VOCAB = 50257
GLOBAL_BATCH = 32
BYTES_PER_VALUE = 2


def gpt2_parameter_count() -> int:
    """GPT-2-like count with tied token/output embeddings and learned positions."""
    embeddings = VOCAB * HIDDEN + SEQ * HIDDEN
    attention = HIDDEN * (3 * HIDDEN) + 3 * HIDDEN + HIDDEN * HIDDEN + HIDDEN
    mlp = HIDDEN * (4 * HIDDEN) + 4 * HIDDEN + (4 * HIDDEN) * HIDDEN + HIDDEN
    layer_norms = 4 * HIDDEN  # two LNs, each with weight and bias
    final_layer_norm = 2 * HIDDEN
    return embeddings + LAYERS * (attention + mlp + layer_norms) + final_layer_norm


def row(name: str, fwd_op: str = "NONE", fwd_bytes: int = 0,
        bwd_op: str = "NONE", bwd_bytes: int = 0,
        dp_op: str = "NONE", dp_bytes: int = 0) -> str:
    # name, dependency, then (compute, op, bytes) for forward, input-gradient,
    # and weight-gradient/data-parallel phases, followed by process_time.
    return "\t".join(map(str, [
        name, -1, 1, fwd_op, fwd_bytes, 1, bwd_op, bwd_bytes,
        1, dp_op, dp_bytes, 100,
    ]))


def write(name: str, header: str, rows: list[str]) -> None:
    (WORKLOADS / name).write_text(header + "\n" + str(len(rows)) + "\n" + "\n".join(rows) + "\n")


def main() -> None:
    WORKLOADS.mkdir(parents=True, exist_ok=True)
    params = gpt2_parameter_count()
    gradient_bytes = params * BYTES_PER_VALUE
    activation_b32 = GLOBAL_BATCH * SEQ * HIDDEN * BYTES_PER_VALUE
    activation_micro_b8 = 8 * SEQ * HIDDEN * BYTES_PER_VALUE

    common = "ep: 1 checkpoints: 0 checkpoint_initiates: 0"

    # DP=4: one logical optimizer-boundary gradient synchronization.
    dp_header = (
        f"HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 1 {common} "
        "pp: 1 vpp: 12 ga: 1 all_gpus: 4 pp_comm: 0"
    )
    write("dp.txt", dp_header, [
        row("optimizer_gradient_sync", dp_op="ALLREDUCE", dp_bytes=gradient_bytes)
    ])

    # TP=4: textbook baseline, two forward and two backward AllReduces/layer.
    tp_header = (
        f"HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 4 {common} "
        "pp: 1 vpp: 12 ga: 1 all_gpus: 4 pp_comm: 0"
    )
    tp_rows = []
    for layer in range(1, LAYERS + 1):
        for site in ("attention", "mlp"):
            tp_rows.append(row(
                f"layer_{layer:02d}_{site}",
                fwd_op="ALLREDUCE", fwd_bytes=activation_b32,
                bwd_op="ALLREDUCE", bwd_bytes=activation_b32,
            ))
    write("tp.txt", tp_header, tp_rows)

    # PP=4: AICB's header-level representation of a three-layer stage and four
    # microbatches. SimAI-Analytical reads pp_comm but does not emit P2P events.
    pp_header = (
        f"HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 1 {common} "
        f"pp: 4 vpp: 3 ga: 4 all_gpus: 4 pp_comm: {activation_micro_b8}"
    )
    write("pp.txt", pp_header, [row(f"stage_layer_{i:02d}") for i in range(1, 4)])

    metadata = {
        "model": {
            "layers": LAYERS, "hidden_size": HIDDEN, "attention_heads": HEADS,
            "sequence_length": SEQ, "vocabulary_size": VOCAB,
            "parameter_count": params, "bytes_per_value": BYTES_PER_VALUE,
            "parameter_count_note": "124.44M with tied token/output embeddings; 2.78% below the 128M pedagogical reference",
        },
        "global_batch_sequences": GLOBAL_BATCH,
        "reference_bytes": {
            "full_model_gradient": gradient_bytes,
            "activation_batch_32": activation_b32,
            "pp_activation_microbatch_8": activation_micro_b8,
        },
        "scenarios": {
            "DP": {"dp": 4, "tp": 1, "pp": 1, "micro_batch": 8, "gradient_accumulation": 1},
            "TP": {"dp": 1, "tp": 4, "pp": 1, "micro_batch": 32, "gradient_accumulation": 1},
            "PP": {"dp": 1, "tp": 1, "pp": 4, "micro_batch": 8, "gradient_accumulation": 4},
        },
        "generation": {
            "format": "SimAI HYBRID_TRANSFORMER_FWD_IN_BCKWD text format",
            "aicb_direct_generation": False,
            "reason": "Pinned AICB non-AIOB path contaminates pure cases; see README audit.",
        },
    }
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
