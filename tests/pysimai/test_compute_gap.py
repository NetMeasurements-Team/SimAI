#!/usr/bin/env python3
"""Validate the Vidur-derived compute-gap wiring end-to-end.

Covers:
  * `BaseExecutionTimePredictor._get_vidur_compute_gap_us(batch)` math;
  * `get_execution_time` passes the derived gap to `TPTimePredictor` when
    `simai_compute_gap_from_vidur=True` (the default);
  * it falls back to the fixed `simai_compute_gap_us` (`compute_gap_us=None`)
    when `simai_compute_gap_from_vidur=False`;
  * a real `pysimai` bridge call accepts the derived gap (if importable).

The probe subclass of `BaseExecutionTimePredictor` uses a constant per-layer
compute time `C_MS` so the expected derived gap is exactly
`12 * C_MS / 2 * 1000` us (12 per-layer compute components, 2 allreduces/block).

Exit code 0 = PASS, 1 = FAIL.

Usage (run on the remote host in the `vidur` env):
  python tests/pysimai/test_compute_gap.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vidur-alibabacloud"))
BIN_DIR = ROOT / "bin"
if any(BIN_DIR.glob("pysimai*")):
    sys.path.insert(0, str(BIN_DIR))

os.environ.setdefault("AS_SEND_LAT", "6")
os.environ.setdefault("AS_NVLS_ENABLE", "1")

from vidur.config import (  # noqa: E402
    MetricsConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SarathiSchedulerConfig,
    SimulationConfig,
)
from vidur.entities import Batch  # noqa: E402
from vidur.execution_time_predictor.base_execution_time_predictor import (  # noqa: E402
    BaseExecutionTimePredictor,
)

MODEL = "meta-llama/Meta-Llama-3-8B"
TOPO = str(ROOT / "Rail_Opti_SingleToR_16g_8gps_400Gbps_H100")
CONF = str(ROOT / "astra-sim-alibabacloud" / "inputs" / "config" / "SimAI.conf")
C_MS = 0.5
NUM_COMPUTE_COMPONENTS = 12
EXPECTED_GAP_US = NUM_COMPUTE_COMPONENTS * C_MS / 2.0 * 1000.0


class ProbePredictor(BaseExecutionTimePredictor):
    """Minimal BaseExecutionTimePredictor: constant compute, zero overheads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._c = C_MS

    def _compute(self, batch):
        return self._c

    def _zero(self, batch):
        return 0.0

    _get_attention_layer_pre_proj_execution_time = _compute
    _get_attention_layer_post_proj_execution_time = _compute
    _get_attention_rope_execution_time = _compute
    _get_attention_kv_cache_save_execution_time = _compute
    _get_attention_decode_execution_time = _compute
    _get_attention_prefill_execution_time = _compute
    _get_mlp_layer_up_proj_execution_time = _compute
    _get_mlp_layer_down_proj_execution_time = _compute
    _get_mlp_layer_act_execution_time = _compute
    _get_mlp_norm_layer_act_execution_time = _compute
    _get_attn_norm_layer_act_execution_time = _compute
    _get_add_layer_act_execution_time = _compute

    _get_tensor_parallel_communication_time = _zero
    _get_pipeline_parallel_communication_time = _zero
    _get_schedule_time = _zero
    _get_sampler_e2e_time = _zero
    _get_prepare_inputs_e2e_time = _zero
    _get_process_model_outputs_time = _zero
    _get_ray_comm_time = _zero


def make_probe(from_vidur: bool):
    predictor_config = RandomForrestExecutionTimePredictorConfig(
        backend="simai_simulation",
        simai_dir=str(ROOT),
        simai_simulation_topo=TOPO,
        simai_simulation_config=CONF,
        simai_compute_gap_us=1000,
        simai_compute_gap_from_vidur=from_vidur,
        nccl_cpu_launch_overhead_ms=0.0,
        nccl_cpu_skew_overhead_per_device_ms=0.0,
    )
    replica_config = ReplicaConfig(
        model_name=MODEL,
        num_pipeline_stages=1,
        tensor_parallel_size=16,
    )
    metrics_config = MetricsConfig(cache_dir=str(ROOT / ".validate_cache"))
    simulation_config = SimulationConfig()
    return ProbePredictor(
        predictor_config=predictor_config,
        replica_config=replica_config,
        replica_scheduler_config=SarathiSchedulerConfig(),
        metrics_config=metrics_config,
        simulation_config=simulation_config,
    )


def main() -> int:
    batch = Batch(replica_id=0, requests=[], num_tokens=[256])

    probe = make_probe(from_vidur=True)
    derived = probe._get_vidur_compute_gap_us(batch)
    print(f"derived compute gap (from_vidur=True ) : {derived:12.1f} us  (expected {EXPECTED_GAP_US:.1f})")
    assert abs(derived - EXPECTED_GAP_US) < 1e-6, "derived gap math mismatch"

    calls = []
    real = probe._tp_time_predictor.get_execution_time

    def spy(batch_, compute_gap_us=None):
        calls.append(compute_gap_us)
        return 0.0

    probe._tp_time_predictor.get_execution_time = spy
    probe.get_execution_time(batch, pipeline_stage=0)
    assert len(calls) == 1 and calls[0] == derived, f"expected derived gap {derived}, got {calls}"
    print(f"get_execution_time passed compute_gap_us : {calls[0]:.1f} us  OK")

    probe_off = make_probe(from_vidur=False)
    calls_off = []
    probe_off._tp_time_predictor.get_execution_time = (
        lambda b_, compute_gap_us=None: calls_off.append(compute_gap_us) or 0.0
    )
    probe_off.get_execution_time(batch, pipeline_stage=0)
    assert calls_off == [None], f"expected fallback None, got {calls_off}"
    print("from_vidur=False falls back to fixed gap (compute_gap_us=None)  OK")

    try:
        from pysimai import PySimAIBridge  # noqa: F401
    except ImportError:
        print("pysimai not importable; skipping real bridge call")
        print("PASS")
        return 0

    probe._tp_time_predictor.get_execution_time = real
    lat_us = probe._tp_time_predictor.get_execution_time(batch, compute_gap_us=derived)
    lat_us_fixed = probe._tp_time_predictor.get_execution_time(batch, compute_gap_us=1000)
    print(f"real bridge latency (derived gap {derived:.0f}us) : {lat_us * 1000:9.2f} us")
    print(f"real bridge latency (fixed gap  1000us) : {lat_us_fixed * 1000:9.2f} us")
    assert lat_us >= 0 and lat_us_fixed >= 0, "bridge returned -1"
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
