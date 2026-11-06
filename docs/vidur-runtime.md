# Running Vidur with the in-memory SimAI runtime

`SimAIRuntime` keeps the ns-3 network and Astra-Sim `Sys` instances resident in
memory and exposes them to Python through the `pysimai` extension, so Vidur can
simulate each TP collective in-process instead of spawning `SimAI_simulator`
and parsing CSV files.

The in-memory path is selected automatically: if `pysimai` is importable, Vidur
uses it; otherwise it falls back to the legacy subprocess path. Nothing in the
Vidur command line changes.

## Build

Install `pybind11` in the active env first (otherwise the `pysimai` target is
skipped), then build the ns-3 backend:

```bash
cd <repo>
pip install pybind11
./scripts/build.sh -c ns3 -d optimized
```

This produces `bin/SimAI_simulator` and `bin/pysimai.cpython-310-*.so`.

## Run the Llama-3-8B example

Same example as `vidur-alibabacloud/README.md` → *Run Llama-3-8B with SimAI
Simulation*, with `<repo>/bin` added to `PYTHONPATH` so `pysimai` is importable:

```bash
cd <repo>

# Generate the topology once
python3 astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

cd vidur-alibabacloud

AS_SEND_LAT=6 AS_NVLS_ENABLE=1 \
PYTHONPATH="$PWD/../bin:$PWD" \
python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 2048 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --random_forrest_execution_time_predictor_config_backend simai_simulation \
  --random_forrest_execution_time_predictor_config_simai_dir .. \
  --random_forrest_execution_time_predictor_config_simai_simulation_topo ../Spectrum-X_128g_8gps_100Gbps_A100 \
  --random_forrest_execution_time_predictor_config_simai_simulation_config ../astra-sim-alibabacloud/inputs/config/SimAI.conf
```

To use the legacy subprocess path instead, drop `../bin` from `PYTHONPATH`.

## Further reading

- [simai-runtime-architecture.md](simai-runtime-architecture.md) — threading
  model, collective lifecycle, and the compute-gap behavior.
- [tests/pysimai/README.md](../tests/pysimai/README.md) — runtime tests and the
  legacy-parity benchmark.
