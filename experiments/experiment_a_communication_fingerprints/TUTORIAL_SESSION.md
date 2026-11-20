# Tutorial: Communication Fingerprints of DP, TP, and PP

Run every command from the SimAI repository root:

```bash
cd /home/settimo/Documents/AIxNEXT/SimAI
```

## 1. Show the model and experiment parameters

Regenerate the experiment metadata/workloads, then print the presentation view:

```bash
python3 experiments/experiment_a_communication_fingerprints/generate_workloads.py
python3 experiments/experiment_a_communication_fingerprints/show_parameters.py
```

`metadata.json` remains the machine-readable source used by the scripts; there is no need to display its raw JSON during the tutorial.

Parameters to highlight:

| Parameter | Value |
|---|---:|
| Transformer layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Sequence length | 1,024 tokens |
| Vocabulary | 50,257 |
| Global batch | 32 sequences |
| Parameters | 124,439,808 |
| Communicated value size | 2 bytes (BF16 assumption) |
| Simulated GPUs | 4 |

The model profile is GPT-2 Small with tied input/output embeddings. The DP payload assumes gradients are communicated in BF16; it is an explicit experimental assumption, not a claim about every Megatron configuration.

## 2. Show the generated DP, TP, and PP workloads

```bash
sed -n '1,8p' experiments/experiment_a_communication_fingerprints/workloads/dp.txt
sed -n '1,12p' experiments/experiment_a_communication_fingerprints/workloads/tp.txt
sed -n '1,12p' experiments/experiment_a_communication_fingerprints/workloads/pp.txt
```

What to explain:

| Strategy | Logical communication pattern | Payload | Operations per iteration |
|---|---|---:|---:|
| DP | One gradient AllReduce | 248,879,616 B | 1 |
| TP | Activation/gradient AllReduces | 50,331,648 B each | 48 |
| PP = 4 | Adjacent-stage point-to-point transfers | 12,582,912 B each | 24 globally |

The counts come from:

- **DP:** `parameters × 2 bytes`.
- **TP:** `global batch × sequence × hidden × 2 bytes`; 12 layers × 2 tensor-parallel regions × forward/backward = 48 collectives.
- **PP:** `microbatch 8 × sequence × hidden × 2 bytes`; four microbatches cross three stage boundaries once forward and once backward: `4 × 3 × 2 = 24` transfers.

The PP visual uses a 1F1B schedule with four microbatches. SimAI Analytical does not produce a detailed point-to-point event timeline here, so PP timing and message placement are kept as a clearly separated analytical reference.

## 3. Show the network configuration

```bash
python3 experiments/experiment_a_communication_fingerprints/show_network.py
```

This prints only the settings used by the experiment:

| Setting | Meaning |
|---|---|
| `-g 4 -g_p_s 4` | Four GPUs, all in one server |
| `-nv 360` | 360 GB/s modeled intra-server GPU bandwidth |
| `-nic 48.5` | 48.5 GB/s effective NIC bandwidth |
| `-g_type A100` | A100 communication model |

### Why exactly 48.5 GB/s?

A 400 Gb/s NIC has a theoretical rate of `400 ÷ 8 = 50 GB/s`. SimAI uses 97% of that rate—**48.5 GB/s**—as a simple allowance for protocol and implementation overhead.

## 4. Run SimAI Analytical and inspect the results

Run all three cases and wait for the command to return:

```bash
./experiments/experiment_a_communication_fingerprints/run_simulations.sh
```

Expected final message:

```text
SimAI runs complete: DP, TP, and PP results were generated and parsed successfully.
```

Inspect the raw SimAI summaries:

```bash
head -n 2 experiments/experiment_a_communication_fingerprints/results/dp/Analytical_test_EndToEnd.csv
head -n 2 experiments/experiment_a_communication_fingerprints/results/tp/Analytical_test_EndToEnd.csv
head -n 2 experiments/experiment_a_communication_fingerprints/results/pp/Analytical_test_EndToEnd.csv
```

Inspect the parsed events and validation report:

```bash
sed -n '1,12p' experiments/experiment_a_communication_fingerprints/parsed/events.csv
python3 -m json.tool experiments/experiment_a_communication_fingerprints/parsed/validation.json
```

Expected headline values for the current configuration:

| Strategy | Result to discuss |
|---|---:|
| DP | 1,792 µs exposed communication |
| TP | 9,375 µs exposed communication |
| PP | 24-message analytical inventory; do not present it as a SimAI event timeline |

## 5. Generate and open the interactive visualizations

```bash
python3 experiments/experiment_a_communication_fingerprints/interactive_plot.py
python3 experiments/experiment_a_communication_fingerprints/collective_vs_p2p.py

xdg-open experiments/experiment_a_communication_fingerprints/plots/communication_fingerprints_interactive.html
```

In the first page, change the model/payload controls and click a DP, TP, or PP element to open the communication-detail page. If the link is blocked by the browser, open it directly:

```bash
xdg-open experiments/experiment_a_communication_fingerprints/plots/collective_vs_p2p.html
```

