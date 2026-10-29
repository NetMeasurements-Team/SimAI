#!/usr/bin/env python3
"""Reconstruct the DeepSeek-671B training time from the partial simulations of artifact
experiment E4 and plot it (paper Sec. 6.3, NCCL bars of Fig. 11 and Fig. 17).

E4 simulates one pipeline stage (of 16) on 128 GPUs in two parts per failure profile:
  DeepSeek671B-transformer  one transformer block, for the per-GPU share of the batch
  DeepSeek671B-grad         the gradient-synchronization phase of the stage
The training time on one 16K-token batch of the stage is

  T_batch = T_transformer * TRANSFORMERS_PER_STAGE * BATCH_SIZE / (NUM_GPUS / TP) + T_grad

Usage:
  python3 benchmark/analyze_results.py <run dir>/E4-e2e/results.csv [--out DIR]

Writes DIR/combined_results.csv and DIR/plots/DeepSeek671B_training_time.{png,pdf}
(DIR defaults to the directory of the CSV). Requires pandas and matplotlib.
"""

import argparse
import os
import sys

import pandas as pd

from plot_results import FAILURE_ORDER, plot_training_time

WORKLOAD = 'DeepSeek671B'
TP = 1
TRANSFORMERS_PER_STAGE = 3
BATCH_SIZE = 16384
NUM_GPUS = 128          # one pipeline stage, as simulated by artifact/run.sh e2e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', help='E4-e2e/results.csv written by artifact/run.sh e2e (or all)')
    ap.add_argument('--out', help='output directory (default: directory of the CSV)')
    args = ap.parse_args()
    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv))

    df = pd.read_csv(args.csv)
    ok = df[df['status'] == 'SUCCESS']
    t = {(r.workload, r.failure_type): int(r.sim_time_ns) for r in ok.itertuples()}

    rows, missing = [], []
    for ft in FAILURE_ORDER:
        tt, tg = t.get((f'{WORKLOAD}-transformer', ft)), t.get((f'{WORKLOAD}-grad', ft))
        if tt is None or tg is None:
            missing.append(ft)
            continue
        total = int(tt * TRANSFORMERS_PER_STAGE * BATCH_SIZE / (NUM_GPUS / TP)) + tg
        rows.append({'workload': WORKLOAD, 'failure_type': ft, 'algorithm': 'nccl', 'num_gpus': NUM_GPUS,
                     'transformer_ns': tt, 'grad_ns': tg, 'simulation_time_ns': total})

    print(f"Runs: {len(df)}, successful: {len(ok)}")
    for ft in missing:
        print(f"WARNING: {ft}: transformer or gradient run missing/failed; not reconstructed")
    if not rows:
        print("Nothing to reconstruct.")
        return 1

    combined = pd.DataFrame(rows)
    base = combined.loc[combined['failure_type'] == 'no_failure', 'simulation_time_ns']
    combined['vs_no_failure'] = (combined['simulation_time_ns'] / base.iloc[0]).round(2) if len(base) else None

    print("\nReconstructed training time on one batch (one pipeline stage, 128 GPUs, NCCL):")
    show = combined[['failure_type', 'transformer_ns', 'grad_ns', 'simulation_time_ns', 'vs_no_failure']].copy()
    show['training_time_s'] = (show['simulation_time_ns'] / 1e9).round(2)
    print(show.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(os.path.join(out_dir, 'combined_results.csv'), index=False)
    print(f"\nSaved: {os.path.join(out_dir, 'combined_results.csv')}")
    plot_training_time(combined, os.path.join(out_dir, 'plots'))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
