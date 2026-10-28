#!/usr/bin/env python3
"""Reconstruct full-batch training time from the partial end-to-end simulations and plot it
(paper Sec. 6.3, Fig. 11, 16, 17).

The end-to-end workloads are simulated in two parts per model:
  <Model>-transformer  one transformer block of one pipeline stage, for the per-GPU share of the
                       batch processed in one simulated pass
  <Model>-grad         the gradient-synchronization phase of the stage
The training time on one batch for one pipeline stage is reconstructed as

  T_batch = T_transformer * TRANSFORMERS_PER_STAGE[model] * BATCH_SIZE / (num_gpus / TP[model])
            + T_grad                      (gradient phase; leave it out with --no-grad)

Usage:
  python3 benchmark/analyze_results.py <simulation_results.csv>... [--no-grad]
                                       [--out DIR] [--exclude ALGO,...]
Accepts both CSV formats: benchmark/run_*.sh (simulation_time_ns, num_gpus) and
artifact/run.sh e2e (sim_time_ns; 128 GPUs). Writes DIR/combined_results.csv and DIR/plots/
(DIR defaults to the directory of the first CSV). Requires pandas and matplotlib.
"""

import argparse
import os

import pandas as pd

from plot_results import plot_all

TP = {'Llama405B': 16, 'DeepSeek671B': 1}
TRANSFORMERS_PER_STAGE = {'Llama405B': 8, 'DeepSeek671B': 3}
BATCH_SIZE = 16384
ARTIFACT_E3_GPUS = 128  # artifact/run.sh e2e simulates one pipeline stage on 128 GPUs

# Add algorithms to this list to filter them out from the plots
EXCLUDED_ALGORITHMS = ['ForestColl']


def load(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if 'sim_time_ns' in df.columns and 'simulation_time_ns' not in df.columns:
            df = df.rename(columns={'sim_time_ns': 'simulation_time_ns'})
        if 'num_gpus' not in df.columns:
            df['num_gpus'] = ARTIFACT_E3_GPUS
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def process_workload(df_success, workload, add_grad):
    full_batch_df = df_success[df_success['workload'] == f'{workload}-transformer'].copy()
    if full_batch_df.empty:
        return full_batch_df
    full_batch_df['simulation_time_ns'] = (
            full_batch_df['simulation_time_ns'].astype('int64') *
            TRANSFORMERS_PER_STAGE[workload] *
            (BATCH_SIZE / (full_batch_df['num_gpus'] / TP[workload]))
    ).astype('int64')
    full_batch_df['workload'] = workload

    if add_grad:
        # Add *-grad simulation times to matching entries (same failure, algorithm, GPU count)
        grad_df = df_success[df_success['workload'] == f'{workload}-grad']
        keys = ['failure_type', 'algorithm', 'num_gpus']
        have = set(map(tuple, grad_df[keys].values.tolist()))
        for key in map(tuple, full_batch_df[keys].values.tolist()):
            if key not in have:
                print(f"WARNING: no successful {workload}-grad run for {key}; "
                      f"its batch time excludes the gradient phase")
        for _, grad_row in grad_df.iterrows():
            match_mask = (
                    (full_batch_df['failure_type'] == grad_row['failure_type']) &
                    (full_batch_df['algorithm'] == grad_row['algorithm']) &
                    (full_batch_df['num_gpus'] == grad_row['num_gpus'])
            )
            if match_mask.any():
                full_batch_df.loc[match_mask, 'simulation_time_ns'] += int(grad_row['simulation_time_ns'])
    return full_batch_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='+', help='simulation_results.csv / results.csv file(s)')
    ap.add_argument('--no-grad', dest='add_grad', action='store_false',
                    help='leave the gradient phase out of the batch time (transformer only)')
    ap.add_argument('--out', help='output directory (default: directory of the first CSV)')
    ap.add_argument('--exclude', default=','.join(EXCLUDED_ALGORITHMS),
                    help='comma-separated algorithms to leave out of the plots')
    args = ap.parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv[0]))
    df = load(args.csv)

    # Filter only successful runs and reused results
    df_success = df[df['status'].isin(['SUCCESS', 'REUSED_FROM_NCCL'])].copy()
    df_success['algorithm'] = df_success['algorithm'].astype(str).str.strip()

    combined_df = pd.concat([process_workload(df_success, w, args.add_grad) for w in TP],
                            ignore_index=True)

    exclusions = [a.strip().lower() for a in args.exclude.split(',') if a.strip()]
    if exclusions and not combined_df.empty:
        combined_df = combined_df[~combined_df['algorithm'].str.lower().isin(exclusions)]
    combined_df = combined_df.reset_index(drop=True)

    print(f"Total runs: {len(df)}")
    print(f"Successful runs: {len(df_success)}")
    print(f"Failed runs: {len(df) - len(df_success)}")
    print(f"Filtered out algorithms: {exclusions}")
    print(f"Gradient phase added: {args.add_grad}")
    print()

    if combined_df.empty:
        print("No successful *-transformer runs to reconstruct.")
        return 1

    cols = ['workload', 'failure_type', 'algorithm', 'num_gpus', 'simulation_time_ns']
    print("Reconstructed training time on one batch (one pipeline stage):")
    show = combined_df[cols].copy()
    show['training_time_s'] = show['simulation_time_ns'] / 1e9
    print(show.to_string(index=False))
    print()

    os.makedirs(out_dir, exist_ok=True)
    combined_df.to_csv(os.path.join(out_dir, 'combined_results.csv'), index=False)
    print(f"Processed data saved to {os.path.join(out_dir, 'combined_results.csv')}")

    print("\nGenerating bar plots...")
    plot_all(combined_df, os.path.join(out_dir, 'plots'))
    print(f"\nAnalysis complete! Check {os.path.join(out_dir, 'plots')} for visualizations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
