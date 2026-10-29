#!/usr/bin/env python3
"""Bar plot of the reconstructed training time of DeepSeek-671B (artifact experiment E4).

One bar per failure profile (no failure, Fail1-3): training time on one 16K-token batch of one
pipeline stage on 128 GPUs with the NCCL-equivalent schedules, in the style of the paper's
Fig. 16-17. Used by analyze_results.py; can also be run on its output:

    python3 benchmark/plot_results.py <combined_results.csv> [output_dir]

Requires pandas and matplotlib.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FAILURE_ORDER = ['no_failure', 'failure_1', 'failure_2', 'failure_3']
FAILURE_LABELS = {'no_failure': 'No Fail', 'failure_1': 'Fail 1',
                  'failure_2': 'Fail 2', 'failure_3': 'Fail 3'}
OUTPUT_NAME = 'DeepSeek671B_training_time'

# paper style
plt.rcdefaults()
plt.rc('font', size=16)
plt.rc('pdf', fonttype=42)
plt.rcParams.update({
    'legend.fontsize': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})


def plot_training_time(df, output_dir):
    """df: columns failure_type, simulation_time_ns (one row per failure profile)."""
    df = df.set_index('failure_type').reindex([f for f in FAILURE_ORDER if f in set(df['failure_type'])])
    if df.empty:
        print("Nothing to plot")
        return
    seconds = df['simulation_time_ns'] / 1e9

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True)
    ax.bar(range(len(seconds)), seconds.values, 0.6, label='NCCL', edgecolor='black')
    ax.set_xticks(range(len(seconds)))
    ax.set_xticklabels([FAILURE_LABELS[f] for f in seconds.index])
    ax.set_ylabel('Training time\non one batch (s)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(output_dir, f'{OUTPUT_NAME}.{ext}'), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, OUTPUT_NAME)}.{{png,pdf}}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(csv_path)), 'plots')
    plot_training_time(pd.read_csv(csv_path), out_dir)
