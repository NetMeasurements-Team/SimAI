#!/usr/bin/env python3
"""Bar plots of reconstructed full-batch training time (paper Fig. 11, 16, 17).

Used by analyze_results.py; can also be run on its own on a combined CSV:
    python3 benchmark/plot_results.py [combined_results.csv] [output_dir]
The CSV needs the columns workload, failure_type, algorithm, num_gpus, simulation_time_ns.
Requires pandas and matplotlib (seaborn optional).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:  # seaborn only sets the base style, which rcdefaults() resets anyway
    pass

# style
plt.rcdefaults()
params = {
    'legend.fontsize': 16,
    'legend.title_fontsize': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.titlesize': 25,
}
plt.rc('font', size=16)
plt.rc('pdf', fonttype=42)
pylab.rcParams.update(params)
default_palette = plt.rcParams['axes.prop_cycle']

# Algorithm configuration
ALGORITHM_ORDER = ['nccl', 'forestcoll', 'mccl']
ALGORITHM_LABELS = {
    'nccl': 'NCCL',
    'forestcoll': 'ForestColl',
    'mccl': 'MCCL',
    'teccl': 'TECCL'  # In case it exists in data
}

DEFAULT_PLOT_DIR = 'benchmark/results/plots'


# Bar plot function
def plot_bar_comparison(workload_name, failure_type, df_data, output_filename, log_scale=False,
                        output_dir=DEFAULT_PLOT_DIR):
    """
    Create a grouped bar plot comparing algorithms for different GPU counts.

    Args:
        workload_name: Name of the workload (e.g., 'Llama405B')
        failure_type: Type of failure (e.g., 'failure_1')
        df_data: DataFrame containing the data
        output_filename: Name of the output file (without extension)
        log_scale: Use log scale for y-axis
        output_dir: Directory for the .png/.pdf files
    """
    # Filter data for this workload and failure type
    df_filtered = df_data[
        (df_data['workload'] == workload_name) &
        (df_data['failure_type'] == failure_type)
        ].copy()

    if len(df_filtered) == 0:
        print(f"No data for {workload_name} with {failure_type}")
        return

    # Pivot data for plotting
    pivot_df = df_filtered.pivot(index='num_gpus', columns='algorithm', values='simulation_time_ns')

    # Filter and reorder columns based on ALGORITHM_ORDER
    # Only include algorithms that exist in the data and are in our order list
    available_algorithms = [alg for alg in ALGORITHM_ORDER if alg in pivot_df.columns]
    pivot_df = pivot_df[available_algorithms]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True)

    # Get GPU counts and algorithms
    gpu_counts = pivot_df.index.values
    algorithms = pivot_df.columns.values

    # Set up bar positions
    x = np.arange(len(gpu_counts))
    width = 0.25  # Width of each bar
    multiplier = 0

    # Plot bars for each algorithm
    for algorithm in algorithms:
        offset = width * multiplier
        values = pivot_df[algorithm].values / 1e9
        # Use custom label
        label = ALGORITHM_LABELS.get(algorithm, algorithm.upper())
        ax.bar(x + offset, values, width, label=label, edgecolor='black')
        multiplier += 1

    # Customize plot
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Training time\non one batch (s)')
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(gpu_counts)
    ax.legend(loc='upper right')
    if log_scale:
        ax.set_yscale('log')
        min_, max_ = pivot_df.values.min() / 1e9, pivot_df.values.max() / 1e9
        ax.set_ylim([min_*0.6, max_*1.8])
    ax.grid(True, alpha=0.3, axis='y')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{output_filename}.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{output_filename}.pdf'), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, output_filename)}")


def plot_all(df, output_dir=DEFAULT_PLOT_DIR):
    for workload in ('Llama405B', 'DeepSeek671B'):
        for failure in ('no_failure', 'failure_1', 'failure_2', 'failure_3'):
            plot_bar_comparison(workload, failure, df, f'{workload}_{failure}', output_dir=output_dir)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'benchmark/results/combined_results.csv'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PLOT_DIR
    df = pd.read_csv(csv_path)
    print("\nGenerating bar plots...")
    plot_all(df, out_dir)
    print(f"Done! Check {out_dir} for visualizations.")
