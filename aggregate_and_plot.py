import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re

env_name = 'metaworld_sweep-into-v2'
base_dir = os.path.join(os.path.dirname(__file__), 'RIME/', env_name)
output_dir = os.path.join(os.path.dirname(__file__), 'other_results/RIME', env_name)
os.makedirs(output_dir, exist_ok=True)


def get_csv_files(base_dir, csv_type):
    csv_files_path = os.path.join(base_dir, '**', f'{csv_type}*.csv')
    print(f'Looking for files in: {csv_files_path}')
    csv_files = sorted(glob.glob(csv_files_path, recursive=True))
    return csv_files

# Define a function to process each CSV type
def process_csv_files(csv_type):
    # Find all relevant CSV files (train, eval, reward)
    csv_files = get_csv_files(base_dir, csv_type)
    dfs = [pd.read_csv(f) for f in csv_files if os.path.getsize(f) > 0]
    
    if not dfs:
        print(f'No {csv_type}.csv files found or all are empty.')
        return None

    # Align by step (or episode)
    key = 'step' if 'step' in dfs[0].columns else 'episode'
    metrics = [col for col in dfs[0].columns if col not in [key, 'episode', 'labeled_feedback', 'total_feedback']]

    # Concatenate all DataFrames
    all_data = pd.concat(dfs, ignore_index=True)

    # Group by step or episode and calculate mean and std
    if 'step' in all_data.columns:
        grouped = all_data.groupby('step')[metrics].agg(['mean', 'std']).reset_index()
        x_values = 'step'
    elif 'episode' in all_data.columns:
        grouped = all_data.groupby('episode')[metrics].agg(['mean', 'std']).reset_index()
        x_values = 'episode'
    else:
        print(f'No step or episode column found in {csv_type}.csv files.')
        return None

    return grouped, metrics, x_values


def plot_metrics(grouped, metrics, x_values, csv_type, dfs=None, csv_files=None):
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # --- Plot individual runs if provided ---
        if dfs is not None and csv_files is not None:
            for df, fname in zip(dfs, csv_files):
                if x_values in df.columns and metric in df.columns:
                    match = re.search(r'seed_\d+', fname)
                    label = match.group(0) if match else os.path.basename(os.path.dirname(fname))
                    plt.plot(df[x_values], df[metric], alpha=0.3, label=label)

        # --- Plot aggregated mean/std ---
        metric_mean = grouped[(metric, 'mean')]
        metric_std = grouped[(metric, 'std')]
        plt.plot(grouped[x_values], metric_mean, color="black", linewidth=2, label=f'Mean {metric}')
        plt.fill_between(grouped[x_values],
                         metric_mean - metric_std,
                         metric_mean + metric_std,
                         alpha=0.2,
                         color="gray",
                         label='Std')

        plt.xlabel(x_values)
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(os.path.join(output_dir, f'{csv_type}_{metric}_with_runs.png'))
        plt.close()
        print(f'Plot saved: {csv_type}_{metric}_with_runs.png')


# Process and plot for train.csv, reward.csv, and eval.csv
for csv_type in ['train', 'reward', 'eval']:
    print(f'Processing {csv_type}.csv...')
    csv_files = get_csv_files(base_dir, csv_type)
    dfs = [pd.read_csv(f) for f in csv_files if os.path.getsize(f) > 0]
    if not dfs:
        continue
    
    grouped, metrics, x_values = process_csv_files(csv_type)
    if grouped is not None:
        plot_metrics(grouped, metrics, x_values, csv_type, dfs=dfs, csv_files=csv_files)

print('All metrics plotted.')
