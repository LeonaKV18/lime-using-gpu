"""
plot_results.py
---------------
Reads a CSV of timing results produced by convert_timings_sweep.py and
generates three plots:

  1. Total pipeline time vs B (one line per D) — GPU and CPU
  2. Per-stage breakdown (gen / infer / weights) for GPU
  3. GPU vs CPU speed curves + speedup heatmap

Usage:
    python plot_results.py                          # reads timings.csv by default
    python plot_results.py --csv=my_timings.csv
    python plot_results.py --csv=timings.csv --out=plots/
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path)
    required = {'D', 'B', 'gpu_gen', 'gpu_infer', 'gpu_weights',
                 'gpu_total', 'cpu_total'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def savefig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}")


# ── Plot 1: Total time vs B ───────────────────────────────────────────────────

def plot_total_time(df, out_dir):
    D_vals = sorted(df['D'].unique())
    B_vals = sorted(df['B'].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors

    for idx, D_val in enumerate(D_vals):
        sub = df[df['D'] == D_val].sort_values('B')
        ax.plot(sub['B'], sub['gpu_total'], marker='o', color=colors[idx],
                label=f'GPU D={D_val}')
        ax.plot(sub['B'], sub['cpu_total'], marker='s', linestyle='--',
                color=colors[idx], alpha=0.5, label=f'CPU D={D_val}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('B (number of samples)')
    ax.set_ylabel('Total time (ms)')
    ax.set_title('Total Pipeline Time — GPU vs CPU')
    ax.set_xticks(B_vals)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    savefig(fig, out_dir, 'plot1_total_time.png')
    plt.show()


# ── Plot 2: Per-stage breakdown (GPU only) ────────────────────────────────────

def plot_stage_breakdown(df, out_dir):
    D_vals = sorted(df['D'].unique())
    stages = ['gpu_gen', 'gpu_infer', 'gpu_weights']
    labels = ['Generation', 'Inference', 'Weights']
    colors = ['#4C72B0', '#DD8452', '#55A868']

    fig, axes = plt.subplots(1, len(D_vals), figsize=(5 * len(D_vals), 5), sharey=True)
    if len(D_vals) == 1:
        axes = [axes]

    for ax, D_val in zip(axes, D_vals):
        sub  = df[df['D'] == D_val].sort_values('B')
        B_vals = sub['B'].values
        bottom = np.zeros(len(B_vals))
        for stage, label, color in zip(stages, labels, colors):
            vals = sub[stage].values
            ax.bar(range(len(B_vals)), vals, bottom=bottom, label=label, color=color)
            bottom += vals
        ax.set_xticks(range(len(B_vals)))
        ax.set_xticklabels(B_vals, rotation=45, fontsize=8)
        ax.set_title(f'D={D_val}')
        ax.set_xlabel('B (samples)')
        if ax == axes[0]:
            ax.set_ylabel('Time (ms)')
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    axes[0].legend(fontsize=9)
    fig.suptitle('GPU Per-Stage Breakdown', fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig(fig, out_dir, 'plot2_stage_breakdown.png')
    plt.show()


# ── Plot 3: Speedup heatmap ───────────────────────────────────────────────────

def plot_speedup_heatmap(df, out_dir):
    df = df.copy()
    df['speedup'] = df['cpu_total'] / df['gpu_total']

    D_vals = sorted(df['D'].unique())
    B_vals = sorted(df['B'].unique())
    pivot  = df.pivot(index='D', columns='B', values='speedup')

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap='RdYlGn',
                   norm=mcolors.LogNorm(vmin=max(0.5, pivot.values.min()),
                                        vmax=pivot.values.max()))
    ax.set_xticks(range(len(B_vals))); ax.set_xticklabels(B_vals)
    ax.set_yticks(range(len(D_vals))); ax.set_yticklabels(D_vals)
    ax.set_xlabel('B (samples)'); ax.set_ylabel('D (features)')
    ax.set_title('GPU Speedup over CPU (total pipeline)')

    for i in range(len(D_vals)):
        for j in range(len(B_vals)):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}x",
                    ha='center', va='center', fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Speedup')
    plt.tight_layout()
    savefig(fig, out_dir, 'plot3_speedup_heatmap.png')
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot timing results from CSV")
    parser.add_argument("--csv", type=str, default="timings.csv", help="Input CSV file")
    parser.add_argument("--out", type=str, default=".",           help="Output directory for plots")
    args = parser.parse_args()

    print(f"Loading {args.csv}")
    df = load_csv(args.csv)
    print(f"  {len(df)} rows   D values: {sorted(df['D'].unique())}   B values: {sorted(df['B'].unique())}")

    print("\nGenerating plots...")
    plot_total_time(df, args.out)
    plot_stage_breakdown(df, args.out)
    plot_speedup_heatmap(df, args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()
