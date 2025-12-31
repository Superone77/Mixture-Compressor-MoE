#!/usr/bin/env python3
"""
Visualization script for Experiment 1.1 and 1.2

This script creates visualizations from the CSV outputs:
- Experiment 1.1: Expert activation heatmaps
- Experiment 1.2: Cross-domain performance bar charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def visualize_experiment_1_1(csv_path: str, output_path: str = None):
    """Visualize Experiment 1.1: Expert Activation Heatmaps."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(12 * n_datasets, 8))
    if n_datasets == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_df = df[df['dataset'] == dataset]
        
        # Pivot to create heatmap: layers x experts
        pivot_df = dataset_df.pivot_table(
            index='layer',
            columns='expert_id',
            values='utilization_rate',
            fill_value=0.0
        )
        
        # Create heatmap
        sns.heatmap(
            pivot_df,
            ax=ax,
            cmap='YlOrRd',
            cbar_kws={'label': 'Utilization Rate (%)'},
            vmin=0,
            vmax=100,
            fmt='.1f',
            annot=False,  # Set to True if you want numbers on cells
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_title(f'Expert Utilization Rate by Layer\n{dataset.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Expert ID', fontsize=12)
        ax.set_ylabel('Layer Index', fontsize=12)
        ax.invert_yaxis()  # Layer 0 at top
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    else:
        plt.savefig('experiment_1_1_heatmap.png', bbox_inches='tight')
        print("Saved heatmap to experiment_1_1_heatmap.png")
    
    plt.close()
    
    # Also create a bar chart showing average utilization per layer
    fig, axes = plt.subplots(1, n_datasets, figsize=(12 * n_datasets, 6))
    if n_datasets == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_df = df[df['dataset'] == dataset]
        
        # Calculate average utilization per layer
        layer_avg = dataset_df.groupby('layer')['utilization_rate'].mean().reset_index()
        
        # Calculate number of experts with 0% activation per layer
        layer_zero = dataset_df.groupby('layer').apply(
            lambda x: (x['utilization_rate'] == 0).sum()
        ).reset_index(name='zero_experts')
        
        # Create bar chart
        x = layer_avg['layer']
        width = 0.35
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, layer_avg['utilization_rate'], width, 
                      label='Avg Utilization Rate (%)', color='steelblue', alpha=0.7)
        bars2 = ax2.bar(x + width/2, layer_zero['zero_experts'], width,
                       label='Experts with 0% Activation', color='coral', alpha=0.7)
        
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Average Utilization Rate (%)', fontsize=12, color='steelblue')
        ax2.set_ylabel('Number of Experts with 0% Activation', fontsize=12, color='coral')
        ax.set_title(f'Expert Utilization Summary by Layer\n{dataset.upper()}', 
                    fontsize=14, fontweight='bold')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        bar_path = output_path.replace('.png', '_bar_chart.png')
        plt.savefig(bar_path, bbox_inches='tight')
        print(f"Saved bar chart to {bar_path}")
    else:
        plt.savefig('experiment_1_1_bar_chart.png', bbox_inches='tight')
        print("Saved bar chart to experiment_1_1_bar_chart.png")
    
    plt.close()


def visualize_experiment_1_2(csv_path: str, output_path: str = None):
    """Visualize Experiment 1.2: Cross-Domain Performance Drop."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for grouped bar chart
    test_datasets = df['test_dataset'].unique()
    models = df['model'].unique()
    
    x = np.arange(len(test_datasets))
    width = 0.35
    
    # Create bars for each model
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        perplexities = [model_data[model_data['test_dataset'] == ds]['perplexity'].values[0] 
                        for ds in test_datasets]
        
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, perplexities, width, label=model, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Test Dataset', fontsize=12)
    ax.set_ylabel('Perplexity (PPL)', fontsize=12)
    ax.set_title('Cross-Domain Performance: Model vs Test Dataset', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(test_datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved bar chart to {output_path}")
    else:
        plt.savefig('experiment_1_2_bar_chart.png', bbox_inches='tight')
        print("Saved bar chart to experiment_1_2_bar_chart.png")
    
    plt.close()
    
    # Also create a heatmap showing the performance matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create pivot table: models x test_datasets
    pivot_df = df.pivot_table(
        index='model',
        columns='test_dataset',
        values='perplexity',
        fill_value=0.0
    )
    
    sns.heatmap(
        pivot_df,
        ax=ax,
        cmap='RdYlGn_r',  # Reversed: lower perplexity (better) = darker green
        cbar_kws={'label': 'Perplexity (PPL)'},
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        linecolor='gray',
        square=True
    )
    
    ax.set_title('Cross-Domain Performance Heatmap\n(Lower is Better)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Test Dataset', fontsize=12)
    ax.set_ylabel('Model (Calibration Dataset)', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        heatmap_path = output_path.replace('.png', '_heatmap.png')
        plt.savefig(heatmap_path, bbox_inches='tight')
        print(f"Saved heatmap to {heatmap_path}")
    else:
        plt.savefig('experiment_1_2_heatmap.png', bbox_inches='tight')
        print("Saved heatmap to experiment_1_2_heatmap.png")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Experiment 1.1 and 1.2 results"
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['1.1', '1.2', 'both'],
        default='both',
        help='Which experiment to visualize'
    )
    parser.add_argument(
        '--csv_1_1',
        type=str,
        default='experiment_1_1_expert_activation.csv',
        help='CSV file for Experiment 1.1'
    )
    parser.add_argument(
        '--csv_1_2',
        type=str,
        default='experiment_1_2_cross_domain.csv',
        help='CSV file for Experiment 1.2'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Visualizing Experiment Results")
    print("="*80)
    
    if args.experiment in ['1.1', 'both']:
        csv_path = args.csv_1_1
        if not Path(csv_path).exists():
            print(f"Warning: CSV file not found: {csv_path}")
        else:
            output_path = output_dir / 'experiment_1_1_heatmap.png'
            visualize_experiment_1_1(csv_path, str(output_path))
    
    if args.experiment in ['1.2', 'both']:
        csv_path = args.csv_1_2
        if not Path(csv_path).exists():
            print(f"Warning: CSV file not found: {csv_path}")
        else:
            output_path = output_dir / 'experiment_1_2_bar_chart.png'
            visualize_experiment_1_2(csv_path, str(output_path))
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()

