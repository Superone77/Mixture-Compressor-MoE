#!/usr/bin/env python3
"""
Visualization script for Experiment 1.1 and 1.2

- Experiment 1.1: Expert Activation Heatmaps
- Experiment 1.2: Cross-Domain Performance Drop (Grouped Bar Chart)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from pathlib import Path


def plot_experiment_1_1(csv_path, output_path=None):
    """Plot expert activation heatmap for Experiment 1.1."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Create pivot table: layer x expert_id, with dataset as separate columns
    datasets = df['dataset'].unique()
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(12 * len(datasets), 8))
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        df_dataset = df[df['dataset'] == dataset]
        pivot = df_dataset.pivot_table(
            values='activation_rate',
            index='layer',
            columns='expert_id',
            aggfunc='mean'
        )
        
        # Sort by layer
        pivot = pivot.sort_index()
        
        # Create heatmap
        sns.heatmap(
            pivot,
            annot=False,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Activation Rate'},
            ax=axes[idx],
            vmin=0,
            vmax=1.0
        )
        
        axes[idx].set_title(f'Expert Activation Rate - {dataset.upper()}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Expert ID', fontsize=12)
        axes[idx].set_ylabel('Layer', fontsize=12)
        axes[idx].invert_yaxis()  # Layer 0 at top
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    else:
        plt.savefig('experiment_1_1_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved heatmap to experiment_1_1_heatmap.png")
    
    plt.close()
    
    # Also create a bar chart showing average activation rate per layer
    fig, ax = plt.subplots(figsize=(14, 6))
    
    layer_avg = df.groupby(['layer', 'dataset'])['activation_rate'].mean().reset_index()
    
    x = np.arange(len(layer_avg['layer'].unique()))
    width = 0.35
    
    for i, dataset in enumerate(datasets):
        dataset_data = layer_avg[layer_avg['dataset'] == dataset]
        layers = dataset_data['layer'].values
        rates = dataset_data['activation_rate'].values
        ax.bar(x + i * width, rates, width, label=dataset.upper(), alpha=0.8)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Average Activation Rate', fontsize=12)
    ax.set_title('Average Expert Activation Rate per Layer', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(layer_avg['layer'].unique(), rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    bar_output = output_path.replace('.png', '_bar.png') if output_path else 'experiment_1_1_bar.png'
    plt.savefig(bar_output, dpi=300, bbox_inches='tight')
    print(f"Saved bar chart to {bar_output}")
    plt.close()


def plot_experiment_1_2(csv_path, output_path=None):
    """Plot grouped bar chart for Experiment 1.2."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    models = df['model'].unique()
    test_datasets = df['test_dataset'].unique()
    
    x = np.arange(len(test_datasets))
    width = 0.35
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        perplexities = []
        for test_ds in test_datasets:
            ppl = model_data[model_data['test_dataset'] == test_ds]['perplexity'].values
            perplexities.append(ppl[0] if len(ppl) > 0 else 0)
        
        bars = ax.bar(x + i * width, perplexities, width, label=model, alpha=0.8, color=colors[i])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Test Dataset', fontsize=12)
    ax.set_ylabel('Perplexity (PPL)', fontsize=12)
    ax.set_title('Cross-Domain Performance Drop\n(Calibration Domain vs Test Domain)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([ds.upper() for ds in test_datasets])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved chart to {output_path}")
    else:
        plt.savefig('experiment_1_2_performance.png', dpi=300, bbox_inches='tight')
        print("Saved chart to experiment_1_2_performance.png")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    for model in models:
        print(f"\n{model}:")
        model_data = df[df['model'] == model]
        for test_ds in test_datasets:
            ppl = model_data[model_data['test_dataset'] == test_ds]['perplexity'].values
            if len(ppl) > 0:
                calib_ds = model_data[model_data['test_dataset'] == test_ds]['calibration_dataset'].values[0]
                print(f"  {test_ds.upper()}: {ppl[0]:.4f} (calibrated on {calib_ds})")


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument('--experiment', type=str, choices=['1.1', '1.2', 'both'], default='both',
                       help='Which experiment to visualize')
    parser.add_argument('--csv_1_1', type=str, default='experiment_1_1_activation_rates.csv',
                       help='CSV file for Experiment 1.1')
    parser.add_argument('--csv_1_2', type=str, default='experiment_1_2_perplexity.csv',
                       help='CSV file for Experiment 1.2')
    parser.add_argument('--output_1_1', type=str, default=None,
                       help='Output path for Experiment 1.1 visualization')
    parser.add_argument('--output_1_2', type=str, default=None,
                       help='Output path for Experiment 1.2 visualization')
    
    args = parser.parse_args()
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    if args.experiment in ['1.1', 'both']:
        csv_path = Path(args.csv_1_1)
        if csv_path.exists():
            print("\n" + "="*80)
            print("Visualizing Experiment 1.1: Expert Activation Heatmaps")
            print("="*80)
            plot_experiment_1_1(str(csv_path), args.output_1_1)
        else:
            print(f"Warning: CSV file not found: {csv_path}")
    
    if args.experiment in ['1.2', 'both']:
        csv_path = Path(args.csv_1_2)
        if csv_path.exists():
            print("\n" + "="*80)
            print("Visualizing Experiment 1.2: Cross-Domain Performance Drop")
            print("="*80)
            plot_experiment_1_2(str(csv_path), args.output_1_2)
        else:
            print(f"Warning: CSV file not found: {csv_path}")
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()

