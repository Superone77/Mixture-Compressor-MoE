#!/usr/bin/env python3
"""
Visualize expert activation from CSV data.

This script reads the expert activation CSV and creates scatter plots for each layer.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path


def parse_experts_and_weights(experts_str):
    """Parse experts_and_weights string into list of (expert_id, weight) tuples."""
    experts_data = []
    if pd.isna(experts_str) or experts_str == '':
        return experts_data
    
    for pair in str(experts_str).split(';'):
        if ',' in pair:
            try:
                expert_id, weight = pair.split(',')
                experts_data.append((int(float(expert_id)), float(weight)))
            except ValueError:
                continue
    
    return experts_data


def create_visualizations(csv_file, output_dir='expert_activation_plots'):
    """Create scatter plots for each layer showing expert activation patterns."""
    
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique layers
    layers = df['layer'].unique()
    print(f"Found {len(layers)} layers")
    
    # Process each layer
    for layer in layers:
        print(f"\nProcessing layer: {layer}")
        
        # Filter data for this layer
        layer_data = df[df['layer'] == layer].copy()
        
        # Parse expert data
        all_tokens = []
        all_experts = []
        all_weights = []
        
        for _, row in layer_data.iterrows():
            token_idx = row['token_index']
            experts_data = parse_experts_and_weights(row['experts_and_weights'])
            
            for expert_id, weight in experts_data:
                all_tokens.append(token_idx)
                all_experts.append(expert_id)
                all_weights.append(weight)
        
        if not all_tokens:
            print(f"No data for layer {layer}")
            continue
        
        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Normalize weights for color mapping (0=dark, 1=bright)
        if all_weights:
            weights_array = np.array(all_weights)
            # Invert so that higher weights are brighter
            normalized_weights = (weights_array - weights_array.min()) / (weights_array.max() - weights_array.min() + 1e-10)
            colors = normalized_weights
            
            # Create scatter plot
            scatter = ax.scatter(
                all_tokens,
                all_experts,
                c=colors,
                cmap='YlOrRd',  # Yellow-Orange-Red colormap (bright = high weight)
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Normalized Weight (darker=lower, brighter=higher)', rotation=270, labelpad=20)
        
        # Set labels and title
        ax.set_xlabel('Token Index', fontsize=12)
        ax.set_ylabel('Expert ID', fontsize=12)
        ax.set_title(f'Expert Activation Pattern - {layer}', fontsize=14, fontweight='bold')
        
        # Set y-axis to integer values
        max_expert = max(all_experts) if all_experts else 0
        ax.set_yticks(range(max_expert + 1))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        # Clean layer name for filename
        safe_layer_name = layer.replace('/', '_').replace('\\', '_').replace('.', '_')
        output_file = os.path.join(output_dir, f'{safe_layer_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        
        plt.close()
    
    print(f"\nVisualization complete! Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize expert activation from CSV data"
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to the CSV file containing expert activation data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='expert_activation_plots',
        help='Output directory for plots (default: expert_activation_plots)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    print("="*80)
    print("Expert Activation Visualization")
    print("="*80)
    print(f"CSV file: {args.csv}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    create_visualizations(args.csv, args.output_dir)


if __name__ == "__main__":
    main()

