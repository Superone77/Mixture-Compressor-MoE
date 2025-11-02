#!/usr/bin/env python3
"""
Visualize mixed precision bit assignments as a heatmap.

This script reads the layer-expert bit assignments CSV file and creates
a heatmap where:
- X-axis: Layer index
- Y-axis: Expert ID
- Color: Bit value (lighter = smaller bit, darker = larger bit)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path


def load_bit_assignments(csv_file):
    """Load bit assignments from CSV file."""
    print(f"Loading bit assignments from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Validate required columns
    required_columns = ['layer', 'expert_id', 'bit']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}. Found: {df.columns.tolist()}")
    
    print(f"Loaded {len(df)} bit assignment records")
    return df


def create_heatmap(df, output_file=None, figsize=(16, 10), dpi=300):
    """
    Create a heatmap of bit assignments.
    
    Args:
        df: DataFrame with columns ['layer', 'expert_id', 'bit']
        output_file: Path to save the figure (if None, will display)
        figsize: Figure size tuple
        dpi: Resolution for saved figure
    """
    # Create pivot table: rows = experts, cols = layers
    pivot_df = df.pivot(index='expert_id', columns='layer', values='bit')
    
    # Sort by layer and expert
    pivot_df = pivot_df.sort_index(axis=0).sort_index(axis=1)
    
    # Convert to numpy array
    matrix = pivot_df.values
    
    print(f"Creating heatmap with {matrix.shape[0]} experts and {matrix.shape[1]} layers")
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a colormap where lighter = smaller bit (smaller value)
    # Using reversed colormap so smaller values are lighter
    # 'viridis_r' or 'plasma_r' are good options (reversed colormaps)
    cmap = 'viridis_r'  # reversed viridis (lighter for smaller values, darker for larger values)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Set labels
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expert ID', fontsize=12, fontweight='bold')
    ax.set_title('Mixed Precision Bit Assignment Heatmap\n(Lighter = Smaller Bit, Darker = Larger Bit)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set tick labels
    layer_labels = [str(int(l)) for l in pivot_df.columns]
    expert_labels = [str(int(e)) for e in pivot_df.index]
    
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=0)
    ax.set_yticks(range(len(expert_labels)))
    ax.set_yticklabels(expert_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Bit Width', rotation=270, labelpad=20, fontsize=11)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(layer_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(expert_labels)) - 0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Add text annotations for bit values (optional, may be crowded for large matrices)
    # Uncomment if you want to see exact bit values in each cell
    # for i, expert_row in enumerate(pivot_df.index):
    #     for j, layer_col in enumerate(pivot_df.columns):
    #         bit_val = matrix[i, j]
    #         if not np.isnan(bit_val):
    #             ax.text(j, i, f'{bit_val:.1f}',
    #                    ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Heatmap saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total layers: {len(pivot_df.columns)}")
    print(f"Total experts per layer: {len(pivot_df.index)}")
    bit_values = df['bit'].values
    if len(bit_values) > 0:
        print(f"Bit range: {bit_values.min():.2f} - {bit_values.max():.2f}")
        print(f"Average bit: {bit_values.mean():.2f}")
        unique_bits = sorted(df['bit'].unique())
        print(f"Unique bit values: {unique_bits}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mixed precision bit assignments as a heatmap"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the CSV file containing layer-expert bit assignments'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the heatmap image (default: input_file_name.png)'
    )
    parser.add_argument(
        '--figsize',
        type=str,
        default='16,10',
        help='Figure size as width,height (default: 16,10)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution for saved figure (default: 300)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Parse figsize
    try:
        figsize = tuple(map(int, args.figsize.split(',')))
    except ValueError:
        print("Warning: Invalid figsize format, using default (16,10)")
        figsize = (16, 10)
    
    # Determine output file
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_heatmap.png")
    
    print("="*80)
    print("Mixed Precision Bit Assignment Visualization")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Figure size: {figsize}")
    print(f"DPI: {args.dpi}")
    print("="*80)
    
    # Load bit assignments
    df = load_bit_assignments(args.input)
    
    # Create heatmap
    create_heatmap(df, args.output, figsize, args.dpi)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

