#!/usr/bin/env python3
"""
Visualize mixed precision bit assignments as a heatmap.

This script reads the layer-expert bit assignments pickle file and creates
a heatmap where:
- X-axis: Layer index
- Y-axis: Expert ID
- Color: Bit value (darker = lower bit, lighter = higher bit)
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path


def load_bit_assignments(pickle_file):
    """Load bit assignments from pickle file."""
    print(f"Loading bit assignments from {pickle_file}...")
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data


def create_heatmap(layer_expert_bits, output_file=None, figsize=(16, 10), dpi=300):
    """
    Create a heatmap of bit assignments.
    
    Args:
        layer_expert_bits: Dictionary {layer_idx: {expert_idx: bit_value}}
        output_file: Path to save the figure (if None, will display)
        figsize: Figure size tuple
        dpi: Resolution for saved figure
    """
    # Get all layers and experts
    layer_indices = sorted(layer_expert_bits.keys())
    all_experts = set()
    for layer_data in layer_expert_bits.values():
        all_experts.update(layer_data.keys())
    expert_indices = sorted(all_experts)
    
    print(f"Found {len(layer_indices)} layers and {len(expert_indices)} experts")
    
    # Create matrix: rows = experts, cols = layers
    matrix = np.zeros((len(expert_indices), len(layer_indices)), dtype=np.float32)
    
    # Fill matrix with bit values
    for layer_idx, layer_col in enumerate(layer_indices):
        if layer_col in layer_expert_bits:
            layer_data = layer_expert_bits[layer_col]
            for expert_idx, expert_row in enumerate(expert_indices):
                if expert_idx in layer_data:
                    matrix[expert_row, layer_idx] = layer_data[expert_idx]
                else:
                    # If expert not found, use NaN (will be shown as white)
                    matrix[expert_row, layer_idx] = np.nan
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a colormap where darker = lower bit (smaller value)
    # Using reversed colormap so smaller values are darker
    # 'viridis_r' or 'plasma_r' are good options (reversed colormaps)
    cmap = 'viridis_r'  # reversed viridis (darker for lower values, lighter for higher values)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Set labels
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expert ID', fontsize=12, fontweight='bold')
    ax.set_title('Mixed Precision Bit Assignment Heatmap\n(Darker = Lower Bit, Lighter = Higher Bit)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set tick labels
    ax.set_xticks(range(len(layer_indices)))
    ax.set_xticklabels([str(l) for l in layer_indices], rotation=0)
    ax.set_yticks(range(len(expert_indices)))
    ax.set_yticklabels([str(e) for e in expert_indices])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Bit Width', rotation=270, labelpad=20, fontsize=11)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(layer_indices)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(expert_indices)) - 0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Add text annotations for bit values (optional, may be crowded)
    # Uncomment if you want to see exact bit values in each cell
    # for expert_row in range(len(expert_indices)):
    #     for layer_col in range(len(layer_indices)):
    #         bit_val = matrix[expert_indices[expert_row], layer_col]
    #         if not np.isnan(bit_val):
    #             ax.text(layer_col, expert_row, f'{int(bit_val)}',
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
    print(f"Total layers: {len(layer_indices)}")
    print(f"Total experts per layer: {len(expert_indices)}")
    bit_values = [bit for layer_data in layer_expert_bits.values() 
                  for bit in layer_data.values()]
    if bit_values:
        print(f"Bit range: {min(bit_values)} - {max(bit_values)}")
        print(f"Average bit: {np.mean(bit_values):.2f}")
        print(f"Unique bit values: {sorted(set(bit_values))}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mixed precision bit assignments as a heatmap"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the pickle file containing layer-expert bit assignments'
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
        args.output = input_path.parent / f"{input_path.stem}_heatmap.png"
    
    print("="*80)
    print("Mixed Precision Bit Assignment Visualization")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Figure size: {figsize}")
    print(f"DPI: {args.dpi}")
    print("="*80)
    
    # Load bit assignments
    layer_expert_bits = load_bit_assignments(args.input)
    
    # Create heatmap
    create_heatmap(layer_expert_bits, args.output, figsize, args.dpi)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

