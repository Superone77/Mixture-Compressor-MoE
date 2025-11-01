#!/bin/bash
# Run script for visualizing expert activation

# Default values
CSV_FILE="expert_activations.csv"
OUTPUT_DIR="expert_activation_plots"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv)
            CSV_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --csv FILE           CSV file containing expert activation data (default: expert_activations.csv)"
            echo "  --output_dir DIR     Output directory for plots (default: expert_activation_plots)"
            echo "  --help               Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "========================================"
echo "Expert Activation Visualization"
echo "========================================"
echo "CSV file: $CSV_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Determine python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    echo "Please run monitor_expert_activation.py first to generate the CSV"
    exit 1
fi

# Run the visualization script
$PYTHON_CMD visualize_expert_activation.py \
    --csv "$CSV_FILE" \
    --output_dir "$OUTPUT_DIR"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Visualization completed successfully!"
    echo "Plots saved to: $OUTPUT_DIR"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Visualization failed with errors"
    echo "========================================"
    exit 1
fi

