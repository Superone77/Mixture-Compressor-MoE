#!/bin/bash
# Run script for visualizing mixed precision bit assignments

# Default values
INPUT_FILE=""
OUTPUT_FILE=""
FIG_SIZE="16,10"
DPI=300

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --figsize)
            FIG_SIZE="$2"
            shift 2
            ;;
        --dpi)
            DPI="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input FILE          Pickle file containing layer-expert bit assignments (required)"
            echo "  --output FILE         Output path for heatmap image (optional, default: input_file_name_heatmap.png)"
            echo "  --figsize WIDTH,HEIGHT Figure size (default: 16,10)"
            echo "  --dpi DPI            Resolution for saved figure (default: 300)"
            echo "  --help               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --input layer_expert_bits.pkl --output heatmap.png"
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
echo "Mixed Precision Bit Assignment Visualization"
echo "========================================"

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

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: --input argument is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo "Please run main.py with --save_bit_assignments first to generate the pickle file"
    exit 1
fi

# Build command
CMD="$PYTHON_CMD visualization_mixed_precision.py --input \"$INPUT_FILE\" --figsize \"$FIG_SIZE\" --dpi $DPI"

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output \"$OUTPUT_FILE\""
fi

# Print configuration
echo "Input file: $INPUT_FILE"
if [ -n "$OUTPUT_FILE" ]; then
    echo "Output file: $OUTPUT_FILE"
fi
echo "Figure size: $FIG_SIZE"
echo "DPI: $DPI"
echo "========================================"
echo ""

# Run the visualization script
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Visualization completed successfully!"
    if [ -n "$OUTPUT_FILE" ]; then
        echo "Heatmap saved to: $OUTPUT_FILE"
    fi
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Visualization failed with errors"
    echo "========================================"
    exit 1
fi

