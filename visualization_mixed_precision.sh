#!/bin/bash
# Shell script to visualize mixed precision bit assignments from CSV file

set -e  # Exit on error

# Default values
INPUT_CSV=""
OUTPUT_PNG=""
FIGSIZE="16,10"
DPI=300

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT_CSV="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_PNG="$2"
            shift 2
            ;;
        --figsize|-f)
            FIGSIZE="$2"
            shift 2
            ;;
        --dpi|-d)
            DPI="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --input <csv_file> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input, -i    Path to CSV file containing bit assignments (required)"
            echo "  --output, -o   Output path for heatmap PNG (default: <input_name>_heatmap.png)"
            echo "  --figsize, -f  Figure size as width,height (default: 16,10)"
            echo "  --dpi, -d      Resolution for saved figure (default: 300)"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --input bit_assignments.csv --output heatmap.png"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if input file is provided
if [ -z "$INPUT_CSV" ]; then
    echo "Error: --input argument is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input file not found: $INPUT_CSV"
    exit 1
fi

# Build command
CMD="python visualization_mixed_precision.py --input \"$INPUT_CSV\" --figsize \"$FIGSIZE\" --dpi $DPI"

if [ -n "$OUTPUT_PNG" ]; then
    CMD="$CMD --output \"$OUTPUT_PNG\""
fi

# Run the visualization script
echo "Running visualization..."
echo "Command: $CMD"
echo ""

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "Visualization completed successfully!"
    if [ -z "$OUTPUT_PNG" ]; then
        # Extract output filename (same as input but with _heatmap.png)
        OUTPUT_PNG="${INPUT_CSV%.*}_heatmap.png"
    fi
    echo "Output saved to: $OUTPUT_PNG"
else
    echo "Error: Visualization failed"
    exit 1
fi

