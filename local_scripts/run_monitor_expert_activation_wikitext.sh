#!/bin/bash
# Run script for monitoring expert activation with WikiText dataset

# Default values
MODEL_NAME="allenai/OLMoE-1B-7B-0125-Instruct"
NUM_SAMPLES=5
MAX_NEW_TOKENS=256
OUTPUT="expert_activations_wikitext.csv"
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_name MODEL       Model name or path (default: allenai/OLMoE-1B-7B-0125-Instruct)"
            echo "  --num_samples N           Number of WikiText samples to process (default: 5)"
            echo "  --max_new_tokens N       Maximum number of new tokens to generate (default: 256)"
            echo "  --output FILE            Output CSV file path (default: expert_activations_wikitext.csv)"
            echo "  --device DEVICE          Device to use: cuda or cpu (default: cuda)"
            echo "  --help                   Show this help message"
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
echo "OLMoE Expert Activation Monitor - WikiText"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Device: $DEVICE"
echo "Number of samples: $NUM_SAMPLES"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Output file: $OUTPUT"
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

# Run the Python script
$PYTHON_CMD monitor_expert_activation_wikitext.py \
    --model_name "$MODEL_NAME" \
    --num_samples "$NUM_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output "$OUTPUT" \
    --device "$DEVICE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Script completed successfully!"
    echo "Results saved to: $OUTPUT"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Script failed with errors"
    echo "========================================"
    exit 1
fi

