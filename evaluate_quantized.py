#!/usr/bin/env python
"""
Evaluation script for quantized models using lm_eval

This script loads a quantized model and evaluates it on various tasks using lm_eval.
Results are displayed in table format.

Usage:
    python evaluate_quantized.py \
        --model_path /path/to/quantized/model \
        --tasks mmlu,hellaswag,winogrande \
        --batch_size 4

Example:
    python evaluate_quantized.py \
        --model_path ./quantized_models/Mixtral-8x7B-v0.1-atten_4-e_0.25 \
        --tasks mmlu,hellaswag,winogrande,arc,truthfulqa \
        --batch_size 4 \
        --device cuda
"""

import os
import sys
import argparse
import torch

# Add AlphaQuant to path for eval_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AlphaQuant'))

from transformers import AutoTokenizer
from inference import load_quantized_model

from eval_utils import tasks_evaluate



def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate quantized MoE models using lm_eval"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the quantized model directory (contains qmodel.pt and config.json)"
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu,hellaswag,winogrande,arc,truthfulqa",
        help="Comma-separated list of tasks to evaluate on (default: mmlu,hellaswag,winogrande,arc,truthfulqa)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 4)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation (default: cuda)"
    )
    
    parser.add_argument(
        "--load_args",
        type=str,
        nargs='*',
        default=[],
        help="Additional arguments to pass when loading the model (e.g., device_map=auto torch_dtype=torch.float16)"
    )
    
    return parser.parse_args()


def parse_load_args(args_str):
    """Parse key=value arguments for model loading"""
    result = {}
    for arg in args_str:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Convert string values to appropriate types
            if value.lower() == 'true':
                result[key] = True
            elif value.lower() == 'false':
                result[key] = False
            else:
                try:
                    result[key] = int(value)
                except ValueError:
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value
        else:
            result[arg] = True
    return result


def main():
    args = parse_args()
    
    print("="*80)
    print("Quantized Model Evaluation")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Setup kwargs for model loading
    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16
    }
    
    # Parse additional arguments if provided
    if args.load_args:
        additional = parse_load_args(args.load_args)
        kwargs.update(additional)
    
    print(f"\nLoading quantized model from: {args.model_path}")
    print(f"Model loading kwargs: {kwargs}")
    
    # Load the quantized model
    model = load_quantized_model(args.model_path, kwargs)
    model.eval()
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except:
        # If tokenizer not found, try loading from the original model path
        print("Warning: Tokenizer not found in quantized model directory")
        print("Make sure tokenizer files are in the model path")
    
    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80 + "\n")
    
    # Parse tasks
    task_list = [task.strip() for task in args.tasks.split(',')]
    
    # Run evaluation
    tasks_evaluate(
        model=model,
        tasks=task_list,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()

