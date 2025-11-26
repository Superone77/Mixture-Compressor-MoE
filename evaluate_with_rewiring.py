#!/usr/bin/env python
"""
Evaluation script for quantized models with Rewiring-MoE optimization

This script loads a quantized model, applies Rewiring-MoE optimization to each test sequence,
and evaluates perplexity on wikitext dataset.

Usage:
    python evaluate_with_rewiring.py \
        --model_path /path/to/quantized/model \
        --steps 5 \
        --lr 0.005

Example:
    python evaluate_with_rewiring.py \
        --model_path ./quantized_models/Mixtral-8x7B-v0.1-atten_4-e_0.25 \
        --steps 5 \
        --lr 0.005 \
        --device cuda
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from inference import load_quantized_model
from rewiring_mixtral import add_rewiring_support_to_mixtral
from datautils import get_wikitext2

# Add rewiring-moe to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rewiring-moe'))
from ours.ours import (
    compute_cross_entropy_loss,
    calc_intensity
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate quantized MoE models with Rewiring-MoE on wikitext"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the quantized model directory (contains qmodel.pt and config.json)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of optimization steps for rewiring (default: 5)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate for delta parameter optimization (default: 0.005)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="mixtral",
        choices=["mixtral", "deepseek", "olmoe", "qwen3"],
        help="Model architecture name (default: mixtral)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation (default: cuda)"
    )
    
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length for evaluation (default: 2048)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--load_args",
        type=str,
        nargs='*',
        default=[],
        help="Additional arguments to pass when loading the model"
    )
    
    return parser.parse_args()


def parse_load_args(args_str):
    """Parse key=value arguments for model loading"""
    result = {}
    for arg in args_str:
        if '=' in arg:
            key, value = arg.split('=', 1)
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


def setup_layer_weights_mixtral(model, input_ids, topk_num=2):
    """
    Setup layer weights for Mixtral model based on routing intensity.
    
    Args:
        model: The model with rewiring support
        input_ids: Input token ids
        topk_num: Number of top experts to consider
    
    Returns:
        ece_weights: Dictionary mapping layer indices to weights
        use_idx: List of layer indices to use
    """
    model.eval()
    with torch.no_grad():
        compute_cross_entropy_loss(model, input_ids)
    
    scores = model.get_scores()
    if len(scores) == 0:
        # Fallback: use all layers with equal weight
        layer_indices = {i for i, layer in enumerate(model.model.layers)
                        if hasattr(layer, 'block_sparse_moe')}
        return {idx: 1.0 for idx in layer_indices}, list(layer_indices)
    
    scores = torch.stack(scores).detach().cpu().numpy()
    n_layer, n_token, n_expert = scores.shape
    
    layer_score = np.zeros((n_layer, 3))
    
    # Calculate intensity for each layer
    for i_nlayer in range(n_layer):
        layer_score[i_nlayer, 1] = calc_intensity(scores[i_nlayer], topk_num)
    
    score_min = np.min(layer_score[:, 1])
    score_max = np.max(layer_score[:, 1])
    
    # Normalize weights
    ece_weights = {}
    layer_indices = {i for i, layer in enumerate(model.model.layers)
                    if hasattr(layer, 'block_sparse_moe')}
    
    for i, idx in enumerate(sorted(layer_indices)):
        if i < len(layer_score):
            weight = (layer_score[i, 1] - score_min) / (score_max - score_min + 1e-6)
            ece_weights[idx] = float(weight)
        else:
            ece_weights[idx] = 0.0
    
    return ece_weights, list(layer_indices)


def optimize_sequence_with_rewiring(model, input_ids, steps, lr, model_name="mixtral", topk_num=2):
    """
    Optimize delta parameters for a single sequence using rewiring.
    
    Args:
        model: Model with rewiring support
        input_ids: Input sequence token ids [1, seq_len]
        steps: Number of optimization steps
        lr: Learning rate for optimization
        model_name: Model architecture name
        topk_num: Number of top experts for intensity calculation
    
    Returns:
        Optimized model state (deltas are updated in-place)
    """
    # Freeze all parameters except deltas
    for param in model.parameters():
        param.requires_grad = False
    
    # Find MoE layers
    layer_indices = {i for i, layer in enumerate(model.model.layers)
                    if hasattr(layer, 'block_sparse_moe')}
    
    if len(layer_indices) == 0:
        print("Warning: No MoE layers found, skipping rewiring optimization")
        return
    
    # Enable delta for MoE layers
    model.enable_delta_for_layers(layer_indices)
    
    # Reset deltas to zero
    for param in model.get_delta_parameters():
        param.data.zero_()
    
    # Setup layer weights
    ece_weights, use_idx = setup_layer_weights_mixtral(model, input_ids, topk_num)
    model.set_weights(layer_indices, ece_weights)
    
    # Get delta parameters and setup optimizer
    delta_params = model.get_delta_parameters()
    if len(delta_params) == 0:
        print("Warning: No delta parameters found, skipping optimization")
        return
    
    optimizer = torch.optim.AdamW(delta_params, lr=lr, weight_decay=0.00001)
    
    # Optimize
    model.train()
    for step in range(steps):
        loss = compute_cross_entropy_loss(model, input_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    
    model.eval()


def evaluate_ppl_with_rewiring(model, testenc, device, steps, lr, model_name="mixtral", seqlen=2048):
    """
    Evaluate perplexity on wikitext with rewiring optimization for each sequence.
    
    Args:
        model: Model with rewiring support
        testenc: Encoded test data
        device: Device to use
        steps: Number of optimization steps per sequence
        lr: Learning rate for optimization
        model_name: Model architecture name
        seqlen: Sequence length
    
    Returns:
        perplexity: Perplexity score
    """
    print("Evaluating perplexity with rewiring...")
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # Process sequences one by one with rewiring
    nlls = []
    
    print(f"Processing {nsamples} sequences with rewiring optimization...")
    for i in tqdm(range(nsamples), desc="Evaluating sequences"):
        # Extract sequence
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
        
        # Optimize with rewiring for this sequence
        try:
            optimize_sequence_with_rewiring(model, batch, steps, lr, model_name)
        except Exception as e:
            print(f"Warning: Rewiring optimization failed for sequence {i}: {e}")
            # Continue with unoptimized model
        
        # Compute loss for this sequence
        with torch.no_grad():
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
        
        # Reset deltas for next sequence
        for param in model.get_delta_parameters():
            param.data.zero_()
        
        torch.cuda.empty_cache()
    
    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    
    model.config.use_cache = use_cache
    
    return ppl.item()


if __name__ == "__main__":
    args = parse_args()
    
    print("="*80)
    print("Quantized Model Evaluation with Rewiring-MoE")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Model name: {args.model_name}")
    print(f"Steps: {args.steps}")
    print(f"Learning rate: {args.lr}")
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
    
    # Load the quantized model
    model = load_quantized_model(args.model_path, kwargs)
    model.eval()
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except:
        print("Warning: Tokenizer not found in quantized model directory")
        raise
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\nModel loaded successfully!")
    
    # Add rewiring support
    if args.model_name == "mixtral":
        print("Adding rewiring support to Mixtral model...")
        model = add_rewiring_support_to_mixtral(model)
        print("Rewiring support added!")
    else:
        print(f"Warning: Rewiring support for {args.model_name} may not be fully implemented")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load wikitext data
    print("\nLoading wikitext dataset...")
    _, testenc = get_wikitext2(nsamples=0, seed=args.seed, seqlen=args.seqlen, 
                               model=args.model_path, tokenizer=tokenizer)
    print(f"Test data loaded: {testenc.input_ids.shape[1]} tokens")
    
    # Move model to device if needed
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*80)
    print("Starting evaluation with rewiring...")
    print("="*80 + "\n")
    
    # Evaluate perplexity with rewiring
    try:
        ppl = evaluate_ppl_with_rewiring(
            model=model,
            testenc=testenc,
            device=device,
            steps=args.steps,
            lr=args.lr,
            model_name=args.model_name,
            seqlen=args.seqlen
        )
        
        print("\n" + "="*80)
        print("Evaluation Results")
        print("="*80)
        print(f"Perplexity (PPL): {ppl:.4f}")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise

