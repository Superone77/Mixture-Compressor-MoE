#!/usr/bin/env python3
"""
Experiment 1.1: Expert Activation Heatmaps

Goal: Prove that calibration data fails to activate deep-layer experts.

Method:
- Take a pre-trained MoE model (e.g., DeepSeekV2-Lite)
- Run inference on the WikiText2 calibration set
- Run inference on GSM8K (and perhaps C4 or Code) validation sets
- Output: CSV with "Expert Utilization Rate" per layer

Expected Result: WikiText2 shows 0% activation for certain experts in deep layers, 
whereas GSM8K activates them.
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from deepseek_moe.modeling_deepseek import DeepseekV2ForCausalLM
from deepseek_moe.configuration_deepseek import DeepseekV2Config
from datautils import get_loaders
from eval_ppl_utils import llama_eval


class ExpertActivationHook:
    """Hook to capture expert activations during forward pass."""
    
    def __init__(self, layer_idx: int, num_experts: int, num_experts_per_tok: int = 2):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.activation_counts = defaultdict(int)  # {expert_id: count}
        self.total_tokens = 0
        
    def __call__(self, module, input_tuple, output_tuple):
        """Record expert activations from MoE forward."""
        try:
            # For DeepSeek, we hook on the DeepseekV2MoE module
            # The gate is stored in module.gate
            if hasattr(module, 'gate'):
                gate = module.gate
                route = gate.get_route()  # Get selected expert indices
                
                if route is not None:
                    # route shape: (batch * seq_len, top_k)
                    route_flat = route.view(-1)
                    num_tokens = route_flat.shape[0] // self.num_experts_per_tok
                    self.total_tokens += num_tokens
                    
                    # Count activations for each expert
                    for expert_id in route_flat.cpu().numpy():
                        if 0 <= expert_id < self.num_experts:
                            self.activation_counts[int(expert_id)] += 1
            # Alternative: hook on the layer's mlp module
            elif hasattr(module, 'mlp') and hasattr(module.mlp, 'gate'):
                gate = module.mlp.gate
                route = gate.get_route()
                
                if route is not None:
                    route_flat = route.view(-1)
                    num_tokens = route_flat.shape[0] // self.num_experts_per_tok
                    self.total_tokens += num_tokens
                    
                    for expert_id in route_flat.cpu().numpy():
                        if 0 <= expert_id < self.num_experts:
                            self.activation_counts[int(expert_id)] += 1
        except Exception as e:
            # Silently skip if there's an error
            pass
    
    def get_utilization_rates(self):
        """Get expert utilization rates as percentages."""
        if self.total_tokens == 0:
            return {expert_id: 0.0 for expert_id in range(self.num_experts)}
        
        utilization = {}
        for expert_id in range(self.num_experts):
            count = self.activation_counts.get(expert_id, 0)
            # Each token activates num_experts_per_tok experts
            # So utilization rate = count / (total_tokens * num_experts_per_tok)
            utilization[expert_id] = (count / (self.total_tokens * self.num_experts_per_tok)) * 100.0
        
        return utilization
    
    def reset(self):
        """Reset counters."""
        self.activation_counts = defaultdict(int)
        self.total_tokens = 0


@torch.no_grad()
def evaluate_expert_activation(model, dataloader, device, dataset_name: str, num_layers: int, num_experts: int):
    """Evaluate expert activation rates on a dataset."""
    print(f"\nEvaluating expert activation on {dataset_name}...")
    
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # Register hooks for all layers
    hooks = []
    layers = model.model.layers
    
    for i in range(num_layers):
        layer = layers[i]
        # For DeepSeek, mlp is DeepseekV2MoE which has a gate
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            hook = ExpertActivationHook(i, num_experts, num_experts_per_tok)
            # Hook on the mlp (DeepseekV2MoE) module
            handle = layer.mlp.register_forward_hook(hook)
            hooks.append((i, hook, handle))
    
    # Process data
    layers[0] = layers[0].to(device)
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    
    dtype = next(iter(model.parameters())).dtype
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    
    # Collect input activations
    testenc = dataloader.input_ids
    nsamples = testenc.numel() // model.seqlen
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(device)
        try:
            model(batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()
    
    # Forward through all layers
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    for i in range(num_layers):
        layer = layers[i].to(device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    # Collect results
    results = []
    for layer_idx, hook, handle in hooks:
        utilization = hook.get_utilization_rates()
        for expert_id, rate in utilization.items():
            results.append({
                'dataset': dataset_name,
                'layer': layer_idx,
                'expert_id': expert_id,
                'utilization_rate': rate
            })
    
    # Remove hooks
    for _, _, handle in hooks:
        handle.remove()
    
    model.config.use_cache = use_cache
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1.1: Expert Activation Heatmaps"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='deepseek-ai/DeepSeek-V2-Lite',
        help='Model name or path'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='experiment_1_1_expert_activation.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--nsamples',
        type=int,
        default=128,
        help='Number of calibration samples'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--seqlen',
        type=int,
        default=2048,
        help='Sequence length'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--attn_implementation',
        type=str,
        default='eager',
        choices=['eager', 'sdpa', 'flash_attention_2'],
        help='Attention implementation'
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("Experiment 1.1: Expert Activation Heatmaps")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Number of samples: {args.nsamples}")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    config = DeepseekV2Config.from_pretrained(
        args.model, 
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    model = DeepseekV2ForCausalLM.from_pretrained(
        args.model, 
        config=config, 
        device_map='cpu',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    model.seqlen = args.seqlen
    
    num_layers = len(model.model.layers)
    num_experts = getattr(config, 'n_routed_experts', 64)
    num_experts_per_tok = getattr(config, 'num_experts_per_tok', 2)
    
    print(f"Model has {num_layers} layers, {num_experts} experts, {num_experts_per_tok} experts per token")
    
    # Evaluate on different datasets
    all_results = []
    datasets = ['wikitext2', 'gsm8k']
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Load data
        dataloader, testloader = get_loaders(
            dataset_name,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=args.seqlen,
        )
        
        # Evaluate expert activation
        results = evaluate_expert_activation(
            model, 
            dataloader, 
            args.device, 
            dataset_name, 
            num_layers, 
            num_experts
        )
        all_results.extend(results)
    
    # Save results to CSV
    if all_results:
        print(f"\nSaving results to {args.output_csv}...")
        df = pd.DataFrame(all_results)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved {len(all_results)} rows to {args.output_csv}")
        
        # Print summary
        print("\n" + "="*80)
        print("Summary Statistics")
        print("="*80)
        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset]
            print(f"\n{dataset}:")
            print(f"  Total layers: {dataset_df['layer'].nunique()}")
            print(f"  Average utilization rate: {dataset_df['utilization_rate'].mean():.2f}%")
            print(f"  Experts with 0% activation: {(dataset_df['utilization_rate'] == 0).sum()}")
            print(f"  Experts with >0% activation: {(dataset_df['utilization_rate'] > 0).sum()}")
    else:
        print("No activation data collected")
    
    print("\n" + "="*80)
    print("Experiment 1.1 complete!")
    print("="*80)


if __name__ == "__main__":
    main()

