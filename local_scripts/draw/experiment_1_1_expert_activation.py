#!/usr/bin/env python3
"""
Experiment 1.1: Expert Activation Heatmaps

Goal: Prove that calibration data fails to activate deep-layer experts.

Method:
- Run inference on WikiText2 calibration set
- Run inference on GSM8K validation set
- Track expert utilization rate per layer
- Output CSV with: layer, expert_id, dataset, activation_rate
"""

import torch
import torch.nn as nn
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from datautils import get_loaders
from tqdm import tqdm
import os
from collections import defaultdict
import re


class ExpertActivationMonitor:
    """Monitor expert activation rates during forward pass."""
    
    def __init__(self, num_experts=8, num_experts_per_tok=2):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.activation_counts = defaultdict(lambda: defaultdict(int))  # {layer_idx: {expert_id: count}}
        self.total_tokens = defaultdict(int)  # {layer_idx: total_tokens}
        
    def __call__(self, module, input_tuple, output_tuple):
        """Record expert activations from MoE forward."""
        try:
            if not input_tuple:
                return
            hidden_states = input_tuple[0]
            
            if not torch.is_tensor(hidden_states):
                return
            
            # Find gate/router layer
            gate_layer = None
            if hasattr(module, 'gate') and isinstance(getattr(module, 'gate'), nn.Module):
                gate_layer = getattr(module, 'gate')
            elif hasattr(module, 'router') and isinstance(getattr(module, 'router'), nn.Module):
                gate_layer = getattr(module, 'router')
            else:
                return
            
            # Ensure hidden_states shape: (batch, seq, hidden)
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)
            if hidden_states.dim() != 3:
                return
            
            # Compute router logits
            with torch.no_grad():
                router_logits_bs = gate_layer(hidden_states)
            
            if router_logits_bs.dim() != 3:
                return
            
            batch_size, seq_len, num_experts = router_logits_bs.shape
            router_logits = router_logits_bs.reshape(batch_size * seq_len, num_experts)
            
            routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
            
            # Extract layer index from module name
            layer_idx = self._extract_layer_idx(module)
            if layer_idx is None:
                return
            
            # Count activations for each token position
            num_positions = batch_size * seq_len
            self.total_tokens[layer_idx] += num_positions
            
            for pos in range(num_positions):
                for expert_id in selected_experts[pos].cpu().tolist():
                    if 0 <= expert_id < self.num_experts:
                        self.activation_counts[layer_idx][int(expert_id)] += 1
                        
        except Exception as e:
            # Silently skip errors
            pass
    
    def _extract_layer_idx(self, module):
        """Extract layer index from module name."""
        # Try to find layer index from module's name or parent
        # We'll set this via the hook wrapper
        return getattr(self, '_current_layer_idx', None)
    
    def get_activation_rates(self):
        """Get activation rates per layer and expert."""
        results = []
        for layer_idx in sorted(self.activation_counts.keys()):
            total = self.total_tokens[layer_idx]
            if total == 0:
                continue
            for expert_id in range(self.num_experts):
                count = self.activation_counts[layer_idx][expert_id]
                rate = count / total if total > 0 else 0.0
                results.append({
                    'layer': layer_idx,
                    'expert_id': expert_id,
                    'activation_count': count,
                    'total_tokens': total,
                    'activation_rate': rate
                })
        return results
    
    def reset(self):
        """Reset counters."""
        self.activation_counts = defaultdict(lambda: defaultdict(int))
        self.total_tokens = defaultdict(int)


def find_moe_modules(model):
    """Find all MoE modules in the model."""
    moe_modules = []
    for name, module in model.named_modules():
        if 'block_sparse_moe' in name and hasattr(module, 'gate'):
            moe_modules.append((name, module))
    return moe_modules


@torch.no_grad()
def evaluate_dataset(model, dataloader, device, dataset_name, num_experts=8, num_experts_per_tok=2):
    """Evaluate model on a dataset and track expert activations."""
    print(f"\nEvaluating on {dataset_name}...")
    
    model.eval()
    model.config.use_cache = False
    
    # Find MoE modules and register hooks
    moe_modules = find_moe_modules(model)
    print(f"Found {len(moe_modules)} MoE modules")
    
    monitor = ExpertActivationMonitor(num_experts, num_experts_per_tok)
    hooks = []
    
    for name, module in moe_modules:
        # Store layer index in monitor for extraction
        match = re.search(r'layers\.(\d+)\.block_sparse_moe', name)
        if match:
            layer_idx = int(match.group(1))
            # Create a wrapper to pass layer_idx
            def make_hook(layer_idx):
                def hook(module, input_tuple, output_tuple):
                    # Temporarily store layer_idx in monitor
                    monitor._current_layer_idx = layer_idx
                    monitor(module, input_tuple, output_tuple)
                    monitor._current_layer_idx = None
                return hook
            handle = module.register_forward_hook(make_hook(layer_idx))
            hooks.append((name, handle))
    
    # Process data
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    layers[0] = layers[0].to(device)
    
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
    
    # Collect inputs
    nsamples = len(dataloader)
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    
    layers[0] = Catcher(layers[0])
    for i, (inp, _) in enumerate(tqdm(dataloader, desc=f"Collecting inputs ({dataset_name})")):
        batch = inp.to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()
    
    # Process through layers
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    for i in range(len(layers)):
        print(f"Processing layer {i}/{len(layers)}...")
        layer = layers[i].to(device)
        
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    # Get activation rates
    activation_rates = monitor.get_activation_rates()
    
    # Add dataset name
    for result in activation_rates:
        result['dataset'] = dataset_name
    
    # Cleanup hooks
    for _, handle in hooks:
        handle.remove()
    
    return activation_rates


def main():
    parser = argparse.ArgumentParser(description="Experiment 1.1: Expert Activation Heatmaps")
    parser.add_argument('--model', type=str, default='mistralai/Mixtral-8x7B-v0.1', help='Model name or path')
    parser.add_argument('--output', type=str, default='experiment_1_1_activation_rates.csv', help='Output CSV file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    parser.add_argument('--attn_implementation', type=str, default='eager', choices=['eager', 'sdpa', 'flash_attention_2'])
    
    args = parser.parse_args()
    
    print("="*80)
    print("Experiment 1.1: Expert Activation Heatmaps")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Number of samples: {args.nsamples}")
    print("="*80)
    
    # Load model
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        config=config, 
        device_map='cpu',
        torch_dtype=torch.float16
    )
    
    assert isinstance(model, MixtralForCausalLM), 'Model must be Mixtral!'
    model.seqlen = args.seqlen
    
    num_experts = getattr(config, 'num_local_experts', 8)
    num_experts_per_tok = getattr(config, 'num_experts_per_tok', 2)
    
    print(f"Model has {num_experts} experts, selecting {num_experts_per_tok} per token")
    
    # Evaluate on WikiText2
    print("\n" + "="*80)
    print("Evaluating on WikiText2 calibration set...")
    print("="*80)
    os.environ.pop('GSM8K_FIELD', None)  # Ensure default field
    dataloader_wiki, _ = get_loaders('wikitext2', nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, model=args.model)
    results_wiki = evaluate_dataset(model, dataloader_wiki, args.device, 'wikitext2', num_experts, num_experts_per_tok)
    
    # Evaluate on GSM8K
    print("\n" + "="*80)
    print("Evaluating on GSM8K validation set...")
    print("="*80)
    os.environ['GSM8K_FIELD'] = 'question'  # Use question field
    dataloader_gsm8k, _ = get_loaders('gsm8k', nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, model=args.model)
    results_gsm8k = evaluate_dataset(model, dataloader_gsm8k, args.device, 'gsm8k', num_experts, num_experts_per_tok)
    
    # Combine results
    all_results = results_wiki + results_gsm8k
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    df = df[['layer', 'expert_id', 'dataset', 'activation_rate', 'activation_count', 'total_tokens']]
    df.to_csv(args.output, index=False)
    
    print("\n" + "="*80)
    print("Results saved to:", args.output)
    print(f"Total rows: {len(all_results)}")
    print("="*80)
    
    # Print summary
    print("\nSummary:")
    print(f"WikiText2: {len(results_wiki)} layer-expert pairs")
    print(f"GSM8K: {len(results_gsm8k)} layer-expert pairs")
    
    # Show some statistics
    wiki_df = df[df['dataset'] == 'wikitext2']
    gsm8k_df = df[df['dataset'] == 'gsm8k']
    
    print(f"\nWikiText2 - Average activation rate: {wiki_df['activation_rate'].mean():.4f}")
    print(f"GSM8K - Average activation rate: {gsm8k_df['activation_rate'].mean():.4f}")
    
    # Show layers with zero activations in WikiText2
    zero_activation_wiki = wiki_df[wiki_df['activation_rate'] == 0.0]
    if len(zero_activation_wiki) > 0:
        print(f"\nWikiText2 has {len(zero_activation_wiki)} layer-expert pairs with 0% activation")
        print("Sample (first 10):")
        print(zero_activation_wiki[['layer', 'expert_id']].head(10))


if __name__ == "__main__":
    main()

