#!/usr/bin/env python3
"""
Monitor expert activation during OLMoE inference on WikiText dataset.

This script monitors which experts are activated and their router weights
for each token generated during inference on WikiText samples.
"""

import torch
import torch.nn as nn
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import csv
from typing import Dict, List, Tuple
import sys
import os


class ExpertMonitoringHook:
    """Hook to capture expert activations and weights during forward pass."""
    
    def __init__(self, num_experts: int, num_experts_per_tok: int = 2):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.data = []
        self.generation_step = 0  # Track generation step
        
    def __call__(self, module, input_tuple, output_tuple):
        """Record expert activations and weights from MoE forward."""
        try:
            hidden_states = input_tuple[0]
            
            # Check if we have router_logits from the output
            if isinstance(output_tuple, tuple) and len(output_tuple) == 2:
                final_hidden_states, router_logits = output_tuple
            else:
                # If no router_logits in output, skip
                return
            
            # Handle router_logits shape: (batch * seq_len, num_experts)
            if hidden_states.dim() == 3:
                batch_size, seq_len, hidden_dim = hidden_states.shape
            else:
                batch_size = 1
                seq_len = 1
                hidden_dim = hidden_states.shape[-1]
            
            # Ensure router_logits has the right shape
            if router_logits.dim() != 2:
                return
            
            # Compute routing from router_logits
            routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
            
            # Get normalized weights for selected experts
            selected_weights = routing_weights.gather(1, selected_experts)
            selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)
            
            # For generation, we're typically only interested in the last token (the one being generated)
            # Only record the last position (most recent generated token)
            num_positions = batch_size * seq_len
            last_pos = min(num_positions, router_logits.shape[0]) - 1
            
            if last_pos >= 0:
                # Get selected experts and their weights for the last position
                pos_selected = selected_experts[last_pos].cpu().tolist()
                pos_weights = selected_weights[last_pos].cpu().tolist()
                
                # Only store activated experts with their weights
                activated_experts = []
                for expert_id, weight in zip(pos_selected, pos_weights):
                    if expert_id < self.num_experts:  # Safety check
                        activated_experts.append((expert_id, float(weight)))
                
                self.data.append({
                    'activated_experts': activated_experts,
                    'generation_step': self.generation_step
                })
            
            self.generation_step += 1
            
        except Exception as e:
            # Silently skip if there's an error in hook
            pass
    
    def reset(self):
        self.data = []
        self.generation_step = 0


def find_moe_modules(model):
    """Find all MoE modules in the model."""
    moe_modules = []
    for name, module in model.named_modules():
        # Look for typical MoE module patterns
        if (hasattr(module, 'num_experts') or 
            hasattr(module, 'experts') or 
            hasattr(module, 'gate') or
            'moe' in name.lower() or
            'block_sparse_moe' in name):
            # Additional check to ensure it's actually a MoE module
            if (hasattr(module, 'gate') and 
                (hasattr(module, 'experts') or hasattr(module, 'num_experts'))):
                moe_modules.append((name, module))
    return moe_modules


def load_wikitext_dataset(num_samples: int = 5, seq_len: int = 128, sample_ids: List[int] = None):
    """Load WikiText dataset and return samples."""
    print(f"Loading WikiText dataset (need {num_samples} valid samples)...")
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    except Exception as e:
        print(f"Error loading WikiText: {e}")
        print("Trying to load from local cache or downloading...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir='./cache')
    
    # Select samples
    selected_samples = []
    
    if sample_ids is not None:
        # Use specified sample IDs
        print(f"Using specified sample IDs: {sample_ids}")
        for sample_id in sample_ids:
            if sample_id >= len(dataset):
                print(f"Warning: Sample ID {sample_id} is out of range (dataset size: {len(dataset)})")
                continue
                
            text = dataset[sample_id]['text']
            
            # Skip empty or very short samples
            if not text or len(text.strip()) < 20:
                print(f"Warning: Sample {sample_id} is empty or too short")
                continue
            
            prompt = text[:seq_len].strip()
            
            if len(prompt) < 20:
                print(f"Warning: Prompt for sample {sample_id} is too short")
                continue
            
            selected_samples.append({
                'text': text,
                'prompt': prompt,
                'sample_id': sample_id
            })
            print(f"  Loaded sample {len(selected_samples)} (ID: {sample_id})")
    else:
        # Find first num_samples valid samples
        checked = 0
        
        for i in range(len(dataset)):
            checked += 1
            text = dataset[i]['text']
            
            # Skip empty or very short samples
            if not text or len(text.strip()) < 20:
                continue
            
            # Take first seq_len characters as prompt
            prompt = text[:seq_len].strip()
            
            # Skip if prompt is too short or empty
            if len(prompt) < 20:
                continue
                
            selected_samples.append({
                'text': text,
                'prompt': prompt,
                'sample_id': i
            })
            
            print(f"  Loaded sample {len(selected_samples)}/{num_samples} (checked {checked} samples)")
            
            if len(selected_samples) >= num_samples:
                break
    
    if not selected_samples:
        # Try train split as fallback
        print("\nNo valid test samples found, trying train split...")
        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            for i in range(len(dataset)):
                text = dataset[i]['text']
                if not text or len(text.strip()) < 20:
                    continue
                
                prompt = text[:seq_len].strip()
                if len(prompt) < 20:
                    continue
                    
                selected_samples.append({
                    'text': text,
                    'prompt': prompt
                })
                
                if len(selected_samples) >= num_samples:
                    break
        except Exception as e:
            print(f"Error loading train split: {e}")
    
    print(f"\nLoaded {len(selected_samples)} valid samples")
    return selected_samples


@torch.no_grad()
def generate_with_monitoring(model, tokenizer, prompt: str, max_new_tokens: int = 256, hooks: List = None):
    """Generate text while monitoring expert activations."""
    # Reset hooks
    if hooks:
        for _, hook, _ in hooks:
            hook.reset()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_ids = inputs['input_ids']
    
    # Try different generation strategies
    with torch.no_grad():
        try:
            # First try with output_router_logits
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_router_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            )
        except RuntimeError as e:
            if "load_balancing_loss_func" in str(e) or "size of tensor" in str(e):
                print(f"Warning: Generation error, trying manual generation: {e}")
                # Use manual generation instead
                return manual_generate_with_monitoring(model, tokenizer, prompt, max_new_tokens, hooks)
            else:
                raise e
    
    generated_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text, generated_ids


@torch.no_grad()
def manual_generate_with_monitoring(model, tokenizer, prompt: str, max_new_tokens: int = 256, hooks: List = None):
    """Manual generation with monitoring when model.generate fails."""
    # Reset hooks
    if hooks:
        for _, hook, _ in hooks:
            hook.reset()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_ids = inputs['input_ids']
    
    generated_ids = input_ids.clone()
    
    # Generate tokens one by one
    for step in range(max_new_tokens):
        # Forward pass
        outputs = model(input_ids=generated_ids, output_router_logits=True)
        
        # Get logits
        logits = outputs.logits[:, -1, :]  # Last token logits
        
        # Get next token
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Append to sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # Check for EOS token
        if tokenizer.eos_token_id and next_token.item() == tokenizer.eos_token_id:
            break
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text, generated_ids[0]


def main():
    parser = argparse.ArgumentParser(
        description="Monitor OLMoE expert activations during WikiText inference"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='allenai/OLMoE-1B-7B-0125-Instruct',
        help='Model name or path'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of WikiText samples to process (default: 5)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=256,
        help='Maximum number of new tokens to generate (default: 256)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='expert_activations_wikitext.csv',
        help='Output CSV file path (default: expert_activations_wikitext.csv)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--prompt_length',
        type=int,
        default=128,
        help='Length of prompt in characters (default: 128)'
    )
    parser.add_argument(
        '--sample_ids',
        type=str,
        default=None,
        help='Comma-separated list of sample IDs to use (e.g., "0,5,10"). If specified, --num_samples is ignored.'
    )
    
    args = parser.parse_args()
    
    # Parse sample_ids if provided
    sample_ids = None
    if args.sample_ids:
        try:
            sample_ids = [int(x.strip()) for x in args.sample_ids.split(',')]
            print(f"Will process specific sample IDs: {sample_ids}")
        except ValueError:
            print(f"Error: Invalid sample_ids format: {args.sample_ids}")
            print("Example: --sample_ids '0,5,10'")
            return
    
    # Auto-detect device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("OLMoE Expert Activation Monitor - WikiText")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Prompt length: {args.prompt_length}")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
        device_map=args.device
    )
    model.eval()
    
    # Get model configuration
    config = model.config
    num_experts = getattr(config, 'num_experts', 8)
    num_experts_per_tok = getattr(config, 'num_experts_per_tok', 2)
    
    print(f"Model has {num_experts} experts, selecting {num_experts_per_tok} per token")
    
    # Find MoE modules
    print("\nFinding MoE modules...")
    moe_modules = find_moe_modules(model)
    print(f"Found {len(moe_modules)} MoE modules")
    
    # Register hooks
    hooks = []
    for name, module in moe_modules:
        hook = ExpertMonitoringHook(num_experts, num_experts_per_tok)
        handle = module.register_forward_hook(hook)
        hooks.append((name, hook, handle))
        print(f"Registered hook for {name}")
    
    # Load WikiText dataset
    print("\nLoading WikiText dataset...")
    wikitext_samples = load_wikitext_dataset(args.num_samples, args.prompt_length, sample_ids)
    
    if not wikitext_samples:
        print("Error: No valid samples loaded from WikiText dataset")
        return
    
    print(f"Loaded {len(wikitext_samples)} samples")
    
    # Process each sample
    all_results = []
    
    for sample_idx, sample in enumerate(wikitext_samples):
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx + 1}/{len(wikitext_samples)}")
        print(f"{'='*80}")
        print(f"Prompt: {sample['prompt'][:100]}...")
        print(f"{'='*80}")
        
        # Generate response
        prompt = sample['prompt']
        generated_text, generated_ids = generate_with_monitoring(
            model, tokenizer, prompt, args.max_new_tokens, hooks
        )
        
        print(f"\nGenerated text (first 200 chars): {generated_text[:200]}...")
        
        # Collect activation data for this sample
        # Organize by generation step first
        activation_by_step = {}  # {step: {layer: data}}
        
        for layer_name, hook, _ in hooks:
            for data in hook.data:
                gen_step = data.get('generation_step', 0)
                if gen_step not in activation_by_step:
                    activation_by_step[gen_step] = {}
                activation_by_step[gen_step][layer_name] = data['activated_experts']
        
        # Now record data organized by token index
        for gen_step in sorted(activation_by_step.keys()):
            for layer_name, activated_experts in activation_by_step[gen_step].items():
                # Format: expert_id1,weight1;expert_id2,weight2;...
                experts_str = ';'.join([f'{eid},{w:.6f}' for eid, w in activated_experts])
                
                row = {
                    'sample_index': sample_idx,
                    'original_sample_id': sample.get('sample_id', sample_idx),
                    'layer': layer_name,
                    'token_index': gen_step,
                    'num_activated_experts': len(activated_experts),
                    'experts_and_weights': experts_str
                }
                all_results.append(row)
        
        # Reset hooks for next sample
        for _, hook, _ in hooks:
            hook.reset()
    
    # Save results
    if all_results:
        print(f"\nSaving results to {args.output}...")
        df = pd.DataFrame(all_results)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(all_results)} rows to {args.output}")
    else:
        print("No activation data to save")
    
    # Cleanup
    for _, _, handle in hooks:
        handle.remove()
    
    print("\n" + "="*80)
    print("Monitoring complete!")
    print(f"Results saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()

