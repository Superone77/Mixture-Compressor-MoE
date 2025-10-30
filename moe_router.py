#!/usr/bin/env python3
"""
Monitor expert activation during Mixtral MoE inference on GSM8K dataset.

This script monitors which experts are activated and their router weights
for each token generated during inference on GSM8K problems.
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
    """Hook to capture expert activations and weights during forward pass (Mixtral MoE).

    For Mixtral, router logits are computed by the module's gate/router linear layer
    given the input hidden_states. We compute softmax and select top-k experts.
    """
    
    def __init__(self, num_experts: int, num_experts_per_tok: int = 2):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.data = []
        self.generation_step = 0  # Track generation step
        
    def __call__(self, module, input_tuple, output_tensor):
        """Record expert activations and weights from Mixtral MoE forward.
        We recompute router logits using module.gate/module.router on incoming hidden_states.
        """
        try:
            if not input_tuple:
                return
            hidden_states = input_tuple[0]

            # Accept only tensor hidden states
            if not torch.is_tensor(hidden_states):
                return

            # Discover gate/router linear
            gate_layer = None
            if hasattr(module, 'gate') and isinstance(getattr(module, 'gate'), nn.Module):
                gate_layer = getattr(module, 'gate')
            elif hasattr(module, 'router') and isinstance(getattr(module, 'router'), nn.Module):
                gate_layer = getattr(module, 'router')
            else:
                # Cannot compute router logits from this module
                return

            # Ensure hidden_states shape: (batch, seq, hidden)
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)
            if hidden_states.dim() != 3:
                return

            # Compute router logits (batch, seq, num_experts)
            with torch.no_grad():
                router_logits_bs = gate_layer(hidden_states)

            if router_logits_bs.dim() != 3:
                return

            batch_size, seq_len, num_experts = router_logits_bs.shape
            # Flatten positions
            router_logits = router_logits_bs.reshape(batch_size * seq_len, num_experts)

            routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)

            # Get normalized weights for selected experts
            selected_weights = routing_weights.gather(1, selected_experts)
            selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)

            # Only record the last position (most recent generated token)
            num_positions = batch_size * seq_len
            last_pos = min(num_positions, router_logits.shape[0]) - 1
            
            if last_pos >= 0:
                pos_selected = selected_experts[last_pos].detach().cpu().tolist()
                pos_weights = selected_weights[last_pos].detach().cpu().tolist()
                
                activated_experts = []
                for expert_id, weight in zip(pos_selected, pos_weights):
                    if int(expert_id) < self.num_experts:
                        activated_experts.append((int(expert_id), float(weight)))
                
                self.data.append({
                    'activated_experts': activated_experts,
                    'generation_step': self.generation_step
                })
            
            self.generation_step += 1
            
        except Exception:
            # Silently skip if there's an error in hook
            pass
    
    def reset(self):
        self.data = []
        self.generation_step = 0


def find_moe_modules(model):
    """Find Mixtral MoE modules in the model.
    We look for modules named with 'block_sparse_moe' that have gate/router and experts.
    """
    moe_modules = []
    for name, module in model.named_modules():
        if 'block_sparse_moe' in name:
            has_gate = hasattr(module, 'gate') or hasattr(module, 'router')
            has_experts = hasattr(module, 'experts') or hasattr(module, 'num_experts')
            if has_gate and has_experts:
                moe_modules.append((name, module))
    return moe_modules


def load_gsm8k_dataset(num_samples: int = 5, sample_ids: List[int] = None):
    """Load GSM8K dataset and return samples."""
    print(f"Loading GSM8K dataset...")
    try:
        dataset = load_dataset('gsm8k', 'main', split='test')
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        print("Trying to load from local cache or downloading...")
        dataset = load_dataset('gsm8k', 'main', split='test', cache_dir='./cache')
    
    # Select samples
    selected_samples = []
    
    if sample_ids is not None:
        # Use specified sample IDs
        print(f"Using specified sample IDs: {sample_ids}")
        for sample_id in sample_ids:
            if sample_id < len(dataset):
                selected_samples.append({
                    'question': dataset[sample_id]['question'],
                    'answer': dataset[sample_id]['answer'],
                    'sample_id': sample_id
                })
            else:
                print(f"Warning: Sample ID {sample_id} is out of range (dataset size: {len(dataset)})")
    else:
        # Use first num_samples samples
        print(f"Using first {num_samples} samples")
        for i in range(min(num_samples, len(dataset))):
            selected_samples.append({
                'question': dataset[i]['question'],
                'answer': dataset[i]['answer'],
                'sample_id': i
            })
    
    return selected_samples


@torch.no_grad()
def generate_with_monitoring(model, tokenizer, prompt: str, max_new_tokens: int = 512, hooks: List = None):
    """Generate text while monitoring expert activations.
    For Mixtral, we do not rely on output_router_logits; hooks compute them.
    """
    # Reset hooks
    if hooks:
        for _, hook, _ in hooks:
            hook.reset()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_ids = inputs['input_ids']
    
    # Use standard generation; hooks will compute router info
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        )
    
    generated_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text, generated_ids


@torch.no_grad()
def manual_generate_with_monitoring(model, tokenizer, prompt: str, max_new_tokens: int = 512, hooks: List = None):
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
        outputs = model(input_ids=generated_ids)
        
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
        description="Monitor Mixtral MoE expert activations during GSM8K inference"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='mistralai/Mixtral-8x7B-v0.1',
        help='Model name or path'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of GSM8K samples to process (default: 5)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='Maximum number of new tokens to generate (default: 512)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='expert_activations.csv',
        help='Output CSV file path (default: expert_activations.csv)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--sample_ids',
        type=str,
        default=None,
        help='Comma-separated list of sample IDs to use (e.g., "0,5,10"). If specified, --num_samples is ignored.'
    )
    parser.add_argument(
        '--quantized_model_dir',
        type=str,
        default=None,
        help='Directory of the quantized model (from main.py). If set, overrides --model_name.'
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
    print("Mixtral MoE Expert Activation Monitor")
    print("="*80)
    print(f"Model: {args.model_name if not args.quantized_model_dir else args.quantized_model_dir}")
    print(f"Device: {args.device}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("="*80)
    
    # Helper: load quantized model if requested
    def load_quantized_model(model_dir, torch_dtype, device):
        """
        Load a quantized model saved by main.py (Mixtral format).
        """
        import torch
        import os
        from transformers import AutoConfig, AutoModelForCausalLM

        config_path = os.path.join(model_dir, 'config.json')
        model_path = os.path.join(model_dir, 'qmodel.pt')
        assert os.path.exists(config_path) and os.path.exists(model_path), \
            f"Quantized model not found in {model_dir}"

        config = AutoConfig.from_pretrained(config_path)
        model_loaded = AutoModelForCausalLM.from_config(config)
        state = torch.load(model_path, map_location=device)
        model_loaded.load_state_dict(state, strict=False)
        model_loaded = model_loaded.to(device)
        model_loaded = model_loaded.to(dtype=torch_dtype)
        print(f"Loaded quantized model from {model_path}")
        return model_loaded
    
    # Load tokenizer and model
    if args.quantized_model_dir:
        print("Loading quantized model and tokenizer from directory:", args.quantized_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.quantized_model_dir)
        model = load_quantized_model(
            args.quantized_model_dir,
            torch.float16 if args.device == 'cuda' else torch.float32,
            args.device
        )
    else:
        print("Loading HF model and tokenizer from:", args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
            device_map=args.device
        )
    model.eval()
    
    # Get model configuration (fallback defaults if not present)
    config = model.config
    num_experts = getattr(config, 'num_local_experts', None) or getattr(config, 'num_experts', 8)
    num_experts_per_tok = getattr(config, 'num_experts_per_tok', 2)
    
    print(f"Model has {num_experts} experts, selecting {num_experts_per_tok} per token")
    
    # Find MoE modules
    print("\nFinding Mixtral MoE modules...")
    moe_modules = find_moe_modules(model)
    print(f"Found {len(moe_modules)} MoE modules")
    
    # Register hooks
    hooks = []
    for name, module in moe_modules:
        hook = ExpertMonitoringHook(num_experts, num_experts_per_tok)
        handle = module.register_forward_hook(hook)
        hooks.append((name, hook, handle))
        print(f"Registered hook for {name}")
    
    # Load GSM8K dataset
    print("\nLoading GSM8K dataset...")
    gsm8k_samples = load_gsm8k_dataset(args.num_samples, sample_ids)
    
    # Process each sample
    all_results = []
    
    for sample_idx, sample in enumerate(gsm8k_samples):
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx + 1}/{len(gsm8k_samples)}")
        print(f"{'='*80}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"{'='*80}")
        
        # Generate response
        prompt = sample['question']
        try:
            generated_text, generated_ids = generate_with_monitoring(
                model, tokenizer, prompt, args.max_new_tokens, hooks
            )
        except RuntimeError as e:
            print(f"Warning: Generation error, trying manual generation: {e}")
            generated_text, generated_ids = manual_generate_with_monitoring(
                model, tokenizer, prompt, args.max_new_tokens, hooks
            )
        
        print(f"\nGenerated text (first 200 chars): {generated_text[:200]}...")
        
        # Collect activation data for this sample
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
