#!/usr/bin/env python3
"""
Monitor expert activation during OLMoE inference on GSM8K with optional MoE-layer quantization.

Quantization modes (applied only to MoE experts' Linear layers):
  - none: no quantization
  - w-int8, w-int4, w-mxfp4: quantize weights only
  - wa-int8, wa-mxfp4: quantize both weights and activations

Notes:
  - Lightweight, standalone quantization wrappers (do not depend on AlphaQuant runtime).
  - MXFP4 here is implemented as a simple 4-bit block floating style approximation
    (shared scale per tensor with 16 levels), suitable for quick experimentation.
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


class ExpertMonitoringHook:
    """Hook to capture expert activations and weights during forward pass (OLMoE).

    We attempt to find a router/gate linear layer on the MoE module to compute
    routing logits, apply softmax, and keep top-k selection for the latest token.
    """

    def __init__(self, num_experts: int, num_experts_per_tok: int = 2):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.data = []
        self.generation_step = 0

    def __call__(self, module, input_tuple, output_tensor):
        try:
            if not input_tuple:
                return
            hidden_states = input_tuple[0]

            if not torch.is_tensor(hidden_states):
                return

            # Find router linear module
            gate_layer = None
            if hasattr(module, 'gate') and isinstance(getattr(module, 'gate'), nn.Module):
                gate_layer = getattr(module, 'gate')
            elif hasattr(module, 'router') and isinstance(getattr(module, 'router'), nn.Module):
                gate_layer = getattr(module, 'router')
            else:
                # Heuristic: sometimes router is named 'routing' or lives as Linear attr
                for attr_name in ['routing', 'gate_proj', 'router_proj']:
                    if hasattr(module, attr_name) and isinstance(getattr(module, attr_name), nn.Module):
                        gate_layer = getattr(module, attr_name)
                        break
            if gate_layer is None:
                return

            # Ensure 3D: (batch, seq, hidden)
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)
            if hidden_states.dim() != 3:
                return

            with torch.no_grad():
                router_logits_bs = gate_layer(hidden_states)
            if router_logits_bs.dim() != 3:
                return

            bsz, seq_len, n_experts = router_logits_bs.shape
            logits_flat = router_logits_bs.reshape(bsz * seq_len, n_experts)
            routing_weights = torch.softmax(logits_flat, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
            selected_weights = routing_weights.gather(1, selected_experts)
            selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)

            last_pos = min(bsz * seq_len, logits_flat.shape[0]) - 1
            if last_pos >= 0:
                pos_selected = selected_experts[last_pos].detach().cpu().tolist()
                pos_weights = selected_weights[last_pos].detach().cpu().tolist()
                activated_experts = []
                for expert_id, weight in zip(pos_selected, pos_weights):
                    if int(expert_id) < self.num_experts:
                        activated_experts.append((int(expert_id), float(weight)))
                self.data.append({
                    'activated_experts': activated_experts,
                    'generation_step': self.generation_step,
                })
            self.generation_step += 1
        except Exception:
            # best-effort monitoring; ignore errors
            pass

    def reset(self):
        self.data = []
        self.generation_step = 0


def find_olmoe_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Find OLMoE modules by heuristics.

    We look for module names containing 'moe' and that also expose gate/router and experts.
    """
    moe_modules: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        lname = name.lower()
        if 'moe' in lname:
            has_gate = hasattr(module, 'gate') or hasattr(module, 'router')
            has_experts = hasattr(module, 'experts') or hasattr(module, 'num_experts')
            if has_gate and has_experts:
                moe_modules.append((name, module))
    return moe_modules


# -------------------------- Lightweight quantization --------------------------

def _symmetric_affine_quant_dequant(x: torch.Tensor, num_bits: int, signed: bool = True) -> torch.Tensor:
    if x.numel() == 0:
        return x
    x_fp = x.detach()
    qmax = (2 ** (num_bits - 1) - 1) if signed else (2 ** num_bits - 1)
    qmin = - (2 ** (num_bits - 1)) if signed else 0
    scale = x_fp.abs().max().clamp(min=1e-8) / qmax
    x_int = torch.clamp(torch.round(x_fp / scale), qmin, qmax)
    return (x_int * scale).to(dtype=x.dtype)


def _mxfp4_dequant(x: torch.Tensor) -> torch.Tensor:
    """Very simple MXFP4 approximation: per-tensor 4-bit levels with power-of-two scale.
    We quantize to 16 levels symmetric around zero using a power-of-two scale.
    """
    if x.numel() == 0:
        return x
    max_abs = x.abs().max().clamp(min=1e-8)
    # choose scale as nearest power-of-two of max_abs / 7 (similar dynamic to int4)
    target = (max_abs / 7.0).item()
    exp = int(round(torch.log2(torch.tensor(target)).item()))
    scale = torch.tensor(2.0 ** exp, device=x.device, dtype=x.dtype)
    q = torch.clamp(torch.round(x / scale), -7, 7)  # 16 levels total with zero
    return (q * scale).to(dtype=x.dtype)


class QuantLinearWrapper(nn.Module):
    """Quantized Linear wrapper (eval-only, CPU/GPU) with per-tensor quantization.

    mode:
      - 'none'
      - 'w-int8', 'w-int4', 'w-mxfp4'
      - 'wa-int8', 'wa-mxfp4'
    """

    def __init__(self, linear: nn.Linear, mode: str):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone())
        self.mode = mode

        # store original weight in fp to quant-dequant lazily once
        self.register_buffer('_cached_w', None, persistent=False)
        self.register_buffer('_cached_w_device', torch.tensor(0), persistent=False)
        self.weight_fp = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)

    def _quantize_weight_once(self) -> torch.Tensor:
        if self._cached_w is not None and self._cached_w_device.device == self.weight_fp.device:
            return self._cached_w
        w = self.weight_fp
        if self.mode in ('w-int8', 'wa-int8'):
            qw = _symmetric_affine_quant_dequant(w, num_bits=8, signed=True)
        elif self.mode in ('w-int4',):
            qw = _symmetric_affine_quant_dequant(w, num_bits=4, signed=True)
        elif self.mode in ('w-mxfp4', 'wa-mxfp4'):
            qw = _mxfp4_dequant(w)
        elif self.mode == 'none':
            qw = w
        else:
            qw = w
        self._cached_w = qw
        self._cached_w_device = torch.tensor(0, device=w.device)
        return qw

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'wa-int8':
            return _symmetric_affine_quant_dequant(x, num_bits=8, signed=True)
        if self.mode == 'wa-mxfp4':
            return _mxfp4_dequant(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qw = self._quantize_weight_once()
        xq = self._quantize_activation(x)
        y = torch.nn.functional.linear(xq, qw, self.bias)
        return y


def _replace_linear_in_module(module: nn.Module, mode: str):
    for name, sub in list(module.named_children()):
        if isinstance(sub, nn.Linear):
            setattr(module, name, QuantLinearWrapper(sub, mode))
        else:
            _replace_linear_in_module(sub, mode)


def apply_quant_to_olmoe_experts(model: nn.Module, mode: str) -> int:
    """Apply quantization wrappers to Linear layers under OLMoE expert modules.

    Returns number of Linear layers replaced.
    """
    if mode == 'none':
        return 0
    replaced = 0
    for name, module in model.named_modules():
        # Heuristic: quantize linears under modules that have 'experts' attribute
        if hasattr(module, 'experts') and isinstance(getattr(module, 'experts'), (list, nn.ModuleList, tuple)):
            experts_container = getattr(module, 'experts')
            if isinstance(experts_container, (list, tuple, nn.ModuleList)):
                for expert in experts_container:
                    before = _count_linears(expert)
                    _replace_linear_in_module(expert, mode)
                    after = _count_linears(expert)
                    # we cannot easily diff replaced, so approximate using before count
                    replaced += before
        # Some OLMoE implementations embed FFNs directly under the moe module
        lname = name.lower()
        if 'moe' in lname and not hasattr(module, 'experts'):
            before = _count_linears(module)
            _replace_linear_in_module(module, mode)
            replaced += before
    return replaced


def _count_linears(module: nn.Module) -> int:
    return sum(1 for _ in module.modules() if isinstance(_, nn.Linear))


def load_gsm8k_dataset(num_samples: int = 5, sample_ids: Optional[List[int]] = None):
    print("Loading GSM8K dataset...")
    try:
        dataset = load_dataset('gsm8k', 'main', split='test')
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        print("Trying to load from local cache or downloading...")
        dataset = load_dataset('gsm8k', 'main', split='test', cache_dir='./cache')

    selected_samples = []
    if sample_ids is not None:
        print(f"Using specified sample IDs: {sample_ids}")
        for sample_id in sample_ids:
            if sample_id < len(dataset):
                selected_samples.append({
                    'question': dataset[sample_id]['question'],
                    'answer': dataset[sample_id]['answer'],
                    'sample_id': sample_id,
                })
            else:
                print(f"Warning: Sample ID {sample_id} is out of range (dataset size: {len(dataset)})")
    else:
        print(f"Using first {num_samples} samples")
        for i in range(min(num_samples, len(dataset))):
            selected_samples.append({
                'question': dataset[i]['question'],
                'answer': dataset[i]['answer'],
                'sample_id': i,
            })
    return selected_samples


@torch.no_grad()
def generate_with_monitoring(model, tokenizer, prompt: str, max_new_tokens: int = 512, hooks: Optional[List] = None):
    if hooks:
        for _, hook, _ in hooks:
            hook.reset()
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_ids = inputs['input_ids']
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
        )
    generated_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, generated_ids


@torch.no_grad()
def manual_generate_with_monitoring(model, tokenizer, prompt: str, max_new_tokens: int = 512, hooks: Optional[List] = None):
    if hooks:
        for _, hook, _ in hooks:
            hook.reset()
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_ids = inputs['input_ids']
    generated_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        if tokenizer.eos_token_id and next_token.item() == tokenizer.eos_token_id:
            break
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text, generated_ids[0]


def main():
    parser = argparse.ArgumentParser(description="Monitor OLMoE expert activations on GSM8K with optional MoE quantization")
    parser.add_argument('--model_name', type=str, default='allenai/OLMoE-1B-7B', help='HF model id or path')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of GSM8K samples to process')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max new tokens to generate')
    parser.add_argument('--output', type=str, default='olmoe_expert_activations.csv', help='Output CSV path')
    parser.add_argument('--device', type=str, default=None, help='Device (default: auto)')
    parser.add_argument('--sample_ids', type=str, default=None, help='Comma-separated list of sample IDs (overrides --num_samples)')
    parser.add_argument('--quant_mode', type=str, default='none', choices=['none', 'w-int8', 'w-int4', 'w-mxfp4', 'wa-int8', 'wa-mxfp4'], help='MoE quantization mode')
    parser.add_argument('--quantized_model_dir', type=str, default=None, help='Directory of a quantized model (overrides --model_name)')

    args = parser.parse_args()

    sample_ids = None
    if args.sample_ids:
        try:
            sample_ids = [int(x.strip()) for x in args.sample_ids.split(',') if x.strip() != '']
            print(f"Will process specific sample IDs: {sample_ids}")
        except ValueError:
            print(f"Error: Invalid sample_ids format: {args.sample_ids}")
            print("Example: --sample_ids '0,5,10'")
            return

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("OLMoE Expert Activation Monitor")
    print("=" * 80)
    print(f"Model: {args.model_name if not args.quantized_model_dir else args.quantized_model_dir}")
    print(f"Device: {args.device}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Quant mode: {args.quant_mode}")
    print("=" * 80)

    def load_quantized_model(model_dir, torch_dtype, device):
        config_path = os.path.join(model_dir, 'config.json')
        model_path = os.path.join(model_dir, 'qmodel.pt')
        assert os.path.exists(config_path) and os.path.exists(model_path), f"Quantized model not found in {model_dir}"
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
            args.device,
        )
    else:
        print("Loading HF model and tokenizer from:", args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
            device_map=args.device,
        )
    model.eval()

    # Get config values
    config = model.config
    num_experts = getattr(config, 'num_local_experts', None) or getattr(config, 'num_experts', 8)
    num_experts_per_tok = getattr(config, 'num_experts_per_tok', 2)
    print(f"Model has {num_experts} experts, selecting {num_experts_per_tok} per token")

    # Find MoE modules
    print("\nFinding OLMoE modules...")
    moe_modules = find_olmoe_modules(model)
    print(f"Found {len(moe_modules)} MoE modules")

    # Apply quantization to experts if requested
    if args.quant_mode != 'none':
        print(f"Applying quantization mode '{args.quant_mode}' to experts' Linear layers...")
        replaced = apply_quant_to_olmoe_experts(model, args.quant_mode)
        print(f"Replaced approximately {replaced} Linear layers under MoE experts")

    # Register hooks
    hooks: List[Tuple[str, ExpertMonitoringHook, any]] = []
    for name, module in moe_modules:
        hook = ExpertMonitoringHook(num_experts, num_experts_per_tok)
        handle = module.register_forward_hook(hook)
        hooks.append((name, hook, handle))
        print(f"Registered hook for {name}")

    # Dataset
    print("\nLoading GSM8K dataset...")
    gsm8k_samples = load_gsm8k_dataset(args.num_samples, sample_ids)

    all_results: List[Dict] = []
    for sample_idx, sample in enumerate(gsm8k_samples):
        print(f"\n{'=' * 80}")
        print(f"Sample {sample_idx + 1}/{len(gsm8k_samples)}")
        print(f"{'=' * 80}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"{'=' * 80}")

        prompt = sample['question']
        try:
            generated_text, generated_ids = generate_with_monitoring(model, tokenizer, prompt, args.max_new_tokens, hooks)
        except RuntimeError as e:
            print(f"Warning: Generation error, trying manual generation: {e}")
            generated_text, generated_ids = manual_generate_with_monitoring(model, tokenizer, prompt, args.max_new_tokens, hooks)

        print(f"\nGenerated text (first 200 chars): {generated_text[:200]}...")

        activation_by_step: Dict[int, Dict[str, List[Tuple[int, float]]]] = {}
        for layer_name, hook, _ in hooks:
            for data in hook.data:
                gen_step = data.get('generation_step', 0)
                if gen_step not in activation_by_step:
                    activation_by_step[gen_step] = {}
                activation_by_step[gen_step][layer_name] = data['activated_experts']

        for gen_step in sorted(activation_by_step.keys()):
            for layer_name, activated_experts in activation_by_step[gen_step].items():
                experts_str = ';'.join([f"{eid},{w:.6f}" for eid, w in activated_experts])
                row = {
                    'sample_index': sample_idx,
                    'original_sample_id': sample.get('sample_id', sample_idx),
                    'layer': layer_name,
                    'token_index': gen_step,
                    'num_activated_experts': len(activated_experts),
                    'experts_and_weights': experts_str,
                }
                all_results.append(row)

        # reset for next sample
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

    print("\n" + "=" * 80)
    print("Monitoring complete!")
    print(f"Results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()


