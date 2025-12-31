#!/usr/bin/env python3
"""
Experiment 1.2: Cross-Domain Performance Drop

Goal: Prove that data-driven calibration overfits to the calibration domain.

Method:
- Quantize the model using calibration data from WikiText2. Call this Model_Wiki.
- Quantize the model using calibration data from GSM8K. Call this Model_GSM.
- Evaluate Model_Wiki on WikiText2 Test and GSM8K Test.
- Evaluate Model_GSM on WikiText2 Test and GSM8K Test.
- Output: CSV with Perplexity (PPL) results

Expected Result: Model_Wiki beats Model_GSM on WikiText2 but loses significantly on GSM8K.
"""

import os
import sys
import torch
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from deepseek_moe.modeling_deepseek import DeepseekV2ForCausalLM
from deepseek_moe.configuration_deepseek import DeepseekV2Config
from datautils import get_loaders
from eval_ppl_utils import llama_eval
from deepseek_main import deepseek_sequential


def quantize_model(model_path, calibration_dataset, output_path, args_template):
    """Quantize a model using calibration data from a specific dataset."""
    print(f"\n{'='*80}")
    print(f"Quantizing model with {calibration_dataset} calibration data...")
    print(f"{'='*80}")
    
    # Create a copy of args for this quantization
    import copy
    args = copy.deepcopy(args_template)
    args.dataset = calibration_dataset
    args.saving_path = output_path
    args.save = True
    
    # Load model
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
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Load calibration data
    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    
    # Quantize
    device = args.device
    quantizers = deepseek_sequential(model, dataloader, device, bit_config=None)
    
    print(f"Quantization complete. Model saved to {output_path}")
    return output_path


@torch.no_grad()
def evaluate_perplexity(model, testloader, device, dataset_name: str):
    """Evaluate perplexity on a test dataset."""
    print(f"\nEvaluating perplexity on {dataset_name}...")
    
    # Use the existing llama_eval function
    llama_eval(model, testloader, device, dataset_name)
    
    # We need to extract the perplexity value
    # Since llama_eval prints it, we'll recompute it here for CSV output
    testenc = testloader.input_ids
    nsamples = testenc.numel() // model.seqlen
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    layers[0] = layers[0].to(device)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}
    
    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError
    
    layers[0] = Catcher(layers[0])
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
    
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    
    for i in range(len(layers)):
        layer = layers[i].to(device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)
    
    testenc = testenc.to(device)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    
    return ppl.item()


def load_quantized_model(model_path, device='cpu', attn_implementation='eager'):
    """Load a quantized DeepSeek model."""
    from evaluate_quantized_deepseek import load_quantized_deepseek
    model = load_quantized_deepseek(
        save_dir=model_path,
        attn_implementation=attn_implementation,
        device=device,
        compute_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
        device_map='none'
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1.2: Cross-Domain Performance Drop"
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
        default='experiment_1_2_cross_domain.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--model_cache_dir',
        type=str,
        default='./quantized_models',
        help='Directory to cache quantized models'
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
        '--wbits',
        type=str,
        default='4bit',
        choices=['1bit', '2bit', '3bit', '4bit', '5bit', '6bit', '7bit', '8bit'],
        help='Weight bit-width'
    )
    parser.add_argument(
        '--attn_bits',
        type=str,
        default='4bit',
        choices=['1bit', '2bit', '3bit', '4bit', '5bit', '6bit', '7bit', '8bit'],
        help='Attention weight bit-width'
    )
    parser.add_argument(
        '--mixed_type',
        type=str,
        default='uniform',
        choices=['uniform', 'mixed', 'random', 'manual', 'mixed_with_alpha', 'no_calib_auto_programming', 'no_calib_auto_programming_expert_level'],
        help='Quantization type'
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
    parser.add_argument(
        '--skip_quantization',
        action='store_true',
        help='Skip quantization and only evaluate existing models'
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert bit strings to integers
    args.wbits = int(args.wbits[0])
    args.attn_bits = int(args.attn_bits[0])
    
    print("="*80)
    print("Experiment 1.2: Cross-Domain Performance Drop")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Quantization: {args.mixed_type}, wbits={args.wbits}, attn_bits={args.attn_bits}")
    print("="*80)
    
    # Create model cache directory
    os.makedirs(args.model_cache_dir, exist_ok=True)
    
    # Define model paths
    model_wiki_path = os.path.join(args.model_cache_dir, f"model_wiki_{args.mixed_type}_w{args.wbits}_a{args.attn_bits}")
    model_gsm_path = os.path.join(args.model_cache_dir, f"model_gsm_{args.mixed_type}_w{args.wbits}_a{args.attn_bits}")
    
    # Quantize models if needed
    if not args.skip_quantization:
        # Quantize with WikiText2
        if not os.path.exists(model_wiki_path):
            quantize_model(args.model, 'wikitext2', model_wiki_path, args)
        else:
            print(f"Model_Wiki already exists at {model_wiki_path}, skipping quantization")
        
        # Quantize with GSM8K
        if not os.path.exists(model_gsm_path):
            quantize_model(args.model, 'gsm8k', model_gsm_path, args)
        else:
            print(f"Model_GSM already exists at {model_gsm_path}, skipping quantization")
    else:
        print("Skipping quantization (--skip_quantization flag set)")
        if not os.path.exists(model_wiki_path):
            print(f"Error: Model_Wiki not found at {model_wiki_path}")
            return
        if not os.path.exists(model_gsm_path):
            print(f"Error: Model_GSM not found at {model_gsm_path}")
            return
    
    # Load test datasets
    print("\nLoading test datasets...")
    _, wiki_test = get_loaders('wikitext2', seed=args.seed, seqlen=args.seqlen, model=args.model)
    _, gsm_test = get_loaders('gsm8k', seed=args.seed, seqlen=args.seqlen, model=args.model)
    
    # Evaluate models
    results = []
    
    # Load and evaluate Model_Wiki
    print(f"\n{'='*80}")
    print("Evaluating Model_Wiki (quantized with WikiText2)")
    print(f"{'='*80}")
    model_wiki = load_quantized_model(model_wiki_path, device=args.device, attn_implementation=args.attn_implementation)
    model_wiki.eval()
    model_wiki.seqlen = args.seqlen
    
    ppl_wiki_on_wiki = evaluate_perplexity(model_wiki, wiki_test, args.device, 'wikitext2')
    results.append({
        'model': 'Model_Wiki',
        'calibration_dataset': 'wikitext2',
        'test_dataset': 'wikitext2',
        'perplexity': ppl_wiki_on_wiki
    })
    
    ppl_wiki_on_gsm = evaluate_perplexity(model_wiki, gsm_test, args.device, 'gsm8k')
    results.append({
        'model': 'Model_Wiki',
        'calibration_dataset': 'wikitext2',
        'test_dataset': 'gsm8k',
        'perplexity': ppl_wiki_on_gsm
    })
    
    # Load and evaluate Model_GSM
    print(f"\n{'='*80}")
    print("Evaluating Model_GSM (quantized with GSM8K)")
    print(f"{'='*80}")
    model_gsm = load_quantized_model(model_gsm_path, device=args.device, attn_implementation=args.attn_implementation)
    model_gsm.eval()
    model_gsm.seqlen = args.seqlen
    
    ppl_gsm_on_wiki = evaluate_perplexity(model_gsm, wiki_test, args.device, 'wikitext2')
    results.append({
        'model': 'Model_GSM',
        'calibration_dataset': 'gsm8k',
        'test_dataset': 'wikitext2',
        'perplexity': ppl_gsm_on_wiki
    })
    
    ppl_gsm_on_gsm = evaluate_perplexity(model_gsm, gsm_test, args.device, 'gsm8k')
    results.append({
        'model': 'Model_GSM',
        'calibration_dataset': 'gsm8k',
        'test_dataset': 'gsm8k',
        'perplexity': ppl_gsm_on_gsm
    })
    
    # Save results
    print(f"\nSaving results to {args.output_csv}...")
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")
    
    # Print summary
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Experiment 1.2 complete!")
    print("="*80)


if __name__ == "__main__":
    main()

