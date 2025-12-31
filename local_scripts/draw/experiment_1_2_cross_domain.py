#!/usr/bin/env python3
"""
Experiment 1.2: Cross-Domain Performance Drop

Goal: Prove that data-driven calibration overfits to the calibration domain.

Method:
- Quantize model using WikiText2 calibration data (Model_Wiki)
- Quantize model using GSM8K calibration data (Model_GSM)
- Evaluate Model_Wiki on WikiText2 Test and GSM8K Test
- Evaluate Model_GSM on WikiText2 Test and GSM8K Test
- Output CSV with: model, test_dataset, perplexity
"""

import torch
import torch.nn as nn
import pandas as pd
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import main.py functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from datautils import get_loaders
from eval_ppl_utils import llama_eval
from main import mixtral_sequential
import time


@torch.no_grad()
def evaluate_perplexity(model, testloader, device, dataset_name):
    """Evaluate perplexity on a test dataset."""
    print(f"\nEvaluating perplexity on {dataset_name}...")
    
    # Use the existing llama_eval function
    # We need to modify it slightly to return the perplexity value
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
    cache = {"i": 0, "attention_mask": None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
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
    
    for i in range(len(layers)):
        print(f"Processing layer {i}/{len(layers)}...")
        layer = layers[i].to(device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
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
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    
    return ppl.item()


def quantize_model(model, calibration_dataset, device, args_template):
    """Quantize model using calibration data."""
    print(f"\nQuantizing model with {calibration_dataset} calibration data...")
    
    # Set up args for quantization
    import types
    args = types.SimpleNamespace()
    args.model = args_template.model
    args.dataset = calibration_dataset
    args.wbits = args_template.wbits
    args.attn_bits = args_template.attn_bits
    args.nsamples = args_template.nsamples
    args.seed = args_template.seed
    args.percdamp = args_template.percdamp
    args.groupsize = args_template.groupsize
    args.sym = args_template.sym
    args.act_order = args_template.act_order
    args.pack = args_template.pack
    args.mixed_type = args_template.mixed_type
    args.attn_implementation = args_template.attn_implementation
    args.cache_dir = args_template.cache_dir
    args.save_bit_assignments = None  # Don't save bit assignments for this experiment
    
    # Set GSM8K field if needed
    if calibration_dataset == 'gsm8k':
        os.environ['GSM8K_FIELD'] = 'question'
    else:
        os.environ.pop('GSM8K_FIELD', None)
    
    # Get calibration data
    dataloader, _ = get_loaders(
        calibration_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    
    # Quantize
    quantizers = mixtral_sequential(model, dataloader, device, bit_config=None)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Experiment 1.2: Cross-Domain Performance Drop")
    parser.add_argument('--model', type=str, default='mistralai/Mixtral-8x7B-v0.1', help='Model name or path')
    parser.add_argument('--output', type=str, default='experiment_1_2_perplexity.csv', help='Output CSV file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--wbits', type=str, default='4bit', choices=['1bit', '2bit', '3bit', '4bit', '5bit', '6bit', '7bit', '8bit'], help='Weight bit-width')
    parser.add_argument('--attn_bits', type=str, default='4bit', choices=['1bit', '2bit', '3bit', '4bit', '5bit', '6bit', '7bit', '8bit'], help='Attention bit-width')
    parser.add_argument('--percdamp', type=float, default=0.01, help='Percent dampening')
    parser.add_argument('--groupsize', type=int, default=128, help='Group size')
    parser.add_argument('--sym', action='store_true', help='Symmetric quantization')
    parser.add_argument('--act-order', action='store_true', help='Activation order')
    parser.add_argument('--pack', action='store_true', help='Pack quantized model')
    parser.add_argument('--mixed_type', type=str, default='uniform', choices=['uniform', 'mixed', 'random', 'manual', 'mixed_with_alpha', 'no_calib_auto_programming', 'no_calib_auto_programming_expert_level', 'no_calib_expert_level_layerwise'], help='Mixed precision type')
    parser.add_argument('--attn_implementation', type=str, default='eager', choices=['eager', 'sdpa', 'flash_attention_2'])
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory for alpha values')
    parser.add_argument('--skip_quantization', action='store_true', help='Skip quantization and use pre-quantized models')
    parser.add_argument('--model_wiki_path', type=str, default=None, help='Path to pre-quantized Model_Wiki')
    parser.add_argument('--model_gsm_path', type=str, default=None, help='Path to pre-quantized Model_GSM')
    
    args = parser.parse_args()
    
    # Convert bit strings to integers
    args.wbits = int(args.wbits[0])
    args.attn_bits = int(args.attn_bits[0])
    
    print("="*80)
    print("Experiment 1.2: Cross-Domain Performance Drop")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Quantization: {args.wbits}-bit weights, {args.attn_bits}-bit attention")
    print(f"Mixed type: {args.mixed_type}")
    print("="*80)
    
    results = []
    
    if args.skip_quantization and args.model_wiki_path and args.model_gsm_path:
        # Load pre-quantized models
        print("\nLoading pre-quantized models...")
        def load_quantized_model(model_dir, device):
            """Load quantized model from saved directory."""
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM
            
            config_path = os.path.join(model_dir, 'config.json')
            model_path = os.path.join(model_dir, 'qmodel.pt')
            
            if not os.path.exists(config_path) or not os.path.exists(model_path):
                raise FileNotFoundError(f"Quantized model not found in {model_dir}")
            
            config = AutoConfig.from_pretrained(config_path)
            model_loaded = AutoModelForCausalLM.from_config(config)
            weights = torch.load(model_path, map_location='cpu')
            
            # Load weights into model
            for name, module in model_loaded.named_modules():
                if name in weights:
                    try:
                        module.load_state_dict(weights[name], strict=False)
                    except:
                        pass  # Skip if can't load
            
            model_loaded.seqlen = 2048
            return model_loaded
        
        print(f"Loading Model_Wiki from {args.model_wiki_path}")
        model_wiki = load_quantized_model(args.model_wiki_path, args.device)
        model_wiki.eval()
        for param in model_wiki.parameters():
            param.requires_grad = False
        
        print(f"Loading Model_GSM from {args.model_gsm_path}")
        model_gsm = load_quantized_model(args.model_gsm_path, args.device)
        model_gsm.eval()
        for param in model_gsm.parameters():
            param.requires_grad = False
    else:
        # Quantize models
        print("\n" + "="*80)
        print("Step 1: Quantizing Model_Wiki (WikiText2 calibration)")
        print("="*80)
        
        # Load fresh model for WikiText2 quantization
        def get_model():
            import torch
            def skip(*args, **kwargs):
                pass
            torch.nn.init.kaiming_uniform_ = skip
            torch.nn.init.uniform_ = skip
            torch.nn.init.normal_ = skip

            config = AutoConfig.from_pretrained(
                args.model, attn_implementation=args.attn_implementation
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model, config=config, device_map='cpu', torch_dtype=torch.float16
            )
            assert isinstance(model, MixtralForCausalLM), 'Model must be Mixtral!'
            model.seqlen = 2048
            return model
        
        model_wiki = get_model()
        model_wiki.eval()
        for param in model_wiki.parameters():
            param.requires_grad = False
        
        model_wiki = quantize_model(model_wiki, 'wikitext2', args.device, args)
        
        print("\n" + "="*80)
        print("Step 2: Quantizing Model_GSM (GSM8K calibration)")
        print("="*80)
        
        # Load fresh model for GSM8K quantization
        model_gsm = get_model()
        model_gsm.eval()
        for param in model_gsm.parameters():
            param.requires_grad = False
        
        model_gsm = quantize_model(model_gsm, 'gsm8k', args.device, args)
    
    # Evaluate models
    print("\n" + "="*80)
    print("Step 3: Evaluating models on test sets")
    print("="*80)
    
    # Get test loaders
    os.environ.pop('GSM8K_FIELD', None)
    _, testloader_wiki = get_loaders('wikitext2', seed=args.seed, seqlen=2048, model=args.model)
    
    os.environ['GSM8K_FIELD'] = 'question'
    _, testloader_gsm8k = get_loaders('gsm8k', seed=args.seed, seqlen=2048, model=args.model)
    
    # Evaluate Model_Wiki on WikiText2
    print("\nEvaluating Model_Wiki on WikiText2 Test...")
    ppl_wiki_on_wiki = evaluate_perplexity(model_wiki, testloader_wiki, args.device, 'wikitext2')
    results.append({
        'model': 'Model_Wiki',
        'calibration_dataset': 'wikitext2',
        'test_dataset': 'wikitext2',
        'perplexity': ppl_wiki_on_wiki
    })
    print(f"Perplexity: {ppl_wiki_on_wiki:.4f}")
    
    # Evaluate Model_Wiki on GSM8K
    print("\nEvaluating Model_Wiki on GSM8K Test...")
    ppl_wiki_on_gsm8k = evaluate_perplexity(model_wiki, testloader_gsm8k, args.device, 'gsm8k')
    results.append({
        'model': 'Model_Wiki',
        'calibration_dataset': 'wikitext2',
        'test_dataset': 'gsm8k',
        'perplexity': ppl_wiki_on_gsm8k
    })
    print(f"Perplexity: {ppl_wiki_on_gsm8k:.4f}")
    
    # Evaluate Model_GSM on WikiText2
    print("\nEvaluating Model_GSM on WikiText2 Test...")
    ppl_gsm_on_wiki = evaluate_perplexity(model_gsm, testloader_wiki, args.device, 'wikitext2')
    results.append({
        'model': 'Model_GSM',
        'calibration_dataset': 'gsm8k',
        'test_dataset': 'wikitext2',
        'perplexity': ppl_gsm_on_wiki
    })
    print(f"Perplexity: {ppl_gsm_on_wiki:.4f}")
    
    # Evaluate Model_GSM on GSM8K
    print("\nEvaluating Model_GSM on GSM8K Test...")
    ppl_gsm_on_gsm8k = evaluate_perplexity(model_gsm, testloader_gsm8k, args.device, 'gsm8k')
    results.append({
        'model': 'Model_GSM',
        'calibration_dataset': 'gsm8k',
        'test_dataset': 'gsm8k',
        'perplexity': ppl_gsm_on_gsm8k
    })
    print(f"Perplexity: {ppl_gsm_on_gsm8k:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    
    print("\n" + "="*80)
    print("Results saved to:", args.output)
    print("="*80)
    print("\nResults Summary:")
    print(df.to_string(index=False))
    
    # Print analysis
    print("\n" + "="*80)
    print("Analysis:")
    print("="*80)
    print(f"Model_Wiki on WikiText2: {ppl_wiki_on_wiki:.4f}")
    print(f"Model_Wiki on GSM8K: {ppl_wiki_on_gsm8k:.4f}")
    print(f"Model_GSM on WikiText2: {ppl_gsm_on_wiki:.4f}")
    print(f"Model_GSM on GSM8K: {ppl_gsm_on_gsm8k:.4f}")
    print(f"\nExpected: Model_Wiki should beat Model_GSM on WikiText2")
    print(f"  Actual: {ppl_wiki_on_wiki:.4f} vs {ppl_gsm_on_wiki:.4f} ({'✓' if ppl_wiki_on_wiki < ppl_gsm_on_wiki else '✗'})")
    print(f"\nExpected: Model_GSM should beat Model_Wiki on GSM8K")
    print(f"  Actual: {ppl_gsm_on_gsm8k:.4f} vs {ppl_wiki_on_gsm8k:.4f} ({'✓' if ppl_gsm_on_gsm8k < ppl_wiki_on_gsm8k else '✗'})")


if __name__ == "__main__":
    main()

