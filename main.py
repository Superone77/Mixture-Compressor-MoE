import os
import pickle
import time
import torch
import logging
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, Dict

from gptq import GPTQ
from modelutils import find_layers
from datautils import get_loaders
from quant.QLinear import *

atten_modules = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]
expert_modules = [
    "block_sparse_moe.experts.0.w1",
    "block_sparse_moe.experts.1.w1",
    "block_sparse_moe.experts.2.w1",
    "block_sparse_moe.experts.3.w1",
    "block_sparse_moe.experts.4.w1",
    "block_sparse_moe.experts.5.w1",
    "block_sparse_moe.experts.6.w1",
    "block_sparse_moe.experts.7.w1",
    "block_sparse_moe.experts.0.w3",
    "block_sparse_moe.experts.1.w3",
    "block_sparse_moe.experts.2.w3",
    "block_sparse_moe.experts.3.w3",
    "block_sparse_moe.experts.4.w3",
    "block_sparse_moe.experts.5.w3",
    "block_sparse_moe.experts.6.w3",
    "block_sparse_moe.experts.7.w3",
    "block_sparse_moe.experts.0.w2",
    "block_sparse_moe.experts.1.w2",
    "block_sparse_moe.experts.2.w2",
    "block_sparse_moe.experts.3.w2",
    "block_sparse_moe.experts.4.w2",
    "block_sparse_moe.experts.5.w2",
    "block_sparse_moe.experts.6.w2",
    "block_sparse_moe.experts.7.w2",
]


logger = logging.getLogger(__name__)


def get_moe_module_name(layer):
    """Detect the MoE module name in a layer"""
    if hasattr(layer, 'block_sparse_moe'):
        return 'block_sparse_moe'
    elif hasattr(layer, 'mlp'):
        return 'mlp'
    elif hasattr(layer, 'moe'):
        return 'moe'
    else:
        raise ValueError(f"Could not find MoE module in layer. Available attributes: {dir(layer)}")


def get_num_experts_from_moe_block(moe_block):
    """Get the number of experts from an MoE block"""
    if hasattr(moe_block, 'num_experts'):
        return moe_block.num_experts
    elif hasattr(moe_block, 'experts'):
        return len(moe_block.experts)
    else:
        raise ValueError("Could not determine number of experts")


def generate_expert_modules(num_experts=8):
    """Generate expert module names for the given number of experts"""
    expert_modules = []
    for expert_idx in range(num_experts):
        expert_modules.extend([
            f"block_sparse_moe.experts.{expert_idx}.w1",
            f"block_sparse_moe.experts.{expert_idx}.w3",
            f"block_sparse_moe.experts.{expert_idx}.w2",
        ])
    return expert_modules


@torch.no_grad()
def alpha_hill_from_weight(
    W: torch.Tensor,
    k: Optional[int] = None,
    k_frac: float = 0.1,
    eps: float = 1e-12,
) -> Tuple[float, int, int]:
    """Compute PL_Alpha_Hill for a weight tensor W.
    
    Returns: alpha, k_used, n_eigs
    """
    # Ensure dense & 2D
    if W.is_sparse:
        W = W.to_dense()
    if W.ndim > 2:
        W = W.reshape(W.shape[0], -1)

    m, n = W.shape
    min_dim = min(m, n)
    if min_dim < 2:
        return float("nan"), 1, min_dim

    # Compute SVD on CPU in float32 for stability
    W_ = W.to(dtype=torch.float32, device="cpu")
    s = torch.linalg.svdvals(W_)
    lam = (s ** 2)
    
    lam, _ = torch.sort(lam)
    n_eigs = lam.numel()
    if n_eigs < 2:
        return float("nan"), 1, n_eigs
    
    if k is None:
        k_used = max(10, int(n_eigs * k_frac))
    else:
        k_used = k
    k_used = max(1, min(k_used, n_eigs - 1))
    
    eps_t = torch.tensor(eps, dtype=lam.dtype, device=lam.device)
    lam_ref = torch.clamp(lam[-k_used-1], min=eps_t)
    top = lam[-k_used:]
    denom = torch.log(top / lam_ref).sum().clamp_min(eps_t)
    alpha = float(1.0 + (k_used / float(denom)))
    return alpha, k_used, n_eigs


def compute_alpha_values(model, cache_dir=None):
    """Compute alpha values for all linear layers in MoE experts."""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "alpha_values.csv")
        if os.path.exists(cache_path):
            logger.info(f"Loading alpha values from cache: {cache_path}")
            return load_alpha_from_csv(cache_path)
    
    logger.info("Computing alpha values for all layers...")
    results = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach()
            if weight is None:
                continue
                
            try:
                alpha, k_used, n_eigs = alpha_hill_from_weight(weight)
                results[name] = alpha
            except Exception as e:
                logger.warning(f"Failed to compute alpha for {name}: {e}")
                results[name] = float('nan')
    
    if cache_path:
        logger.info(f"Saving alpha values to: {cache_path}")
        save_alpha_to_csv(results, cache_path)
    
    return results


def save_alpha_to_csv(alpha_results, filename):
    """Save alpha results to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer_name', 'alpha'])
        for name, alpha in alpha_results.items():
            writer.writerow([name, alpha])


def load_alpha_from_csv(filename):
    """Load alpha values from CSV file."""
    alpha_results = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                alpha_results[row['layer_name']] = float(row['alpha'])
            except (ValueError, KeyError):
                continue
    return alpha_results


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
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)

    # Check if model has MoE structure
    has_moe = hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0
    if has_moe:
        first_layer = model.model.layers[0]
        has_moe = hasattr(first_layer, 'block_sparse_moe') or hasattr(first_layer, 'mlp')
    
    if not has_moe:
        raise ValueError('Model does not have a valid MoE structure!')
    
    model.seqlen = 2048
    return model


@torch.no_grad()
def moe_sequential(model, dataloader, dev, bit_config=None):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    # Detect MoE structure and number of experts
    first_layer = layers[0]
    moe_module_name = get_moe_module_name(first_layer)
    moe_block = getattr(first_layer, moe_module_name)
    num_experts = get_num_experts_from_moe_block(moe_block)
    
    print(f"Detected MoE structure: {moe_module_name} with {num_experts} experts")
    
    # Generate expert modules dynamically based on detected number
    global expert_modules
    expert_modules = generate_expert_modules(num_experts)
    
    # Compute alpha values if needed
    alpha_values = None
    if args.mixed_type == "mixed_with_alpha":
        alpha_values = compute_alpha_values(model, cache_dir=args.cache_dir)
        print(f"Computed alpha values for {len(alpha_values)} layers")

    # Move necessary components to device
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    
    # Ensure rotary embeddings in the first layer are also moved to device
    # This is important for models like OLMoE where rotary_emb might have CPU buffers
    if hasattr(layers[0], 'self_attn') and hasattr(layers[0].self_attn, 'rotary_emb'):
        rotary_emb = layers[0].self_attn.rotary_emb
        rotary_emb = rotary_emb.to(dev)
        # Explicitly move all buffers to device
        for buffer in rotary_emb.buffers():
            if buffer.device != dev:
                buffer.data = buffer.data.to(dev)
        layers[0].self_attn.rotary_emb = rotary_emb

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    print('Ready.')
    quantizers = {}
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+--------------------------------+------------+------------+------------+---------+')
        print('|              name              |weight_error| fp_inp_SNR | q_inp_SNR  |  time   |')
        print('+================================+============+============+============+=========+')

        layer = layers[i].to(dev)
        
        # Ensure rotary embeddings are moved to device
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
            rotary_emb = layer.self_attn.rotary_emb
            rotary_emb = rotary_emb.to(dev)
            # Explicitly move all buffers to device
            for buffer in rotary_emb.buffers():
                if buffer.device != dev:
                    buffer.data = buffer.data.to(dev)
            layer.self_attn.rotary_emb = rotary_emb
        
        full = find_layers(layer)

        sequential = [list(full.keys())]
        
        # Initialize variables
        low_bit_experts = []
        high_bit_experts = []
        
        # random generation
        if args.mixed_type == "random":
            import random
            numbers = list(range(num_experts))
            low_bit_config = random.sample(numbers, 2)
            for num in low_bit_config:
                numbers.remove(num)
            high_bit_config = random.sample(numbers, 2)
            low_bit_experts = ["block_sparse_moe.experts."+str(j) for j in low_bit_config]   
            high_bit_experts = ["block_sparse_moe.experts."+str(j) for j in high_bit_config]
        elif args.mixed_type == "manual":
            if bit_config is not None:
                _, indices_max = torch.topk(bit_config[i], args.h_experts)
                _, indices_min = torch.topk(bit_config[i], args.l_experts, largest=False)
                low_bit_experts = ["block_sparse_moe.experts."+str(j.item()) for j in indices_min]   
                high_bit_experts = ["block_sparse_moe.experts."+str(j.item()) for j in indices_max]
            else:
                print("Please generate the high_experts.pkl and low_experts.pkl first!")
                exit()
        elif args.mixed_type == "mixed":
             if bit_config is not None:
                low_bit_experts = []
                high_bit_experts = []
                for expert_index in bit_config[i].keys():
                    if bit_config[i][expert_index] == 1:
                        low_bit_experts.append("block_sparse_moe.experts."+str(expert_index))
                    elif bit_config[i][expert_index] == 3:
                        high_bit_experts.append("block_sparse_moe.experts."+str(expert_index))
             else:
                print("Please generate the high_experts.pkl and low_experts.pkl first!")
                exit()
        elif args.mixed_type == "mixed_with_alpha":
            # Compute alpha values for expert modules in this layer
            expert_alpha_values = {}
            layer_name_prefix = f'model.layers.{i}.block_sparse_moe.experts.'
            
            # Get alpha values for all experts in this layer
            for expert_idx in range(num_experts):
                expert_prefix = f'{layer_name_prefix}{expert_idx}'
                # Get alpha values for all three weight matrices (w1, w2, w3)
                alpha_sum = 0.0
                count = 0
                for weight_name in ['w1', 'w2', 'w3']:
                    weight_full_name = f'{expert_prefix}.{weight_name}'
                    if weight_full_name in alpha_values:
                        alpha_val = alpha_values[weight_full_name]
                        # Check if alpha is valid (not NaN and not infinite)
                        if isinstance(alpha_val, (int, float)) and alpha_val == alpha_val and abs(alpha_val) != float('inf'):
                            alpha_sum += alpha_val
                            count += 1
                
                if count > 0:
                    expert_alpha_values[expert_idx] = alpha_sum / count
                else:
                    # If no valid alpha found, use a default value
                    expert_alpha_values[expert_idx] = 0.0
            
            # Sort experts by alpha value (ascending: smaller alpha = more sensitive = higher precision)
            sorted_experts = sorted(expert_alpha_values.items(), key=lambda x: x[1])
            
            # Calculate number of experts for each precision
            total_experts = num_experts
            n_high_bit = int(total_experts * args.high_bit_experts_ratio)
            n_low_bit = int(total_experts * args.low_bit_experts_ratio)
            
            # Smaller alpha gets higher precision (wbits+1)
            high_bit_experts = ["block_sparse_moe.experts."+str(sorted_experts[j][0]) for j in range(n_high_bit)]
            # Larger alpha gets lower precision (wbits-1)
            low_bit_experts = ["block_sparse_moe.experts."+str(sorted_experts[total_experts - j - 1][0]) for j in range(n_low_bit)]
            
            print(f"Layer {i}: High bit experts (alpha_sorted): {[expert_alpha_values[int(e.split('.')[-1])] for e in high_bit_experts]}")
            print(f"Layer {i}: Low bit experts (alpha_sorted): {[expert_alpha_values[int(e.split('.')[-1])] for e in low_bit_experts]}")
        


        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:

                gptq[name] = GPTQ(subset[name], logger, name, args.wbits)

                if args.mixed_type == "uniform":
                    gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False, pack=args.pack) 
                    gptq[name].wbits = args.wbits
                else:
                    if name not in expert_modules:
                        gptq[name].quantizer.configure(args.attn_bits, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                        gptq[name].wbits = args.attn_bits
                    else:
                        # Handle mxfp4 special case
                        if isinstance(args.wbits, str) and args.wbits == "mxfp4":
                            gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.wbits
                        elif name[:-3] in high_bit_experts and not isinstance(args.wbits, str):
                            gptq[name].quantizer.configure(args.wbits+1, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.wbits+1
                        elif name[:-3] in low_bit_experts and not isinstance(args.wbits, str):
                            gptq[name].quantizer.configure(args.wbits-1, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.wbits-1
                        else:
                            gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.wbits
            # print(layer)
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                # quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)
                quantizers['model.layers.%d.%s' % (i, name)] = None
                if args.pack:
                    # real quant for compact memory
                    quant_config = BaseQuantizeConfig(nbits=gptq[name].wbits, group_size=args.groupsize)
                    name_parts = name.split('.')
                    if len(name_parts) == 2: # atten layer
                        _module = getattr(layer, name_parts[-2])
                        linear_layer = getattr(_module, name_parts[-1])
                    else: 
                        experts = getattr(layer.block_sparse_moe, "experts")
                        _module = experts[int(name_parts[-2])]
                        linear_layer = getattr(_module, name_parts[-1])
                    quant_layer = QLinear(quant_config=quant_config, device=linear_layer.weight.device)
                    quant_layer.replace_quantized_weight(linear_layer.weight, scale, zero)
                    setattr(_module, name_parts[-1], quant_layer)
                    print(getattr(_module, name_parts[-1]).W_q.dtype)
                gptq[name].free()
            
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print('+--------------------------------+------------+------------+------------+---------+')
        print('\n')

    model.config.use_cache = use_cache

    return quantizers


if __name__ == "__main__":
    import argparse

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "--wbits",
        type=str,
        choices=["1bit", "2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit", "mxfp4"],
        help="weight bit-width",
    )
    parser.add_argument(
        "--attn_bits",
        type=str,
        choices=["1bit", "2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit", "mxfp4"],
        help="attention weight bit-width",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "mix"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help="Group size",
    )
    parser.add_argument(
        "--num_fewshot", 
        type=int, 
        default=0
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="batch size."
    )
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument(
        '--sym', 
        action='store_true', 
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--act-order', 
        action='store_true', 
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        "--multigpu",
        action="store_true",
    )
    parser.add_argument(
        "--eval_ppl", action="store_true", help="Evaluate perplexity."
    )
    parser.add_argument(
        "--tasks", 
        type=str,
        default="",
        help="Test datasets",
    )
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--pack", action="store_true", help="Whether to save the packed model."
    )
    parser.add_argument(
        "--use_flash_attention_2", action="store_true", help="Whether to use flash_attention2 for inference."
    )
    parser.add_argument(
        '--r', type=int, default=7, help='Number of experts to preserve'
    )
    parser.add_argument(
        "--mixed_type",
        type=str,
        choices=["uniform", "mixed", "random", "manual", "mixed_with_alpha"],
        help='Whether to use mixed-precision',
    )
    parser.add_argument(
        "--h_experts", 
        type=int, 
        default=2, 
        help="batch size." 
    )
    parser.add_argument(
        "--l_experts", 
        type=int, 
        default=2, 
        help="batch size." 
    )
    parser.add_argument(
        "--precisions", type=str, help="the file path of experts precision"
    )
    parser.add_argument(
        "--saving_path", type=str, help="the saving path of quantized model"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Directory to cache alpha values"
    )
    parser.add_argument(
        "--high_bit_experts_ratio", type=float, default=0.25, help="Ratio of high bit experts (for mixed_with_alpha)"
    )
    parser.add_argument(
        "--low_bit_experts_ratio", type=float, default=0.25, help="Ratio of low bit experts (for mixed_with_alpha)"
    )

    args = parser.parse_args()
    print(f'Arguments: {args}')

    groupsize = args.groupsize
    # Handle mxfp4 as special case
    if args.wbits == "mxfp4":
        args.wbits = "mxfp4"
    else:
        args.wbits = int(args.wbits[0])
    
    if args.attn_bits == "mxfp4":
        args.attn_bits = "mxfp4"
    else:
        args.attn_bits = int(args.attn_bits[0])

    model = get_model()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    bit_config = None
    if args.mixed_type == "manual" or args.mixed_type == "mixed":
        high_bit = args.precisions
        if os.path.exists(high_bit):
            with open(high_bit, 'rb') as file:
                bit_config = pickle.load(file)
        else:
            print("Please generate the high_experts.pkl and low_experts.pkl first!")
            exit()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    device = "cuda:0"
    tick = time.time()
    quantizers = moe_sequential(model, dataloader, device, bit_config)
    print("quantization time:", time.time() - tick, "s")
    print(model)

    if args.eval_ppl:
        for dataset in ["wikitext2"]:#, "c4", "ptb"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, seqlen=2048, model=args.model
            )
            print(dataset)
            from eval_ppl_utils import llama_eval
            t1 = time.time()
            llama_eval(model, testloader, device, dataset)
            print("Time: ", time.time() - t1)
    if args.save:
        average_bits = int(args.precisions[-9:-7])/8 if args.precisions else 0
        model_name = os.path.basename(args.model)
        saving_path = args.saving_path + f"{model_name}-atten_{args.attn_bits}-e_{average_bits}"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(saving_path)
        from utils.pack import save_quantized
        save_quantized(model, saving_path)