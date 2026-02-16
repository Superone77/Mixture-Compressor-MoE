#!/usr/bin/env python3
"""
Qwen3-Coder-Next: load AlphaQ bit recipe (CSV), run GPTQ with per-layer/per-expert bit width,
save quantized model for inference. Supports multi-GPU via device_map="auto".

Usage (on GPU machine):
  cd Mixture-Compressor-MoE
  python qwen3_gptq_from_recipe.py \
    --model Qwen/Qwen3-Coder-Next \
    --recipe_csv qwen3_bit_recipes/qwen3_coder_next_gamma10.0_bpp4.0.csv \
    --output_dir ./out_qwen3_bpp4 \
    --nsamples 128 --seqlen 2048 \
    --device_map auto

Output: output_dir with config.json, model.safetensors (or pytorch_model.bin), tokenizer files.
Inference: use inference_qwen3_quantized.py or AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto").
"""

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add repo root so we can import gptq, utils
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from gptq import GPTQ

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_recipe(csv_path: str) -> dict:
    """Load recipe CSV; return dict: full_name -> bit_width (int)."""
    out = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            bit = row.get("bit_width", "4").strip()
            if name and bit.isdigit():
                out[name] = int(bit)
    return out


def get_wikitext_calibration(nsamples: int, seqlen: int, tokenizer, device: str, seed: int = 42):
    """Return list of (input_ids,) for calibration; input_ids shape (1, seqlen)."""
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    import random
    random.seed(seed)
    loader = []
    n = trainenc.input_ids.shape[1]
    for _ in range(nsamples):
        i = random.randint(0, max(0, n - seqlen - 1))
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        loader.append((inp,))
    return loader


def run_gptq_qwen3_from_recipe(
    model,
    tokenizer,
    recipe: dict,
    trainloader,
    device_map: str,
    seqlen: int,
    blocksize: int = 128,
    percdamp: float = 0.01,
    groupsize: int = -1,
    actorder: bool = False,
):
    """Run GPTQ on model using recipe (name -> bit). Only expert slices and Linear in recipe are quantized."""
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    num_layers = len(layers)
    hidden_size = model.config.hidden_size

    # Build layer inputs: run forward to get input to each layer
    logger.info("Building calibration inputs ...")
    embeds = []
    for batch in trainloader:
        inp = batch[0]
        embeds.append(model.model.embed_tokens(inp))
    layer_inps = [torch.cat(embeds, dim=0)]
    nsamples = layer_inps[0].shape[0]
    seqlen_act = layer_inps[0].shape[1]
    first_dev = next(model.parameters()).device
    attention_mask = torch.ones(nsamples, seqlen_act, dtype=torch.long, device=first_dev)
    position_ids = torch.arange(seqlen_act, device=first_dev).unsqueeze(0).expand(nsamples, -1)

    first_dev = next(model.parameters()).device
    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        layer_dev = next(layer.parameters()).device
        cur_inp = layer_inps[layer_idx].to(layer_dev)
        attn = attention_mask.to(layer_dev)
        pos = position_ids.to(layer_dev)
        with torch.no_grad():
            out = layer(cur_inp, attention_mask=attn, position_ids=pos)[0]
        layer_inps.append(out.cpu() if out.device.type != "cpu" else out)
        if (layer_idx + 1) % 12 == 0:
            logger.info(f"  Calib layer {layer_idx+1}/{num_layers}")

    # Optional: quantize non-expert Linears (linear_attn, mlp.gate, etc.) by name
    def get_bit(name):
        return recipe.get(name)

    # 1) Quantize expert slices per layer
    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        prefix = f"model.model.layers.{layer_idx}."
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
            continue
        experts = layer.mlp.experts
        layer_dev = next(layers[layer_idx].parameters()).device
        mlp_inp = layer_inps[layer_idx].to(layer_dev)
        if mlp_inp.dim() == 3:
            mlp_inp_flat = mlp_inp.reshape(-1, hidden_size)
        else:
            mlp_inp_flat = mlp_inp.reshape(-1, hidden_size)
        gate_up = experts.gate_up_proj
        down = experts.down_proj
        inter = gate_up.shape[1] // 2
        dev = next(layer.parameters()).device

        for e in range(gate_up.shape[0]):
            name_gu = f"{prefix}mlp.experts.gate_up_proj.expert_{e}"
            bit = get_bit(name_gu)
            if bit is None:
                continue
            W = gate_up.data[e].float().clone().to(dev)
            wrapper = nn.Linear(gate_up.shape[2], gate_up.shape[1], bias=False, device=dev)
            wrapper.weight.data = W.t().clone()
            gptq = GPTQ(wrapper, logger, name_gu, bit)
            gptq.quantizer.configure(bit, perchannel=True, sym=True, mse=False, pack=False)
            out_flat = mlp_inp_flat @ W.t()
            gptq.add_batch(mlp_inp_flat, out_flat)
            gptq.fasterquant(blocksize=blocksize, percdamp=percdamp, groupsize=groupsize, actorder=actorder, name=name_gu)
            q_w = wrapper.weight.data.t().clone()
            experts.gate_up_proj.data[e] = q_w.to(experts.gate_up_proj.dtype).to(experts.gate_up_proj.device)
            gptq.free()

        for e in range(down.shape[0]):
            name_d = f"{prefix}mlp.experts.down_proj.expert_{e}"
            bit = get_bit(name_d)
            if bit is None:
                continue
            W_gu = experts.gate_up_proj.data[e].float().to(dev)
            out1 = mlp_inp_flat @ W_gu.t()
            gate_part = out1[:, :inter]
            up_part = out1[:, inter:]
            mid = torch.nn.functional.silu(up_part) * gate_part
            W_d = down.data[e].float().clone().to(dev)
            wrapper = nn.Linear(inter, down.shape[2], bias=False, device=dev)
            wrapper.weight.data = W_d.t().clone()
            gptq = GPTQ(wrapper, logger, name_d, bit)
            gptq.quantizer.configure(bit, perchannel=True, sym=True, mse=False, pack=False)
            out_flat = mid @ W_d.t()
            gptq.add_batch(mid, out_flat)
            gptq.fasterquant(blocksize=blocksize, percdamp=percdamp, groupsize=groupsize, actorder=actorder, name=name_d)
            q_w = wrapper.weight.data.t().clone()
            experts.down_proj.data[e] = q_w.to(experts.down_proj.dtype).to(experts.down_proj.device)
            gptq.free()

        if (layer_idx + 1) % 6 == 0:
            logger.info(f"  GPTQ experts layer {layer_idx+1}/{num_layers}")
        # Re-run quantized layer to get correct input for next layer
        if layer_idx + 1 < num_layers:
            layer = layer.to(dev)
            cur = layer_inps[layer_idx].to(dev)
            with torch.no_grad():
                next_inp = layer(cur, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0]
            layer_inps[layer_idx + 1] = next_inp.cpu()
        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return model


def main():
    p = argparse.ArgumentParser(description="Qwen3-Coder-Next GPTQ from AlphaQ recipe CSV")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-Next", help="HuggingFace model name")
    p.add_argument("--recipe_csv", type=str, required=True, help="Path to recipe CSV (name, bit_width)")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for quantized model")
    p.add_argument("--nsamples", type=int, default=128, help="Calibration samples")
    p.add_argument("--seqlen", type=int, default=2048, help="Sequence length for calibration")
    p.add_argument("--device_map", type=str, default="auto", help="Device map (auto for multi-GPU)")
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--groupsize", type=int, default=-1)
    p.add_argument("--act_order", action="store_true")
    args = p.parse_args()

    recipe_path = Path(args.recipe_csv)
    if not recipe_path.is_absolute():
        recipe_path = REPO_ROOT / recipe_path
    if not recipe_path.exists():
        logger.error(f"Recipe not found: {recipe_path}")
        return 1
    recipe = load_recipe(str(recipe_path))
    logger.info(f"Loaded recipe: {len(recipe)} entries from {recipe_path}")

    logger.info(f"Loading model: {args.model} (device_map={args.device_map})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
    )

    trainloader = get_wikitext_calibration(args.nsamples, args.seqlen, tokenizer, next(model.parameters()).device)
    logger.info(f"Calibration: {len(trainloader)} samples, seqlen={args.seqlen}")

    run_gptq_qwen3_from_recipe(
        model,
        tokenizer,
        recipe,
        trainloader,
        args.device_map,
        args.seqlen,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        groupsize=args.groupsize,
        actorder=args.act_order,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model and tokenizer to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
