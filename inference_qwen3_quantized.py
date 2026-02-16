#!/usr/bin/env python3
"""
Load Qwen3-Coder-Next quantized model (saved by qwen3_gptq_from_recipe.py) and run inference.
Supports multi-GPU via device_map="auto".

Usage:
  python inference_qwen3_quantized.py --model_path ./out_qwen3_bpp4 --prompt "Hello, world" --max_new_tokens 64
  python inference_qwen3_quantized.py --model_path ./out_qwen3_bpp4 --device_map auto
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description="Inference with Qwen3-Coder-Next quantized model")
    p.add_argument("--model_path", type=str, required=True, help="Path to quantized model dir (output of qwen3_gptq_from_recipe.py)")
    p.add_argument("--device_map", type=str, default="auto", help="Device map for multi-GPU")
    p.add_argument("--prompt", type=str, default="The meaning of life is")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    print(f"Loading tokenizer and model from {args.model_path} (device_map={args.device_map}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    if args.device_map == "auto" and next(model.parameters()).device.type == "cuda":
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("Output:", text)
    return 0


if __name__ == "__main__":
    exit(main())
