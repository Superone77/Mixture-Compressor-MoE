#!/usr/bin/env python3
"""
Load Qwen3-Coder-Next and run inference. Works for both the original model and the
quantize-then-dequantize model saved by qwen3_gptq_from_recipe.py (same API: from_pretrained).
Supports multi-GPU via device_map="auto".

Usage:
  python inference_qwen3_quantized.py --model_path Qwen/Qwen3-Coder-Next --prompt "Hello" --max_new_tokens 64
  python inference_qwen3_quantized.py --model_path ./out_qwen3_bpp4 --prompt "Hello" --max_new_tokens 64
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description="Inference with Qwen3-Coder-Next (original or GPTQ-roundtrip model)")
    p.add_argument("--model_path", type=str, required=True, help="HuggingFace model id (e.g. Qwen/Qwen3-Coder-Next) or local dir with config.json")
    p.add_argument("--device_map", type=str, default="auto", help="Device map for multi-GPU")
    p.add_argument("--prompt", type=str, default="The meaning of life is")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    print(f"Loading tokenizer and model from {args.model_path} (device_map={args.device_map}) ...")
    # Same loader for original HF model or local GPTQ-roundtrip dir
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
