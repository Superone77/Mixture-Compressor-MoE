"""
Evaluate a quantized OLMoE model with test-time rerouting using lm-evaluation-harness.

Example:
    python run_quantized_olmoe_reroute.py \
        --model_path /path/to/quantized/OLMoE-1B-7B \
        --tasks hellaswag,arc_easy \
        --batch_size 1 \
        --reroute_steps 3 --reroute_chunk_size 32 --reroute_lr 0.005 --reroute_log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoTokenizer

from inference import load_quantized_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate quantized OLMoE with rerouting via lm_eval"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Directory of quantized model (contains qmodel.pt, config.json, tokenizer files)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="wikitext",
        help="Comma-separated list of tasks to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--load_args",
        type=str,
        nargs="*",
        default=[],
        help="Additional key=value pairs for load_quantized_model (e.g., device_map=auto torch_dtype=torch.float16)",
    )

    # Generation / rerouting controls
    parser.add_argument("--max_gen_toks", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--reroute_steps", type=int, default=3)
    parser.add_argument("--reroute_lr", type=float, default=5e-3)
    parser.add_argument("--reroute_chunk_size", type=int, default=32)
    parser.add_argument("--reroute_layer_start", type=int, default=0)
    parser.add_argument(
        "--reroute_log",
        action="store_true",
        help="Enable verbose rerouter logging (weights, delta norms).",
    )

    parser.add_argument(
        "--harness_path",
        type=Path,
        default=None,
        help="Path to local lm-evaluation-harness (defaults to ./lm-evaluation-harness relative to this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save full results JSON",
    )
    return parser.parse_args()


def parse_load_args(args_str):
    result = {}
    for arg in args_str:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if value.lower() == "true":
                result[key] = True
            elif value.lower() == "false":
                result[key] = False
            else:
                try:
                    result[key] = int(value)
                except ValueError:
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value
        else:
            result[arg] = True
    return result


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    harness_root = args.harness_path or (repo_root / "lm-evaluation-harness")
    sys.path.insert(0, str(harness_root))

    import lm_eval  # noqa: WPS433

    print("=" * 80)
    print("Quantized OLMoE Evaluation with Rerouting")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Reroute steps: {args.reroute_steps}, lr: {args.reroute_lr}, chunk: {args.reroute_chunk_size}")
    print("=" * 80)

    load_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    load_kwargs.update(parse_load_args(args.load_args))

    print("\nLoading quantized model...")
    model = load_quantized_model(str(args.model_path), load_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Build HFLM wrapper with rerouting enabled
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
        reroute_moe=True,
        reroute_steps=args.reroute_steps,
        reroute_lr=args.reroute_lr,
        reroute_chunk_size=args.reroute_chunk_size,
        reroute_layer_start=args.reroute_layer_start,
        reroute_log=args.reroute_log,
    )

    gen_kwargs = {
        "max_gen_toks": args.max_gen_toks,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"\nEvaluating tasks: {task_list}")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=args.batch_size,
        gen_kwargs=gen_kwargs,
    )

    print(json.dumps(results.get("results", {}), indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Saved full results to {args.output}")


if __name__ == "__main__":
    main()
