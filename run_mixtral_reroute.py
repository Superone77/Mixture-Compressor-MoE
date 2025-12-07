"""
Example Python-only entrypoint to run Mixtral 8x7B with test-time MoE rerouting
inside lm-evaluation-harness.

Usage:
    python run_mixtral_reroute.py --model mistralai/Mixtral-8x7B-v0.1 --tasks hellaswag arc_easy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def build_model_args(args: argparse.Namespace) -> str:
    """Convert parsed arguments into the model_args string expected by simple_evaluate."""
    pairs: Dict[str, Any] = {
        "pretrained": args.model,
        "trust_remote_code": True,
        "reroute_moe": True,
        "reroute_steps": args.reroute_steps,
        "reroute_lr": args.reroute_lr,
        "reroute_chunk_size": args.reroute_chunk_size,
        "reroute_layer_start": args.reroute_layer_start,
        "reroute_log": args.reroute_log,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "use_fast_tokenizer": args.use_fast_tokenizer,
    }
    return ",".join(f"{k}={v}" for k, v in pairs.items() if v is not None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixtral rerouting eval via Python API")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model id or local path (e.g., mistralai/Mixtral-8x7B-v0.1)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["hellaswag", "arc_easy"],
        help="Task names to evaluate",
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=None, help="Integer or fraction")
    parser.add_argument("--device", type=str, default=None, help="cuda:0, cpu, etc.")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True)

    parser.add_argument("--max_gen_toks", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--reroute_steps", type=int, default=5)
    parser.add_argument("--reroute_lr", type=float, default=5e-3)
    parser.add_argument("--reroute_chunk_size", type=int, default=64)
    parser.add_argument("--reroute_layer_start", type=int, default=0)
    parser.add_argument(
        "--reroute_log",
        action="store_true",
        help="Enable verbose rerouter logging (layer weights, delta norms).",
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
        help="Optional path to save the raw results JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure local lm-evaluation-harness is on path
    repo_root = Path(__file__).resolve().parent
    harness_root = args.harness_path or (repo_root / "lm-evaluation-harness")
    sys.path.insert(0, str(harness_root))

    import lm_eval  # noqa: WPS433

    model_args = build_model_args(args)
    gen_kwargs: Dict[str, Any] = {
        "max_gen_toks": args.max_gen_toks,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        device=args.device,
        gen_kwargs=gen_kwargs,
    )

    print(json.dumps(results["results"], indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Saved full results to {args.output}")


if __name__ == "__main__":
    main()
