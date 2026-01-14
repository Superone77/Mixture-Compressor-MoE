#!/usr/bin/env python3
import argparse
import csv
import math
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM

from datautils import get_loaders
from eval_ppl_utils import llama_eval
from utils.quantizer_moe import Quantizer


MOE_WEIGHT_NAMES = {"w1", "w2", "w3"}


def _is_moe_linear(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if ".block_sparse_moe.experts." not in name:
        return False
    return name.split(".")[-1] in MOE_WEIGHT_NAMES


def load_alpha_stats(csv_path: Path, alpha_prefix: Optional[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("layer_name", "")
            if not name:
                continue
            if alpha_prefix and not name.startswith(alpha_prefix):
                continue
            if ".block_sparse_moe.experts." not in name:
                continue
            if name.split(".")[-1] not in MOE_WEIGHT_NAMES:
                continue
            try:
                alpha_val = float(row.get("alpha", "nan"))
                variance_val = float(row.get("variance", "nan"))
            except ValueError:
                alpha_val = float("nan")
                variance_val = float("nan")
            stats[name] = {"alpha": alpha_val, "variance": variance_val}
    return stats


def compute_model_alpha_value(
    layer_stats: Dict[str, Dict[str, float]],
    layer_bits: Dict[str, int],
    gamma: float,
    eps: float = 1e-8,
) -> float:
    alphas = [
        stats["alpha"]
        for stats in layer_stats.values()
        if isinstance(stats.get("alpha"), (int, float)) and math.isfinite(stats["alpha"])
    ]
    if not alphas:
        return float("nan")

    alpha_median = statistics.median(alphas)
    total = 0.0
    for layer_name, stats in layer_stats.items():
        alpha = stats.get("alpha", float("nan"))
        variance = stats.get("variance", float("nan"))
        if not (isinstance(alpha, (int, float)) and math.isfinite(alpha)):
            continue
        if not (isinstance(variance, (int, float)) and math.isfinite(variance)):
            continue
        bit = layer_bits.get(layer_name)
        if bit is None:
            continue
        x = alpha_median / max(alpha, eps)
        y = x ** gamma
        total += y * variance * (2.0 ** (-2 * bit))
    return total


def build_random_bit_assignments(
    layer_names: List[str],
    candidate_bits: List[int],
    rng: random.Random,
) -> Dict[str, int]:
    return {name: rng.choice(candidate_bits) for name in layer_names}


def quantize_moe_layers(
    model: nn.Module,
    layer_bits: Dict[str, int],
    original_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    for name, module in model.named_modules():
        if name not in layer_bits:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if original_weights is not None:
            module.weight.data = original_weights[name].to(module.weight.device).clone()
        q = Quantizer()
        q.configure(layer_bits[name], perchannel=True, sym=True, mse=False, pack=False)
        q.find_params(module.weight.data, weight=True)
        q_weight = q.quantize(module.weight.data)
        module.weight.data = q_weight.reshape_as(module.weight.data)


def snapshot_moe_weights(model: nn.Module, layer_names: List[str]) -> Dict[str, torch.Tensor]:
    weights: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if name in layer_names:
            weights[name] = module.weight.detach().cpu().clone()
    return weights


def get_model(args) -> nn.Module:
    def _skip(*_args, **_kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = _skip
    torch.nn.init.uniform_ = _skip
    torch.nn.init.normal_ = _skip

    config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    if not isinstance(model, MixtralForCausalLM):
        raise TypeError("Loaded model is not MixtralForCausalLM.")
    model.seqlen = args.seqlen
    return model


def resolve_testenc(loaders):
    if isinstance(loaders, tuple) and len(loaders) == 2:
        return loaders[1]
    return loaders


def plot_scatter(alpha_vals: List[float], ppl_vals: List[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(alpha_vals, ppl_vals, s=30, alpha=0.8, edgecolors="black", linewidths=0.4)
    ax.set_xlabel("Model Alpha")
    ax.set_ylabel("Perplexity (Wikitext)")
    ax.set_title("Alpha vs Perplexity (Random Mixed Precision)")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_bits(bits_str: str) -> List[int]:
    bits = [int(x.strip()) for x in bits_str.split(",") if x.strip()]
    if not bits:
        raise ValueError("candidate_bits is empty.")
    if any(b <= 0 for b in bits):
        raise ValueError("candidate_bits must be positive integers.")
    return bits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="HF model path, e.g. mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--alpha_csv", type=str, default="data/mixtral_alpha_FARMS.csv")
    parser.add_argument(
        "--alpha_prefix",
        type=str,
        default="model.layers.0.block_sparse_moe.experts",
        help="Prefix filter for alpha CSV rows (set empty to include all MoE layers).",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--candidate_bits", type=str, default="2,3,4,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--reload_each_trial", action="store_true")
    parser.add_argument("--output_csv", type=str, default="random_moe_experiments.csv")
    parser.add_argument("--output_plot", type=str, default="alpha_ppl_scatter.png")

    args = parser.parse_args()

    candidate_bits = parse_bits(args.candidate_bits)
    alpha_prefix = args.alpha_prefix.strip() or None

    alpha_csv_path = Path(args.alpha_csv)
    if not alpha_csv_path.exists():
        raise FileNotFoundError(f"Alpha CSV not found: {alpha_csv_path}")

    alpha_stats = load_alpha_stats(alpha_csv_path, alpha_prefix)
    if not alpha_stats:
        raise ValueError("No alpha stats matched the prefix/filter.")

    rng = random.Random(args.seed)
    results = []

    loaders = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        model=args.model,
    )
    testenc = resolve_testenc(loaders)

    for trial in range(args.trials):
        print(f"Trial {trial + 1}/{args.trials}")
        if args.reload_each_trial or trial == 0:
            model = get_model(args)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

            moe_layer_names = [
                name
                for name, module in model.named_modules()
                if _is_moe_linear(name, module)
            ]
            original_weights = None
            if not args.reload_each_trial:
                original_weights = snapshot_moe_weights(model, moe_layer_names)
        else:
            moe_layer_names = list(original_weights.keys())

        layer_bits = build_random_bit_assignments(moe_layer_names, candidate_bits, rng)
        quantize_moe_layers(model, layer_bits, original_weights=original_weights)

        ppl = llama_eval(model, testenc, args.device, args.dataset)
        model_alpha = compute_model_alpha_value(alpha_stats, layer_bits, args.gamma)
        print(f"Alpha: {model_alpha:.6f}, PPL: {ppl:.4f}")

        results.append(
            {
                "trial": trial,
                "alpha": model_alpha,
                "ppl": ppl,
            }
        )

        if args.reload_each_trial:
            del model
            torch.cuda.empty_cache()

    output_csv_path = Path(args.output_csv)
    with output_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "alpha", "ppl"])
        writer.writeheader()
        writer.writerows(results)

    alpha_vals = [r["alpha"] for r in results]
    ppl_vals = [r["ppl"] for r in results]
    plot_scatter(alpha_vals, ppl_vals, Path(args.output_plot))

    print(f"Saved results to {output_csv_path}")
    print(f"Saved scatter plot to {args.output_plot}")


if __name__ == "__main__":
    main()
