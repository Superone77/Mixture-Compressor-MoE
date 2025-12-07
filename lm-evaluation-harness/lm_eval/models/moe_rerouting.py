from __future__ import annotations

import logging
from typing import Dict, Iterable

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def calc_intensity(layer_probs: torch.Tensor, topk: int) -> float:
    """Compute routing confidence score for a layer."""
    if layer_probs.numel() == 0:
        return 0.0
    # shape: [seq_len, experts] (batch size is assumed to be 1)
    k = min(topk, layer_probs.shape[-1])
    topk_weights, _ = torch.topk(layer_probs, k=k, dim=-1)
    # Higher intensity -> more confident routing
    intensity = (-torch.log(topk_weights + 1e-12)).sum(dim=-1).mean()
    return float(intensity.item())


def compute_layer_weights(
    score_dict: Dict[int, torch.Tensor], topk: int, start_layer: int = 0
) -> Dict[int, float]:
    """Normalize routing confidence across layers to derive per-layer weights."""
    if not score_dict:
        return {}

    intensities: Dict[int, float] = {}
    for layer_idx, scores in score_dict.items():
        if scores is None or layer_idx < start_layer:
            continue
        # scores are stored as [seq_len, experts] on CPU
        intensities[layer_idx] = calc_intensity(scores, topk)

    if not intensities:
        return {}

    vals = list(intensities.values())
    min_val, max_val = min(vals), max(vals)
    denom = (max_val - min_val) + 1e-6

    return {idx: (score - min_val) / denom for idx, score in intensities.items()}


class MixtralRerouter:
    """Utility to inject delta parameters into Mixtral MoE layers at runtime."""

    def __init__(self, model) -> None:
        self.model = model
        self.blocks: list[tuple[int, torch.nn.Module]] = []
        self.top_k: int | None = None
        self.active_layers: set[int] = set()
        self._patch_model()

    def _patch_model(self) -> None:
        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            logger.warning("Rerouting requested but Mixtral layers could not be found.")
            return

        for layer_idx, layer in enumerate(layers):
            block = getattr(layer, "block_sparse_moe", None)
            if block is None or not hasattr(block, "gate"):
                continue

            # Avoid double patching
            if hasattr(block, "reroute_delta"):
                self.blocks.append((layer_idx, block))
                continue

            device = block.gate.weight.device
            delta = torch.nn.Parameter(
                torch.zeros(block.num_experts, device=device, dtype=torch.float32),
                requires_grad=False,
            )
            block.register_parameter("reroute_delta", delta)
            block.reroute_weight = 1.0
            block.reroute_use_delta = False
            block.reroute_scores = None
            block.reroute_router_logits = None

            def _gate_hook(module, inputs, output, block=block):
                logits = output
                block.reroute_router_logits = logits.detach()
                if block.reroute_use_delta:
                    logits = logits + block.reroute_delta.unsqueeze(0) * float(
                        block.reroute_weight
                    )
                # Store on CPU to reduce GPU memory pressure
                block.reroute_scores = F.softmax(
                    logits, dim=-1, dtype=torch.float32
                ).detach().cpu()
                return logits

            block.gate._reroute_parent = block  # type: ignore[attr-defined]
            block._reroute_hook = block.gate.register_forward_hook(_gate_hook)
            self.blocks.append((layer_idx, block))
            if self.top_k is None:
                self.top_k = getattr(block, "top_k", None)

    @property
    def has_blocks(self) -> bool:
        return len(self.blocks) > 0

    def enable(self, layer_indices: Iterable[int] | None = None) -> None:
        indices = set(layer_indices) if layer_indices is not None else {
            idx for idx, _ in self.blocks
        }
        self.active_layers = indices
        for idx, block in self.blocks:
            is_active = idx in indices
            block.reroute_use_delta = is_active
            block.reroute_delta.requires_grad = is_active
            block.reroute_delta.data.zero_()

    def reset(self) -> None:
        for _, block in self.blocks:
            block.reroute_use_delta = False
            block.reroute_delta.requires_grad = False
            block.reroute_delta.data.zero_()
            block.reroute_scores = None
            block.reroute_router_logits = None
        self.active_layers = set()

    def get_delta_parameters(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for _, block in self.blocks:
            if block.reroute_use_delta:
                params.append(block.reroute_delta)
        return params

    def get_scores(self) -> Dict[int, torch.Tensor]:
        scores: Dict[int, torch.Tensor] = {}
        for idx, block in self.blocks:
            if block.reroute_scores is not None:
                scores[idx] = block.reroute_scores
        return scores

    def set_weights(self, weights: Dict[int, float]) -> None:
        for idx, block in self.blocks:
            block.reroute_weight = float(weights.get(idx, 1.0))
