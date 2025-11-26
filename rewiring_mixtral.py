"""
Rewiring support for Mixtral models

This module adds delta parameter support and rewiring methods to Mixtral models,
similar to the implementation in rewiring-moe/models/deepseek_moe/modeling_deepseek.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def patch_mixtral_gate_for_rewiring(gate_module):
    """
    Patch a Mixtral gate module to support delta parameters for rewiring.
    
    Args:
        gate_module: The gate (router) module from MixtralSparseMoeBlock
    """
    if hasattr(gate_module, 'use_delta'):
        # Already patched
        return
    
    # Add delta parameter attributes
    gate_module.use_delta = False
    gate_module.delta = None
    gate_module.weights = 1.0
    gate_module.score = None
    gate_module.route_logits = None
    
    # Store original forward
    original_forward = gate_module.forward
    
    def patched_forward(hidden_states):
        # Compute router logits
        router_logits = original_forward(hidden_states)
        gate_module.route_logits = router_logits
        
        # Apply delta if enabled
        if gate_module.use_delta and gate_module.delta is not None:
            router_logits = router_logits + gate_module.delta.unsqueeze(0) * gate_module.weights
        
        # Compute scores (softmax probabilities)
        scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        gate_module.score = scores
        
        return router_logits
    
    gate_module.forward = patched_forward
    
    def enable_delta():
        """Enable delta for test-time fine-tuning"""
        gate_module.use_delta = True
        if gate_module.delta is None:
            num_experts = gate_module.weight.shape[0] if hasattr(gate_module, 'weight') else 8
            device = next(gate_module.parameters()).device
            gate_module.delta = nn.Parameter(
                torch.zeros(num_experts, device=device, dtype=torch.float32),
                requires_grad=True
            )
    
    def disable_delta():
        """Disable delta"""
        gate_module.use_delta = False
    
    def set_weights(weights):
        """Set layer weights for delta scaling"""
        gate_module.weights = weights
    
    def get_score():
        """Get routing scores"""
        return gate_module.score
    
    def get_route_logits():
        """Get routing logits"""
        return gate_module.route_logits
    
    # Add methods to gate module
    gate_module.enable_delta = enable_delta
    gate_module.disable_delta = disable_delta
    gate_module.set_weights = set_weights
    gate_module.get_score = get_score
    gate_module.get_route_logits = get_route_logits


def add_rewiring_support_to_mixtral(model):
    """
    Add rewiring support methods to a Mixtral model.
    
    Args:
        model: The Mixtral model (quantized or regular)
    """
    # Find all MoE layers
    moe_layers = []
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
            moe_layers.append((idx, layer))
            # Patch the gate
            patch_mixtral_gate_for_rewiring(layer.block_sparse_moe.gate)
    
    if len(moe_layers) == 0:
        raise ValueError("No MoE layers found in model. Make sure this is a Mixtral MoE model.")
    
    def enable_delta_for_layers(layer_indices):
        """Enable delta for specific MoE layers"""
        for idx, layer in moe_layers:
            if idx in layer_indices:
                layer.block_sparse_moe.gate.enable_delta()
    
    def disable_all_deltas():
        """Disable all delta parameters"""
        for idx, layer in moe_layers:
            if hasattr(layer.block_sparse_moe.gate, 'disable_delta'):
                layer.block_sparse_moe.gate.disable_delta()
    
    def get_delta_parameters():
        """Get all delta parameters for TTF"""
        delta_params = []
        for idx, layer in moe_layers:
            gate = layer.block_sparse_moe.gate
            if hasattr(gate, 'delta') and gate.delta is not None:
                delta_params.append(gate.delta)
        return delta_params
    
    def get_scores():
        """Get all routing scores"""
        scores = []
        for idx, layer in moe_layers:
            gate = layer.block_sparse_moe.gate
            if hasattr(gate, 'get_score'):
                score = gate.get_score()
                if score is not None:
                    scores.append(score)
        return scores
    
    def set_weights(layer_indices, weights):
        """Set weights for specific layers"""
        for idx, layer in moe_layers:
            if idx in layer_indices:
                gate = layer.block_sparse_moe.gate
                if hasattr(gate, 'set_weights'):
                    # weights is a dict mapping layer_idx to weight value
                    layer_weight = weights.get(idx, 1.0)
                    gate.set_weights(layer_weight)
    
    # Add methods to model
    model.enable_delta_for_layers = enable_delta_for_layers
    model.disable_all_deltas = disable_all_deltas
    model.get_delta_parameters = get_delta_parameters
    model.get_scores = get_scores
    model.set_weights = set_weights
    
    return model

