# SPDX-License-Identifier: Apache-2.0
"""Llama model adapter.

This module provides the adapter for Llama-based models
(including Mistral, Mixtral, and other Llama derivatives).
"""

import torch
from vllm_blend.adapters.base import BaseModelAdapter
from vllm_blend.adapters.registry import register_adapter


@register_adapter("llama")
class LlamaAdapter(BaseModelAdapter):
    """
    Adapter for Llama-based models.

    Supports:
    - Llama 1, 2, 3, and variants
    - Mistral 7B
    - Mixtral 8x7B
    - Other Llama-architecture models
    """

    def __init__(self, vllm_model):
        """
        Initialize the Llama adapter.

        Args:
            vllm_model: vLLM LlamaForCausalLM model
        """
        super().__init__(vllm_model)

        # Cache layer references for efficiency
        self._layers = self.vllm_model.model.layers

    def compute_layer_qkv(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute QKV projection for a Llama layer.

        Llama QKV projection:
        1. Input layernorm
        2. QKV projection (single fused op)
        3. Split into Q, K, V
        4. Reshape for attention

        Args:
            layer_id: Layer index [0, num_layers)
            hidden_states: [seq_len, hidden_dim]
            residual: [seq_len, hidden_dim]

        Returns:
            q: [seq_len, num_q_heads, head_dim]
            k: [seq_len, num_kv_heads, head_dim]
            v: [seq_len, num_kv_heads, head_dim]
            residual: [seq_len, hidden_dim]
        """
        self._validate_layer_id(layer_id)

        layer = self._layers[layer_id]

        # Input layernorm
        # Llama uses RMSNorm
        hidden_states = layer.input_layernorm(hidden_states)

        # QKV projection
        # Llama fuses Q, K, V projections into one operation
        qkv, _ = layer.self_attn.qkv_proj(hidden_states)

        # Get head configuration
        num_q_heads = layer.self_attn.num_q_heads
        num_kv_heads = layer.self_attn.num_kv_heads
        head_dim = layer.self_attn.head_dim

        # Split into Q, K, V
        qkv_shape = qkv.shape
        assert len(qkv_shape) == 2, f"Unexpected QKV shape: {qkv_shape}"

        # Reshape and split
        # Original: [seq_len, (num_q_heads + 2*num_kv_heads) * head_dim]
        qkv = qkv.reshape(qkv_shape[0], num_q_heads + 2 * num_kv_heads, head_dim)

        q, k, v = torch.split(
            qkv,
            [
                num_q_heads,  # Query
                num_kv_heads,  # Key
                num_kv_heads,  # Value
            ],
            dim=1,
        )

        # Note: We keep the 2D shape [seq_len, num_heads, head_dim]
        # This is compatible with the blending operations

        return q, k, v, residual

    def get_layer_input_norm(self, layer_id: int):
        """
        Get input layer normalization for a Llama layer.

        Llama uses RMSNorm without bias.
        """
        self._validate_layer_id(layer_id)
        return self._layers[layer_id].input_layernorm

    def get_rotary_emb(self, layer_id: int):
        """
        Get rotary embedding for a Llama layer.

        Llama uses rotary position encoding (RoPE).
        """
        self._validate_layer_id(layer_id)
        return self._layers[layer_id].self_attn.rotary_emb

    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position encoding to Q and K.

        Args:
            q: [seq_len, num_q_heads, head_dim]
            k: [seq_len, num_kv_heads, head_dim]
            positions: [seq_len] position indices
            layer_id: Layer index

        Returns:
            q_rot: [seq_len, num_q_heads, head_dim]
            k_rot: [seq_len, num_kv_heads, head_dim]
        """
        rotary_emb = self.get_rotary_emb(layer_id)

        # Apply RoPE
        # Llama's rotary_emb implements __call__(positions, q, k)
        # Note: The exact signature may vary
        try:
            q_rot, k_rot = rotary_emb(positions, q, k)
        except Exception as e:
            # Fallback: try alternative signature
            # Some implementations might have different order
            inv_freq = rotary_emb.inv_freq
            # Manual RoPE application could go here
            raise

        return q_rot, k_rot

    def get_model_type(self) -> str:
        """Return 'llama' for Llama-based models."""
        return "llama"

    def supports_sparse_attention(self) -> bool:
        """
        Check if this Llama model supports sparse attention.

        Returns:
            True if the model has sparse attention support
        """
        # Check if model has sliding window or other sparse attention
        # For Llama, this would depend on the specific variant
        return False  # Standard Llama doesn't use sparse attention

    def get_head_config(self, layer_id: int) -> dict:
        """
        Get attention head configuration.

        Returns:
            Dict with head configuration details
        """
        self._validate_layer_id(layer_id)
        attn = self._layers[layer_id].self_attn

        return {
            "num_q_heads": attn.num_q_heads,
            "num_kv_heads": attn.num_kv_heads,
            "head_dim": attn.head_dim,
            "use_gqa": attn.num_kv_heads < attn.num_q_heads,
            "total_q_heads": attn.num_q_heads,
            "total_kv_heads": attn.num_kv_heads,
        }

    def get_attention_pattern(self) -> str:
        """
        Get the attention pattern used by this model.

        Returns:
            Attention pattern: "causal", "bidirectional", etc.
        """
        return "causal"


# Auto-register for convenience
ModelAdapterRegistry = None
# Import registry here to avoid circular dependency
from vllm_blend.adapters.registry import ModelAdapterRegistry

# The adapter is already registered via the decorator above
# This just confirms it's available
