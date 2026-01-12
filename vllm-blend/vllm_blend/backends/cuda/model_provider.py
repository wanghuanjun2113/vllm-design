# SPDX-License-Identifier: Apache-2.0
"""CUDA model provider implementation.

This module provides model computation provider for CUDA systems,
using adapter pattern to support different model architectures.
"""

from typing import Tuple

import torch

from vllm_blend.adapters import ModelAdapterRegistry
from vllm_blend.providers.model_provider import ModelProviderInterface


class CUDAModelProvider(ModelProviderInterface):
    """
    Model computation provider for CUDA systems.

    This provider uses model adapters to perform layer-wise QKV computation
    for different model architectures (Llama, Qwen, etc.).

    Example:
        >>> provider = CUDAModelProvider(vllm_model, model_runner)
        >>> num_layers = provider.get_num_layers()
        >>> q, k, v, residual = provider.compute_layer_qkv(0, hidden_states, residual)
    """

    def __init__(self, vllm_model, model_runner=None):
        """
        Initialize the CUDA model provider.

        Args:
            vllm_model: vLLM model instance (e.g., LlamaForCausalLM)
            model_runner: Optional vLLM model runner for additional context
        """
        self.vllm_model = vllm_model
        self.model_runner = model_runner

        # Get appropriate adapter for this model
        self.adapter = ModelAdapterRegistry.get_adapter(vllm_model)

        # Cache model info
        self.num_layers = self.adapter.get_num_layers()
        self.model_type = self.adapter.get_model_type()

    def get_num_layers(self) -> int:
        """
        Get total number of layers in the model.

        Returns:
            Number of transformer layers
        """
        return self.num_layers

    def compute_layer_qkv(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute QKV projection for a specific layer.

        This method delegates to the model-specific adapter to perform
        the QKV projection, which varies between model architectures.

        Args:
            layer_id: Layer index [0, num_layers)
            hidden_states: Input hidden states [seq_len, hidden_dim]
            residual: Residual connection [seq_len, hidden_dim]

        Returns:
            (q, k, v, residual) tuple

        Note:
            The adapter handles model-specific details like:
            - Different QKV projection layouts
            - Multi-query vs multi-head attention
            - GQA (Grouped Query Attention) variations
        """
        # Delegate to adapter
        return self.adapter.compute_layer_qkv(layer_id, hidden_states, residual)

    def get_rotary_emb(self, layer_id: int):
        """
        Get rotary embedding for a specific layer.

        Args:
            layer_id: Layer index

        Returns:
            Rotary embedding object from the model
        """
        return self.adapter.get_rotary_emb(layer_id)

    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position encoding to Q and K tensors.

        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            positions: Position indices [seq_len]
            layer_id: Layer index

        Returns:
            (q_rotated, k_rotated) tuple

        Note:
            Delegates to the adapter to handle model-specific RoPE application.
        """
        return self.adapter.apply_rotary_emb(q, k, positions, layer_id)

    def get_model_type(self) -> str:
        """
        Get the model type/architecture name.

        Returns:
            Model type string (e.g., "llama", "qwen2")
        """
        return self.model_type

    def supports_sparse_attention(self) -> bool:
        """
        Check if model supports sparse attention optimization.

        Returns:
            True if sparse attention is supported by this model
        """
        # Check if model has sparse attention support
        # This could be model-specific (e.g., Qwen3 with sparse attention)
        return self.adapter.supports_sparse_attention()

    def get_layer_input_norm(self, layer_id: int):
        """
        Get input layer normalization for a specific layer.

        Args:
            layer_id: Layer index

        Returns:
            Layer normalization module
        """
        return self.adapter.get_layer_input_norm(layer_id)

    def get_attention_module(self, layer_id: int):
        """
        Get the attention module for a specific layer.

        Args:
            layer_id: Layer index

        Returns:
            Attention module (e.g., LlamaAttention)
        """
        # Access through vLLM model structure
        # Model-specific - adapter might provide helper
        return self.vllm_model.model.layers[layer_id].self_attn

    def get_head_config(self, layer_id: int) -> dict:
        """
        Get attention head configuration for a layer.

        Returns:
            Dictionary with:
            - num_q_heads: Number of query heads
            - num_kv_heads: Number of key/value heads
            - head_dim: Head dimension
            - use_gqa: Whether using grouped query attention
        """
        attn_module = self.get_attention_module(layer_id)

        return {
            "num_q_heads": attn_module.num_q_heads,
            "num_kv_heads": attn_module.num_kv_heads,
            "head_dim": attn_module.head_dim,
            "use_gqa": attn_module.num_kv_heads < attn_module.num_q_heads,
        }


class CUDALayerwiseModelProvider(ModelProviderInterface):
    """
    Layer-wise model provider for CUDA systems.

    This provider is optimized for layer-wise execution, which is useful
    for blending scenarios where you want to process one layer at a time.

    It provides generator-based layer execution that can be interleaved
    with cache retrieval and blending operations.
    """

    def __init__(self, vllm_model, model_runner=None):
        """
        Initialize the layer-wise model provider.

        Args:
            vllm_model: vLLM model instance
            model_runner: Optional model runner
        """
        self.vllm_model = vllm_model
        self.model_runner = model_runner
        self.adapter = ModelAdapterRegistry.get_adapter(vllm_model)
        self.num_layers = self.adapter.get_num_layers()

    def get_num_layers(self) -> int:
        """Get number of layers."""
        return self.num_layers

    def compute_layer_qkv(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute QKV for a single layer.

        Similar to CUDAModelProvider but optimized for layer-wise execution.
        """
        return self.adapter.compute_layer_qkv(layer_id, hidden_states, residual)

    def get_rotary_emb(self, layer_id: int):
        """Get rotary embedding for a layer."""
        return self.adapter.get_rotary_emb(layer_id)

    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding."""
        return self.adapter.apply_rotary_emb(q, k, positions, layer_id)

    def get_model_type(self) -> str:
        """Get model type."""
        return self.adapter.get_model_type()

    def supports_sparse_attention(self) -> bool:
        """Check sparse attention support."""
        return self.adapter.supports_sparse_attention()

    def execute_layer_generator(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ):
        """
        Execute layers one by one as a generator.

        This is useful for layer-wise blending where you want to:
        1. Retrieve cache for layer N
        2. Compute layer N
        3. Blend results
        4. Move to layer N+1

        Args:
            hidden_states: Initial hidden states
            positions: Position indices

        Yields:
            (layer_id, q, k, v, residual) for each layer

        Example:
            >>> for layer_id, q, k, v, residual in provider.execute_layer_generator(h, pos):
            ...     # Retrieve cache for this layer
            ...     cached_kv = cache_provider.retrieve_layer(...)
            ...     # Blend with cached KV
            ...     q, k, v, residual = blender.process_qkv(q, k, v, residual, layer_id)
        """
        residual = hidden_states.clone()

        for layer_id in range(self.num_layers):
            # Compute QKV for this layer
            q, k, v, residual = self.compute_layer_qkv(
                layer_id, hidden_states, residual
            )

            # Apply RoPE
            q, k = self.apply_rotary_emb(q, k, positions, layer_id)

            # Yield results
            yield layer_id, q, k, v, residual

            # Prepare for next layer (if needed)
            # Note: This is simplified; actual layer execution would
            # involve attention, FFN, etc.
            # For blending, we mainly need the QKV computation
