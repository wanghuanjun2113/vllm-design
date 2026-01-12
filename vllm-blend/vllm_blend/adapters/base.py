# SPDX-License-Identifier: Apache-2.0
"""Base model adapter.

This module defines the abstract base class for model adapters.
"""

from abc import abstractmethod

import torch
from vllm_blend.providers.model_provider import ModelProviderInterface


class BaseModelAdapter(ModelProviderInterface):
    """
    Base class for model-specific adapters.

    Each model architecture (Llama, Qwen, Mistral, etc.) should implement
    this adapter to provide model-specific operations.

    Subclasses must implement:
    - get_num_layers()
    - compute_layer_qkv()
    - get_rotary_emb()
    - apply_rotary_emb()
    - get_layer_input_norm()

    Optional implementations:
    - get_model_type()
    - supports_sparse_attention()
    """

    def __init__(self, vllm_model):
        """
        Initialize the adapter.

        Args:
            vllm_model: vLLM model instance
        """
        self.vllm_model = vllm_model
        self.num_layers = len(vllm_model.model.layers)

    def get_num_layers(self) -> int:
        """Get number of layers."""
        return self.num_layers

    @abstractmethod
    def compute_layer_qkv(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute QKV projection for a specific layer.

        Args:
            layer_id: Layer index
            hidden_states: Input hidden states [seq_len, hidden_dim]
            residual: Residual connection [seq_len, hidden_dim]

        Returns:
            (q, k, v, residual) tuple

        Note:
            This is the core method that encapsulates model-specific
            QKV projection logic.
        """
        pass

    @abstractmethod
    def get_layer_input_norm(self, layer_id: int):
        """
        Get input layer normalization for a layer.

        Args:
            layer_id: Layer index

        Returns:
            Layer normalization module
        """
        pass

    @abstractmethod
    def get_rotary_emb(self, layer_id: int):
        """
        Get rotary embedding for a layer.

        Args:
            layer_id: Layer index

        Returns:
            Rotary embedding object with __call__ method
        """
        pass

    @abstractmethod
    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position encoding.

        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            positions: Position indices [seq_len]
            layer_id: Layer index

        Returns:
            (q_rot, k_rot) tuple
        """
        pass

    def get_model_type(self) -> str:
        """
        Get model type string.

        Returns:
            Model type (e.g., "llama", "qwen2")
        """
        # Default implementation
        return type(self.vllm_model).__name__

    def supports_sparse_attention(self) -> bool:
        """
        Check if model supports sparse attention.

        Returns:
            True if sparse attention is supported
        """
        return False

    def _validate_layer_id(self, layer_id: int):
        """Validate layer ID is in range."""
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(
                f"Invalid layer_id {layer_id}. "
                f"Must be in [0, {self.num_layers})"
            )
