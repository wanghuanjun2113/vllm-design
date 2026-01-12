# SPDX-License-Identifier: Apache-2.0
"""Model provider interface.

This module defines the abstract interface for model computation.
Implementations provide layer-wise QKV computation for different model architectures.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class ModelProviderInterface(ABC):
    """
    Abstract interface for model computation.

    Implementations provide access to model-specific operations like
    QKV projection and rotary embedding application.
    """

    @abstractmethod
    def get_num_layers(self) -> int:
        """
        Get total number of layers in the model.

        Returns:
            Number of transformer layers
        """
        pass

    @abstractmethod
    def compute_layer_qkv(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute QKV projection for a specific layer.

        This method performs the QKV projection for a single layer,
        allowing Blend to recompute specific tokens.

        Args:
            layer_id: Layer index
            hidden_states: Input hidden states [seq_len, hidden_dim]
            residual: Residual connection [seq_len, hidden_dim]

        Returns:
            (q, k, v, residual) tuple of tensors

        Example:
            >>> provider = LlamaAdapter(vllm_model)
            >>> h = torch.randn(10, 4096)
            >>> residual = torch.randn(10, 4096)
            >>> q, k, v, residual = provider.compute_layer_qkv(0, h, residual)
        """
        pass

    @abstractmethod
    def get_rotary_emb(self, layer_id: int):
        """
        Get rotary embedding for a specific layer.

        Args:
            layer_id: Layer index

        Returns:
            Rotary embedding object (implementation-dependent)
        """
        pass

    @abstractmethod
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
        """
        pass

    def get_model_type(self) -> str:
        """
        Get the model type/architecture name.

        Returns:
            Model type string (e.g., "llama", "qwen2")
        """
        return "unknown"

    def supports_sparse_attention(self) -> bool:
        """
        Check if model supports sparse attention optimization.

        Returns:
            True if sparse attention is supported
        """
        return False
