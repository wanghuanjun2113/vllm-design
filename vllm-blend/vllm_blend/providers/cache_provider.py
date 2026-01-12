# SPDX-License-Identifier: Apache-2.0
"""Cache provider interface.

This module defines the abstract interface for KV cache providers.
Implementations can include LMCache, CPU memory, remote storage, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class CacheProviderInterface(ABC):
    """
    Abstract interface for KV cache providers.

    Implementations of this interface provide storage and retrieval of
    KV caches, allowing Blend to work with different backends.
    """

    @abstractmethod
    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        layer_id: int,
        metadata: Dict[str, Any],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve cached KV tensors for a specific layer.

        Args:
            tokens: Token sequence to lookup
            layer_id: Layer index
            metadata: Additional metadata (e.g., positions, mask)

        Returns:
            (k_cache, v_cache) tuple if cache hit, None if miss

        Example:
            >>> provider = LMCacheProvider(...)
            >>> tokens = torch.tensor([1, 2, 3, 4, 5])
            >>> result = provider.retrieve_layer(tokens, layer_id=0, metadata={})
            >>> if result is not None:
            >>>     k_cache, v_cache = result
        """
        pass

    @abstractmethod
    def store_layer(
        self,
        tokens: torch.Tensor,
        layer_id: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Store KV tensors for a specific layer.

        Args:
            tokens: Token sequence associated with the KV cache
            layer_id: Layer index
            k_cache: Key tensor to store
            v_cache: Value tensor to store
            metadata: Additional metadata
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with statistics like:
            - hit_rate: Cache hit rate
            - total_entries: Number of entries in cache
            - size_bytes: Cache size in bytes
        """
        pass

    def supports_layerwise_retrieval(self) -> bool:
        """
        Check if provider supports layer-wise retrieval.

        Returns:
            True if retrieve_layer() is implemented, False otherwise
        """
        return True
