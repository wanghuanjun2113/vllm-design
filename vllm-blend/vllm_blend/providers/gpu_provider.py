# SPDX-License-Identifier: Apache-2.0
"""GPU provider interface.

This module defines the abstract interface for GPU KV cache access.
Different hardware platforms (CUDA, NPU, ROCm) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class GPUProviderInterface(ABC):
    """
    Abstract interface for GPU KV cache access.

    Implementations provide access to KV cache stored in GPU memory,
    allowing Blend to read and update cached tensors.
    """

    @abstractmethod
    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current KV tensors from GPU memory.

        Args:
            layer_id: Layer index

        Returns:
            (k_gpu, v_gpu) tuple of tensors from GPU memory

        Example:
            >>> provider = CUDAGPUProvider(model_runner)
            >>> k_gpu, v_gpu = provider.get_kv(layer_id=0)
            >>> print(f"K shape: {k_gpu.shape}, V shape: {v_gpu.shape}")
        """
        pass

    @abstractmethod
    def update_kv(
        self,
        layer_id: int,
        k_update: torch.Tensor,
        v_update: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """
        Update specific positions in GPU KV cache.

        This method is used during blending to replace selected tokens
        in the GPU cache with newly computed values.

        Args:
            layer_id: Layer index
            k_update: New K values to write
            v_update: New V values to write
            indices: Token indices to update

        Example:
            >>> # During blending, update important tokens
            >>> provider.update_kv(
            ...     layer_id=0,
            ...     k_update=new_k[selected_indices],
            ...     v_update=new_v[selected_indices],
            ...     indices=selected_indices,
            ... )
        """
        pass

    @abstractmethod
    def get_kv_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of KV cache tensors.

        Returns:
            Shape tuple (e.g., (max_blocks, block_size, num_heads, head_dim))
        """
        pass

    def get_device(self) -> torch.device:
        """
        Get the device where KV cache is stored.

        Returns:
            torch.device object (e.g., torch.device("cuda:0"))
        """
        # Default implementation for CUDA
        return torch.device("cuda")
