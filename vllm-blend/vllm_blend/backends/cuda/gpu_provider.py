# SPDX-License-Identifier: Apache-2.0
"""CUDA GPU provider implementation.

This module provides CUDA-specific implementation for GPU KV cache access.
"""

from typing import Tuple

import torch

from vllm_blend.providers.gpu_provider import GPUProviderInterface


class CUDAGPUProvider(GPUProviderInterface):
    """
    CUDA GPU KV cache provider.

    This provider accesses KV cache stored in GPU memory through vLLM's
    KV connector interface.

    Example:
        >>> from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        >>> provider = CUDAGPUProvider(model_runner)
        >>> k_gpu, v_gpu = provider.get_kv(layer_id=0)
    """

    def __init__(self, model_runner):
        """
        Initialize the CUDA GPU provider.

        Args:
            model_runner: vLLM ModelRunner instance.
                        Must implement KVConnectorModelRunnerMixin or
                        provide get_kv/set_kv methods.
        """
        self.model_runner = model_runner
        self.device = torch.device("cuda")

        # Try to verify KV connector is available
        if not hasattr(model_runner, "get_kv_from_connector"):
            # If not using KV connector, we might need alternative access
            # For now, we'll store a reference and handle it in get_kv
            self.uses_kv_connector = False
        else:
            self.uses_kv_connector = True

    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV tensors from GPU memory for a specific layer.

        Args:
            layer_id: Layer index

        Returns:
            (k_gpu, v_gpu) tuple from GPU memory

        Note:
            This method attempts to use the KV connector if available.
            If not, it will raise NotImplementedError.
        """
        if self.uses_kv_connector:
            # Use vLLM's KV connector interface
            try:
                k_gpu, v_gpu = self.model_runner.get_kv_from_connector(layer_id)
                return k_gpu, v_gpu
            except AttributeError as e:
                raise RuntimeError(
                    f"Model runner does not support KV connector: {e}"
                )
        else:
            # Alternative: Direct KV cache access
            # This would need to be implemented based on how the model
            # runner stores KV cache
            raise NotImplementedError(
                "Direct KV access without KV connector is not yet implemented. "
                "Please use a model runner with KV connector support."
            )

    def update_kv(
        self,
        layer_id: int,
        k_update: torch.Tensor,
        v_update: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """
        Update specific positions in GPU KV cache.

        Args:
            layer_id: Layer index
            k_update: New K values [n_selected, num_kv_heads, head_dim]
            v_update: New V values [n_selected, num_kv_heads, head_dim]
            indices: Token positions to update [n_selected]

        Note:
            This performs an in-place update on GPU memory.
            The operation is synchronous but typically fast for small selections.
        """
        k_gpu, v_gpu = self.get_kv(layer_id)

        # Validate shapes
        if len(indices) != k_update.shape[0]:
            raise ValueError(
                f"Length mismatch: indices ({len(indices)}) vs "
                f"k_update ({k_update.shape[0]})"
            )

        # Perform in-place update
        # This is efficient for small selections (~15% of tokens)
        try:
            k_gpu[indices] = k_update
            v_gpu[indices] = v_update
        except IndexError as e:
            raise ValueError(
                f"Index out of bounds in KV cache update: {e}. "
                f"KV shape: {k_gpu.shape}, indices: {indices}"
            )

        # Optional: Synchronize to ensure update is complete
        # torch.cuda.synchronize()  # Usually not needed for correctness

    def get_kv_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of KV cache tensors.

        Returns:
            Shape tuple, typically (max_blocks, block_size, num_heads, head_dim)
            or (max_len, num_heads, head_dim) for non-paged cache
        """
        # Try to get shape from layer 0
        try:
            k_gpu, _ = self.get_kv(0)
            return k_gpu.shape
        except Exception:
            # Fallback: return a default shape
            # This would need to be configured based on model
            return (0, 0, 0, 0)  # Placeholder

    def get_device(self) -> torch.device:
        """Get the CUDA device."""
        return self.device

    def supports_indexed_update(self) -> bool:
        """Check if indexed updates are supported."""
        return True

    def get_memory_usage(self) -> dict:
        """
        Get GPU memory usage statistics.

        Returns:
            Dictionary with memory stats in bytes
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            return {
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "allocated_gb": allocated / 1024**3,
                "reserved_gb": reserved / 1024**3,
            }
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
        }
