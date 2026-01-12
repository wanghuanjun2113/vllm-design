# SPDX-License-Identifier: Apache-2.0
"""Core blending logic.

This module implements the main BlendBlender class that orchestrates
KV cache blending using the provider interfaces.
"""

from typing import Optional, Tuple

import torch

from vllm_blend.core.metadata import BlendCommonMetadata, BlendMetadata
from vllm_blend.core.selector import TokenSelector
from vllm_blend.providers.cache_provider import CacheProviderInterface
from vllm_blend.providers.gpu_provider import GPUProviderInterface
from vllm_blend.providers.model_provider import ModelProviderInterface


class BlendBlender:
    """
    Core blending logic, fully decoupled from specific implementations.

    This class orchestrates the KV cache blending process by coordinating
    cache, GPU, and model providers to selectively recompute and blend tokens.

    The blending process:
    1. Retrieve cached KV from cache provider
    2. Compare with GPU KV to identify differences
    3. Select important tokens for recomputation
    4. Update GPU cache with blended results

    Example:
        >>> blender = BlendBlender(
        ...     cache_provider=LMCacheProvider(...),
        ...     gpu_provider=CUDAGPUProvider(...),
        ...     model_provider=LlamaAdapter(...),
        ...     common_metadata=BlendCommonMetadata(
        ...         check_layers=[0],
        ...         recomp_ratios=[0.15]
        ...     ),
        ... )
        >>>
        >>> q, k, v, residual = blender.process_qkv(
        ...     q=torch.randn(100, 32, 128),
        ...     k=torch.randn(100, 32, 128),
        ...     v=torch.randn(100, 32, 128),
        ...     residual=torch.randn(100, 4096),
        ...     layer_id=0,
        ... )
    """

    def __init__(
        self,
        cache_provider: CacheProviderInterface,
        gpu_provider: GPUProviderInterface,
        model_provider: ModelProviderInterface,
        common_metadata: BlendCommonMetadata,
    ):
        """
        Initialize the blender with required providers.

        Args:
            cache_provider: KV cache storage provider
            gpu_provider: GPU KV cache access provider
            model_provider: Model computation provider
            common_metadata: Fixed blending hyperparameters
        """
        self.cache_provider = cache_provider
        self.gpu_provider = gpu_provider
        self.model_provider = model_provider
        self.common_metadata = common_metadata

        # Runtime metadata (reset for each blending operation)
        self.metadata = BlendMetadata()

        # Token selector
        self.selector = TokenSelector(common_metadata)

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process QKV tensors with blending.

        This is called during model execution for each layer. It attempts to
        blend cached KV with newly computed KV, selectively recomputing tokens
        that differ significantly.

        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            residual: Residual connection [seq_len, hidden_dim]
            layer_id: Layer index

        Returns:
            (q, k, v, residual) tuple, possibly with only selected tokens
            if blending was performed at this layer
        """
        # Get positions from metadata (used for RoPE)
        if self.metadata.positions is None:
            # Default positions if not set
            self.metadata.positions = torch.arange(
                q.shape[0],
                device=q.device,
                dtype=torch.int64,
            )

        # Try to retrieve cached KV
        cached_k, cached_v = self.cache_provider.retrieve_layer(
            tokens=self.metadata.positions if self.metadata.positions is not None else None,
            layer_id=layer_id,
            metadata={},
        )

        if cached_k is None:
            # Cache miss - return tensors as-is
            return q, k, v, residual

        # Get GPU KV for comparison
        gpu_k, gpu_v = self.gpu_provider.get_kv(layer_id)

        # Apply rotary embedding if positions are available
        if self.metadata.positions is not None:
            q, k = self.model_provider.apply_rotary_emb(
                q, k, self.metadata.positions, layer_id
            )

        # Check if this layer needs blending
        if layer_id in self.common_metadata.check_layers:
            # Get recompute ratio for this layer
            ratio = self.selector.get_recompute_ratio_for_layer(layer_id)
            if ratio is None:
                # Use first ratio as default
                ratio = self.common_metadata.recomp_ratios[0]

            # Select important tokens for recomputation
            imp_indices = self.selector.select_important_tokens(
                new_k=k,
                old_k=gpu_k,
                ratio=ratio,
            )

            # Update GPU cache with selected tokens
            self.gpu_provider.update_kv(
                layer_id=layer_id,
                k_update=k[imp_indices],
                v_update=v[imp_indices],
                indices=imp_indices,
            )

            # Update metadata
            self.metadata.imp_indices = imp_indices
            self.metadata.positions = self.metadata.positions[imp_indices]

            # Return only selected tokens
            q = q[imp_indices]
            residual = residual[imp_indices]

            # Return blended KV from GPU after update
            gpu_k, gpu_v = self.gpu_provider.get_kv(layer_id)
            return q, gpu_k, gpu_v, residual

        # No blending for this layer - return as-is
        return q, k, v, residual

    def reset_metadata(self):
        """Reset runtime metadata for a new blending operation."""
        self.metadata.clean()

    def is_active(self) -> bool:
        """Check if blending is currently active."""
        return self.metadata.is_active()

    def get_selected_indices(self) -> Optional[torch.Tensor]:
        """Get the indices of tokens selected for recomputation."""
        return self.metadata.imp_indices

    def blend(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layer-wise blending for a sequence of tokens.

        This is a high-level method that coordinates the entire blending
        process across all layers.

        Args:
            tokens: Input token sequence
            mask: Optional attention mask
            **kwargs: Additional arguments passed to providers
        """
        # Reset metadata
        self.reset_metadata()

        # Initialize positions
        self.metadata.positions = torch.arange(
            len(tokens), device=tokens.device, dtype=torch.int64
        )

        # Layer-wise blending would be handled by the worker
        # This method is a placeholder for future enhancement
        pass
