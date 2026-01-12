# SPDX-License-Identifier: Apache-2.0
"""Token selection strategies for blending.

This module implements algorithms to select important tokens for recomputation
during KV cache blending.
"""

from typing import Optional

import torch
from vllm_blend.core.metadata import BlendCommonMetadata


class TokenSelector:
    """
    Select important tokens for recomputation based on various strategies.

    The default strategy uses L2 distance between new and cached K tensors
    to identify tokens that have changed the most and need recomputation.
    """

    def __init__(self, common_metadata: BlendCommonMetadata):
        """
        Initialize the token selector.

        Args:
            common_metadata: Fixed hyperparameters for blending.
        """
        self.common_metadata = common_metadata

    def select_important_tokens(
        self,
        new_k: torch.Tensor,
        old_k: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """
        Select top-K tokens with the largest KV differences.

        This method computes the L2 distance between new and old K tensors
        (averaged over attention heads and head dimensions), then selects
        the tokens with the largest distances.

        Args:
            new_k: Newly computed K tensor with shape [seq_len, num_heads, head_dim]
            old_k: Cached K tensor with shape [seq_len, num_heads, head_dim]
            ratio: Fraction of tokens to select (0.0-1.0)

        Returns:
            indices: Sorted tensor of important token indices

        Example:
            >>> selector = TokenSelector(BlendCommonMetadata(check_layers=[0], recomp_ratios=[0.1]))
            >>> new_k = torch.randn(100, 32, 128)
            >>> old_k = torch.randn(100, 32, 128)
            >>> indices = selector.select_important_tokens(new_k, old_k, ratio=0.15)
            >>> assert len(indices) == 15  # 15% of 100 tokens
        """
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        if new_k.shape != old_k.shape:
            raise ValueError(
                f"Shape mismatch: new_k {new_k.shape} vs old_k {old_k.shape}"
            )

        # Compute L2 distance between new and old K
        # Average over heads and head_dim dimensions
        diff_k = torch.sum(
            (new_k.to(torch.float32) - old_k.to(torch.float32)) ** 2,
            dim=[1, 2],
        )

        total_len = diff_k.shape[0]
        topk_num = max(int(total_len * ratio), 1)

        # Get top-k indices (tokens with largest differences)
        top_indices = torch.topk(diff_k, k=topk_num).indices
        top_indices, _ = torch.sort(top_indices)

        return top_indices

    def select_by_threshold(
        self,
        new_k: torch.Tensor,
        old_k: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """
        Select tokens with KV difference above a threshold.

        This is an alternative strategy that selects all tokens where
        the difference exceeds a threshold value.

        Args:
            new_k: Newly computed K tensor
            old_k: Cached K tensor
            threshold: Minimum difference threshold

        Returns:
            indices: Sorted tensor of token indices above threshold
        """
        # Compute L2 distance
        diff_k = torch.sum(
            (new_k.to(torch.float32) - old_k.to(torch.float32)) ** 2,
            dim=[1, 2],
        )

        # Select tokens above threshold
        mask = diff_k > threshold
        indices = torch.nonzero(mask, as_tuple=True)[0]

        return indices

    def get_recompute_ratio_for_layer(self, layer_id: int) -> Optional[float]:
        """
        Get the recompute ratio for a specific layer.

        Args:
            layer_id: Layer index

        Returns:
            Recompute ratio for this layer, or None if layer not in check_layers
        """
        if self.common_metadata.recomp_ratios is None:
            return None

        try:
            idx = self.common_metadata.check_layers.index(layer_id)
            return self.common_metadata.recomp_ratios[idx]
        except ValueError:
            return None
