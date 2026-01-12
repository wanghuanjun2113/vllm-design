# SPDX-License-Identifier: Apache-2.0
"""Blend metadata definitions.

This module defines the metadata structures used for KV cache blending.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class BlendCommonMetadata:
    """
    Common metadata (fixed hyperparameters) for blending operations.

    Attributes:
        check_layers: List of layer indices to perform blending checks.
        recomp_ratios: Fraction of tokens to recompute for each checked layer.
        thresholds: Optional thresholds for decision-based blending (not yet implemented).
    """

    check_layers: List[int]
    recomp_ratios: Optional[List[float]] = None
    thresholds: Optional[List[float]] = None

    def __post_init__(self):
        """Validate the metadata."""
        if not self.check_layers:
            raise ValueError("check_layers cannot be empty")

        if self.recomp_ratios is not None:
            if len(self.recomp_ratios) != len(self.check_layers):
                raise ValueError(
                    f"recomp_ratios length ({len(self.recomp_ratios)}) must match "
                    f"check_layers length ({len(self.check_layers)})"
                )

            for i, ratio in enumerate(self.recomp_ratios):
                if not 0.0 <= ratio <= 1.0:
                    raise ValueError(
                        f"recomp_ratios[{i}] must be in [0, 1], got {ratio}"
                    )


@dataclass
class BlendMetadata:
    """
    Runtime metadata (determined during execution) for blending operations.

    Attributes:
        imp_indices: Indices of important tokens selected for recomputation.
        attn_mask: Attention mask for the current blending operation.
        positions: Position indices for rotary embedding.
    """

    imp_indices: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None

    def clean(self):
        """Reset all runtime metadata."""
        self.imp_indices = None
        self.attn_mask = None
        self.positions = None

    def is_active(self) -> bool:
        """Check if blending is currently active."""
        return self.imp_indices is not None
