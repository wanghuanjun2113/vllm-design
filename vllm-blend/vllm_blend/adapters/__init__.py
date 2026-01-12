# SPDX-License-Identifier: Apache-2.0
"""Model adapters for Blend plugin.

This package contains adapters for different model architectures.
"""

from vllm_blend.adapters.base import BaseModelAdapter
from vllm_blend.adapters.registry import ModelAdapterRegistry, register_adapter

# Import specific adapters to trigger registration
from vllm_blend.adapters import llama  # noqa: F401

__all__ = [
    "BaseModelAdapter",
    "ModelAdapterRegistry",
    "register_adapter",
]
