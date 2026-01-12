# SPDX-License-Identifier: Apache-2.0
"""Provider interfaces for Blend plugin.

This package contains abstract interfaces for:
- Cache providers (KV cache storage)
- GPU providers (GPU KV access)
- Model providers (model computation)
"""

from vllm_blend.providers.cache_provider import CacheProviderInterface
from vllm_blend.providers.gpu_provider import GPUProviderInterface
from vllm_blend.providers.model_provider import ModelProviderInterface

__all__ = [
    "CacheProviderInterface",
    "GPUProviderInterface",
    "ModelProviderInterface",
]
