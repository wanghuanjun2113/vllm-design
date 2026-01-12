# SPDX-License-Identifier: Apache-2.0
"""vLLM Blend Plugin.

A plugin for vLLM that enables KV cache blending for efficient inference
in RAG, multi-document QA, and other scenarios with repeated text segments.
"""

from vllm_blend.config import BlendConfig, create_default_config

__version__ = "0.1.0"

__all__ = [
    "BlendConfig",
    "create_default_config",
]


def register():
    """Register the Blend platform plugin."""
    return "vllm_blend.platform.BlendPlatform"


def register_worker():
    """Register Blend worker customization."""
    # This would install BlendWorker patches
    pass


def register_config():
    """Register Blend configuration."""
    from vllm_blend.config import register_blend_config
    register_blend_config()
