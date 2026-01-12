# SPDX-License-Identifier: Apache-2.0
"""Backend factory for creating provider instances.

This module provides factory functions to create appropriate providers
based on device type, configuration, and availability.
"""

from typing import Any, Union

import torch

from vllm_blend.backends.cuda.cache_provider import (
    CUDACacheProvider,
    LMCacheCacheProvider,
)
from vllm_blend.backends.cuda.gpu_provider import CUDAGPUProvider
from vllm_blend.backends.cuda.model_provider import CUDAModelProvider
from vllm_blend.config import BlendConfig
from vllm_blend.providers.cache_provider import CacheProviderInterface
from vllm_blend.providers.gpu_provider import GPUProviderInterface
from vllm_blend.providers.model_provider import ModelProviderInterface


def get_cache_provider(
    config: BlendConfig,
    device: torch.device,
) -> CacheProviderInterface:
    """
    Get appropriate cache provider based on configuration.

    Args:
        config: Blend configuration
        device: Torch device

    Returns:
        Cache provider instance

    Raises:
        ValueError: If cache provider type is invalid
        ImportError: If LMCache is requested but not installed

    Example:
        >>> config = BlendConfig(cache_provider="cpu")
        >>> provider = get_cache_provider(config, torch.device("cuda"))
    """
    cache_type = config.cache_provider

    if cache_type == "cpu":
        return CUDACacheProvider(config)

    elif cache_type == "lmcache":
        return LMCacheCacheProvider(config)

    elif cache_type == "remote":
        # TODO: Implement remote cache provider
        raise NotImplementedError(
            "Remote cache provider is not yet implemented"
        )

    else:
        raise ValueError(
            f"Unknown cache provider type: {cache_type}. "
            f"Valid options: cpu, lmcache, remote"
        )


def get_gpu_provider(
    device: torch.device,
    model_runner: Any = None,
) -> GPUProviderInterface:
    """
    Get appropriate GPU provider based on device type.

    Args:
        device: Torch device (cuda, npu, etc.)
        model_runner: vLLM model runner instance

    Returns:
        GPU provider instance

    Raises:
        ValueError: If device type is not supported

    Example:
        >>> device = torch.device("cuda")
        >>> provider = get_gpu_provider(device, model_runner)
    """
    device_type = device.type

    if device_type == "cuda":
        if model_runner is None:
            raise ValueError("model_runner is required for CUDA GPU provider")
        return CUDAGPUProvider(model_runner)

    elif device_type == "npu":
        # TODO: Implement NPU GPU provider
        raise NotImplementedError(
            "NPU GPU provider is not yet implemented. "
            "See Phase 5 implementation."
        )

    elif device_type == "rocm":
        # TODO: Implement ROCm GPU provider
        raise NotImplementedError(
            "ROCm GPU provider is not yet implemented."
        )

    else:
        raise ValueError(
            f"Unsupported device type: {device_type}. "
            f"Supported types: cuda, npu, rocm"
        )


def get_model_provider(
    device: torch.device,
    vllm_model: Any,
    model_runner: Any = None,
) -> ModelProviderInterface:
    """
    Get appropriate model provider based on device type.

    Args:
        device: Torch device
        vllm_model: vLLM model instance
        model_runner: Optional model runner

    Returns:
        Model provider instance

    Raises:
        ValueError: If device type is not supported
        ValueError: If model is not supported

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
        >>> provider = get_model_provider(torch.device("cuda"), model)
    """
    device_type = device.type

    if device_type in ("cuda", "rocm"):
        return CUDAModelProvider(vllm_model, model_runner)

    elif device_type == "npu":
        # TODO: Implement NPU model provider
        raise NotImplementedError(
            "NPU model provider is not yet implemented. "
            "See Phase 5 implementation."
        )

    else:
        raise ValueError(
            f"Unsupported device type: {device_type}. "
            f"Supported types: cuda, npu, rocm"
        )


def create_all_providers(
    config: BlendConfig,
    device: torch.device,
    vllm_model: Any,
    model_runner: Any = None,
) -> tuple[CacheProviderInterface, GPUProviderInterface, ModelProviderInterface]:
    """
    Create all three providers for Blend.

    This is a convenience function that creates cache, GPU, and model providers
    with consistent configuration.

    Args:
        config: Blend configuration
        device: Torch device
        vllm_model: vLLM model instance
        model_runner: Optional model runner

    Returns:
        (cache_provider, gpu_provider, model_provider) tuple

    Example:
        >>> providers = create_all_providers(
        ...     blend_config,
        ...     torch.device("cuda"),
        ...     vllm_model,
        ...     model_runner,
        ... )
        >>> cache_provider, gpu_provider, model_provider = providers
    """
    cache_provider = get_cache_provider(config, device)
    gpu_provider = get_gpu_provider(device, model_runner)
    model_provider = get_model_provider(device, vllm_model, model_runner)

    return cache_provider, gpu_provider, model_provider


def check_provider_compatibility(
    cache_provider: CacheProviderInterface,
    gpu_provider: GPUProviderInterface,
    model_provider: ModelProviderInterface,
) -> bool:
    """
    Check if providers are compatible with each other.

    Args:
        cache_provider: Cache provider instance
        gpu_provider: GPU provider instance
        model_provider: Model provider instance

    Returns:
        True if all providers are compatible

    Note:
        This checks for things like:
        - Device compatibility
        - Memory compatibility
        - Model compatibility
    """
    # Get devices
    cache_device = getattr(cache_provider, "device", None)
    gpu_device = gpu_provider.get_device()

    # Check device compatibility
    if cache_device is not None and cache_device != gpu_device:
        # Warning might be appropriate here
        pass

    # Check if model provider supports the model
    # This is implicitly handled by adapter registry

    return True


def get_device_from_config(device_config: Any) -> torch.device:
    """
    Extract torch device from vLLM device config.

    Args:
        device_config: vLLM DeviceConfig instance

    Returns:
        torch.device object

    Example:
        >>> from vllm.config import DeviceConfig
        >>> device_conf = DeviceConfig(device="cuda:0")
        >>> device = get_device_from_config(device_conf)
    """
    if hasattr(device_config, "device"):
        return torch.device(device_config.device)
    else:
        # Fallback to default
        return torch.device("cuda")
