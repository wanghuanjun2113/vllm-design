# SPDX-License-Identifier: Apache-2.0
"""Model adapter registry.

This module provides a registry for model adapters, allowing dynamic
lookup of the appropriate adapter for a given vLLM model.
"""

from typing import Dict, Type

from vllm_blend.adapters.base import BaseModelAdapter


class ModelAdapterRegistry:
    """
    Registry for model adapters.

    This class maintains a mapping from model type names to adapter classes,
    allowing automatic adapter selection based on the vLLM model type.

    Example:
        >>> # Register adapters
        >>> ModelAdapterRegistry.register("llama", LlamaAdapter)
        >>> ModelAdapterRegistry.register("qwen2", Qwen2Adapter)
        >>>
        >>> # Get adapter for a model
        >>> model = LlamaForCausalLM(...)
        >>> adapter = ModelAdapterRegistry.get_adapter(model)
    """

    # Class-level registry
    _adapters: Dict[str, Type[BaseModelAdapter]] = {}

    # Model type to adapter name mapping
    _model_type_mapping: Dict[str, str] = {
        "LlamaForCausalLM": "llama",
        "LlamaForCausalLMWithValueHead": "llama",
        "Qwen2ForCausalLM": "llama",  # Qwen2 uses Llama architecture
        "Qwen2MoeForCausalLM": "llama",
        "Qwen2ForRewardModel": "llama",
        "Qwen2ForSequenceClassification": "llama",
        "MistralForCausalLM": "llama",  # Mistral uses Llama architecture
        "MixtralForCausalLM": "llama",
        "Phi3ForCausalLM": "llama",
        # Add more mappings as needed
    }

    @classmethod
    def register(cls, adapter_name: str, adapter_cls: Type[BaseModelAdapter]):
        """
        Register a model adapter.

        Args:
            adapter_name: Name for the adapter (e.g., "llama", "qwen2")
            adapter_cls: Adapter class (subclass of BaseModelAdapter)

        Example:
            >>> class MyModelAdapter(BaseModelAdapter):
            ...     # Implementation
            ...
            >>> ModelAdapterRegistry.register("mymodel", MyModelAdapter)
        """
        if not issubclass(adapter_cls, BaseModelAdapter):
            raise TypeError(
                f"Adapter must be a subclass of BaseModelAdapter, "
                f"got {type(adapter_cls)}"
            )

        cls._adapters[adapter_name] = adapter_cls

    @classmethod
    def get_adapter(cls, vllm_model) -> BaseModelAdapter:
        """
        Get the appropriate adapter for a vLLM model.

        Args:
            vllm_model: vLLM model instance

        Returns:
            Model adapter instance

        Raises:
            ValueError: If no adapter is registered for this model type

        Example:
            >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
            >>> adapter = ModelAdapterRegistry.get_adapter(model)
        """
        # Get model type
        model_type = type(vllm_model).__name__

        # Look up adapter name
        adapter_name = cls._model_type_mapping.get(model_type)

        if adapter_name is None:
            # Try direct lookup
            adapter_name = model_type.replace("ForCausalLM", "").lower()

        # Get adapter class
        adapter_cls = cls._adapters.get(adapter_name)

        if adapter_cls is None:
            raise ValueError(
                f"No adapter registered for model type '{model_type}'. "
                f"Available adapters: {list(cls._adapters.keys())}. "
                f"Please implement and register an adapter for this model."
            )

        # Create and return adapter instance
        return adapter_cls(vllm_model)

    @classmethod
    def list_adapters(cls) -> list[str]:
        """
        List all registered adapters.

        Returns:
            List of adapter names
        """
        return list(cls._adapters.keys())

    @classmethod
    def list_supported_models(cls) -> list[str]:
        """
        List all supported vLLM model types.

        Returns:
            List of model class names
        """
        return [
            model_type
            for model_type, adapter_name in cls._model_type_mapping.items()
            if adapter_name in cls._adapters
        ]

    @classmethod
    def is_model_supported(cls, vllm_model) -> bool:
        """
        Check if a model is supported.

        Args:
            vllm_model: vLLM model instance

        Returns:
            True if adapter is available for this model
        """
        try:
            cls.get_adapter(vllm_model)
            return True
        except ValueError:
            return False


# Convenience function for registration
def register_adapter(adapter_name: str):
    """
    Decorator for registering model adapters.

    Example:
        >>> @register_adapter("llama")
        >>> class LlamaAdapter(BaseModelAdapter):
        ...     pass
    """
    def decorator(adapter_cls: Type[BaseModelAdapter]):
        ModelAdapterRegistry.register(adapter_name, adapter_cls)
        return adapter_cls

    return decorator
