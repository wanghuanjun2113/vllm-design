# SPDX-License-Identifier: Apache-2.0
"""Blend configuration system.

This module defines the configuration class for Blend functionality.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vllm.config import VllmConfig
from vllm_blend.core.metadata import BlendCommonMetadata


@dataclass
class BlendConfig:
    """
    Configuration for Blend functionality.

    This class contains all configuration parameters for the Blend plugin,
    including layer selection, recompute ratios, and cache backend settings.

    Attributes:
        enabled: Whether Blend functionality is enabled
        check_layers: Layer indices to perform blending checks
        recompute_ratios: Fraction of tokens to recompute for each checked layer
        thresholds: Optional thresholds for decision-based blending
        cache_provider: Cache backend to use ("cpu", "lmcache", "remote")
        cache_config: Additional configuration for the cache provider
    """

    # Enable/disable Blend
    enabled: bool = False

    # Layers to check for blending
    check_layers: List[int] = field(default_factory=lambda: [0, 16, 32])

    # Fraction of tokens to recompute (0.0-1.0)
    recompute_ratios: List[float] = field(default_factory=lambda: [0.1])

    # Thresholds for decision-based blending (optional)
    thresholds: Optional[List[float]] = None

    # Cache provider configuration
    cache_provider: str = "cpu"  # Options: cpu, lmcache, remote

    # Additional cache configuration
    cache_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.enabled:
            if not self.check_layers:
                raise ValueError("check_layers cannot be empty when Blend is enabled")

            if not self.recompute_ratios:
                raise ValueError("recompute_ratios cannot be empty when Blend is enabled")

            # Validate recompute ratios
            for i, ratio in enumerate(self.recompute_ratios):
                if not 0.0 <= ratio <= 1.0:
                    raise ValueError(
                        f"recompute_ratios[{i}] must be in [0, 1], got {ratio}"
                    )

            # Validate lengths match
            if len(self.recompute_ratios) != len(self.check_layers):
                if len(self.recompute_ratios) == 1:
                    # Broadcast single ratio to all layers
                    self.recompute_ratios = self.recompute_ratios * len(self.check_layers)
                else:
                    raise ValueError(
                        f"recompute_ratios length ({len(self.recompute_ratios)}) "
                        f"must match check_layers length ({len(self.check_layers)}) "
                        "or be a single value to broadcast"
                    )

            # Validate cache provider
            valid_providers = ["cpu", "lmcache", "remote"]
            if self.cache_provider not in valid_providers:
                raise ValueError(
                    f"cache_provider must be one of {valid_providers}, "
                    f"got {self.cache_provider}"
                )

    @property
    def common_metadata(self) -> BlendCommonMetadata:
        """Convert to BlendCommonMetadata for use by Blender."""
        return BlendCommonMetadata(
            check_layers=self.check_layers,
            recomp_ratios=self.recompute_ratios,
            thresholds=self.thresholds,
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "BlendConfig":
        """
        Create BlendConfig from VllmConfig.

        This method extracts Blend configuration from vLLM's additional_config,
        which can be populated via command-line arguments or Python API.

        Args:
            vllm_config: vLLM configuration object

        Returns:
            BlendConfig instance

        Example:
            >>> vllm_config = VllmConfig(
            ...     model="meta-llama/Llama-2-7b-chat-hf",
            ...     additional_config={
            ...         "blend_config": {
            ...             "enabled": True,
            ...             "check_layers": [0, 16],
            ...             "recompute_ratios": [0.15],
            ...         }
            ...     }
            ... )
            >>> blend_config = BlendConfig.from_vllm_config(vllm_config)
        """
        additional_config = vllm_config.additional_config or {}
        blend_config_dict = additional_config.get("blend_config", {})

        # If no config provided, return default (disabled)
        if not blend_config_dict:
            return cls()

        return cls(**blend_config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BlendConfig":
        """
        Create BlendConfig from a dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            BlendConfig instance

        Example:
            >>> config = BlendConfig.from_dict({
            ...     "enabled": True,
            ...     "check_layers": [0, 16, 32],
            ...     "recompute_ratios": [0.1, 0.15, 0.1],
            ... })
        """
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "enabled": self.enabled,
            "check_layers": self.check_layers,
            "recompute_ratios": self.recompute_ratios,
            "thresholds": self.thresholds,
            "cache_provider": self.cache_provider,
            "cache_config": self.cache_config,
        }

    def __repr__(self) -> str:
        """String representation of the configuration."""
        if not self.enabled:
            return "BlendConfig(disabled)"

        return (
            f"BlendConfig(enabled=True, "
            f"check_layers={self.check_layers}, "
            f"recompute_ratios={self.recompute_ratios}, "
            f"cache_provider='{self.cache_provider}')"
        )


def register_blend_config():
    """
    Register Blend configuration with vLLM's argument parser.

    This function is called during plugin initialization to add Blend-specific
    command-line arguments to vLLM.

    Example:
        After registration, users can use:
        ```bash
        vllm serve model_name --enable-blend --blend-check-layers 0 16 32
        ```
    """
    # This would integrate with vLLM's argument system
    # For now, it's a placeholder showing how registration would work
    pass


def create_default_config() -> BlendConfig:
    """
    Create a default BlendConfig with common settings.

    Returns:
        BlendConfig with sensible defaults for typical use cases

    Example:
        >>> config = create_default_config()
        >>> config.enabled = True
    """
    return BlendConfig(
        enabled=False,
        check_layers=[0, 16, 32],
        recompute_ratios=[0.1],
        thresholds=None,
        cache_provider="cpu",
        cache_config={},
    )
