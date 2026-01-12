# SPDX-License-Identifier: Apache-2.0
"""Setup script for vLLM Blend plugin."""

from setuptools import find_packages, setup

setup(
    name="vllm-blend",
    version="0.1.0",
    description="KV cache blending plugin for vLLM",
    long_description=open("README.md").read() if True else "",
    long_description_content_type="text/markdown",
    author="vLLM Blend Contributors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "vllm>=0.7.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "lmcache": [
            "lmcache>=0.1.0",
        ],
    },
    entry_points={
        "vllm.platform_plugins": [
            "blend = vllm_blend:register",
        ],
        "vllm.general_plugins": [
            "blend_worker = vllm_blend:register_worker",
            "blend_config = vllm_blend:register_config",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
