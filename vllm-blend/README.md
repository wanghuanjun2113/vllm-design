# vLLM Blend Plugin

A plugin for vLLM that enables KV cache blending for efficient inference in RAG, multi-document QA, and other scenarios with repeated text segments.

## Overview

Blend allows vLLM to reuse KV caches from previous requests even when text segments appear in different orders or contexts. This is particularly useful for:

- **RAG (Retrieval-Augmented Generation)**: Reuse cached document KV caches
- **Multi-document QA**: Handle documents in any order with minimal recomputation
- **Long context processing**: Process long documents efficiently
- **Conversational AI**: Reuse conversation history

## Installation

```bash
pip install vllm-blend
```

## Quick Start

### Command Line

```bash
vllm serve meta-llama/Llama-2-7b-chat-hf \
    --enable-blend \
    --blend-check-layers 0 16 32 \
    --blend-recompute-ratios 0.1
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_blend=True,
    blend_check_layers=[0, 16, 32],
    blend_recompute_ratios=[0.1],
)

outputs = llm.generate("Hello, world!", SamplingParams(max_tokens=10))
```

## Architecture

Blend is built on a provider abstraction that makes it hardware-agnostic:

- **CacheProviderInterface**: Abstract KV cache storage
- **GPUProviderInterface**: Abstract GPU KV access
- **ModelProviderInterface**: Abstract model computation

This allows Blend to work with:
- CUDA (NVIDIA GPUs)
- NPU (Ascend hardware)
- ROCm (AMD GPUs)
- Different cache backends (LMCache, CPU, remote storage)

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_blend` | bool | False | Enable Blend functionality |
| `blend_check_layers` | List[int] | [0, 16, 32] | Layers to perform blending |
| `blend_recompute_ratios` | List[float] | [0.1] | Fraction of tokens to recompute |
| `blend_cache_provider` | str | "cpu" | Cache backend (cpu, lmcache, remote) |

## How It Works

Blend intelligently reuses KV caches by:

1. **Segment Storage**: Documents are stored as independent cache segments
2. **Smart Selection**: Only tokens that change significantly are recomputed
3. **Layer-wise Blending**: Blending occurs at specific layers for efficiency

In RAG scenarios with 3 document chunks in different orders:
- Without Blend: 100% recomputation for each query
- With Blend: ~15% recomputation (only at connection points)

## Performance

Expected improvements:
- **TTFT**: 30-60% reduction in cached scenarios
- **Throughput**: 1.5-2x increase for repeated content
- **Quality**: <1% perplexity degradation

## Development

See [DESIGN.md](DESIGN.md) for detailed architecture documentation.

## License

Apache License 2.0
