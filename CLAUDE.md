# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This workspace contains three related projects:

1. **vLLM** (`vllm/`) - A high-throughput and memory-efficient inference engine for Large Language Models (LLMs). Written primarily in Python with C++/CUDA kernels for GPU acceleration. Key innovations include **PagedAttention** (efficient KV cache management) and **continuous batching** for optimal throughput.

2. **vLLM-Ascend** (`vllm-ascend/`) - A hardware plugin that extends vLLM to run on Huawei Ascend NPUs. This is **NOT a fork** but a plugin that integrates with vLLM through the platform plugin interface (RFC #11162), maintaining full API compatibility with the main vLLM project.

3. **LMCache** (`LMCache/`) - A high-performance KV cache management system that extends vLLM with multi-tier KV cache storage (GPU/CPU/Disk/Remote). Reduces TTFT by 3-10x through intelligent cache reuse, cross-instance sharing, and disaggregated prefill-decode. Integrates with vLLM v1 via the KV connector interface.


## Code Architecture

### v1 Engine (Current Default)

The main engine has migrated to `vllm/v1/`. The legacy `LLMEngine` class in `vllm/engine/llm_engine.py` now redirects to `vllm/v1/engine/llm_engine.py`.

**Key v1 Components:**
- `vllm/v1/engine/llm_engine.py`: Main LLMEngine implementation
- `vllm/v1/engine/core_client.py`: Client interface to engine core
- `vllm/v1/engine/input_processor.py`: Processes input prompts
- `vllm/v1/engine/output_processor.py`: Converts engine outputs to API responses
- `vllm/v1/core/`: Core scheduling and execution logic
- `vllm/v1/worker/`: Worker process implementations
- `vllm/v1/executor/`: Executor backends (gpu, ray, external_launcher)
- `vllm/v1/attention/`: Attention mechanism implementations
- `vllm/v1/sample/`: Sampling logic

### Core Entry Points

1. **Python API** (`vllm/entrypoints/llm.py`):
   ```python
   from vllm import LLM, SamplingParams
   llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
   outputs = llm.generate("Hello, world!", SamplingParams(temperature=0.8))
   ```

2. **CLI** (`vllm` command):
   ```bash
   vllm serve <model_name> --port 8000
   ```

3. **OpenAI-Compatible API** (`vllm/entrypoints/openai/api_server.py`):
   ```bash
   python -m vllm.entrypoints.openai.api_server --model <model_name>
   ```

### Key Directories (Main vLLM)

| Directory | Purpose |
|-----------|---------|
| `vllm/config/` | Configuration system (`VllmConfig` is central) |
| `vllm/model_executor/` | Model loading and execution (300+ model implementations) |
| `vllm/model_executor/models/` | Per-model implementations (LLaMA, Mixtral, Qwen, etc.) |
| `vllm/model_executor/layers/` | Neural network layer implementations |
| `vllm/attention/` | Attention mechanisms (PagedAttention, FlashAttention, FlashInfer) |
| `vllm/distributed/` | Distributed computing (tensor, pipeline, data parallelism) |
| `vllm/entrypoints/` | API servers (OpenAI, Anthropic, gRPC) |
| `vllm/samplers/` | Sampling strategies |
| `vllm/lora/` | LoRA (Low-Rank Adaptation) support |
| `vllm/multimodal/` | Multi-modal model support |
| `vllm/tokenizers/` | Tokenization utilities |
| `vllm/transformers_utils/` | HuggingFace transformers integration |
| `vllm/v1/` | Next-generation engine (current default) |
| `vllm/plugins/` | Plugin system for extensibility |
| `csrc/` | C++/CUDA kernels |
| `cmake/` | CMake build files |

### Configuration System

The `VllmConfig` class (`vllm/config/`) is the central configuration object that contains:
- `ModelConfig`: Model-specific settings
- `CacheConfig`: KV cache configuration
- `ParallelConfig`: Distributed execution settings
- `SchedulerConfig`: Request scheduling
- `DeviceConfig`: Hardware device configuration
- `ObservabilityConfig`: Metrics and tracing

### Model Support

Models are registered in `vllm/model_executor/models/` via the `ModelRegistry`. To add a new model:
1. Create a model file in `vllm/model_executor/models/`
2. Register it using the decorator system
3. The model will be automatically discovered and loaded

## vLLM-Ascend Plugin Architecture

The vLLM-Ascend plugin (`vllm-ascend/`) extends vLLM to support Huawei Ascend NPUs through a clean plugin architecture. It registers with vLLM via entry points defined in `vllm-ascend/setup.py`:

```python
entry_points={
    "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
    "vllm.general_plugins": [
        "ascend_kv_connector = vllm_ascend:register_connector",
        "ascend_model_loader = vllm_ascend:register_model_loader",
        "ascend_service_profiling = vllm_ascend:register_service_profiling"
    ],
}
```

### Key Plugin Components

| Component | Path | Purpose |
|-----------|------|---------|
| **Platform** | `vllm_ascend/platform.py` | `NPUPlatform` class extending vLLM's `Platform` for Ascend hardware |
| **Configuration** | `vllm_ascend/ascend_config.py` | Ascend-specific configuration management |
| **Attention** | `vllm_ascend/attention/` | Ascend-optimized attention (SFA, MLA, CP parallel) |
| **Compilation** | `vllm_ascend/compilation/` | ACL graph compilation and optimization passes |
| **Distributed** | `vllm_ascend/distributed/` | HCCL-based distributed training/inference |
| **MoE** | `vllm_ascend/fused_moe/` | Mixture of Experts implementations |
| **Quantization** | `vllm_ascend/quantization/` | Quantization methods including compressed tensors |
| **Kernels** | `csrc/` | C++ kernels and Python bindings for Ascend operations |

### Ascend Hardware Support

**Supported Hardware Variants:**
- Atlas 800I A2/A3 series
- Atlas A2/A3 Training series
- Atlas 300I Duo
- Chip variants: 910B, 910C, 310P

**Software Stack:**
- **CANN**: 8.3.rc2 (Ascend HDK - equivalent to CUDA for GPUs)
- **torch-npu**: 2.8.0 (PyTorch extension for Ascend NPUs)
- **HCCL**: Huawei Collective Communication Library (replaces NCCL)
- **ACL**: Ascend Computing Language (graph compilation, replaces CUDA graphs)

### Key Architectural Differences from CUDA vLLM

| Aspect | CUDA vLLM | vLLM-Ascend |
|--------|-----------|-------------|
| Communication | NCCL | HCCL |
| Graph Compilation | CUDA graphs | ACL graph compilation |
| Memory Allocator | PyTorch default | Custom `camem_allocator.cpp` |
| Attention Backend | FlashAttention/FlashInfer | Sparse Flash Attention (SFA), MLA |
| Hardware Backend | CUDA kernels | CANN kernels + custom C++ kernels |

### Ascend-Specific Optimizations

- **FlashComm**: Optimized tensor parallelism communication
- **MLAPO** (Multi-Head Latent Attention Parallel Optimization): Specialized optimization for MLA models
- **MatMulAllReduce Fusion**: Fused matrix multiplication with all-reduce for efficiency
- **Dynamic EPLB**: Expert Parallel Load Balancing for MoE models
- **Context Parallel**: Distributed attention across multiple NPUs

## LMCache Architecture

LMCache (`LMCache/`) is a KV cache management system that extends vLLM with multi-tier storage and intelligent cache reuse. Achieves 3-10x TTFT reduction through cross-request and cross-instance KV cache sharing.

### Multi-Tier Storage Architecture

LMCache implements a hierarchical storage system:
- **GPU Memory**: Active KV caches currently in use
- **CPU DRAM**: Hot cache with pinned memory for fast GPU transfers
- **Local Storage** (NVMe/GDS): Large tier for long documents
- **Remote Storage** (Redis/Mooncake/S3): Persistent cross-instance sharing

### Key Components

| Component | Path | Purpose |
|-----------|------|---------|
| **Cache Engine** | `lmcache/v1/cache_engine.py` | Main KV cache management interface |
| **Storage Manager** | `lmcache/v1/storage_backend/` | Multi-tier storage orchestration |
| **Token Database** | `lmcache/v1/token_database/` | Token-to-KV mapping index |
| **GPU Connector** | `lmcache/v1/gpu_connector/` | GPU↔CPU data transfer adapters |
| **Memory Allocator** | `lmcache/v1/memory_management/` | Custom CPU memory allocation |
| **vLLM Integration** | `lmcache/integration/vllm/` | vLLM v1 KV connector |
| **Cache Controller** | `lmcache/v1/cache_controller/` | Distributed coordination |

### vLLM Integration

LMCache integrates with vLLM v1 through the KV connector interface (`vllm/distributed/kv_transfer/kv_connector/v1/`):

**Key Integration Points:**
- `get_num_new_matched_tokens()`: Query cache for reusable KV chunks
- `start_load_kv()`: Begin async KV cache loading
- `wait_for_layer_load()`: Sync point for layer-by-layer loading
- `save_kv_layer()`: Store newly computed KV chunks
- `wait_for_save()`: Ensure async save completion

### Storage Backends

LMCache supports multiple storage backends:

| Backend | Purpose |
|---------|---------|
| **LocalCPUBackend** | Hot cache in pinned CPU memory |
| **LocalDiskBackend** | Local NVMe/disk storage with GDS support |
| **RemoteBackend** | Generic remote storage wrapper |
| **P2PBackend** | Peer-to-peer KV sharing between nodes |
| **Redis/Mooncake** | Distributed cache plugins |

### Operational Modes

**Storage Mode (KV Cache Offloading):**
- Offloads infrequently used KV blocks from GPU to CPU
- Persists popular caches across sessions
- Survives process restarts with disk/remote backends

**Transport Mode (Disaggregated Prefill-Decode):**
- Routes KV cache data between nodes in real-time
- Enables prefill-decode disaggregation (compute on one node, generate on another)
- Uses peer-to-peer channels (NIXL) for low-latency transfers

### Cache Controller API

Runtime cache management operations:
- **lookup()**: Query cache entries for token sequences
- **store()** / **retrieve()**: Basic cache operations
- **clear()**: Purge cache (global or selective)
- **compress()** / **decompress()**: On-demand compression (CacheGen)
- **move()**: Migrate caches between storage tiers
- **pin()** / **unpin()**: Prevent/allow eviction
- **health()**: Check cache system health

### Configuration

Key configuration parameters (`lmcache/v1/config.py`):
- `chunk_size`: Token chunk size (default: 256)
- `max_local_cpu_size`: CPU hot cache size in GB (default: 5.0)
- `max_local_disk_size`: Disk cache size in GB
- `local_disk`: Disk storage path
- `remote_url`: Remote storage endpoint
- `enable_blending`: Enable CacheBlend for RAG scenarios
- `enable_p2p`: Enable peer-to-peer sharing
- `enable_controller`: Enable distributed controller
- `use_layerwise`: Layer-wise KV cache processing

### Performance Optimizations

- **Asynchronous Operations**: Non-blocking store/retrieve with `enable_async_loading`
- **Zero-Copy Transfers**: GPU-direct (GDS) for disk, NIXL for RDMA
- **Intelligent Prefetching**: `async_lookup_and_prefetch()` for proactive loading
- **NUMA Awareness**: NUMA detection and local memory optimization
- **Layer-wise Processing**: Process layer-by-layer to reduce peak memory

### Use Cases

- **Multi-Round QA**: Cache conversation history across rounds (3-10x TTFT reduction)
- **RAG**: Cache document embeddings, CacheBlend for multi-document fusion
- **Long Context**: Offload long documents to CPU/disk, fetch on-demand
- **Disaggregated Inference**: Prefill nodes compute KV, decode nodes receive
- **Multi-Tenant Serving**: Cross-instance cache sharing via remote backends

### Important Notes

1. **Chunk Size**: 256 tokens balances memory overhead and cache granularity
2. **Eviction Policy**: LRU by default; configurable via `cache_policy`
3. **MLA Support**: Special handling for Multi-Head Latent Attention (only rank 0 saves by default)
4. **Freeze Mode**: Can freeze cache to prevent changes while serving from local hot cache
5. **Thread Safety**: AsyncMultiSerializer prevents race conditions in concurrent operations

## Important Implementation Notes

### Main vLLM

1. **Lazy Imports**: The root `__init__.py` uses lazy imports via `__getattr__` for performance. The `MODULE_ATTRS` dictionary maps public names to their actual locations.

2. **Torch Version Pinning**: PyTorch is pinned to exactly 2.9.1. Do not upgrade without checking compatibility.

3. **Hardware Support**: The codebase supports CUDA, ROCm, Intel CPU/GPU, AMD, TPU, and various hardware plugins. Target device is controlled by `VLLM_TARGET_DEVICE`.

4. **v1 vs Legacy**: The v1 engine is now the default. The `vllm/engine/` directory contains compatibility shims that redirect to `vllm/v1/`.

5. **Model Registry**: Models are auto-discovered through a registry pattern. New models must be registered to be loadable.

6. **Third-Party Code**: Code in `vllm/third_party/` is excluded from linting (see `.pre-commit-config.yaml` and `pyproject.toml`).

7. **SPDX Headers**: All Python files must have proper SPDX license headers (enforced by pre-commit hook).

### vLLM-Ascend Specific

1. **Communication**: Uses HCCL (Huawei Collective Communication Library) instead of NCCL for distributed operations.

2. **Graph Compilation**: Uses ACL (Ascend Computing Language) graph compilation framework instead of CUDA graphs.

3. **Memory Management**: Custom memory allocator implementation in `csrc/camem_allocator.cpp` for efficient NPU memory usage.

4. **Attention Implementations**: Ascend-optimized attention mechanisms include:
   - **SFA** (Sparse Flash Attention) v1
   - **MLA** (Multi-Head Latent Attention) v1
   - Context Parallel attention for distributed scenarios

5. **Plugin Entry Points**: The plugin registers itself through multiple entry points:
   - `vllm.platform_plugins`: Platform registration
   - `vllm.general_plugins`: KV cache connector, model loader, service profiling

6. **No Forking**: The plugin architecture avoids forking vLLM, making it easier to maintain compatibility with upstream vLLM updates.

7. **Hardware-Specific Features**: Many optimizations are specific to Ascend hardware architecture:
   - FlashComm for optimized tensor parallelism
   - MLAPO for Multi-Head Latent Attention optimization
   - Dynamic EPLB for MoE load balancing
