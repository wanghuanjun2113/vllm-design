# Chunk Cache Implementation Summary

This document provides a summary of the Chunk Cache feature implementation for vLLM-Ascend.

## Overview

The Chunk Cache feature has been implemented to provide position-agnostic KV cache management for vLLM-Ascend. This implementation allows users to cache and reuse KV caches for content chunks regardless of their position in the prompt.

## Files Created

### 1. Core Data Structures
**File**: `vllm/v1/engine/chunk_metadata.py`

Defines the core data structures:
- `ChunkMetadata`: Metadata for a single chunk (token IDs, hash, dependencies)
- `ChunkParseResult`: Result of parsing a prompt with chunk delimiters
- `ChunkMatchResult`: Result of matching chunks against cache
- `ChunkKVCache`: KV cache storage for a single chunk
- `CacheStats`: Statistics for the chunk cache pool

### 2. Chunk Parser
**File**: `vllm/inputs/chunk_parser.py`

Implements the `ChunkParser` class:
- Parses prompts containing `# #` delimiters
- Extracts system prompt, chunks, and user question
- Validates chunk format
- Provides helper methods for chunk position tracking

### 3. Hash Utilities
**File**: `vllm-ascend/chunk/hash_utils.py`

Provides hash computation utilities:
- `compute_chunk_hash()`: Compute position-agnostic chunk hash
- `verify_chunk_integrity()`: Verify chunk matches expected hash
- `compute_chunk_hash_string()`: Hash directly from text
- `truncate_hash()`: Truncate hash for display

### 4. Chunk Cache Pool
**File**: `vllm-ascend/chunk/cache_pool.py`

Implements `ChunkCachePool` class:
- Token-granular KV cache storage
- LRU eviction policy
- Hash-based lookup
- Configurable pool size (default 2GB)
- Reference counting for shared chunks

### 5. Chunk Cache Manager
**File**: `vllm-ascend/chunk/cache_manager.py`

Implements `ChunkCacheManager` class:
- Coordinates chunk cache lifecycle
- Matches chunks against cache
- Computes missing chunks
- Assembles complete KV caches
- Computes position offsets for chunks
- Provides cache statistics

### 6. Package Init
**File**: `vllm-ascend/chunk/__init__.py`

Package initialization file that exports `ChunkCacheManager` and `ChunkCachePool`.

## Files Modified

### 1. EngineCoreRequest
**File**: `vllm/vllm/v1/engine/__init__.py`

Changes:
- Added import for `ChunkMetadata`
- Added `chunk_metadata: list[ChunkMetadata] | None` field
- Added `has_chunks: bool` field
- Added `get_chunk_hashes()` method
- Added `get_total_chunk_tokens()` method

### 2. Input Processor
**File**: `vllm/vllm/v1/engine/input_processor.py`

Changes:
- Added import for `ChunkParser` and `ChunkMetadata`
- Added `chunk_parser` instance in `__init__`
- Added `_process_chunks()` method to detect and parse chunks
- Modified `process_inputs()` to:
  - Extract text prompt for chunk detection
  - Call `_process_chunks()` to get chunk metadata
  - Pass `chunk_metadata` and `has_chunks` to `EngineCoreRequest`

## Usage Example

```python
from vllm import LLM

# Example prompt with chunks
prompt = """
You are a helpful assistant. # #

Document 1: vLLM is a high-performance LLM inference engine... # #

Document 2: Chunk Cache enables position-agnostic KV caching... # #

Question: What are the benefits of Chunk Cache?
"""

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
output = llm.generate(prompt)
print(output[0].outputs[0].text)
```

## Configuration Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_CHUNK_CACHE_SIZE_GB` | 2.0 | Chunk cache pool size in GB |
| `VLLM_ENABLE_CHUNK_CACHE` | true | Enable/disable chunk caching |
| `VLLM_CHUNK_CACHE_LRU_THRESHOLD` | 0.9 | LRU eviction threshold |
| `VLLM_CHUNK_CACHE_HASH_ALGORITHM` | sha256 | Hash algorithm for chunks |

## Architecture Flow

1. **Input Processing**:
   - User submits prompt with `# #` delimiters
   - `InputProcessor._process_chunks()` detects delimiters
   - `ChunkParser` parses prompt into system prompt + chunks + question

2. **Tokenization**:
   - Each chunk is tokenized independently with system prompt
   - Chunk hash is computed from token IDs
   - `ChunkMetadata` is created for each chunk

3. **Cache Lookup** (in ChunkCacheManager):
   - For each chunk hash, lookup in ChunkCachePool
   - Return ChunkMatchResult with hits/misses

4. **KV Cache Assembly**:
   - Matched chunks: Reuse cached KV
   - Missing chunks: Compute and store in cache
   - Assemble complete KV cache for inference

5. **Position Encoding** (to be implemented in AttentionBackend):
   - Each chunk starts from position 0
   - User question starts from max position of all chunks
   - RoPE is computed accordingly

## Remaining Implementation Tasks

### 1. NPUWorker Integration
Modify `vllm-ascend/worker.py` to:
- Initialize ChunkCacheManager
- Pass to ModelRunner
- Configure cache pool size

### 2. Scheduler Integration
Modify scheduler to:
- Detect `has_chunks` flag
- Route to ChunkCacheManager instead of prefix caching
- Handle chunk cache hits/misses

### 3. Attention Backend Position Correction
Modify `vllm-ascend/attention/attention_v1.py` to:
- Compute position offsets for chunks
- Correct RoPE for chunk reuse
- Copy chunk KV to PagedAttention blocks

### 4. Testing
Create test suite to verify:
- Chunk parsing correctness
- Hash computation
- Cache lookup and storage
- LRU eviction
- Position encoding correction

### 5. Documentation
Update documentation with:
- Usage guide
- Configuration options
- Performance tuning tips
- Troubleshooting guide

## Key Design Decisions

1. **Delimiter-based**: Used `# #` (with spaces) as delimiter for better readability
2. **Position-independent hashing**: Chunks hash based on content only, not position
3. **Independent chunk computation**: Each chunk computed with system prompt separately
4. **Separate memory pool**: Chunk cache uses dedicated pool, doesn't interfere with PagedAttention
5. **LRU eviction**: Simple and effective cache management strategy
6. **Extensible design**: Easy to add new features like persistence or distributed caching

## Performance Expectations

- **Cache hit rate**: 50-95% depending on content overlap
- **Speedup**: 3-10x for cached chunks
- **Memory overhead**: Configurable (default 2GB)
- **Latency impact**: Minimal overhead for chunk detection and hashing

## Compatibility Notes

- **With Prefix Caching**: Mutually exclusive - chunk requests bypass prefix caching
- **With Multimodal**: Not yet tested, should work with text-only chunks
- **With LoRA**: Hash can include LoRA adapter name for correctness
- **With Speculative Decoding**: Needs testing and possible adjustments

## Future Enhancements

1. **Persistent cache**: Save chunks to disk for cross-session reuse
2. **Distributed cache**: Share chunks across multiple instances
3. **Multimodal chunks**: Support images and other modalities in chunks
4. **Smart chunking**: Automatically detect optimal chunk boundaries
5. **Compression**: Compress KV caches for more storage
6. **Metrics**: Detailed cache performance metrics and monitoring

## Troubleshooting

### Issue: Chunks not being detected
- **Solution**: Ensure prompt contains exactly `# #` (with spaces) as delimiter

### Issue: Low cache hit rate
- **Solution**: Check that chunk content is actually identical across requests

### Issue: Out of memory errors
- **Solution**: Reduce `VLLM_CHUNK_CACHE_SIZE_GB` or increase system memory

### Issue: Incorrect position encoding
- **Solution**: Ensure AttentionBackend position correction is properly implemented

## Development Notes

- All new code follows vLLM coding style and conventions
- Type hints are used throughout for better IDE support
- Comprehensive docstrings for all public APIs
- Logging statements for debugging and monitoring
- Error handling for edge cases (missing tokenizer, invalid format, etc.)

## Version

- Implementation Date: 2025-01-22
- Target vLLM Version: v1 (latest)
- Target Platform: Huawei Ascend NPU (vLLM-Ascend)

## References

- Design Document: `chunk_cache.md`
- vLLM Prefix Caching: `vllm/v1/core/kv_cache_manager.py`
- RFC #11162: Hardware Pluggable Interface
