# 位置无关 KV Cache 缓存和复用 - 昇腾 NPU 实现方案

**版本**: v1.0
**创建日期**: 2026-01-16
**目标平台**: 昇腾 NPU (Atlas 910B/910C/310P)
**软件栈**: vLLM v1 + vLLM-Ascend + CANN 8.3.rc2 + torch-npu 2.8.0

---

## 目录

1. [需求概述](#1-需求概述)
2. [系统架构](#2-系统架构)
3. [核心实现](#3-核心实现)
4. [关键接口](#4-关键接口)
5. [性能预期](#5-性能预期)

---

## 1. 需求概述

### 1.1 业务场景

在 RAG (检索增强生成) 应用中，相同内容在不同位置无法复用，Prefix Caching 命中率不高。

**本方案的优势**:
- Chunk 基于内容哈希缓存，位置无关
- 相同内容跨请求复用，大幅减少计算
- Chunk 隔离注意力，避免 chunk 之间相互干扰
- 高效KV拷贝重映射，利用NPU加速，开销仅~10%

### 1.2 核心功能

**Prompt 格式**:
```
sys_prompt + "##" + chunk1 + "##" + chunk2 + "##" + user_question
```

**Chunk 隔离注意力**:
- ✅ Chunk token 可以关注 sys_prompt 的所有 token
- ✅ Chunk token 可以关注同 chunk 内的前序 token (causal attention)
- ❌ Chunk token **不能**关注其他 chunk 的 token
- ✅ User question 可以关注所有内容 (sys_prompt + 所有 chunks)

**位置无关缓存**:
- Chunk 在虚拟位置 [VIRTUAL_POS_START, VIRTUAL_POS_START + max_chunk_len) 计算 KV cache
- 缓存 key 基于内容哈希 (不包含位置信息)
- 使用时**拷贝**KV数据并重映射到实际位置，应用新位置的RoPE编码
- 利用 NPU 加速拷贝和 RoPE，开销仅 ~2-5ms (4K tokens)

**sys_prompt 也作为 chunk 处理**:
- 不使用 vLLM 的标准 prefix caching
- 所有内容 (包括 sys_prompt) 都作为 chunk 处理
- 统一基于内容哈希缓存

### 1.3 核心创新点

1. **位置无关缓存**: Chunk hash 不包含位置信息
2. **虚拟位置计算**: 所有 chunk 在统一虚拟位置计算
3. **高效KV拷贝**: 拷贝 + RoPE 编码 ~2-5ms (4K tokens)
4. **统一chunk处理**: sys_prompt 也作为特殊chunk
5. **Chunk 隔离注意力**: 自定义 attention mask
6. **独立内存池**: 与主 KV cache 分离

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     LLMEngine (vllm/v1/engine/)             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │InputProcessor│→ │ ChunkCache   │→ │ EngineCoreClient │  │
│  │              │  │   Manager    │  │                  │  │
│  │ 解析chunked  │  │ (新增)       │  │ 调度执行         │  │
│  │ prompt       │  │              │  │                  │  │
│  └──────────────┘  └──────┬───────┘  └──────────────────┘  │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Worker (vllm/v1/worker/)                   │  │
│  │                                                       │  │
│  │  GPUModelRunner (修改)                               │  │
│  │  - 新增: parse_chunked_prompt()                      │  │
│  │  - 新增: get_or_compute_chunks()                     │  │
│  │  - 调用: ChunkCacheManager.get_or_compute_chunk()    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 分层架构

```
API Layer (entrypoints)
    ↓
ENGINE LAYER (vllm/v1/engine/)
    - InputProcessor: 解析 chunked prompt
    - LLMEngine: 初始化 ChunkCacheManager
    - ChunkCacheManager: 管理缓存
    ↓
WORKER LAYER (vllm/v1/worker/)
    - GPUModelRunner: 获取/计算 chunks
    - PositionRemapper: KV 拷贝重映射
    ↓
CORE LAYER (vllm/v1/core/)
    - ChunkHashIndex: 哈希索引
    - ChunkBlockPool: 块池管理
    - Chunk Structures: 数据结构
    ↓
ATTENTION LAYER (vllm-ascend/attention/)
    - AscendSFABackend: SFA attention
    - AttentionMaskBuilder: chunk-aware mask
```

### 2.3 新增/修改模块

**新增模块** (6个):
- `vllm/v1/core/chunk_hash_index.py` - 哈希索引
- `vllm/v1/core/chunk_block_pool.py` - 块池管理
- `vllm/v1/core/chunk_cache_manager.py` - 缓存管理器
- `vllm/v1/worker/position_remapper.py` - 位置重映射
- `vllm/v1/core/chunk_structures.py` - 数据结构
- `vllm/config/chunk_cache.py` - 配置

**修改模块** (7个):
- `vllm/config/vllm_config.py` - 添加 ChunkCacheConfig
- `vllm/v1/engine/llm_engine.py` - 集成 ChunkCacheManager
- `vllm/v1/engine/input_processor.py` - 解析 chunked prompt
- `vllm/v1/worker/gpu_model_runner.py` - 新增方法
- `vllm-ascend/attention/utils.py` - 扩展元数据
- `vllm-ascend/attention/sfa_v1.py` - 集成 chunk mask

### 2.4 核心流程

```
User Request ("You are helpful.##Doc1##What?")
    ↓
InputProcessor.parse_chunked_prompt()
    → ChunkedPrompt {sys_prompt, chunks, question}
    ↓
GPUModelRunner.get_or_compute_chunks()
    → for chunk in chunks:
    →   ChunkCacheManager.get_or_compute_chunk()
    →     - hash → lookup → [HIT/MISS]
    →     - [HIT]: remap_and_copy() (3-5ms)
    →     - [MISS]: compute → cache → remap (50-60ms)
    ↓
GPUModelRunner.compute_question_kv()
    ↓
AscendSFABackend.forward()
    → with chunk-aware mask
    ↓
Generated Output
```

### 2.5 内存管理

**双内存池架构**:
```
主 KV Pool (70%):     活跃请求的 KV
Chunk Pool (15%):     位置无关缓存 (CaMemAllocator "chunk_cache")
预留 (15%):           其他用途
```

**重映射内存流**:
1. 从 Chunk Pool 读取缓存的 KV
2. 从主 KV Pool 分配新块
3. 拷贝 KV + 应用新位置 RoPE
4. 请求结束后释放主 Pool 的块，Chunk Pool 保留

---

## 3. 核心实现

### 3.1 数据结构

**ChunkHash** - 内容哈希 (位置无关):
```python
@dataclass(frozen=True)
class ChunkHash:
    hash_bytes: bytes      # XXHash128(16 bytes)
    token_count: int
    num_blocks: int
```

**ChunkKVCache** - 缓存的 Chunk KV:
```python
@dataclass
class ChunkKVCache:
    chunk_hash: ChunkHash
    block_ids: list[int]           # Chunk Pool 中的块
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    virtual_position_start: int    # 虚拟位置
    virtual_position_end: int
```

**RemappedChunkKV** - 重映射后的 Chunk (包含 KV 拷贝):
```python
@dataclass
class RemappedChunkKV:
    block_ids: list[int]           # 主 KV Pool 的新块
    key_cache: torch.Tensor        # 已应用新位置 RoPE
    value_cache: torch.Tensor
    position_start: int
    position_end: int
    chunk_hash: ChunkHash
```

**ChunkedPrompt** - 分块提示词:
```python
@dataclass
class ChunkedPrompt:
    sys_prompt: list[int]          # 作为特殊 chunk
    chunks: list[list[int]]
    user_question: list[int]
    separator: str = "##"
```

### 3.2 LLMEngine 集成

**初始化** (`vllm/v1/engine/llm_engine.py`):
```python
class LLMEngine:
    def __init__(self, vllm_config: VllmConfig, ...):
        # 新增: 初始化 ChunkCacheManager
        if vllm_config.chunk_cache_config.enable_chunk_cache:
            ascend_allocator = self._init_ascend_allocator()
            self.chunk_cache_manager = ChunkCacheManager(
                config=vllm_config.chunk_cache_config,
                kv_cache_spec=self.model_config.kv_cache_spec,
                block_size=self.cache_config.block_size,
                ascend_allocator=ascend_allocator,
            )
        else:
            self.chunk_cache_manager = None
```

**请求处理**:
```python
def step(self):
    # 检测 chunked prompt
    if scheduled_req.use_chunk_cache:
        chunk_kvs = self._get_or_compute_chunks(chunked_prompt)
        # 传递给 Worker 执行
```

### 3.3 InputProcessor 扩展

**解析 chunked prompt** (`vllm/v1/engine/input_processor.py`):
```python
class InputProcessor:
    def process_inputs(self, prompts, ...):
        # 检测 "##" 分隔符
        if "##" in prompt_str:
            chunked_prompt = self.parse_chunked_prompt(
                prompt_tokens, separator="##"
            )
            request_metadata["use_chunk_cache"] = True
            request_metadata["chunked_prompt"] = chunked_prompt
```

### 3.4 ChunkCacheManager 核心

**获取或计算 Chunk** (`vllm/v1/core/chunk_cache_manager.py`):
```python
class ChunkCacheManager:
    def get_or_compute_chunk(
        self, chunk_tokens, position_offset, model_runner
    ) -> RemappedChunkKV:
        # 1. 计算哈希
        chunk_hash = self.chunk_hash_index.compute_hash(chunk_tokens)

        # 2. 查找缓存
        cached_chunk = self.chunk_hash_index.lookup(chunk_hash)

        if cached_chunk:
            # 3a. 命中: 拷贝重映射 (3-5ms)
            return self.position_remapper.remap_and_copy(
                cached_chunk, position_offset
            )

        # 3b. 未命中: 计算并缓存 (50-60ms)
        # 淘汰 LRU
        self.chunk_block_pool.evict_lru_until_enough(required_blocks)

        # 在虚拟位置计算
        virtual_pos = self._get_virtual_position(len(chunk_tokens))
        computed_kv = model_runner.compute_chunk_kv(
            chunk_tokens, virtual_pos
        )

        # 存储
        self.chunk_hash_index.insert(chunk_hash, computed_kv)

        # 拷贝重映射到目标位置
        return self.position_remapper.remap_and_copy(
            computed_kv, position_offset
        )
```

### 3.5 PositionRemapper 实现

**KV 拷贝重映射** (`vllm/v1/worker/position_remapper.py`):
```python
class PositionRemapper:
    def remap_and_copy(
        self, chunk_kv_cache, new_position_offset
    ) -> RemappedChunkKV:
        # 1. 从主 KV Pool 分配新块
        new_block_ids = self._allocate_from_main_pool(num_blocks)

        # 2. 计算新位置序列
        new_positions = torch.arange(
            new_position_offset,
            new_position_offset + num_tokens,
            device="npu"
        )

        # 3. 拷贝 KV 并应用 RoPE (NPU 加速)
        cos, sin = self._compute_rope(new_positions)
        remapped_k, remapped_v = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_input=None,
            cos=cos, sin=sin,
            positions=new_positions,
            key_cache=src_k,
            value_cache=src_v,
            block_ids=new_block_ids,
        )

        # 4. 返回重映射后的 KV
        return RemappedChunkKV(
            block_ids=new_block_ids,
            key_cache=remapped_k,
            value_cache=remapped_v,
            position_start=new_position_offset,
            position_end=new_position_offset + num_tokens,
        )
```

### 3.6 GPUModelRunner 扩展

**处理 Chunks** (`vllm/v1/worker/gpu_model_runner.py`):
```python
class GPUModelRunner:
    def get_or_compute_chunks(
        self, chunked_prompt
    ) -> List[RemappedChunkKV]:
        remapped_kvs = []
        current_position = 0

        # 处理所有 chunks (包括 sys_prompt)
        for chunk_tokens in chunked_prompt.get_all_chunks():
            remapped_kv = self.chunk_cache_manager.get_or_compute_chunk(
                chunk_tokens, current_position, self
            )
            remapped_kvs.append(remapped_kv)
            current_position += len(chunk_tokens)

        return remapped_kvs
```

### 3.7 Chunk-Aware Attention Mask

**构建隔离 Mask** (`vllm-ascend/attention/attention_mask.py`):
```python
def get_chunk_aware_mask(
    num_tokens, chunk_ids, chunk_boundaries, dtype, device
) -> torch.Tensor:
    """
    Chunk 隔离的 causal attention mask

    规则:
    - sys_prompt: 标准 causal
    - chunk tokens: 可看 sys_prompt + 同 chunk (causal)
    - question: 可看所有 (causal)
    """
    mask = torch.zeros(num_tokens, num_tokens, dtype=dtype, device=device)

    for query_pos in range(num_tokens):
        query_chunk_id = chunk_ids[query_pos]

        if query_chunk_id >= 0:  # Chunk token
            # 可看 sys_prompt
            mask[query_pos, :sys_end] = 0
            # 可看同 chunk 内的前序 token
            mask[query_pos, chunk_start:query_pos] = 0

        elif query_chunk_id == -2:  # Question
            mask[query_pos, :query_pos] = 0

        else:  # sys_prompt
            mask[query_pos, :query_pos] = 0

    return mask
```

### 3.8 配置

**ChunkCacheConfig** (`vllm/config/chunk_cache.py`):
```python
@dataclass
class ChunkCacheConfig:
    enable_chunk_cache: bool = False
    chunk_separator: str = "##"
    max_chunks: int = 500
    max_chunk_tokens: int = 4096
    chunk_cache_gpu_memory_utilization: float = 0.15
    chunk_hash_algo: str = "xxhash"
```

**VllmConfig 扩展**:
```python
@dataclass
class VllmConfig:
    # ... 现有字段 ...
    chunk_cache_config: ChunkCacheConfig = field(
        default_factory=ChunkCacheConfig
    )
```

---

## 4. 关键接口

### 4.1 ChunkCacheManager

```python
class ChunkCacheManager:
    def get_or_compute_chunk(
        self, chunk_tokens, position_offset, model_runner
    ) -> RemappedChunkKV
        """获取或计算 chunk KV cache"""

    def get_stats(self) -> Dict
        """获取缓存统计信息"""

    def clear(self)
        """清空缓存"""
```

### 4.2 PositionRemapper

```python
class PositionRemapper:
    def remap_and_copy(
        self, chunk_kv_cache, new_position_offset
    ) -> RemappedChunkKV
        """拷贝并重映射 chunk KV"""

    def release_remapped_blocks(self, block_ids)
        """释放重映射块 (请求结束后)"""
```

### 4.3 GPUModelRunner

```python
class GPUModelRunner:
    def parse_chunked_prompt(
        self, prompt_tokens, separator
    ) -> ChunkedPrompt
        """解析分块提示词"""

    def get_or_compute_chunks(
        self, chunked_prompt
    ) -> List[RemappedChunkKV]
        """获取或计算所有 chunks"""

    def compute_chunk_kv(
        self, tokens, position
    ) -> ChunkKVCache
        """在指定位置计算 chunk KV"""
```

---

## 5. 性能预期

### 5.1 加速比

| 场景 | 加速比 | 说明 |
|------|--------|------|
| 单个 chunk 缓存命中 | ~12x | 5ms vs 60ms (含拷贝) |
| 3-5 个 chunk 缓存命中 | ~30-50x | 累积收益 |
| TTFT 减少 | 50-80% | RAG 场景 |
| 端到端加速 | 2-5x | 80%+ 命中率 |

### 5.2 开销分析

**缓存命中** (4K tokens):
- Hash 计算: ~0.01ms
- 查找: O(1)
- KV 拷贝: ~1-3ms (NPU)
- RoPE 编码: ~0.5-1ms
- **总开销**: ~2-5ms
- **节省**: ~55ms (相比完整计算)

**缓存未命中** (4K tokens):
- Hash: ~0.01ms
- 计算 KV: ~50ms
- 拷贝重映射: ~3-5ms
- **总时间**: ~55ms
- **后续复用**: ~5ms

### 5.3 内存占用

**配置** (32GB GPU, block_size=16, head_dim=128, num_heads=32):
- 主 KV Pool: 20GB (70%)
- Chunk Pool: 4.3GB (15%)
- 单 block: 256KB
- Chunk Pool 容量: ~17,000 blocks
- 可缓存 tokens: ~272K

### 5.4 命中率预期

**典型 RAG 场景**:
- 文档复用率: 60-80%
- 预期命中率: 70-90%
- 平均加速: 10-20x

---

## 总结

本方案设计了一个位置无关的 KV Cache 缓存系统，核心特点：

1. **位置无关**: 基于 content hash，支持跨位置复用
2. **高效拷贝**: NPU 加速，开销仅 10%
3. **统一处理**: sys_prompt 也作为 chunk
4. **隔离注意力**: chunk 之间互不干扰
5. **独立内存池**: 与主 KV cache 分离
6. **向后兼容**: 默认禁用，检测 "##" 启用

**实施要点**:
- 新增 6 个模块，修改 7 个模块
- 最小化现有代码改动
- 利用 vLLM-Ascend CaMemAllocator
- 12x 加速 (单 chunk)，30-50x (多 chunks)
