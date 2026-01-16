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

#### 场景1: 缓存未命中 - 生成Chunk KV并缓存

```
User Request: "You are helpful.##Doc1##What?"
    ↓
[1] Prompt解析
    InputProcessor.parse_chunked_prompt()
    → ChunkedPrompt {
        sys_prompt: [1,2,3],
        chunks: [[10,11,...]],
        user_question: [100,101]
      }
    ↓
[2] Chunk哈希查找 (每个chunk)
    ChunkHashIndex.compute_hash(chunk_tokens)
    → hash_abc → lookup → [MISS]
    ↓
[3] 生成Chunk KV (虚拟位置)
    分配 Chunk Pool 块
    GPUModelRunner.compute_chunk_kv(
        tokens=chunk_tokens,
        position=VIRTUAL_POS_START  # 虚拟位置 [0, max_chunk_len)
    )
    → 应用虚拟位置的RoPE编码
    → ChunkKVCache {
        block_ids: [100, 101],
        key_cache, value_cache,
        virtual_position: [0, 100)
      }
    ↓
[4] 缓存Chunk KV
    ChunkHashIndex.insert(hash_abc, ChunkKVCache)
    → 存储到Chunk Pool (持久化)
    ↓
[5] 拷贝并重映射到实际位置
    PositionRemapper.remap_and_copy(
        cached_kv, new_position=0  # sys_prompt在位置0
    )
    → 分配主KV Pool块 [5000, 5001]
    → 拷贝KV数据 (100→5000, 101→5001)
    → 应用实际位置的RoPE编码
    → RemappedChunkKV {
        block_ids: [5000, 5001],
        key_cache, value_cache,  # 已应用位置0的RoPE
        position: [0, 100)
      }
    ↓
[6] 构建Chunk-Aware Attention Mask
    AttentionMaskBuilder.get_chunk_aware_mask()
    → sys_prompt tokens: 标准 causal attention
    → chunk tokens: 可看 sys_prompt + 同chunk内causal
    → question tokens: 可看所有内容
    ↓
[7] 执行推理
    AscendSFABackend.forward(
        query, key, value,
        chunk_attn_mask,
        remapped_kv
    )
    → Generated Output
```

#### 场景2: 缓存命中 - 拷贝拼接KV并推理

```
User Request: "Different sys.##Doc1##Tell me more"
    ↓
[1] Prompt解析
    InputProcessor.parse_chunked_prompt()
    → ChunkedPrompt {
        sys_prompt: [5,6,7],  # 不同！
        chunks: [[10,11,...]],  # 相同 → 缓存命中
        user_question: [200,201]
      }
    ↓
[2] Chunk哈希查找 (每个chunk)
    ChunkHashIndex.compute_hash([10,11,...])
    → hash_def → lookup → [HIT]  ✓
    ↓
[3] 拷贝并重映射到新位置
    PositionRemapper.remap_and_copy(
        cached_kv,  # 来自缓存，虚拟位置[0,100)
        new_position=3  # 新sys_prompt在位置0-2
    )
    → 分配主KV Pool块 [6000, 6001]
    → 拷贝KV数据 (100→6000, 101→6001)
    → 应用位置3的RoPE编码 (覆盖原虚拟位置RoPE)
    → RemappedChunkKV {
        block_ids: [6000, 6001],
        key_cache, value_cache,  # 已应用位置3的RoPE
        position: [3, 103)
      }
    跳过步骤[4-5] (已缓存)
    ↓
[4] 计算未命中的sys_prompt
    sys_prompt [5,6,7] → 缓存未命中
    → compute_chunk_kv(position=0)
    → 缓存到hash_xyz
    → remap_and_copy(new_position=0)
    ↓
[5] 拼接所有KV
    合并:
    - sys_prompt KV (blocks [7000,7001], pos [0,3))
    - Doc1 KV (blocks [6000,6001], pos [3,103))  来自缓存
    - question KV (blocks [7002,7003], pos [103,105))
    → merged_blocks: [7000,7001,6000,6001,7002,7003]
    ↓
[6] 构建Chunk-Aware Attention Mask
    根据新的chunk边界构建mask
    → chunk_ids: [-1,-1,-1, 0,0,..., -2,-2]
    → chunk隔离: Doc1不能看新sys_prompt的token
    ↓
[7] 执行推理
    AscendSFABackend.forward(
        query, key, value,
        chunk_attn_mask,
        merged_kv
    )
    → Generated Output
```

**核心差异对比**:

| 步骤 | 缓存未命中 | 缓存命中 |
|------|-----------|---------|
| 哈希查找 | MISS | HIT |
| KV计算 | ✓ (50ms) | ✗ |
| 缓存存储 | ✓ | ✗ |
| KV拷贝 | ✓ (3-5ms) | ✓ (3-5ms) |
| RoPE处理 | 虚拟位置→实际位置 | 虚拟位置→实际位置 |
| Attention Mask | chunk隔离 | chunk隔离 (新边界) |
| 总时间 | ~55ms | ~5ms |
| 加速比 | 1x | **12x** |

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

**职责**: Chunk缓存管理器，协调ChunkHashIndex和ChunkBlockPool

**位置**: `vllm/v1/core/chunk_cache_manager.py`

**主要接口**:

```python
class ChunkCacheManager:
    """Chunk缓存管理器核心接口"""

    def __init__(
        self,
        config: ChunkCacheConfig,
        kv_cache_spec: KVCacheSpec,
        block_size: int,
        ascend_allocator: Optional[CaMemAllocator] = None,
    ):
        """
        初始化ChunkCacheManager

        功能:
            - 初始化ChunkHashIndex (哈希索引)
            - 初始化ChunkBlockPool (块池管理)
            - 初始化PositionRemapper (位置重映射)
            - 初始化统计信息收集器

        Args:
            config: Chunk缓存配置
            kv_cache_spec: KV缓存规格 {num_kv_heads, head_dim, dtype}
            block_size: 块大小 (tokens per block)
            ascend_allocator: 昇腾内存分配器 (可选)

        Raises:
            ValueError: 配置参数无效
        """

    def get_or_compute_chunk(
        self,
        chunk_tokens: List[int],
        position_offset: int,
        model_runner: "GPUModelRunner",
    ) -> RemappedChunkKV:
        """
        获取或计算chunk KV cache (核心方法)

        功能:
            1. 计算chunk的内容哈希
            2. 查找缓存
            3. [命中] 拷贝重映射到目标位置
            4. [未命中] 淘汰LRU → 虚拟位置计算 → 缓存 → 拷贝重映射

        Args:
            chunk_tokens: chunk的token序列
            position_offset: 目标位置偏移
            model_runner: 模型运行器引用

        Returns:
            RemappedChunkKV: 重映射后的chunk KV (在主KV池中)

        Raises:
            ValueError: chunk_tokens为空
            MemoryError: 内存不足 (即使LRU淘汰后)
            RuntimeError: model_runner未初始化

        时间复杂度:
            缓存命中: O(1) + O(n) 拷贝重映射
            缓存未命中: O(n) 计算 + O(n) 拷贝重映射
        """

    def get_stats(self) -> ChunkCacheStats:
        """
        获取缓存统计信息

        Returns:
            ChunkCacheStats: 统计信息副本 (线程安全)
            - total_chunks: 总chunk数
            - cache_hits/misses: 命中/未命中次数
            - hit_rate: 命中率
            - total_tokens_cached: 缓存token总数
            - evicted_chunks: 淘汰chunk数
            - current_memory_mb: 当前内存使用(MB)
        """

    def clear(self):
        """清空所有缓存 (测试用)"""

    def cleanup_remapped_blocks(self, block_ids: List[int]):
        """
        清理重映射使用的块 (请求结束后调用)

        Args:
            block_ids: 要释放的块ID列表 (主KV池)
        """
```

**依赖模块**:
- `ChunkHashIndex`: 哈希索引 (计算hash、查找、插入、淘汰)
- `ChunkBlockPool`: 块池管理 (分配、释放、LRU淘汰)
- `PositionRemapper`: 位置重映射 (KV拷贝+RoPE编码)

### 4.2 PositionRemapper

**职责**: 位置重映射器，负责KV拷贝和RoPE重新编码

**位置**: `vllm/v1/worker/position_remapper.py`

**主要接口**:

```python
class PositionRemapper:
    """位置重映射器 - KV拷贝和RoPE重新编码"""

    def __init__(
        self,
        block_pool: ChunkBlockPool,
        kv_cache_spec: KVCacheSpec,
        device: torch.device,
    ):
        """
        初始化

        功能:
            - 保存块池引用
            - 保存KV缓存规格 (num_kv_heads, head_dim, dtype)
            - 初始化RoPE缓存 (位置 -> cos/sin)

        Args:
            block_pool: Chunk块池 (用于分配)
            kv_cache_spec: KV缓存规格
            device: 设备 (npu或cuda)
        """

    def remap_and_copy(
        self,
        chunk_kv_cache: ChunkKVCache,
        new_position_offset: int,
    ) -> RemappedChunkKV:
        """
        拷贝并重映射chunk KV cache (核心方法)

        功能:
            1. 从主KV池分配新块
            2. 计算新位置序列 (position_offset to position_offset + num_tokens)
            3. 获取或计算RoPE编码 (带缓存)
            4. 拷贝KV数据并应用新位置的RoPE (使用NPU加速)
            5. 创建RemappedChunkKV对象

        Args:
            chunk_kv_cache: 缓存的chunk KV (在chunk池中)
            new_position_offset: 新的位置偏移

        Returns:
            RemappedChunkKV: 重映射后的KV (在主KV池中)
                - block_ids: 主KV池的新块
                - key_cache/value_cache: 已应用新位置RoPE的KV
                - position_start/end: 实际位置范围

        Raises:
            MemoryError: 主KV池内存不足
            RuntimeError: NPU拷贝失败或RoPE计算失败

        时间复杂度: O(num_tokens)
        空间复杂度: O(num_tokens) (新分配blocks)

        昇腾NPU优化:
            - 使用 torch_npu.npu_kv_rmsnorm_rope_cache()
            - 一次调用完成KV拷贝+RoPE编码
            - 预期开销: ~2-5ms (4K tokens)
        """

    def release_remapped_blocks(self, block_ids: List[int]):
        """
        释放重映射使用的物理块 (请求结束后)

        功能:
            - 将块归还给主KV池
            - 请求结束后调用,清理临时分配

        Args:
            block_ids: 要释放的块ID列表
        """
```

**RoPE缓存优化**:
- 缓存key: `(position_start, position_end)`
- 避免重复计算相同位置范围的RoPE
- 显著减少重复请求的开销

### 4.3 ChunkedPromptParser

**职责**: 分块提示词解析器，识别和处理chunk分隔符

**位置**: `vllm/v1/engine/chunked_prompt_parser.py`

**设计参考**: LMcache `SegmentTokenDatabase` 使用滑动窗口快速匹配

**数据结构**:

```python
@dataclass
class ChunkedPrompt:
    """
    分块提示词结构

    Attributes:
        sys_prompt: 第一个chunk (特殊)
        chunks: 文档chunks列表
        user_question: 用户问题
        separator: 分隔符 (默认 "##")
        chunk_boundaries: 每个chunk的边界 [(start, end), ...]
    """
    sys_prompt: List[int]
    chunks: List[List[int]]
    user_question: List[int]
    separator: str = "##"
    chunk_boundaries: List[Tuple[int, int]]

    def get_all_chunks(self) -> List[List[int]]:
        """获取所有需要缓存的chunks (包括sys_prompt)"""
```

**主要接口**:

```python
class ChunkedPromptParser:
    """分块提示词解析器 - 滑动窗口快速匹配"""

    def __init__(
        self,
        tokenizer,
        separator: str = "##",
        device: torch.device = torch.device("cpu"),
    ):
        """
        初始化

        功能:
            - 编码分隔符为tokens (移除BOS token)
            - 准备滑动窗口匹配的sep_tokens tensor

        Args:
            tokenizer: 分词器
            separator: 分隔符字符串 (默认 "##")
            device: 分隔符tokens所在设备
        """

    def parse(
        self,
        prompt_tokens: Union[torch.Tensor, List[int]],
        prompt_str: Optional[str] = None,
    ) -> ChunkedPrompt:
        """
        解析分块提示词

        功能:
            1. 转换为torch.Tensor (如果是List[int])
            2. 使用滑动窗口快速匹配分隔符
            3. 解析: sys_prompt (第一个分隔符之前)
            4. 解析: chunks (分隔符之间)
            5. 解析: question (最后一个分隔符之后)
            6. 验证: 不允许空chunk

        Args:
            prompt_tokens: token序列 (torch.Tensor或List[int])
            prompt_str: 原始字符串 (可选，用于验证)

        Returns:
            ChunkedPrompt: 解析后的分块结构

        Raises:
            ValueError: 格式错误、分隔符不存在、空chunk
            RuntimeError: 解析失败

        性能:
            - 使用 torch.unfold() 滑动窗口: O((n-m+1) * m)
            - n = len(tokens), m = len(separator_tokens)
            - 比朴素O(n * m)更优
        """

    def _fast_split_by_subtensor(
        self,
        tokens: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        使用滑动窗口快速匹配分隔符

        功能:
            1. 使用 tokens.unfold(0, sep_len, 1) 创建滑动窗口
            2. 比较每个窗口与sep_tokens
            3. 在匹配位置切分tokens

        Args:
            tokens: token张量

        Yields:
            分割后的token张量

        时间复杂度: O((n-m+1) * m)
        """

    def _find_separator_positions(
        self,
        tokens: torch.Tensor,
    ) -> List[int]:
        """
        查找所有分隔符的起始位置

        Returns:
            List[int]: 分隔符起始位置列表
        """

    def validate_and_parse(
        self,
        prompt_str: str,
    ) -> ChunkedPrompt:
        """
        验证并解析提示词 (带字符串验证)

        功能:
            1. 验证分隔符存在
            2. 验证分隔符数量 (至少1个)
            3. Tokenize
            4. 调用parse()

        Args:
            prompt_str: 原始提示词字符串

        Returns:
            ChunkedPrompt: 解析后的结构

        Raises:
            ValueError: 格式验证失败
        """
```

**性能优化**:
- 使用 `torch.unfold()` 创建滑动窗口
- 向量化比较，避免Python循环
- 时间复杂度: O((n-m+1) * m)

### 4.4 GPUModelRunner 扩展

**职责**: GPU模型运行器，新增chunk缓存处理方法

**位置**: `vllm/v1/worker/gpu_model_runner.py`

**新增接口**:

```python
class GPUModelRunner:
    """GPU模型运行器 - Chunk缓存扩展"""

    def set_chunk_cache_manager(self, manager: ChunkCacheManager):
        """
        注入ChunkCacheManager (由LLMEngine调用)

        功能:
            - 保存manager引用
            - 初始化ChunkedPromptParser

        Args:
            manager: Chunk缓存管理器实例
        """

    def parse_chunked_prompt(
        self,
        prompt_tokens: Union[torch.Tensor, List[int]],
        prompt_str: Optional[str] = None,
    ) -> ChunkedPrompt:
        """
        解析分块提示词

        Args:
            prompt_tokens: token序列
            prompt_str: 原始字符串 (可选)

        Returns:
            ChunkedPrompt: 解析后的分块结构

        Raises:
            RuntimeError: chunk_parser未初始化
        """

    def get_or_compute_chunks(
        self,
        chunked_prompt: ChunkedPrompt,
    ) -> List[RemappedChunkKV]:
        """
        获取或计算所有chunks的KV cache (包括sys_prompt)

        功能:
            1. 遍历所有chunks (sys_prompt + chunks)
            2. 对每个chunk调用 chunk_cache_manager.get_or_compute_chunk()
            3. 累加position_offset
            4. 错误处理: 清理已分配的KV

        Args:
            chunked_prompt: 解析后的分块提示词

        Returns:
            List[RemappedChunkKV]: 所有chunk的重映射KV

        Raises:
            RuntimeError: ChunkCacheManager未初始化
            MemoryError: 内存不足

        注意:
            - 错误时自动清理已分配的block_ids
        """

    def compute_chunk_kv(
        self,
        tokens: List[int],
        position: int,
        block_pool: "ChunkBlockPool",
    ) -> ChunkKVCache:
        """
        在指定位置计算chunk KV

        功能:
            1. 从block_pool分配块
            2. 准备input_ids和positions
            3. 执行模型前向传播 (kv_cache初始为空)
            4. 提取key_cache和value_cache
            5. 存储到分配的块中
            6. 计算chunk_hash

        Args:
            tokens: token序列
            position: 起始位置
            block_pool: 块池 (用于分配)

        Returns:
            ChunkKVCache: 计算得到的chunk KV

        Raises:
            RuntimeError: 模型执行失败
            MemoryError: 块池内存不足

        注意:
            - 错误时自动清理已分配的block_ids
        """
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
