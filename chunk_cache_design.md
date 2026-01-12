# 位置无关 KV Cache 缓存和复用 - 技术设计文档

## 文档概述

本文档是位置无关 KV Cache 缓存和复用系统的完整技术设计文档，用于指导 vLLM-Ascend 上的实现工作。

**文档版本**: v1.0
**创建日期**: 2026-01-13
**目标平台**: 昇腾 NPU (vLLM + vLLM-Ascend)
**作者**: Claude Code

## 目录

1. [需求概述](#需求概述)
2. [系统架构](#系统架构)
3. [数据结构](#数据结构)
4. [接口设计](#接口设计)
5. [算法设计](#算法设计)
6. [性能分析](#性能分析)
7. [测试策略](#测试策略)
8. [部署与运维](#部署与运维)
9. [实施计划](#实施计划)

---

## 需求概述

### 核心需求

设计一个位置无关的 KV Cache 缓存和复用功能，支持将 prompt 通过 "##" 分隔符分成不同的块，每个块独立缓存 KV Cache，并支持位置无关的复用。

### 具体需求

1. **分块处理**：
   - Prompt 格式：`sys_prompt + "##" + chunk1 + "##" + chunk2 + "##" + chunk3 + "##" + user_question`
   - 每个 chunk 独立与 sys_prompt 构建 KV Cache
   - chunk 之间没有交叉注意力
   - user question 与所有内容做交叉注意力

2. **缓存机制**：
   - Token 级别的 KV Cache 缓存
   - 缓存到显存中，需要单独管理

3. **位置编码处理**：
   - chunk 的位置编码有重叠
   - user question 从所有 chunk 中最大的位置开始位置编码

4. **目标平台**：昇腾 NPU（基于 vLLM + vLLM-Ascend）

### 用户澄清的关键设计决策

- **sys_prompt 只有一个**：所有 chunk 都基于同一个 sys_prompt 构建 KV Cache
- **缓存策略**：基于 token hash 的精确匹配
- **显存管理**：独立显存池（与主 KV Cache 分离）
- **适用场景**：单轮场景（如 RAG 应用），不适用于多轮对话

---

## 系统架构

### 分层架构设计

系统采用分层架构，从上到下分为四层：

#### 1. API 层
- **Python API**: `vllm.LLM` 类扩展，支持 chunk cache 配置
- **OpenAI API**: `/v1/chat/completions` 端点扩展
- **CLI API**: `--enable-chunk-cache` 参数

#### 2. 引擎层 (Engine Layer)
- **ChunkCacheManager**: 核心 chunk cache 管理器
- **KVCacheManager**: 现有 prefix caching 系统
- **协调机制**: 两个 cache 系统的协同工作

#### 3. 执行层 (Worker Layer)
- **PositionReindexer**: 位置编码重映射
- **ChunkAwareModelRunner**: 扩展的模型运行器

#### 4. 硬件层 (Hardware Layer)
- **CaMemAllocator**: 昇腾内存分配器
- **SFA/MLA Attention**: 昇腾注意力机制
- **HCCL**: 多卡通信

### 模块依赖图

```
┌─────────────────────────────────────────────────────────┐
│                    vllm/config                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ChunkCacheConfig                                │   │
│  │ - enable_chunk_cache: bool                      │   │
│  │ - chunk_separator: str                          │   │
│  │ - position_strategy: Literal                    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                 vllm/v1/core                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ChunkHashIndex                                  │   │
│  │ - compute_chunk_hash()                          │   │
│  │ - lookup() / insert()                           │   │
│  └─────────────────────────────────────────────────┘   │
│                          ↓                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ChunkBlockPool                                  │   │
│  │ - allocate_blocks()                             │   │
│  │ - free_blocks()                                 │   │
│  │ - (uses CaMemAllocator with "chunk_cache" tag)  │   │
│  └─────────────────────────────────────────────────┘   │
│                          ↓                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ChunkCacheManager                               │   │
│  │ - get_or_compute_chunk()                        │   │
│  │ - _reindex_chunk_positions()                    │   │
│  │ - evict_lru()                                   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                vllm/v1/worker                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ PositionReindexer                               │   │
│  │ - reindex_chunk_kv()                            │   │
│  │ - _compute_rope_for_positions()                 │   │
│  │ - _compute_slot_mapping()                       │   │
│  └─────────────────────────────────────────────────┘   │
│                          ↓                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ GPUModelRunner (EXTENDED)                       │   │
│  │ - parse_chunked_prompt()                        │   │
│  │ - get_or_compute_chunks()                       │   │
│  │ - merge_kv_caches()                             │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 核心架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         LLM API Layer                           │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  Python API    │  │  OpenAI API      │  │  CLI API       │ │
│  └────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      vLLM Engine Layer                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    LLMEngine (v1)                         │ │
│  │  ┌────────────────────────────────────────────────────┐ │ │
│  │  │         ChunkCacheManager (NEW)                    │ │ │
│  │  │  - ChunkHashIndex: dict[ChunkHash, ChunkKVCache]   │ │ │
│  │  │  - ChunkBlockPool: 独立 block 池                   │ │ │
│  │  │  - LRUCache: 缓存淘汰策略                          │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  │  ┌────────────────────────────────────────────────────┐ │ │
│  │  │         KVCacheManager (EXISTING)                  │ │ │
│  │  │  - 位置相关的 prefix caching                        │ │ │
│  │  │  - Block Hash 索引                                  │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Worker/Execution Layer                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         PositionReindexer (NEW)                          │ │
│  │  - 虚拟位置空间映射                                       │ │
│  │  - RoPE cos/sin 缓存管理                                 │ │
│  │  - slot_mapping 重计算                                   │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         ChunkAwareModelRunner (EXTEND)                  │ │
│  │  - parse_chunked_prompt(): 解析 "##" 分隔符            │ │
│  │  - get_or_compute_chunks(): 获取/计算 chunk KV          │ │
│  │  - merge_kv_caches(): 合并多源 KV cache                │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Hardware/Platform Layer                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         CaMemAllocator (ASCEND)                          │ │
│  │  - 标签化内存池: "chunk_cache" tag                      │ │
│  │  - sleep/wake 生命周期管理                              │ │
│  │  - 物理内存隔离                                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         SFA/MLA Attention (ASCEND)                      │ │
│  │  - npu_kv_rmsnorm_rope_cache()                          │ │
│  │  - 支持非连续 block IDs                                 │ │
│  │  - KPE/KNOPE 分离 (MLA)                                 │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         HCCL Communication (ASCEND)                     │ │
│  │  - 多卡 chunk 同步                                      │ │
│  │  - 异步通信优化                                         │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 数据结构

### 核心数据结构定义

#### 1. ChunkHash

```python
@dataclass(frozen=True)
class ChunkHash:
    """内容哈希（位置无关）"""

    hash_bytes: bytes          # SHA256 或 XXHash 结果
    token_count: int           # Token 数量
    num_blocks: int            # 占用的 block 数量

    def __hash__(self) -> int:
        return int.from_bytes(self.hash_bytes, byteorder='big')

    def __eq__(self, other) -> bool:
        return self.hash_bytes == other.hash_bytes
```

**设计要点**：
- `frozen=True` 确保 hash 值不可变，可作为字典键
- 仅包含内容信息，不包含位置信息
- `token_count` 和 `num_blocks` 用于快速验证和内存估算

#### 2. ChunkKVCache

```python
@dataclass
class ChunkKVCache:
    """Chunk KV Cache 数据结构"""

    # 基本信息
    chunk_hash: ChunkHash
    num_tokens: int

    # Block 信息
    block_ids: list[int]              # 物理块 ID 列表
    block_size: int

    # 位置信息（缓存时）
    cached_position_start: int        # 缓存时的起始位置（虚拟位置）
    cached_position_end: int

    # RoPE 缓存
    cos_cache: torch.Tensor | None    # cos 缓存
    sin_cache: torch.Tensor | None    # sin 缓存

    # 元数据
    created_at: float                 # 创建时间戳
    last_accessed: float              # 最后访问时间
    access_count: int                 # 访问次数

    # 引用计数
    ref_count: int = 0                # 当前引用计数
```

**设计要点**：
- 缓存时使用虚拟位置 `[0, num_tokens)`
- RoPE 缓存可复用，减少重映射开销
- 元数据支持 LRU 淘汰策略

#### 3. ReindexedChunkKV

```python
@dataclass
class ReindexedChunkKV:
    """重索引后的 Chunk KV Cache"""

    # 原始 cache（不变）
    original_cache: ChunkKVCache

    # 新的位置信息
    new_position_start: int
    new_position_end: int
    new_positions: torch.Tensor       # 位置序列

    # 重计算的 RoPE
    cos: torch.Tensor
    sin: torch.Tensor

    # 新的 slot_mapping
    slot_mapping: torch.Tensor

    # 合并信息
    offset_in_merged_kv: int          # 在合并后的 KV 中的偏移
```

**设计要点**：
- 保留原始 cache 引用，避免 KV 数据拷贝
- 仅更新位置相关元数据
- 零拷贝重映射，最小化开销

#### 4. ChunkedPrompt

```python
@dataclass
class ChunkedPrompt:
    """分块提示词结构"""

    sys_prompt: list[int]             # 系统提示词 tokens
    chunks: list[list[int]]           # 各个 chunk tokens
    user_question: list[int]          # 用户问题 tokens
    separator: str = "##"             # 分隔符

    def validate(self) -> bool:
        """验证提示词结构"""
        if not self.sys_prompt:
            raise ValueError("sys_prompt cannot be empty")
        if not self.chunks:
            raise ValueError("At least one chunk is required")
        if not self.user_question:
            raise ValueError("user_question cannot be empty")
        return True

    def get_total_tokens(self) -> int:
        """计算总 token 数量"""
        total = len(self.sys_prompt) + len(self.user_question)
        total += sum(len(chunk) for chunk in self.chunks)
        # 加上分隔符 tokens
        total += len(self.chunks) * 3  # 假设 "##" 占 2 个 tokens + 1 个 space
        return total
```

#### 5. ChunkCacheMetrics

```python
@dataclass
class ChunkCacheMetrics:
    """Chunk Cache 性能指标"""

    # 查询统计
    lookup_count: int = 0
    hit_count: int = 0
    miss_count: int = 0

    # 内存统计
    total_blocks: int = 0
    used_blocks: int = 0
    cached_chunks: int = 0

    # 时间统计
    avg_lookup_time_ms: float = 0.0
    avg_reindex_time_ms: float = 0.0
    avg_compute_time_ms: float = 0.0

    def hit_rate(self) -> float:
        """计算缓存命中率"""
        if self.lookup_count == 0:
            return 0.0
        return self.hit_count / self.lookup_count

    def memory_utilization(self) -> float:
        """计算内存利用率"""
        if self.total_blocks == 0:
            return 0.0
        return self.used_blocks / self.total_blocks
```

---

## 接口设计

### ChunkCacheManager 接口

```python
class ChunkCacheManager:
    """位置无关的 Chunk KV Cache 管理器"""

    def __init__(
        self,
        config: ChunkCacheConfig,
        kv_cache_spec: KVCacheSpec,
        block_size: int,
        ascend_allocator: CaMemAllocator | None = None,
    ):
        """
        初始化 Chunk Cache Manager

        Args:
            config: Chunk cache 配置
            kv_cache_spec: KV cache 规范
            block_size: Block 大小（tokens）
            ascend_allocator: 昇腾内存分配器（可选）
        """
        pass

    def get_or_compute_chunk(
        self,
        chunk_tokens: list[int],
        position_offset: int,
        model_runner: ModelRunner,
        sys_prompt_kv: KVCache | None = None,
    ) -> ReindexedChunkKV:
        """
        获取或计算 chunk KV cache

        Args:
            chunk_tokens: Chunk token IDs
            position_offset: 目标位置偏移
            model_runner: 模型运行器（用于计算）
            sys_prompt_kv: 系统提示词 KV cache（可选）

        Returns:
            ReindexedChunkKV: 重索引后的 chunk KV cache

        Raises:
            MemoryError: 内存不足
        """
        pass

    def compute_chunk_hash(
        self,
        tokens: list[int],
    ) -> ChunkHash:
        """计算 chunk 内容哈希（位置无关）"""
        pass

    def reindex_chunk_positions(
        self,
        chunk_kv: ChunkKVCache,
        new_position_offset: int,
    ) -> ReindexedChunkKV:
        """重索引 chunk 位置"""
        pass

    def evict_lru(self) -> None:
        """淘汰 LRU chunk"""
        pass

    def get_metrics(self) -> ChunkCacheMetrics:
        """获取性能指标"""
        pass

    def clear_cache(self) -> None:
        """清空所有缓存"""
        pass
```

### PositionReindexer 接口

```python
class PositionReindexer:
    """位置编码重映射器"""

    def __init__(
        self,
        rope_config: RoPEConfig,
        max_position: int,
        position_strategy: Literal["virtual", "absolute"] = "virtual",
    ):
        """
        初始化位置重映射器

        Args:
            rope_config: RoPE 配置
            max_position: 最大位置
            position_strategy: 位置策略
        """
        pass

    def reindex_chunk_kv(
        self,
        chunk_kv_cache: ChunkKVCache,
        new_position_offset: int,
        original_position_offset: int = 0,
    ) -> ReindexedChunkKV:
        """
        重索引 chunk KV cache

        Args:
            chunk_kv_cache: 原始 chunk KV cache
            new_position_offset: 新的位置偏移
            original_position_offset: 原始位置偏移

        Returns:
            ReindexedChunkKV: 重索引后的 chunk KV cache
        """
        pass

    def compute_rope_for_positions(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算指定位置的 RoPE

        Args:
            positions: 位置张量

        Returns:
            (cos, sin): RoPE cos 和 sin 张量
        """
        pass

    def can_reuse_rope(
        self,
        old_offset: int,
        new_offset: int,
    ) -> bool:
        """
        判断是否可以复用 RoPE 缓存

        Args:
            old_offset: 原始偏移
            new_offset: 新偏移

        Returns:
            bool: 是否可以复用
        """
        pass
```

### GPUModelRunner 扩展接口

```python
class GPUModelRunner:
    """扩展的 GPU 模型运行器"""

    def parse_chunked_prompt(
        self,
        prompt: str,
        separator: str = "##",
    ) -> ChunkedPrompt:
        """
        解析分块提示词

        Args:
            prompt: 原始提示词
            separator: 分隔符

        Returns:
            ChunkedPrompt: 分块后的提示词结构
        """
        pass

    def get_or_compute_chunks(
        self,
        chunked_prompt: ChunkedPrompt,
        chunk_cache_manager: ChunkCacheManager,
    ) -> list[ReindexedChunkKV]:
        """
        获取或计算所有 chunks

        Args:
            chunked_prompt: 分块提示词
            chunk_cache_manager: Chunk cache 管理器

        Returns:
            list[ReindexedChunkKV]: 重索引后的 chunk KV caches
        """
        pass

    def merge_kv_caches(
        self,
        sys_kv: KVCache,
        chunk_kvs: list[ReindexedChunkKV],
        question_kv: KVCache,
    ) -> MergedKVCache:
        """
        合并多源 KV cache

        Args:
            sys_kv: 系统提示词 KV
            chunk_kvs: Chunks KV 列表
            question_kv: 问题 KV

        Returns:
            MergedKVCache: 合并后的 KV cache
        """
        pass
```

---

## 算法设计

### 1. Chunk Hash 计算算法

```python
def compute_chunk_hash(tokens: list[int]) -> ChunkHash:
    """
    计算 chunk 内容哈希（位置无关）

    算法:
    1. 使用 XXHash128 或 SHA256 计算哈希
    2. 不包含位置信息，仅基于 token 内容
    3. 附加 token 数量和 block 数量信息

    时间复杂度: O(n)，n = len(tokens)
    空间复杂度: O(1)
    """
    # 选择哈希算法
    if config.chunk_hash_algo == "xxhash":
        import xxhash
        hash_bytes = xxhash.xxh128(tokens).digest()
    else:  # sha256
        import hashlib
        token_bytes = tuple(tokens).__repr__().encode()
        hash_bytes = hashlib.sha256(token_bytes).digest()

    # 计算 block 数量
    num_blocks = (len(tokens) + block_size - 1) // block_size

    return ChunkHash(
        hash_bytes=hash_bytes,
        token_count=len(tokens),
        num_blocks=num_blocks,
    )
```

### 2. 位置重映射算法

```python
def reindex_chunk_kv(
    chunk_kv_cache: ChunkKVCache,
    new_position_offset: int,
) -> ReindexedChunkKV:
    """
    Chunk KV Cache 位置重映射算法

    算法步骤:
    1. 计算新的位置序列
    2. 判断是否需要重新计算 RoPE
    3. 复用或重计算 RoPE cos/sin
    4. 重计算 slot_mapping
    5. KV cache 数据保持不变

    关键优化:
    - 对于标准 RoPE，可以复用 cos/sin 缓存
    - 仅更新元数据，避免 KV 数据拷贝

    时间复杂度: O(n)，n = chunk_kv_cache.num_tokens
    空间复杂度: O(n)，用于存储新的 positions 和 slot_mapping
    """
    num_tokens = chunk_kv_cache.num_tokens

    # Step 1: 计算新的位置序列
    new_positions = torch.arange(
        new_position_offset,
        new_position_offset + num_tokens,
        dtype=torch.long,
        device=chunk_kv_cache.cos_cache.device,
    )

    # Step 2: 判断是否可以复用 RoPE
    if can_reuse_rope(chunk_kv_cache, new_position_offset):
        # 复用 cos/sin 缓存
        cos, sin = get_reused_rope(
            chunk_kv_cache.cos_cache,
            chunk_kv_cache.sin_cache,
            new_position_offset,
        )
    else:
        # 重计算 RoPE
        cos, sin = compute_rope_for_positions(new_positions)

    # Step 3: 重计算 slot_mapping
    # slot_mapping[i] = block_id * block_size + block_offset
    slot_mapping = compute_slot_mapping(
        chunk_kv_cache.block_ids,
        new_positions,
        block_size,
    )

    return ReindexedChunkKV(
        original_cache=chunk_kv_cache,
        new_position_start=new_position_offset,
        new_position_end=new_position_offset + num_tokens,
        new_positions=new_positions,
        cos=cos,
        sin=sin,
        slot_mapping=slot_mapping,
    )
```

### 3. Chunk Cache 查找算法

```python
def get_or_compute_chunk(
    chunk_tokens: list[int],
    position_offset: int,
) -> ReindexedChunkKV:
    """
    Chunk Cache 查找算法

    算法流程:
    1. 计算内容哈希
    2. 在哈希索引中查找
    3. 如果命中，重索引并返回
    4. 如果未命中，计算并缓存

    时间复杂度:
    - 缓存命中: O(1) + O(n) 重索引
    - 缓存未命中: O(n) 计算 + O(n) 重索引

    空间复杂度: O(n) 用于存储 KV cache
    """
    # Step 1: 计算内容哈希
    chunk_hash = compute_chunk_hash(chunk_tokens)

    # Step 2: 查找缓存
    start_time = time.time()
    cached_chunk = chunk_hash_index.lookup(chunk_hash)

    if cached_chunk is not None:
        # 缓存命中
        metrics.hit_count += 1
        metrics.lookup_count += 1

        # 更新 LRU
        lru_cache.touch(chunk_hash)

        # 重索引到目标位置
        reindexed = reindex_chunk_kv(cached_chunk, position_offset)

        # 更新访问时间
        cached_chunk.last_accessed = time.time()
        cached_chunk.access_count += 1

        return reindexed

    # Step 3: 缓存未命中
    metrics.miss_count += 1
    metrics.lookup_count += 1

    # 检查内存是否足够
    if not block_pool.has_enough_blocks(chunk_hash.num_blocks):
        # 淘汰 LRU chunks
        evict_lru_until_enough(chunk_hash.num_blocks)

    # Step 4: 计算 chunk KV cache
    chunk_kv = compute_chunk_kv(
        chunk_tokens,
        position_offset=0,  # 使用虚拟位置
    )

    # Step 5: 存储到缓存
    chunk_hash_index.insert(chunk_hash, chunk_kv)

    # Step 6: 重索引到目标位置
    reindexed = reindex_chunk_kv(chunk_kv, position_offset)

    return reindexed
```

### 4. 多源 KV Cache 合并算法

```python
def merge_kv_caches(
    sys_kv: KVCache,
    chunk_kvs: list[ReindexedChunkKV],
    question_kv: KVCache,
) -> MergedKVCache:
    """
    多源 KV Cache 合并算法

    合并顺序: sys_prompt -> chunk1 -> chunk2 -> ... -> user_question

    算法步骤:
    1. 计算 KV cache 在合并序列中的位置
    2. 合并 slot_mapping
    3. 合并 positions
    4. 创建统一的 block_table

    关键点:
    - 每个 chunk 的位置编码已经重映射
    - 物理块可能不连续，通过 slot_mapping 映射
    - 最终合并成连续的逻辑 KV 序列

    时间复杂度: O(n)，n = 总 token 数
    空间复杂度: O(n)，用于存储合并后的元数据
    """
    all_kvs = [sys_kv] + chunk_kvs + [question_kv]
    merged_slots = []
    merged_positions = []
    merged_block_ids = []

    current_position = 0

    # 按顺序合并所有 KV caches
    for kv in all_kvs:
        if isinstance(kv, ReindexedChunkKV):
            # Chunk KV cache
            num_tokens = kv.original_cache.num_tokens
            merged_slots.extend(kv.slot_mapping.tolist())
            merged_positions.extend(kv.new_positions.tolist())
            merged_block_ids.extend(kv.original_cache.block_ids)
        else:
            # Standard KV cache
            num_tokens = kv.num_tokens
            merged_slots.extend(kv.slot_mapping.tolist())
            merged_positions.extend(kv.positions.tolist())
            merged_block_ids.extend(kv.block_ids)

        current_position += num_tokens

    return MergedKVCache(
        slot_mapping=torch.tensor(merged_slots),
        positions=torch.tensor(merged_positions),
        block_ids=merged_block_ids,
        total_tokens=current_position,
    )
```

---

## 性能分析

### 性能模型

#### 1. 缓存命中率模型

假设 RAG 应用中：
- 文档库大小: N 个文档
- 每个 query 检索: k 个文档 (k << N)
- 文档重复率: p (0 <= p <= 1)

理论缓存命中率:
```
HitRate ≈ 1 - (1 - p)^k
```

例如：
- p = 0.3 (30% 文档重复)
- k = 5 (检索 5 个文档)
- HitRate ≈ 1 - 0.7^5 ≈ 83%

#### 2. 性能加速比

定义:
- T_baseline: 无 chunk cache 的推理时间
- T_with_cache: 有 chunk cache 的推理时间
- α: chunk 计算时间占比
- β: chunk 命中率

加速比:
```
Speedup = T_baseline / T_with_cache
        = 1 / (1 - α * β)
```

例如：
- α = 0.6 (60% 时间用于 chunk 计算)
- β = 0.8 (80% 命中率)
- Speedup ≈ 1 / (1 - 0.6 * 0.8) ≈ 2.0x

#### 3. 内存开销

Chunk Cache 内存开销:
```
Memory = num_chunks * chunk_tokens * head_dim * num_kv_heads * dtype_bytes
```

例如：
- 1000 chunks, 每个 4096 tokens
- head_dim = 128, num_kv_heads = 32
- dtype = bfloat16 (2 bytes)

Memory = 1000 * 4096 * 128 * 32 * 2 = 32 GB

### 性能优化策略

#### 1. 内存优化
- **LRU 淘汰**: 自动淘汰不常用 chunks
- **压缩存储**: 对冷数据启用压缩
- **Sleep/Wake**: 利用 CaMemAllocator 卸载到 CPU

#### 2. 计算优化
- **RoPE 缓存复用**: 避免重复计算 cos/sin
- **零拷贝重映射**: 仅更新元数据
- **批量处理**: 批量处理多个 chunks

#### 3. 通信优化（多卡）
- **智能分布**: 每个 GPU 缓存不同 chunks
- **异步通信**: 计算与通信重叠
- **本地优先**: 优先使用本地缓存

---

## 测试策略

### 单元测试

#### 1. ChunkHashIndex 测试
```python
def test_chunk_hash_computation():
    """测试 chunk hash 计算"""
    tokens1 = [1, 2, 3, 4, 5]
    tokens2 = [1, 2, 3, 4, 5]
    tokens3 = [1, 2, 3, 4, 6]

    hash1 = compute_chunk_hash(tokens1)
    hash2 = compute_chunk_hash(tokens2)
    hash3 = compute_chunk_hash(tokens3)

    assert hash1 == hash2  # 相同内容，相同 hash
    assert hash1 != hash3  # 不同内容，不同 hash
```

#### 2. PositionReindexer 测试
```python
def test_position_reindexing():
    """测试位置重映射"""
    chunk_kv = create_mock_chunk_kv(num_tokens=100)

    # 重索引到不同位置
    reindexed1 = reindexer.reindex_chunk_kv(chunk_kv, 0)
    reindexed2 = reindexer.reindex_chunk_kv(chunk_kv, 1000)

    # 验证位置正确
    assert reindexed1.new_position_start == 0
    assert reindexed2.new_position_start == 1000

    # 验证 KV 数据不变
    assert torch.equal(reindexed1.original_cache.kv_data,
                      reindexed2.original_cache.kv_data)
```

#### 3. ChunkBlockPool 测试
```python
def test_block_pool_allocation():
    """测试 block pool 分配"""
    pool = ChunkBlockPool(num_blocks=100, block_size=16)

    # 分配 10 blocks
    blocks1 = pool.allocate_blocks(10)
    assert len(blocks1) == 10

    # 再分配 20 blocks
    blocks2 = pool.allocate_blocks(20)
    assert len(blocks2) == 20

    # 释放前 10 blocks
    pool.free_blocks(blocks1)

    # 验证可用 blocks
    assert pool.get_num_free_blocks() == 10
```

### 集成测试

#### 1. 端到端 Chunk Cache 测试
```python
def test_end_to_end_chunk_cache():
    """测试完整的 chunk cache 流程"""
    # 初始化
    llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",
              enable_chunk_cache=True)

    # 第一次请求（缓存未命中）
    prompt1 = "You are helpful.##Doc1##What is this?"
    outputs1 = llm.generate([prompt1])

    # 验证 chunk 被缓存
    stats = llm.get_chunk_cache_stats()
    assert stats.cached_chunks == 1
    assert stats.miss_count == 1

    # 第二次请求（缓存命中）
    prompt2 = "You are helpful.##Doc1##Tell me more"
    outputs2 = llm.generate([prompt2])

    # 验证缓存命中
    stats = llm.get_chunk_cache_stats()
    assert stats.hit_count == 1
```

#### 2. 位置无关性测试
```python
def test_position_agnostic_caching():
    """测试位置无关性"""
    chunk = "This is a document chunk."

    # 同一个 chunk 在不同位置
    prompt1 = f"Sys.##{chunk}##Question1?"
    prompt2 = f"Different system prompt.##{chunk}##Question2?"

    outputs1 = llm.generate([prompt1])
    outputs2 = llm.generate([prompt2])

    # 验证 chunk 缓存命中
    stats = llm.get_chunk_cache_stats()
    assert stats.hit_count >= 1  # chunk 应该命中
```

### 性能测试

#### 1. 缓存命中率测试
```python
def test_cache_hit_rate():
    """测试不同场景下的缓存命中率"""
    # 场景 1: 高重复率
    docs = load_documents(repetition_rate=0.5)
    hit_rate = measure_hit_rate(docs)
    assert hit_rate > 0.7

    # 场景 2: 低重复率
    docs = load_documents(repetition_rate=0.1)
    hit_rate = measure_hit_rate(docs)
    assert hit_rate > 0.2
```

#### 2. 性能加速测试
```python
def test_performance_speedup():
    """测试性能提升"""
    # Baseline (无 chunk cache)
    llm_baseline = LLM(model=..., enable_chunk_cache=False)
    time_baseline = benchmark(llm_baseline, test_prompts)

    # With chunk cache
    llm_cached = LLM(model=..., enable_chunk_cache=True)
    time_cached = benchmark(llm_cached, test_prompts)

    speedup = time_baseline / time_cached
    assert speedup > 2.0  # 至少 2x 加速
```

---

## 部署与运维

### 配置指南

#### 基础配置
```python
from vllm import LLM, ChunkCacheConfig

# 推荐配置（RAG 应用）
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_chunk_cache=True,
    chunk_cache_config=ChunkCacheConfig(
        chunk_cache_gpu_memory_utilization=0.15,  # 15% GPU 用于 chunk cache
        max_chunks=500,                           # 最多缓存 500 个 chunks
        max_chunk_tokens=4096,                    # 单个 chunk 最大 4K tokens
        chunk_separator="##",                     # 分隔符
        chunk_hash_algo="xxhash",                 # 哈希算法
        position_strategy="virtual",              # 位置策略
        cache_policy="lru",                       # 缓存策略
        log_chunk_cache_stats=True,               # 记录统计信息
    ),
)
```

#### 内存规划
```
总 GPU 内存 = 100%
├── 模型权重: ~40%
├── 主 KV Cache: ~45%
└── Chunk Cache: ~15%

Chunk Cache 大小计算:
  假设 8GB GPU 内存用于 chunk cache
  单个 chunk (4K tokens): 4K * 128 * 32 * 2B = 32MB
  可缓存 chunks: 8GB / 32MB ≈ 250 chunks
```

### 监控指标

#### 关键指标
1. **缓存命中率** (Hit Rate)
   - 目标: > 70%
   - 告警: < 50%

2. **内存利用率** (Memory Utilization)
   - 目标: 70-90%
   - 告警: > 95% (接近满载)

3. **平均查询时间** (Avg Lookup Time)
   - 目标: < 1ms
   - 告警: > 5ms

4. **重映射时间** (Reindex Time)
   - 目标: < 10ms (4K tokens)
   - 告警: > 50ms

#### 监控接口
```python
# 获取实时统计
stats = llm.get_chunk_cache_stats()
print(f"Hit Rate: {stats.hit_rate:.2%}")
print(f"Memory Used: {stats.used_blocks} / {stats.total_blocks} blocks")
print(f"Avg Lookup: {stats.avg_lookup_time_ms:.2f} ms")
```

### 故障排查

#### 常见问题

**问题 1: 缓存命中率低**
```
症状: hit_rate < 50%

可能原因:
1. chunk_separator 配置错误
2. 文档重复率低
3. max_chunks 设置过小

解决方案:
1. 检查 prompt 格式
2. 增加 max_chunks
3. 调整 LRU 淘汰策略
```

**问题 2: 内存不足**
```
症状: OOM 错误

可能原因:
1. chunk_cache_gpu_memory_utilization 过大
2. 单个 chunk 超过 max_chunk_tokens

解决方案:
1. 降低 chunk_cache_gpu_memory_utilization
2. 增加 max_chunk_tokens
3. 启用 sleep/wake 机制
```

**问题 3: 输出不一致**
```
症状: 有/无 chunk cache 输出不同

可能原因:
1. 位置重映射错误
2. RoPE 计算错误

解决方案:
1. 检查 position_strategy 配置
2. 验证 RoPE 类型支持
3. 运行单元测试验证
```

---

## 实施计划

### Phase 1: 核心基础设施（2-3 周）

**目标**：基本 chunk cache 架构

**任务**：
- [ ] 创建 `chunk_cache_manager.py`
- [ ] 创建 `chunk_block_pool.py`
- [ ] 创建 `chunk_hash_index.py`
- [ ] 创建 `position_reindexer.py`
- [ ] 添加 `ChunkCacheConfig`
- [ ] 基础单元测试

**验收**：
- Chunk cache 可以存储和检索
- 位置重映射基本工作
- 内存隔离验证

### Phase 2: 位置编码与 RoPE（2 周）

**目标**：位置无关的 RoPE 处理

**任务**：
- [ ] PositionReindexer 完整实现
- [ ] 虚拟位置策略
- [ ] RoPE cos/sin 缓存管理
- [ ] 不同 RoPE 类型适配
- [ ] 集成测试

**验收**：
- Chunk 在不同位置复用时输出一致
- RoPE 计算正确
- 性能提升 >2x

### Phase 3: 昇腾 NPU 优化（2-3 周）

**目标**：Ascend 硬件性能优化

**任务**：
- [ ] CaMemAllocator 集成（标签化内存池）
- [ ] Sleep/wake 生命周期管理
- [ ] SFA/MLA 适配
- [ ] ACL 图编译优化
- [ ] HCCL 多卡同步

**验收**：
- 在 Ascend NPU 上稳定运行
- 性能接近理论最优
- 内存使用可控

### Phase 4: API 与用户体验（1-2 周）

**目标**：易用的 API

**任务**：
- [ ] 扩展 LLM 类
- [ ] ChunkedPrompt 辅助类
- [ ] 统计信息 API
- [ ] OpenAI 兼容 API
- [ ] 文档和示例

**验收**：
- API 简单易用
- 向后兼容
- 文档完整

### Phase 5: 测试与优化（2 周）

**目标**：生产就绪

**任务**：
- [ ] 单元测试覆盖率 >80%
- [ ] 集成测试（真实 RAG workload）
- [ ] 压力测试
- [ ] 性能 benchmark
- [ ] 稳定性测试（24h+）
- [ ] 优化迭代

**验收**：
- 所有测试通过
- 性能达标（>2x 加速）
- 无严重 bug

### 关键文件清单

#### 新增文件（核心）

1. `vllm/v1/core/chunk_cache_manager.py` - ChunkCacheManager
2. `vllm/v1/core/chunk_block_pool.py` - ChunkBlockPool
3. `vllm/v1/core/chunk_hash_index.py` - ChunkHashIndex
4. `vllm/v1/worker/position_reindexer.py` - PositionReindexer
5. `vllm/config/chunk_cache.py` - ChunkCacheConfig
6. `vllm/entrypoints/chunk_api.py` - ChunkedPrompt 等

#### 修改文件（集成）

1. `vllm/config/vllm.py` - 添加 chunk_cache_config
2. `vllm/v1/engine/llm_engine.py` - 集成 ChunkCacheManager
3. `vllm/v1/worker/gpu_model_runner.py` - 支持多源 KV cache
4. `vllm_ascend/attention/sfa_v1.py` - Chunk-aware SFA
5. `vllm_ascend/attention/mla_v1.py` - Chunk-aware MLA
6. `vllm_ascend/device_allocator/camem.py` - 标签化内存池集成

### 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 位置重映射不正确 | 生成质量下降 | 严格单元测试，逐步验证，对比 baseline |
| 内存超限 | OOM | 可配置上限，LRU 淘汰，监控告警 |
| Hash 冲突 | 错误缓存复用 | SHA256 默认，可选 token 验证 |
| 重映射开销过大 | 性能提升不明显 | 虚拟位置策略，RoPE 缓存复用，profiling |
| 多卡通信瓶颈 | 扩展性差 | 智能分布，批量同步，异步通信 |
| 与 prefix caching 冲突 | 系统不稳定 | 清晰职责划分，优先级设计，充分测试 |

### 预期收益

**性能提升**：
- 典型 RAG workload（3-5 个 chunks）：**2-5x 加速**
- Chunk 缓存命中率 80%+ 场景：**3-10x 加速**
- TTFT（Time To First Token）减少：**50-80%**

**适用场景**：
- RAG 应用（多文档检索）
- 知识库问答
- 多文档摘要
- 系统提示词 + 多个独立任务

---

**文档结束**
