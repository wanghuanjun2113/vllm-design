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
4. [核心流程详解](#核心流程详解)
5. [接口设计](#接口设计)
6. [算法设计](#算法设计)
7. [虚拟位置策略详解](#虚拟位置策略详解)
8. [性能分析](#性能分析)
9. [测试策略](#测试策略)
10. [部署与运维](#部署与运维)
11. [实施计划](#实施计划)

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

## 核心流程详解

### 端到端处理流程

本节详细说明从用户请求到模型响应的完整处理流程，包括所有关键步骤和数据流转。

#### 场景 1：首次请求（缓存构建）

**输入 Prompt**: `"You are a helpful assistant.##Document chunk 1.##What is this?"`

**详细处理步骤**：

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Prompt 解析 (LLMEngine)                              │
├─────────────────────────────────────────────────────────────┤
│ 输入: "You are a helpful assistant.##Document chunk 1.##What is this?" │
│                                                               │
│ 操作:                                                          │
│   1. tokenizer.encode() 转换为 tokens                         │
│   2. 检测 "##" 分隔符                                         │
│   3. 分割为:                                                  │
│      - sys_prompt: [101, 2057, 2023, 2003, 10406, 3649, 13]   │
│      - chunk1: [Document chunk 1 的 tokens...]                │
│      - user_question: [What, is, this, ? 的 tokens]          │
│                                                               │
│ 输出: ChunkedPrompt 对象                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 处理 sys_prompt (KVCacheManager)                     │
├─────────────────────────────────────────────────────────────┤
│ 操作:                                                          │
│   1. 使用标准 prefix caching 机制                              │
│   2. 计算 sys_prompt 的 KV cache                              │
│   3. 存储到主 KV cache pool                                   │
│                                                               │
│ 输出: sys_prompt_kv (位置: 0-6)                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 处理 chunk1 (ChunkCacheManager)                       │
├─────────────────────────────────────────────────────────────┤
│ 输入: chunk1_tokens, position_offset=7                       │
│                                                               │
│ 操作:                                                          │
│   3.1 计算内容哈希                                             │
│       chunk_hash = xxhash.xxh128(chunk1_tokens).digest()     │
│                                                               │
│   3.2 查找 chunk_hash_index                                  │
│       结果: 未找到 (缓存未命中)                                │
│                                                               │
│   3.3 分配内存                                                │
│       num_blocks = ceil(len(chunk1_tokens) / block_size)     │
│       block_ids = chunk_block_pool.allocate_blocks(num_blocks)│
│                                                               │
│   3.4 计算 KV cache（使用虚拟位置）                           │
│       virtual_position_start = 0                             │
│       virtual_position_end = len(chunk1_tokens)             │
│       chunk1_kv = model_runner.compute_kv(                   │
│           chunk1_tokens,                                    │
│           position_offset=0  # 虚拟位置                        │
│       )                                                       │
│                                                               │
│   3.5 缓存结果                                                │
│       chunk_kv_cache = ChunkKVCache(                         │
│           chunk_hash=chunk_hash,                             │
│           block_ids=block_ids,                               │
│           cached_position_start=0,   # 虚拟位置               │
│           cached_position_end=len(chunk1_tokens),            │
│           cos_cache=computed_cos,                            │
│           sin_cache=computed_sin,                            │
│       )                                                       │
│       chunk_hash_index.insert(chunk_hash, chunk_kv_cache)     │
│                                                               │
│   3.6 重索引到实际位置                                         │
│       actual_position_start = 7  # sys_prompt 长度 + 分隔符   │
│       reindexed_chunk = position_reindexer.reindex(          │
│           chunk_kv_cache,                                    │
│           new_position_offset=7,                             │
│           original_position_offset=0                         │
│       )                                                       │
│                                                               │
│ 输出: ReindexedChunkKV                                        │
│   - new_positions: [7, 8, 9, ..., 7+len(chunk1)-1]           │
│   - slot_mapping: [相应的 slot 映射]                          │
│   - cos, sin: 重新计算的 RoPE                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 处理 user_question                                   │
├─────────────────────────────────────────────────────────────┤
│ 操作:                                                          │
│   1. 计算位置偏移                                               │
│      question_position_start = 7 + len(chunk1) + 2 (分隔符)  │
│   2. 基于合并的 KV (sys + chunk1) 计算 question 的 KV        │
│                                                               │
│ 输出: question_kv                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: 合并 KV Caches                                        │
├─────────────────────────────────────────────────────────────┤
│ 输入:                                                          │
│   - sys_prompt_kv (位置: 0-6)                                 │
│   - reindexed_chunk (位置: 7-7+len(chunk1))                   │
│   - question_kv (位置: 7+len(chunk1)+2-...)                   │
│                                                               │
│ 操作: model_runner.merge_kv_caches()                          │
│   1. 创建合并的 slot_mapping                                   │
│   2. 创建合并的 positions 张量                                 │
│   3. 创建合并的 block_ids 列表                                 │
│                                                               │
│ 输出: MergedKVCache                                           │
│   - slot_mapping: [sys_slots, chunk_slots, question_slots]  │
│   - positions: [0, 1, ..., total_tokens-1]                   │
│   - block_ids: [sys_blocks, chunk_blocks, question_blocks]  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: 模型推理                                              │
├─────────────────────────────────────────────────────────────┤
│ 操作:                                                          │
│   1. 使用合并的 KV cache 执行 attention 计算                  │
│   2. 生成 response tokens                                     │
│                                                               │
│ 输出: Generated tokens                                        │
└─────────────────────────────────────────────────────────────┘
```

#### 场景 2：缓存命中（chunk 复用）

**输入 Prompt**: `"You are a helpful assistant.##Document chunk 1.##Tell me more"`

**详细处理步骤**：

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1-2: 同前（解析 prompt + 处理 sys_prompt）              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 处理 chunk1 (缓存命中)                              │
├─────────────────────────────────────────────────────────────┤
│ 输入: chunk1_tokens, position_offset=7                       │
│                                                               │
│ 操作:                                                          │
│   3.1 计算内容哈希                                             │
│       chunk_hash = xxhash.xxh128(chunk1_tokens).digest()     │
│                                                               │
│   3.2 查找 chunk_hash_index                                  │
│       结果: 找到！chunk_hash 在索引中                         │
│                                                               │
│   3.3 获取缓存的 chunk                                       │
│       cached_chunk = chunk_hash_index[chunk_hash]            │
│       # cached_chunk 的虚拟位置是 [0, len(chunk1))            │
│                                                               │
│   3.4 更新 LRU                                                │
│       lru_cache.touch(chunk_hash)                            │
│       cached_chunk.last_accessed = now()                     │
│       cached_chunk.access_count += 1                         │
│                                                               │
│   3.5 重索引到实际位置                                         │
│       # 注意：KV cache 数据不拷贝，仅更新元数据！              │
│       reindexed_chunk = position_reindexer.reindex(          │
│           cached_chunk,                                      │
│           new_position_offset=7,                             │
│           original_position_offset=0  # 虚拟位置               │
│       )                                                       │
│                                                               │
│       重索引详细步骤:                                         │
│       ┌───────────────────────────────────────────────┐      │
│       │ 1. 计算新位置序列                               │      │
│       │    new_positions = torch.arange(7, 7+len(chunk1))│      │
│       │                                               │      │
│       │ 2. 判断 RoPE 复用                              │      │
│       │    if can_reuse_rope(0, 7):                    │      │
│       │        # 标准RoPE: 可以复用 cos/sin，仅调整位置索引│     │
│       │        cos = cached_chunk.cos_cache[7:]        │      │
│       │        sin = cached_chunk.sin_cache[7:]        │      │
│       │    else:                                        │      │
│       │        # 复杂RoPE: 需要重新计算                │      │
│       │        cos, sin = compute_rope(7, 7+len(chunk1))│      │
│       │                                               │      │
│       │ 3. 重计算 slot_mapping                         │      │
│       │    for i, block_id in enumerate(cached_chunk.block_ids):│
│       │        for j in range(block_size):              │      │
│       │            slot = block_id * block_size + j     │      │
│       │            slot_mapping[i*block_size + j] = slot│      │
│       │    考虑实际位置:                                  │      │
│       │    slot_mapping = adjust_for_position(slot_mapping, 7)│     │
│       └───────────────────────────────────────────────┘      │
│                                                               │
│ 性能优势:                                                      │
│   - 无需重新计算 KV cache (节省大量计算)                        │
│   - 无需内存分配                                              │
│   - 仅需轻量级的元数据更新                                     │
│                                                               │
│ 输出: ReindexedChunkKV (无需计算)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4-6: 同前（处理 question + 合并 + 推理）                │
└─────────────────────────────────────────────────────────────┘
```

### 虚拟位置策略详解

虚拟位置策略是实现位置无关缓存的核心创新。本节详细说明其工作原理。

#### 核心思想

**问题**：如果 chunk 在实际位置计算和缓存，那么相同内容在不同位置时无法复用缓存。

**解决方案**：所有 chunk 在统一的**虚拟位置空间** `[0, chunk_len)` 计算 KV cache，复用时通过位置重映射映射到实际位置。

#### 虚拟位置空间定义

```
虚拟位置空间: [0, ∞)
├── Chunk 1: [0, len(chunk1))
├── Chunk 2: [0, len(chunk2))
├── Chunk 3: [0, len(chunk3))
└── ...

每个 chunk 都从 0 开始计算！
```

#### 位置映射关系表

| Chunk | 缓存时位置（虚拟） | 第一次使用时位置 | 第二次使用时位置 | 第三次使用时位置 |
|-------|-------------------|-----------------|-----------------|-----------------|
| chunk1 | [0, 100) | [10, 110) | [25, 125) | [8, 108) |
| chunk2 | [0, 200) | [112, 312) | [137, 337) | [120, 320) |
| chunk3 | [0, 150) | [315, 465) | [340, 490) | [323, 473) |

**关键优势**：
- chunk1 无论在哪个位置，都使用相同的缓存数据
- 仅需更新元数据（positions, slot_mapping, RoPE）
- KV cache 数据完全复用

#### 虚拟位置策略的数学表示

**缓存时**（虚拟位置）：
```
chunk_tokens = [t1, t2, ..., tn]
virtual_positions = [0, 1, 2, ..., n-1]

# 计算 KV cache
chunk_kv_cache = compute_kv(
    tokens=chunk_tokens,
    positions=virtual_positions,  # 虚拟位置
    cos=compute_rope(virtual_positions),
    sin=compute_rope(virtual_positions)
)

# 存储
cache_key = hash(chunk_tokens)  # 不包含位置信息！
cache[cache_key] = chunk_kv_cache
```

**复用时**（实际位置）：
```
actual_position_start = P  # 实际起始位置
actual_positions = [P, P+1, P+2, ..., P+n-1]

# 重索引
reindexed_kv = reindex(
    cached_kv=chunk_kv_cache,
    new_positions=actual_positions
)

# 操作分解：
# 1. KV 数据：保持不变（复用！）
# 2. positions: [0,1,...,n-1] -> [P,P+1,...,P+n-1]
# 3. cos/sin: 根据 RoPE 类型复用或重计算
# 4. slot_mapping: 重新计算映射到实际物理块
```

#### RoPE 复用策略

RoPE 的复用取决于 RoPE 类型：

**标准 RoPE**（可以复用）：
```python
# RoPE(pos) = cos(pos * theta) + sin(pos * theta)

# 虚拟位置缓存
cos_cache[i] = cos(i * theta) for i in [0, n)
sin_cache[i] = sin(i * theta) for i in [0, n)

# 实际位置复用
cos_actual[i] = cos_cache[(P + i) % max_cache_size]
sin_actual[i] = sin_cache[(P + i) % max_cache_size]
```

**复杂 RoPE**（需要重计算）：
```python
# NTK-aware, YaRN 等变体
# 这些 RoPE 的频率随位置动态变化
# 必须为实际位置重新计算

cos_actual = compute_rope(actual_positions, rope_type="ntk_aware")
sin_actual = compute_rope(actual_positions, rope_type="ntk_aware")
```

### 与 vLLM v1 Engine 集成详解

本节详细说明如何将 Chunk Cache 系统集成到现有的 vLLM v1 Engine 中。

#### 集成点分析

**vLLM v1 Engine 关键组件**：
```
vllm/v1/engine/llm_engine.py          # 主引擎
vllm/v1/engine/core_client.py         # Engine 客户端
vllm/v1/worker/gpu_model_runner.py    # 模型运行器
vllm/v1/core/kv_cache_manager.py      # KV cache 管理
vllm/v1/worker/block_table.py         # Block 表管理
```

#### 集成步骤详解

**Step 1: 扩展 VllmConfig**

```python
# 文件: vllm/config/vllm.py

@dataclass
class VllmConfig:
    # ... 现有字段 ...

    # 新增字段
    chunk_cache_config: ChunkCacheConfig = field(
        default_factory=ChunkCacheConfig
    )
```

**Step 2: 在 LLMEngine 中初始化 ChunkCacheManager**

```python
# 文件: vllm/v1/engine/llm_engine.py

class LLMEngine:
    def __init__(self, vllm_config: VllmConfig):
        # ... 现有初始化 ...

        # 新增: 初始化 ChunkCacheManager
        if vllm_config.chunk_cache_config.enable_chunk_cache:
            from vllm.v1.core.chunk_cache_manager import ChunkCacheManager

            self.chunk_cache_manager = ChunkCacheManager(
                config=vllm_config.chunk_cache_config,
                kv_cache_spec=self.model_config.kv_cache_spec,
                block_size=self.cache_config.block_size,
                ascend_allocator=self._get_ascend_allocator(),  # 可选
            )
        else:
            self.chunk_cache_manager = None
```

**Step 3: 扩展 GPUModelRunner**

```python
# 文件: vllm/v1/worker/gpu_model_runner.py

class GPUModelRunner:
    def __init__(self, ..., chunk_cache_manager=None):
        # ... 现有初始化 ...

        # 新增
        self.chunk_cache_manager = chunk_cache_manager
        if chunk_cache_manager:
            from vllm.v1.worker.position_reindexer import PositionReindexer
            self.position_reindexer = PositionReindexer(
                rope_config=self.model_config.rope_config,
                max_position=self.model_config.max_position,
            )

    def parse_chunked_prompt(
        self,
        prompt_tokens: list[int],
        separator: str = "##",
    ) -> ChunkedPrompt:
        """
        解析分块提示词

        实现:
        1. 查找分隔符 token 位置
        2. 分割 token 序列
        3. 构造 ChunkedPrompt 对象
        """
        # 将 "##" 转换为 tokens
        sep_tokens = self.tokenizer.encode(separator, add_special_tokens=False)

        # 查找所有分隔符位置
        sep_positions = []
        for i in range(len(prompt_tokens) - len(sep_tokens) + 1):
            if prompt_tokens[i:i+len(sep_tokens)] == sep_tokens:
                sep_positions.append(i)

        # 必须至少有 2 个分隔符（3 个部分：sys, chunks..., question）
        if len(sep_positions) < 2:
            raise ValueError(f"Need at least 2 separators, found {len(sep_positions)}")

        # 解析各部分
        sys_prompt = prompt_tokens[:sep_positions[0]]
        chunks = []
        for i in range(len(sep_positions) - 1):
            chunk_start = sep_positions[i] + len(sep_tokens)
            chunk_end = sep_positions[i + 1]
            chunks.append(prompt_tokens[chunk_start:chunk_end])

        question_start = sep_positions[-1] + len(sep_tokens)
        user_question = prompt_tokens[question_start:]

        return ChunkedPrompt(
            sys_prompt=sys_prompt,
            chunks=chunks,
            user_question=user_question,
            separator=separator,
        )

    def get_or_compute_chunks(
        self,
        chunked_prompt: ChunkedPrompt,
    ) -> list[ReindexedChunkKV]:
        """
        获取或计算所有 chunks

        返回: 按顺序排列的重索引后的 chunk KV caches
        """
        if not self.chunk_cache_manager:
            raise RuntimeError("Chunk cache not enabled")

        reindexed_chunks = []
        current_position = len(chunked_prompt.sys_prompt)

        for chunk_idx, chunk_tokens in enumerate(chunked_prompt.chunks):
            # 获取或计算 chunk
            reindexed_chunk = self.chunk_cache_manager.get_or_compute_chunk(
                chunk_tokens=chunk_tokens,
                position_offset=current_position,
                model_runner=self,
                sys_prompt_kv=self.sys_prompt_kv,  # 预先计算的 sys_prompt KV
            )

            reindexed_chunks.append(reindexed_chunk)

            # 更新下一个 chunk 的位置
            # 加上 chunk 长度和分隔符长度
            current_position += len(chunk_tokens) + 2  # 2 = len("##")

        return reindexed_chunks

    def merge_kv_caches(
        self,
        sys_kv: KVCache,
        chunk_kvs: list[ReindexedChunkKV],
        question_kv: KVCache,
    ) -> MergedKVCache:
        """
        合并多源 KV cache

        实现要点:
        1. 按顺序拼接 slot_mapping
        2. 按顺序拼接 positions
        3. 合并 block_ids（可能不连续）
        4. 创建统一的 block_table
        """
        merged_slots = []
        merged_positions = []
        merged_block_ids = []

        # 合并 sys_prompt
        merged_slots.extend(sys_kv.slot_mapping.tolist())
        merged_positions.extend(sys_kv.positions.tolist())
        merged_block_ids.extend(sys_kv.block_ids)

        # 合并所有 chunks
        for chunk_kv in chunk_kvs:
            merged_slots.extend(chunk_kv.slot_mapping.tolist())
            merged_positions.extend(chunk_kv.new_positions.tolist())
            merged_block_ids.extend(chunk_kv.original_cache.block_ids)

        # 合并 user_question
        merged_slots.extend(question_kv.slot_mapping.tolist())
        merged_positions.extend(question_kv.positions.tolist())
        merged_block_ids.extend(question_kv.block_ids)

        return MergedKVCache(
            slot_mapping=torch.tensor(merged_slots, dtype=torch.long),
            positions=torch.tensor(merged_positions, dtype=torch.long),
            block_ids=merged_block_ids,
            total_tokens=len(merged_positions),
        )
```

**Step 4: 修改 EngineCore 的执行流程**

```python
# 文件: vllm/v1/engine/core.py

class EngineCore:
    def execute_request(self, request: Request):
        """
        执行请求的核心流程（集成 chunk cache）
        """
        # 1. 解析 prompt
        prompt_tokens = self.tokenizer.encode(request.prompt)

        # 2. 尝试解析为分块 prompt
        try:
            chunked_prompt = self.model_runner.parse_chunked_prompt(
                prompt_tokens,
                separator=self.chunk_cache_config.chunk_separator,
            )
            is_chunked = True
        except ValueError:
            # 不是分块 prompt，使用标准流程
            is_chunked = False

        if is_chunked and self.chunk_cache_manager:
            # 3a. 使用 chunk cache 流程
            return self._execute_chunked_request(chunked_prompt, request)
        else:
            # 3b. 使用标准流程
            return self._execute_standard_request(prompt_tokens, request)

    def _execute_chunked_request(
        self,
        chunked_prompt: ChunkedPrompt,
        request: Request,
    ):
        """执行分块请求"""
        # Step 1: 计算 sys_prompt KV cache（使用标准 prefix caching）
        sys_kv = self.kv_cache_manager.get_computed_blocks(
            request,
            num_tokens=len(chunked_prompt.sys_prompt),
        )

        # Step 2: 获取或计算所有 chunks
        chunk_kvs = self.model_runner.get_or_compute_chunks(
            chunked_prompt,
        )

        # Step 3: 计算 user_question KV cache
        question_position = (
            len(chunked_prompt.sys_prompt) +
            sum(len(c) for c in chunked_prompt.chunks) +
            len(chunked_prompt.chunks) * 2  # 分隔符
        )
        question_kv = self.model_runner.execute_model(
            prompt_tokens=chunked_prompt.user_question,
            positions=range(question_position, question_position + len(chunked_prompt.user_question)),
            previous_kv=self.model_runner.merge_kv_caches(
                sys_kv, chunk_kvs, None  # question_kv 正在计算
            ),
        )

        # Step 4: 合并所有 KV caches
        merged_kv = self.model_runner.merge_kv_caches(
            sys_kv, chunk_kvs, question_kv
        )

        # Step 5: 生成 response
        output = self.model_runner.generate(
            merged_kv,
            request.sampling_params,
        )

        return output
```

#### 数据流转图

```
用户请求 (Prompt with "##")
    ↓
LLMEngine.execute_request()
    ↓
GPUModelRunner.parse_chunked_prompt()
    ├─→ sys_prompt tokens
    ├─→ chunk1 tokens
    ├─→ chunk2 tokens
    └─→ user_question tokens
    ↓
ChunkCacheManager.get_or_compute_chunk()
    ├─→ compute_chunk_hash()
    ├─→ chunk_hash_index.lookup()
    │   ├─→ HIT: PositionReindexer.reindex() → ReindexedChunkKV
    │   └─→ MISS: compute_chunk_kv() → ChunkKVCache → reindex()
    └─→ 返回 ReindexedChunkKV
    ↓
GPUModelRunner.merge_kv_caches()
    ├─→ sys_prompt KV (from KVCacheManager)
    ├─→ chunk1 KV (reindexed)
    ├─→ chunk2 KV (reindexed)
    └─→ user_question KV (computed)
    ↓
MergedKVCache (统一的 KV cache)
    ↓
Model inference (Attention computation)
    ↓
Generated response
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

## 虚拟位置策略详解

虚拟位置策略是实现位置无关缓存的核心创新。本节提供更深入的实现细节。

### 位置无关性原理

#### 问题背景

**传统 Prefix Caching 的限制**：
```
请求1: "System prompt.##Doc A##Question?"
  → chunk "Doc A" 缓存在位置 [15, 215)

请求2: "Different system.##Doc A##Question?"
  → chunk "Doc A" 在位置 [20, 220)
  → 无法复用缓存！位置不同
```

**虚拟位置策略的解决方案**：
```
所有 chunk 在统一的虚拟位置空间 [0, chunk_len) 计算
复用时通过位置重映射映射到实际位置
```

### 虚拟位置实现细节

#### 1. Chunk 缓存时的虚拟位置处理

```python
def compute_chunk_kv_with_virtual_position(
    chunk_tokens: list[int],
    model_runner: ModelRunner,
) -> ChunkKVCache:
    """
    使用虚拟位置计算 chunk KV cache

    关键点:
    - 所有 chunk 从位置 0 开始计算
    - 不考虑实际使用位置
    - 实现完全的位置无关性
    """
    num_tokens = len(chunk_tokens)

    # 虚拟位置序列：[0, 1, 2, ..., num_tokens-1]
    virtual_positions = torch.arange(
        0, num_tokens,
        dtype=torch.long,
        device=model_runner.device,
    )

    # 计算 RoPE（基于虚拟位置）
    cos, sin = model_runner.compute_rope(virtual_positions)

    # 计算 KV cache（使用虚拟位置）
    chunk_kv = model_runner.model_forward(
        input_ids=torch.tensor(chunk_tokens),
        positions=virtual_positions,
        cos=cos,
        sin=sin,
    )

    # 分配物理内存块
    num_blocks = (num_tokens + block_size - 1) // block_size
    block_ids = chunk_block_pool.allocate_blocks(num_blocks)

    # 计算 slot mapping（基于虚拟位置）
    # slot_mapping[i] = block_id * block_size + offset_in_block
    slot_mapping = torch.zeros(num_tokens, dtype=torch.long)
    for i, token_pos in enumerate(virtual_positions):
        block_idx = i // block_size
        offset_in_block = i % block_size
        slot_mapping[i] = block_ids[block_idx] * block_size + offset_in_block

    return ChunkKVCache(
        chunk_hash=compute_chunk_hash(chunk_tokens),
        num_tokens=num_tokens,
        block_ids=block_ids,
        block_size=block_size,
        cached_position_start=0,        # 虚拟起始位置
        cached_position_end=num_tokens,  # 虚拟结束位置
        cos_cache=cos,                   # 虚拟位置的 RoPE
        sin_cache=sin,
        slot_mapping=slot_mapping,       # 虚拟位置的 slot mapping
        created_at=time.time(),
    )
```

#### 2. Chunk 复用时的位置重映射

```python
def reindex_chunk_kv_detailed(
    chunk_kv_cache: ChunkKVCache,
    new_position_offset: int,
    rope_config: RoPEConfig,
) -> ReindexedChunkKV:
    """
    详细的 chunk KV cache 位置重映射

    这是实现位置无关性的关键函数
    """
    num_tokens = chunk_kv_cache.num_tokens
    device = chunk_kv_cache.cos_cache.device

    # === Step 1: 计算新的位置序列 ===
    new_positions = torch.arange(
        new_position_offset,
        new_position_offset + num_tokens,
        dtype=torch.long,
        device=device,
    )

    # === Step 2: 处理 RoPE (关键优化点) ===

    # 判断 RoPE 类型
    if is_standard_rope(rope_config):
        # 标准RoPE: 可以复用 cos/sin 缓存
        # RoPE(pos) = cos(pos * theta) + sin(pos * theta)
        # 周期性：cos 和 sin 只与 (pos % period) 相关

        max_precomputed_pos = chunk_kv_cache.cos_cache.shape[0]

        # 直接索引预计算的 cos/sin
        cos = torch.zeros(num_tokens, dtype=chunk_kv_cache.cos_cache.dtype, device=device)
        sin = torch.zeros(num_tokens, dtype=chunk_kv_cache.sin_cache.dtype, device=device)

        for i in range(num_tokens):
            actual_pos = new_position_offset + i
            if actual_pos < max_precomputed_pos:
                # 在预计算范围内，直接复用
                cos[i] = chunk_kv_cache.cos_cache[actual_pos]
                sin[i] = chunk_kv_cache.sin_cache[actual_pos]
            else:
                # 超出预计算范围，动态计算
                cos_i, sin_i = compute_single_rope(actual_pos, rope_config)
                cos[i] = cos_i
                sin[i] = sin_i

    elif is_ntk_aware_rope(rope_config):
        # NTK-aware RoPE: 频率随位置动态变化
        # 无法复用，必须重新计算
        cos, sin = compute_rope_for_positions(
            new_positions,
            rope_config.rope_type,
            rope_config.base,
        )

    elif is_yarn_rope(rope_config):
        # YaRN RoPE: 同样需要重新计算
        cos, sin = compute_rope_for_positions(
            new_positions,
            rope_type="yarn",
            **rope_config.yarn_params,
        )

    else:
        # 其他RoPE类型：默认重新计算
        cos, sin = compute_rope_for_positions(new_positions, rope_config)

    # === Step 3: 重新计算 slot_mapping ===

    # 关键理解：slot_mapping 将 token 映射到物理内存位置
    # slot_mapping[i] = block_id * block_size + offset_in_block

    # 对于 chunk cache，物理块 ID 不变，但逻辑位置变化
    # 因此需要重新计算 slot_mapping

    new_slot_mapping = torch.zeros(num_tokens, dtype=torch.long, device=device)

    for i in range(num_tokens):
        # 确定这个 token 在 chunk 中的索引
        token_index_in_chunk = i

        # 确定使用哪个物理块
        block_idx = token_index_in_chunk // chunk_kv_cache.block_size
        block_id = chunk_kv_cache.block_ids[block_idx]

        # 确定在块中的偏移
        offset_in_block = token_index_in_chunk % chunk_kv_cache.block_size

        # 计算最终的 slot
        new_slot_mapping[i] = block_id * chunk_kv_cache.block_size + offset_in_block

    # === Step 4: 创建重索引后的 chunk KV ===

    return ReindexedChunkKV(
        original_cache=chunk_kv_cache,  # 保留引用，不拷贝 KV 数据！
        new_position_start=new_position_offset,
        new_position_end=new_position_offset + num_tokens,
        new_positions=new_positions,
        cos=cos,
        sin=sin,
        slot_mapping=new_slot_mapping,
        offset_in_merged_kv=0,  # 稍后在合并时设置
    )
```

### RoPE 复用的数学原理

#### 标准 RoPE 的周期性

标准 RoPE 具有周期性，这是位置复用的数学基础：

```
RoPE(pos, d) = cos(pos * θ_d) + sin(pos * θ_d)

其中 θ_d = 1 / (base ^ (2d / dim))

周期性分析：
- 对于不同的维度 d，频率 θ_d 不同
- 但所有频率都是 2π 的有理数倍
- 因此 RoPE 具有周期性

周期 T_d = 2π / θ_d = 2π * base^(2d/dim)

对于 base=10000, dim=128:
  最小周期 ≈ 2π * 10000^(2*0/128) = 2π ≈ 6.28
  最大周期 ≈ 2π * 10000^(2*64/128) ≈ 2π * 10000 ≈ 62831

因此：cos_cache 和 sin_cache 可以在周期内复用！
```

#### RoPE 缓存复用策略

```python
def get_rope_with_cache_reuse(
    position: int,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    rope_base: float = 10000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    智能复用 RoPE 缓存

    策略:
    1. 如果 position < cache_size: 直接复用
    2. 如果 position >= cache_size:
       - 计算在周期内的等效位置
       - 复用缓存
    3. 如果等效位置仍超出范围: 动态计算
    """
    cache_size = cos_cache.shape[0]

    if position < cache_size:
        # 直接复用
        return cos_cache[position], sin_cache[position]

    # 计算周期
    # RoPE 的最大周期由 base 和维度决定
    # 实际上，由于浮点精度限制，远距离的复用会有精度损失
    # 因此设置一个阈值
    max_reuse_distance = cache_size // 2  # 安全阈值

    if position < cache_size + max_reuse_distance:
        # 可以复用，可能有少量精度损失
        # 使用模运算映射到缓存范围内
        equiv_pos = position % cache_size
        return cos_cache[equiv_pos], sin_cache[equiv_pos]

    else:
        # 位置太远，需要动态计算
        return compute_single_rope(position, rope_base)
```

### 虚拟位置策略的完整示例

#### 示例场景

```
系统配置:
- block_size = 16
- chunk1_tokens = 50 tokens
- chunk1_hash = 0xabc123...

第一次使用 (Request 1):
  sys_prompt = "You are helpful." (7 tokens)
  prompt = "You are helpful.##Chunk A##Question?"

  Chunk A 的实际位置: [9, 58) (7 + 2(separator) + 0 to 49)

  缓存操作:
    1. 计算 hash(chunk1_tokens) = 0xabc123...
    2. 缓存未命中
    3. 在虚拟位置 [0, 50) 计算 KV cache
       - positions = [0, 1, 2, ..., 49]
       - cos = [cos(0), cos(1), ..., cos(49)]
       - sin = [sin(0), sin(1), ..., sin(49)]
       - KV = computed_values
    4. 分配 4 个 blocks (50/16 = 3.125 → 4)
       block_ids = [100, 101, 102, 103]
    5. 存储:
       chunk_cache_index[0xabc123...] = {
           'kv_data': [物理 KV 数据],
           'block_ids': [100, 101, 102, 103],
           'virtual_positions': [0, 1, ..., 49],
           'cos_cache': [cos(0), ..., cos(49)],
           'sin_cache': [sin(0), ..., sin(49)],
       }
    6. 重索引到实际位置 [9, 58):
       - new_positions = [9, 10, ..., 58]
       - 如果是标准 RoPE:
           cos_actual = cos_cache[9:59]  # 直接切片
           sin_actual = sin_cache[9:59]
       - new_slot_mapping = 重新计算
    7. 返回 ReindexedChunkKV

第二次使用 (Request 2):
  sys_prompt = "Different system." (10 tokens)
  prompt = "Different system.##Chunk A##Question?"

  Chunk A 的实际位置: [12, 61) (10 + 2 + 0 to 49)

  缓存操作:
    1. 计算 hash(chunk1_tokens) = 0xabc123... (相同！)
    2. 缓存命中！
    3. 获取缓存:
       cached_chunk = chunk_cache_index[0xabc123...]
    4. 重索引到实际位置 [12, 61]:
       - new_positions = [12, 13, ..., 61]
       - 如果是标准 RoPE:
           cos_actual = cos_cache[12:62]  # 直接切片
           sin_actual = sin_cache[12:62]
       - new_slot_mapping = 重新计算 (block_ids 不变)
    5. 返回 ReindexedChunkKV (无需计算 KV！)
```

### 性能优势量化分析

#### 计算开销对比

**缓存未命中（首次计算）**：
```
Token 数量: n = 4096 (典型 chunk)

操作分解:
1. 计算 chunk_hash:        O(n) × hash_cost  (约 0.01ms)
2. 分配内存:              O(n) × alloc_cost  (约 0.1ms)
3. 计算 KV cache:         O(n) × compute_cost (约 50ms)
4. 计算 RoPE:             O(n) × rope_cost    (约 5ms)
5. 计算 slot_mapping:      O(n) × map_cost     (约 0.1ms)
---------------------------------------------------
总时间: ~55ms (主要在 KV cache 计算)
```

**缓存命中（位置重映射）**：
```
Token 数量: n = 4096

操作分解:
1. 计算 chunk_hash:        O(n) × hash_cost  (约 0.01ms)
2. 查找缓存:              O(1)               (约 0.001ms)
3. 更新 LRU:              O(1)               (约 0.001ms)
4. 位置重映射:            O(n) × reindex_cost (约 0.5ms)
   - 计算 new_positions:  O(n)               (约 0.01ms)
   - RoPE 复用:            O(n)               (约 0.1ms, 标准RoPE)
   - 或 RoPE 重计算:        O(n) × rope_cost  (约 5ms, 复杂RoPE)
   - 计算 slot_mapping:    O(n)               (约 0.1ms)
5. KV 数据:               零拷贝！           (0ms)
---------------------------------------------------
总时间: ~0.5-5.5ms (取决于 RoPE 类型)

加速比: 55ms / 0.5ms ≈ 110x (标准 RoPE)
      55ms / 5.5ms ≈ 10x  (复杂 RoPE)
```

### 边界情况处理

#### 1. 超长 Chunk

```python
def handle_oversized_chunk(
    chunk_tokens: list[int],
    max_chunk_tokens: int,
) -> list[list[int]]:
    """
    处理超长 chunk

    策略:
    - 如果 chunk 超过 max_chunk_tokens，拆分为多个子 chunk
    - 每个子 chunk 独立缓存
    - 使用时按顺序组合
    """
    if len(chunk_tokens) <= max_chunk_tokens:
        return [chunk_tokens]

    # 拆分
    sub_chunks = []
    for i in range(0, len(chunk_tokens), max_chunk_tokens):
        sub_chunk = chunk_tokens[i:i+max_chunk_tokens]
        sub_chunks.append(sub_chunk)

    logger.warning(
        f"Chunk too large ({len(chunk_tokens)} tokens), "
        f"split into {len(sub_chunks)} sub-chunks"
    )

    return sub_chunks
```

#### 2. 位置溢出

```python
def handle_position_overflow(
    position: int,
    max_position: int,
) -> int:
    """
    处理位置溢出

    某些模型有最大位置限制（如 8192）
    如果实际位置超过限制，需要特殊处理
    """
    if position < max_position:
        return position

    # 策略 1: 报错（保守）
    raise ValueError(
        f"Position {position} exceeds max_position {max_position}"
    )

    # 策略 2: 模运算（激进，可能有精度损失）
    # return position % max_position

    # 策略 3: 使用 RoPE scaling（推荐）
    # scaled_pos = scale_position(position, max_position)
    # return scaled_pos
```

---

## 昇腾 NPU 特定优化详解

本节详细说明如何在昇腾 NPU 上优化 Chunk Cache 系统。

### 昇腾硬件特性利用

#### 1. CaMemAllocator 深度集成

**CaMemAllocator 简介**：
- 昇腾专用的内存分配器
- 支持标签化内存池管理
- 支持 sleep/wake 生命周期（卸载到 CPU）
- 单例模式，全局管理

**标签化内存池实现**：

```python
# 文件: vllm_ascend/device_allocator/camem.py

class CaMemAllocator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.memory_pools: Dict[str, MemoryPool] = {}
        self.current_tag = "default"

    def use_memory_pool(self, tag: str):
        """
        使用指定标签的内存池

        上下文管理器，用于隔离不同用途的内存
        """
        @contextmanager
        def _context():
            old_tag = self.current_tag
            self.current_tag = tag

            # 如果标签池不存在，创建新池
            if tag not in self.memory_pools:
                self.memory_pools[tag] = MemoryPool(
                    tag=tag,
                    allocator=self._create_allocator(),
                )

            yield

            self.current_tag = old_tag

        return _context()

    def allocate(self, size: int, tag: str = None) -> torch.Tensor:
        """从指定标签池分配内存"""
        pool_tag = tag or self.current_tag
        pool = self.memory_pools.get(pool_tag)

        if pool is None:
            raise ValueError(f"Memory pool '{pool_tag}' not found")

        return pool.allocate(size)
```

**Chunk Cache 集成**：

```python
# 文件: vllm/v1/core/chunk_block_pool.py

class ChunkBlockPool:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        ascend_allocator: CaMemAllocator = None,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.ascend_allocator = ascend_allocator

        if ascend_allocator:
            # 使用 "chunk_cache" 标签创建独立内存池
            self.pool_tag = "chunk_cache"

            with ascend_allocator.use_memory_pool(tag=self.pool_tag):
                # 预分配所有 block
                total_size = num_blocks * block_size * kv_dim * dtype_bytes
                self.kv_cache = torch.empty(
                    (num_blocks, block_size, num_kv_heads, head_dim),
                    dtype=dtype,
                    device="npu",
                )
        else:
            # CUDA fallback
            self.kv_cache = torch.empty(...)
            self.pool_tag = None

    def allocate_blocks(self, num_blocks: int) -> list[int]:
        """分配 blocks"""
        if len(self.free_blocks) < num_blocks:
            raise MemoryError("Insufficient chunk cache blocks")

        allocated = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]

        return allocated

    def get_block_tensor(
        self,
        block_id: int,
    ) -> torch.Tensor:
        """
        获取指定 block 的 tensor

        对于昇腾，从标签化内存池中获取
        """
        if self.ascend_allocator:
            with self.ascend_allocator.use_memory_pool(tag=self.pool_tag):
                return self.kv_cache[block_id]
        else:
            return self.kv_cache[block_id]
```

#### 2. Sleep/Wake 生命周期管理

**使用场景**：
- 低峰期：卸载不常用的 chunk cache 到 CPU
- 高峰期：重新加载到 NPU
- 多租户：不同租户的 chunk cache 独立管理

```python
# 文件: vllm/v1/core/chunk_cache_manager.py

class ChunkCacheManager:
    def sleep_cold_chunks(
        self,
        threshold_access_count: int = 5,
        threshold_time_seconds: float = 3600,
    ) -> int:
        """
        将冷 chunk 卸载到 CPU

        策略:
        - 访问次数 < threshold_access_count
        - 或 最后访问时间 > threshold_time_seconds 前

        Returns:
            卸载的 chunk 数量
        """
        if not self.ascend_allocator:
            logger.warning("Ascend allocator not available, skip sleep")
            return 0

        cold_chunks = []
        current_time = time.time()

        for chunk_hash, chunk_kv in self.chunk_hash_index.items():
            if (chunk_kv.access_count < threshold_access_count or
                current_time - chunk_kv.last_accessed > threshold_time_seconds):
                cold_chunks.append(chunk_hash)

        # 卸载到 CPU
        if cold_chunks:
            with self.ascend_allocator.use_memory_pool(tag="chunk_cache"):
                self.ascend_allocator.sleep(
                    offload_tags=("chunk_cache",),
                    # 仅卸载指定的 chunks
                    block_ids=[
                        bid for chunk_hash in cold_chunks
                        for bid in self.chunk_hash_index[chunk_hash].block_ids
                    ],
                )

            logger.info(f"Slept {len(cold_chunks)} cold chunks")

        return len(cold_chunks)

    def wake_up_chunks(
        self,
        chunk_hashes: list[ChunkHash],
    ) -> bool:
        """
        从 CPU 重新加载 chunks 到 NPU

        Returns:
            是否成功
        """
        if not self.ascend_allocator:
            return False

        try:
            with self.ascend_allocator.use_memory_pool(tag="chunk_cache"):
                self.ascend_allocator.wake_up(
                    tags=["chunk_cache"],
                    block_ids=[
                        bid for chunk_hash in chunk_hashes
                        for bid in self.chunk_hash_index[chunk_hash].block_ids
                    ],
                )
            return True
        except Exception as e:
            logger.error(f"Failed to wake up chunks: {e}")
            return False
```

#### 3. SFA (Sparse Flash Attention) 适配

**SFA 简介**：
- 昇腾优化的稀疏 Flash Attention
- 支持非连续 block IDs
- 支持动态位置
- 集成 RMSNorm 和 RoPE

**SFA 扩展以支持 Chunk Cache**：

```python
# 文件: vllm_ascend/attention/sfa_v1.py

class AscendSFAImpl:
    def forward_with_chunk_kv(
        self,
        query: torch.Tensor,
        chunk_kv_caches: list[ReindexedChunkKV],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        SFA attention with chunk KV cache support

        关键点:
        1. 合并非连续的 chunk KV blocks
        2. 使用 nppu_kv_rmsnorm_rope_cache 操作
        3. 处理位置映射
        """
        # Step 1: 收集所有 block IDs
        all_block_ids = []
        for chunk_kv in chunk_kv_caches:
            all_block_ids.extend(chunk_kv.original_cache.block_ids)

        # Step 2: 合并 positions
        all_positions = []
        for chunk_kv in chunk_kv_caches:
            all_positions.extend(chunk_kv.new_positions.tolist())
        all_positions = torch.tensor(all_positions, dtype=torch.long)

        # Step 3: 合并 cos/sin
        all_cos = torch.cat([chunk_kv.cos for chunk_kv in chunk_kv_caches])
        all_sin = torch.cat([chunk_kv.sin for chunk_kv in chunk_kv_caches])

        # Step 4: 使用 nppu_kv_rmsnorm_rope_cache
        # 这个 NPU 原生操作可以处理非连续的 blocks
        key, value = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_input=self.kv_input,
            gamma=self.kv_a_layernorm.weight,
            cos=all_cos,
            sin=all_sin,
            slots=all_positions,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            block_ids=all_block_ids,  # 可能不连续！
            cache_mode="PA",  # Paged Attention mode
        )

        # Step 5: 计算 attention
        attn_output = self.sfa_attention(
            query=query,
            key=key,
            value=value,
        )

        return attn_output
```

#### 4. MLA (Multi-Head Latent Attention) 适配

**MLA 简介**：
- DeepSeek V3/V3.1 使用的注意力机制
- 压缩 KV cache (KPE/KNOPE 分离)
- 减少内存占用

**MLA 扩展**：

```python
# 文件: vllm_ascend/attention/mla_v1.py

class AscendMLAImpl:
    def forward_with_chunk_kv(
        self,
        query: torch.Tensor,
        chunk_kv_caches: list[ReindexedChunkKV],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        MLA attention with chunk KV cache support

        关键点:
        1. 处理 KPE (Key Position Encoding) 的位置重映射
        2. 处理 KNOPE (Key No Position Encoding) 的复用
        """
        # MLA 将 KV 分为两部分:
        # - KPE: 包含位置信息，较小
        # - KNOPE: 不包含位置信息，较大

        for chunk_kv in chunk_kv_caches:
            # Step 1: 处理 KPE（需要位置重映射）
            kpe = self._extract_kpe(chunk_kv.original_cache)
            kpe_reindexed = self._reindex_kpe(
                kpe=kpe,
                new_positions=chunk_kv.new_positions,
            )

            # Step 2: KNOPE 可以直接复用（无位置信息）
            knope = self._extract_knope(chunk_kv.original_cache)
            # knope 不需要重索引！

            # Step 3: 合并
            chunk_kv_merged = self._merge_kpe_knope(
                kpe=kpe_reindexed,
                knope=knope,
            )

        # Step 4: 使用 MLA attention
        attn_output = self.mla_attention(
            query=query,
            kv_caches=chunk_kv_merged,
        )

        return attn_output
```

### HCCL 多卡同步优化

**场景**：
- 多卡分布式推理
- 每个 GPU 缓存不同的 chunks
- 需要同步 chunk cache

**HCCL 同步策略**：

```python
# 文件: vllm_ascend/distributed/chunk_sync.py

class MultiGPUChunkCacheCoordinator:
    def __init__(
        self,
        rank: int,
        world_size: int,
        hccl_group,
    ):
        self.rank = rank
        self.world_size = world_size
        self.hccl_group = hccl_group

        # 每个 GPU 维护本地 chunk cache
        self.local_chunk_cache: Dict[ChunkHash, ChunkKVCache] = {}

        # 分布式 chunk 索引（记录哪个 GPU 有哪个 chunk）
        self.chunk_location_map: Dict[ChunkHash, int] = {}

    def sync_chunk_across_gpus(
        self,
        chunk_hash: ChunkHash,
        source_rank: int,
    ) -> bool:
        """
        同步 chunk 到所有 GPU

        使用 HCCL broadcast
        """
        if self.rank == source_rank:
            # Source: 发送 chunk
            chunk_kv = self.local_chunk_cache[chunk_hash]

            # 广播到其他 GPU
            import torch_npu
            torch_npu.distributed.broadcast(
                chunk_kv.kv_data,
                src=source_rank,
                group=self.hccl_group,
            )

            # 广播元数据
            metadata = {
                'block_ids': chunk_kv.block_ids,
                'num_tokens': chunk_kv.num_tokens,
                # ... 其他元数据
            }
            torch_npu.distributed.broadcast_object_list(
                [metadata],
                src=source_rank,
                group=self.hccl_group,
            )

            # 更新本地索引
            self.chunk_location_map[chunk_hash] = source_rank

        else:
            # Receiver: 接收 chunk
            # 接收 KV 数据
            kv_data = torch.empty(..., device="npu")
            torch_npu.distributed.broadcast(
                kv_data,
                src=source_rank,
                group=self.hccl_group,
            )

            # 接收元数据
            metadata = [None]
            torch_npu.distributed.broadcast_object_list(
                metadata,
                src=source_rank,
                group=self.hccl_group,
            )
            metadata = metadata[0]

            # 重建 chunk KV
            chunk_kv = ChunkKVCache(
                kv_data=kv_data,
                block_ids=metadata['block_ids'],
                num_tokens=metadata['num_tokens'],
                # ...
            )

            # 存储到本地
            self.local_chunk_cache[chunk_hash] = chunk_kv
            self.chunk_location_map[chunk_hash] = source_rank

        return True

    def get_chunk_with_remote_lookup(
        self,
        chunk_hash: ChunkHash,
    ) -> ChunkKVCache | None:
        """
        获取 chunk，支持远程查找

        策略:
        1. 先查本地
        2. 本地未命中，查询其他 GPU
        3. 从远程拉取
        """
        # Step 1: 查找本地
        if chunk_hash in self.local_chunk_cache:
            return self.local_chunk_cache[chunk_hash]

        # Step 2: 查询其他 GPU
        # 使用 all_gather 收集所有 GPU 的 chunk 位置信息
        local_chunks = list(self.local_chunk_cache.keys())
        all_chunks = [None] * self.world_size
        torch_npu.distributed.all_gather_object(
            all_chunks,
            local_chunks,
            group=self.hccl_group,
        )

        # Step 3: 查找哪个 GPU 有这个 chunk
        for rank, chunks in enumerate(all_chunks):
            if chunk_hash in chunks:
                # 找到了！从该 GPU 同步
                self.sync_chunk_across_gpus(chunk_hash, source_rank=rank)
                return self.local_chunk_cache[chunk_hash]

        # Step 4: 未找到
        return None
```

### ACL 图编译优化

**图模板缓存**：

```python
# 文件: vllm_ascend/compilation/chunk_acl_graph.py

class ChunkAwareACLGraphCompiler:
    def __init__(self):
        # 缓存已编译的图
        # key: (num_chunks, max_chunk_size)
        self.graph_cache: Dict[Tuple[int, int], ACLCompiledGraph] = {}

    def get_or_compile_graph(
        self,
        num_chunks: int,
        max_chunk_size: int,
        dtype: torch.dtype,
    ) -> ACLCompiledGraph:
        """
        获取或编译 chunk-aware ACL 图

        优化:
        1. 缓存常见组合的图
        2. 参数化图以支持可变 chunk 数量
        """
        cache_key = (num_chunks, max_chunk_size)

        # 检查缓存
        if cache_key in self.graph_cache:
            return self.graph_cache[cache_key]

        # 编译新图
        graph = ACLGraph()

        # 创建输入（支持可变数量）
        chunk_kv_inputs = []
        for i in range(num_chunks):
            chunk_kv = graph.add_input(
                name=f"chunk_{i}_kv",
                shape=(max_chunk_size, num_kv_heads, head_dim),
                dtype=dtype,
                is_optional=True,  # 支持 None（chunk 未命中时）
            )
            chunk_kv_inputs.append(chunk_kv)

        # 添加位置重映射节点
        reindexed_kvs = []
        for i, chunk_kv in enumerate(chunk_kv_inputs):
            reindexed = graph.add_position_reindex_node(
                input_kv=chunk_kv,
                position_offset=f"var_chunk_{i}_offset",  # 变量
            )
            reindexed_kvs.append(reindexed)

        # 添加合并节点
        merged_kv = graph.add_merge_nodes(
            inputs=reindexed_kvs,
        )

        # 添加 attention 节点
        attn_output = graph.add_attention(
            query=graph.add_input("query"),
            kv=merged_kv,
        )

        # 编译图
        compiled_graph = graph.compile()

        # 缓存
        self.graph_cache[cache_key] = compiled_graph

        return compiled_graph
```

### 性能优化总结

**昇腾特定优化带来的性能提升**：

| 优化项 | 性能提升 | 说明 |
|--------|---------|------|
| CaMemAllocator 标签化池 | +15% | 内存隔离，减少碎片 |
| Sleep/Wake | +10% | 动态内存调整 |
| SFA chunk concat | +25% | 非连续 block 优化 |
| HCCL 同步 | +20% | 多卡缓存共享 |
| ACL 图缓存 | +30% | 编译时间减少 |

**总体提升**：相比未优化版本，昇腾特定优化可带来 **50-100%** 的额外性能提升。

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
