# Prefix Caching (Automatic Prefix Caching)

## 概述

**Automatic Prefix Caching (APC)** 是 vLLM 的核心优化特性之一,通过缓存已处理请求的 KV cache,使得新请求如果与已有请求共享相同前缀时,可以直接复用 KV cache,跳过共享部分的重复计算。

### 核心原理

APC 基于以下关键机制实现:

1. **块哈希链 (Block Hash Chain)**: 每个 KV cache block 计算一个哈希值,该哈希值包含父 block 的哈希、当前 block 的 token IDs 以及额外的键(如多模态输入、LoRA 适配器、cache_salt)。这种链式哈希设计使得相同的前缀必然产生相同的哈希序列。

2. **基于哈希的查找**: 当新请求到达时,vLLM 会将其 token 序列分成固定大小的块(默认 16 个 token/block),计算每个块的哈希值,然后在哈希表中查找是否存在完全匹配的哈希链。如果找到匹配,就直接复用对应的物理 KV cache blocks。

3. **引用计数与 LRU 驱逐**: 每个 block 维护引用计数 (`ref_cnt`)。当 block 被多个请求共享时,`ref_cnt > 1`。当请求完成释放 block 时,`ref_cnt` 减 1。只有当 `ref_cnt == 0` 时,block 才被放入 LRU 驱逐队列。当需要分配新 block 但空闲池不足时,从 LRU 队列头部(最久未使用)驱逐 block。

4. **多注意力类型支持**: vLLM v1 支持多种注意力机制,每种类型的缓存策略不同:
   - **FullAttention**: 标准的完整注意力,从左到右查找最长匹配前缀
   - **SlidingWindowAttention**: 滑动窗口注意力,从右到左查找窗口内的连续 blocks
   - **ChunkedLocalAttention**: 分块局部注意力,只缓存当前 attention 窗口内的 blocks
   - **MambaAttention**: 线性注意力 (Mamba/SSM),只保留最后一个计算 token 的状态
   - **SinkFullAttention**: Sink token + 完整注意力,sink blocks 永不驱逐

### 性能收益

APC 在以下场景中能带来显著的性能提升:

- **长文档问答**: 用户反复对同一长文档(如软件手册、年度报告)提出不同问题时,APC 允许 vLLM 仅处理一次该长文档,所有后续请求都能复用其 KV cache,从而实现更高的吞吐量和更低的延迟。
- **多轮对话**: 用户在同一聊天会话中进行多轮对话时,APC 可以跨对话轮次复用聊天历史的处理结果,避免每轮都重新处理整个历史记录。
- **提示词模板复用**: 当多个请求使用相同的系统提示词或模板时(如 RAG 应用中的固定文档上下文),APC 可以自动复用这些共享前缀。

### 局限性

- APC 仅减少**查询处理时间**(prefill 阶段),不减少**生成新 token 的时间**(decode 阶段)
- 当 vLLM 大部分时间用于生成答案(如答案长度很长)时,APC 的收益有限
- 当新请求与任何现有请求都不共享前缀时,无法实现计算复用
- 需要额外的 GPU 内存来维护 prefix cache,可能影响可服务的并发请求数

## 架构设计

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          vLLM v1 Engine                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────┐      ┌──────────────────────────────────────────┐   │
│  │  Input Processor  │─────│           Scheduler                       │   │
│  └───────────────────┘      │  ┌────────────────────────────────────┐  │   │
│                              │  │  Prefix Cache Hit Detection         │  │   │
│                              │  │  - get_computed_blocks()            │  │   │
│                              │  │  - Update num_cached_tokens         │  │   │
│                              │  └────────────────────────────────────┘  │   │
│                              └──────────────────────────────────────────┘   │
│                                             │                                │
│                                             ▼                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              KVCacheManager (High-Level Orchestration)               │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  KVCacheCoordinator (Multi-Group Coordination)                 │ │   │
│  │  │  - UnitaryKVCacheCoordinator (single KV group)                 │ │   │
│  │  │  - HybridKVCacheCoordinator (multiple KV groups)               │ │   │
│  │  │    * find_longest_cache_hit()                                  │ │   │
│  │  │    * allocate_new_computed_blocks()                            │ │   │
│  │  │    * cache_blocks()                                            │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                        │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  SingleTypeKVCacheManager (Attention-Type Specific)            │ │   │
│  │  │  - FullAttentionManager                                        │ │   │
│  │  │  - SlidingWindowManager                                        │ │   │
│  │  │  - ChunkedLocalAttentionManager                                │ │   │
│  │  │  - MambaManager                                                │ │   │
│  │  │  - SinkFullAttentionManager                                    │ │   │
│  │  │    * find_longest_cache_hit() [class method]                   │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                             │                                │
│                                             ▼                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       BlockPool (Core Cache Store)                   │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  BlockHashToBlockMap (Hash Table Lookup)                       │ │   │
│  │  │  - get_one_block(block_hash, group_ids) -> KVCacheBlock       │ │   │
│  │  │  - insert(block_hash, block)                                   │ │   │
│  │  │  - pop(block_hash, block_id)                                   │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                        │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  KVCacheBlock (Block Metadata)                                 │ │   │
│  │  │  - block_id: Physical block index                              │ │   │
│  │  │  - ref_cnt: Reference count (shared by requests)               │ │   │
│  │  │  - _block_hash: Cached hash (when full and cached)             │ │   │
│  │  │  - is_null: Marker for skipped blocks (sliding window)         │ │   │
│  │  │  - Linked list pointers for free block queue                   │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                        │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  FreeKVCacheBlockQueue (LRU Eviction Queue)                    │ │   │
│  │  │  - Doubly-linked list ordered by eviction priority             │ │   │
│  │  │  - O(1) remove from middle                                     │ │   │
│  │  │  - Non-blocking operations                                     │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                        │   │
│  │  Core Methods:                                                         │   │
│  │  - get_cached_block(block_hash, group_ids) -> KVCacheBlock           │   │
│  │  - cache_full_blocks(req_to_blocks)                                   │   │
│  │  - get_new_blocks(num_blocks) -> List[KVCacheBlock]                   │   │
│  │  - touch(blocks) -> Increase ref_cnt, remove from free queue          │   │
│  │  - free_blocks(blocks) -> Decrease ref_cnt, add to free queue         │   │
│  │  - evict_blocks(block_ids) -> Force eviction from hash table          │   │
│  │  - reset_prefix_cache() -> Clear all cached blocks                    │   │
│  │  - get_num_free_blocks() -> Query available blocks                    │   │
│  │  - get_usage() -> KV cache utilization (0.0-1.0)                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌───────────────────┐      ┌──────────────────────────────────────────┐   │
│  │  Output Processor │◀─────│           Worker                         │   │
│  └───────────────────┘      │  ┌────────────────────────────────────┐  │   │
│                              │  │  BlockTable (Token → Block Mapping)│  │   │
│                              │  │  - Maps token positions to blocks  │  │   │
│                              │  │  - Converts block IDs to slot      │  │   │
│                              │  │    mappings for attention kernels  │  │   │
│                              │  └────────────────────────────────────┘  │   │
│                              └──────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 模块详细说明

#### 1. BlockPool (`vllm/v1/core/block_pool.py`)

**核心职责**: GPU KV cache blocks 的分配、缓存查找和驱逐管理。

**主要类**:

- **`BlockHashToBlockMap`**: 哈希表,将 block 哈希映射到缓存的 blocks
  - `get_one_block(block_hash, group_ids)`: 获取给定哈希的任意缓存 block
  - `insert(block_hash, block)`: 插入新的缓存 block
  - `pop(block_hash, block_id)`: 从缓存中移除特定 block

- **`BlockPool`**: 核心 block 管理
  - `get_cached_block(block_hash, group_ids)`: 通过哈希查询可复用的 blocks
  - `cache_full_blocks(req_to_blocks)`: 缓存已计算的 blocks 以便前缀复用
  - `get_new_blocks(num_blocks)`: 从空闲池分配新 blocks
  - `touch(blocks)`: 增加 block 引用计数(缓存命中时调用)
  - `free_blocks(blocks)`: 释放 blocks,使用 LRU 驱逐
  - `evict_blocks(block_ids)`: 显式驱逐 prefix cache 中的 blocks
  - `reset_prefix_cache()`: 清空所有缓存的 blocks
  - `get_num_free_blocks()`: 查询可用 blocks 数量
  - `get_usage()`: 获取 KV cache 利用率 (0.0-1.0)

**数据结构**:

- **`KVCacheBlock`**: Block 元数据
  - `block_id`: 物理 block 索引 (0 到 num_gpu_blocks-1)
  - `ref_cnt`: 引用计数 (可被多个请求共享)
  - `_block_hash`: 缓存的哈希值 (仅在 block 已满且已缓存时)
  - `is_null`: 跳过 block 的标记 (滑动窗口注意力用)
  - 用于空闲 block 队列的双向链表指针

- **`FreeKVCacheBlockQueue`**: 空闲 block 的双向链表
  - 按驱逐优先级 (LRU) 排序
  - O(1) 从中间移除
  - 非阻塞操作以提高性能

#### 2. KVCacheManager (`vllm/v1/core/kv_cache_manager.py`)

**核心职责**: 高级 KV cache 管理和调度器与 block pool 之间的桥梁。

**主要类**:

- **`KVCacheBlocks`**: 调度器与 KVCacheManager 之间的接口
  - 封装每个 KV cache 组分配的 blocks
  - `get_block_ids()`: 提取 block IDs
  - `get_unhashed_block_ids()`: 获取非缓存的 blocks

- **`KVCacheManager`**: 高级 KV cache 管理
  - `get_computed_blocks(request)`: 查找请求的缓存 blocks (前缀缓存命中)
  - `allocate_slots(request, num_new_tokens, ...)`: 为新/续接 tokens 分配 blocks
  - `free(request)`: 释放请求的 blocks
  - `cache_blocks(req_to_blocks)`: 缓存已计算的 blocks
  - `evict_blocks(block_ids)`: 显式驱逐 blocks
  - `reset_prefix_cache()`: 重置所有缓存状态
  - `get_num_common_prefix_blocks(reqs)`: 查找请求间的共享前缀

**前缀缓存关键方法**:

```python
get_computed_blocks(request):
    # 查找最长缓存命中前缀
    # 返回: (cached_blocks, num_cached_tokens)
    # 如果 enable_caching=False 或 request.skip_reading_prefix_cache=True 则跳过

allocate_slots(request, num_new_tokens, ...):
    # 三阶段分配:
    # 1. 释放不必要的 blocks (滑动窗口)
    # 2. 处理前缀缓存命中 (touch blocks)
    # 3. 为未缓存 tokens 分配新 blocks
    # 分配后缓存完整的 blocks
```

#### 3. KVCacheCoordinator (`vllm/v1/core/kv_cache_coordinator.py`)

**核心职责**: 多 KV cache 组协调,支持混合注意力模型。

**主要类**:

- **`KVCacheCoordinator`** (ABC): 多组支持的协调器基类
- **`KVCacheCoordinatorNoPrefixCache`**: 无缓存支持的协调器
- **`UnitaryKVCacheCoordinator`**: 单 KV cache 组 (大多数模型)
- **`HybridKVCacheCoordinator`**: 多 KV cache 类型 (如混合注意力)

**关键方法**:

- `find_longest_cache_hit(...)`: 跨 KV 组查找最长前缀匹配
- `allocate_new_computed_blocks(...)`: 处理缓存命中的 blocks
- `cache_blocks(req_to_blocks)`: 计算后缓存完整 blocks
- `get_num_common_prefix_blocks(reqs)`: 计算共享前缀长度

**特殊逻辑**:

- **Unitary**: 单组的简单基于哈希的查找
- **Hybrid**: 多注意力类型的迭代固定点算法
  - 检查每种注意力类型 (full, sliding window, chunked local)
  - 减少候选长度直到所有类型一致
  - 确保跨组的 block 对齐

#### 4. SingleTypeKVCacheManager (`vllm/v1/core/single_type_kv_cache_manager.py`)

**核心职责**: 每种注意力类型的专门缓存逻辑。

**管理器类**:

1. **`FullAttentionManager`**: 标准完整注意力缓存
   - 从左到右缓存命中搜索
   - 通过 `ref_cnt == len(req_to_blocks)` 检测公共前缀

2. **`SlidingWindowManager`**: 滑动窗口注意力
   - 从右到左搜索窗口内的连续 blocks
   - 跳过窗口外的 tokens 的 (null) blocks
   - 返回 `[NULL, NULL, cached_block, ...]` 模式

3. **`ChunkedLocalAttentionManager`**: 局部注意力分块
   - 将当前 chunk 外的所有 blocks 标记为 NULL
   - 仅缓存 attention 窗口内的 blocks

4. **`MambaManager`**: 线性注意力 (Mamba/SSM)
   - 从右到左搜索最后匹配的 block
   - 仅保留最后计算 token 的状态

5. **`SinkFullAttentionManager`**: Sink token + 完整注意力
   - 预分配永不驱逐的 sink blocks

**关键方法**:
```python
@classmethod
find_longest_cache_hit(
    block_hashes, max_length, kv_cache_group_ids,
    block_pool, kv_cache_spec, ...
) -> list[KVCacheBlock]:
    # 返回匹配前缀哈希链的缓存 blocks
    # 处理对齐、滑动窗口、分块
```

#### 5. KVCacheUtils (`vllm/v1/core/kv_cache_utils.py`)

**核心职责**: Block 哈希、元数据结构和工具函数。

**Block 哈希**:

- **`BlockHash`**: 类型别名,用于 bytes (token block 的哈希)
- **`BlockHashWithGroupId`**: 将哈希 + group_id 组合成键
- **`hash_block_tokens()`: 使用 parent_hash + token_ids + extra_keys 计算哈希
  - 支持多模态输入 (mm_hash + offset)
  - 支持 LoRA (lora_name)
  - 支持特定请求变体的 cache_salt
  - 使用 `sha256_cbor` 或 `xxhash_cbor` 实现可重现性

**Block 元数据**:

- **`KVCacheBlock`**: Block 元数据 (见 BlockPool 部分)

**空闲 Block 队列**:

- **`FreeKVCacheBlockQueue`**: 空闲 blocks 的双向链表 (见 BlockPool 部分)

#### 6. Scheduler (`vllm/v1/core/sched/scheduler.py`)

**核心职责**: 请求调度和缓存命中检测的集成点。

**缓存集成**:

```python
# 请求调度期间
computed_blocks, num_cached_tokens = kv_cache_manager.get_computed_blocks(request)
# 更新 request.num_cached_tokens
# 在 prefix_cache_stats 中记录指标
```

**分配流程**:

1. 检查前缀缓存命中 → `get_computed_blocks()`
2. 为新 tokens 分配 slots → `allocate_slots()`
3. 缓存新计算的 blocks → `cache_blocks()`
4. 完成时释放 → `free()`

**统计收集**:

- `PrefixCacheStats`: 跟踪请求、查询 (tokens)、命中
- `CachingMetrics`: 最近 N 个请求的滚动命中率
- 区分新请求与被抢占的请求

#### 7. Worker BlockTable (`vllm/v1/worker/block_table.py`)

**核心职责**: Worker 端 block 表管理,将 block IDs 映射到 attention kernels 的 slots。

**主要类**:

- **`BlockTable`**: 将 token 位置映射到物理 blocks
  - 将 block IDs 转换为 attention kernels 的 slot 映射
  - 处理混合 block 大小 (allocation vs. kernel blocks)

- **`MultiGroupBlockTable`**: 混合注意力的多个 block 表
  - 每个 KV cache 组的独立表
  - 跨组的同步操作

## 主要流程

### 1. 前缀缓存命中流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   New Request│     │   Scheduler  │     │KVCacheManager│     │  BlockPool   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │                    │
       │  schedule()        │                    │                    │
       │───────────────────>│                    │                    │
       │                    │                    │                    │
       │                    │  get_computed_blocks(req)                │
       │                    │───────────────────>│                    │
       │                    │                    │                    │
       │                    │                    │  find_longest_cache_hit()
       │                    │                    │  - Compute block hashes
       │                    │                    │  - Lookup in hash table
       │                    │                    │───────────────────>│
       │                    │                    │                    │
       │                    │                    │                    │  get_cached_block(hash, group_id)
       │                    │                    │                    │  - Check BlockHashToBlockMap
       │                    │                    │                    │<───────────────────│
       │                    │                    │                    │
       │                    │                    │  Return cached blocks
       │                    │                    │<───────────────────│
       │                    │                    │                    │
       │                    │  (cached_blocks, num_cached_tokens)      │
       │                    │<───────────────────│                    │
       │                    │                    │                    │
       │                    │  Update request.num_cached_tokens        │
       │                    │  Record prefix cache hit metrics         │
       │                    │                    │                    │
       │  Schedule request  │                    │                    │
       │<───────────────────│                    │                    │
       │                    │                    │                    │
```

### 2. Block 分配流程 (缓存命中 + 新计算)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Scheduler  │     │KVCacheManager│     │  BlockPool   │     │    Worker    │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │                    │
       │  allocate_slots()  │                    │                    │
       │───────────────────>│                    │                    │
       │                    │                    │                    │
       │                    │  Stage 1: Free unnecessary blocks        │
       │                    │  (sliding window)                        │
       │                    │───────────────────>│                    │
       │                    │                    │                    │
       │                    │  Stage 2: Handle prefix cache hit        │
       │                    │  - Touch cached blocks (inc ref_cnt)      │
       │                    │───────────────────>│                    │
       │                    │                    │                    │
       │                    │  Stage 3: Allocate new blocks            │
       │                    │  for uncached tokens                     │
       │                    │───────────────────>│                    │
       │                    │                    │  get_new_blocks()  │
       │                    │                    │  - Check free pool │
       │                    │                    │  - Evict LRU if    │
       │                    │                    │    insufficient    │
       │                    │                    │<───────────────────│
       │                    │                    │                    │
       │                    │  Allocate slots for new tokens           │
       │                    │<───────────────────│                    │
       │                    │                    │                    │
       │  Allocated blocks  │                    │                    │
       │<───────────────────│                    │                    │
       │                    │                    │                    │
       │  Execute model     │                    │                    │
       │───────────────────────────────────────────────────────────>│
       │                    │                    │                    │
       │                    │                    │  After computation │
       │                    │                    │                    │
       │                    │  cache_full_blocks()                    │
       │                    │  - Hash new blocks                       │
       │                    │  - Insert into hash table                │
       │                    │───────────────────>│                    │
       │                    │                    │                    │
```

### 3. 缓存驱逐流程 (LRU)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Request    │     │  BlockPool   │     │FreeBlockQueue│
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │  get_new_blocks()  │                    │
       │───────────────────>│                    │
       │                    │                    │
       │                    │  Check free pool   │
       │                    │  Insufficient blocks│
       │                    │                    │
       │                    │  Need to evict     │
       │                    │───────────────────>│
       │                    │                    │
       │                    │                    │  Pop from head
       │                    │                    │  (oldest block)
       │                    │                    │
       │                    │  Remove from       │
       │                    │  BlockHashToBlockMap│
       │                    │<───────────────────│
       │                    │                    │
       │  Allocate evicted  │                    │
       │  block to request  │                    │
       │<───────────────────│                    │
       │                    │                    │
       │                    │                    │
       │  Request completes │                    │
       │───────────────────>│                    │
       │                    │                    │
       │                    │  free_blocks()     │
       │                    │  - Decrease ref_cnt│
       │                    │  - If ref_cnt == 0:│
       │                    │    Add to tail of  │
       │                    │    free queue      │
       │                    │───────────────────>│
       │                    │                    │
       │                    │                    │
       │  Request completes │                    │
       │───────────────────>│                    │
       │                    │  free_blocks()     │
       │                    │  - Decrease ref_cnt│
       │                    │  - If ref_cnt > 0: │
       │                    │    Keep (still     │
       │                    │    shared)         │
       │                    │                    │
```

### 4. 多轮对话缓存复用流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  User        │     │   Scheduler  │     │KVCacheManager│     │  BlockPool   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │                    │
       │  Round 1:          │                    │                    │
       │  "Hello, how are   │                    │                    │
       │   you?"            │                    │                    │
       │───────────────────>│                    │                    │
       │                    │                    │                    │
       │                    │  No cache hit      │                    │
       │                    │  (first request)   │                    │
       │                    │───────────────────>│                    │
       │                    │                    │                    │
       │                    │                    │  Allocate & compute│
       │                    │                    │───────────────────>│
       │                    │                    │                    │
       │  "I'm doing great!"│                    │                    │
       │<───────────────────│                    │                    │
       │                    │                    │                    │
       │                    │  Cache all blocks  │                    │
       │                    │───────────────────>│                    │
       │                    │                    │                    │
       ├────────────────────────────────────────────────────────────────┤
       │                    │                    │                    │
       │  Round 2:          │                    │                    │
       │  "Tell me a joke"  │                    │                    │
       │───────────────────>│                    │                    │
       │                    │                    │                    │
       │                    │  Cache hit!        │                    │
       │                    │  (reuse conversation│                   │
       │                    │   history)         │                    │
       │                    │───────────────────>│                    │
       │                    │                    │                    │
       │                    │                    │  Touch cached blocks│
       │                    │                    │  (inc ref_cnt)     │
       │                    │                    │───────────────────>│
       │                    │                    │                    │
       │                    │  Only compute new  │                    │
       │                    │  "Tell me a joke"   │                    │
       │                    │───────────────────>│                    │
       │                    │                    │                    │
       │  "Why did the      │                    │                    │
       │   chicken..."      │                    │                    │
       │<───────────────────│                    │                    │
       │                    │                    │                    │
```

### 5. 混合注意力协调流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Scheduler  │     │  Coordinator │     │   FullAttn   │     │SlidingWindow │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │                    │
       │  find_longest_     │                    │                    │
       │  cache_hit()       │                    │                    │
       │───────────────────>│                    │                    │
       │                    │                    │                    │
       │                    │  Check each attention type            │
       │                    │───────────────────>│                    │
       │                    │                    │                    │
       │                    │                    │  Find longest match│
       │                    │                    │  (left-to-right)   │
       │                    │                    │  Returns: 10 blocks│
       │                    │                    │<───────────────────│
       │                    │                    │                    │
       │                    │───────────────────>│                    │
       │                    │                    │  Find longest match│
       │                    │                    │  (right-to-left)   │
       │                    │                    │  Returns: 8 blocks  │
       │                    │                    │<───────────────────│
       │                    │                    │                    │
       │                    │  Iterative fixed-point:                │
       │                    │  - Full provides initial bound: 10     │
       │                    │  - SlidingWindow bound: 8              │
       │                    │  - Reduce to min: 8 blocks             │
       │                    │  - Re-verify all types agree on 8      │
       │                    │                    │
       │  Return agreed     │                    │                    │
       │  prefix length: 8  │                    │                    │
       │<───────────────────│                    │                    │
       │                    │                    │                    │
```

## 相关代码

### 核心文件列表

| 文件路径 | 主要类/函数 | 功能描述 |
|---------|-----------|---------|
| `vllm/v1/core/block_pool.py` | `BlockPool`, `BlockHashToBlockMap`, `KVCacheBlock`, `FreeKVCacheBlockQueue` | 核心 block 管理、哈希表、LRU 驱逐 |
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager`, `KVCacheBlocks` | 高级 KV cache 管理和调度器接口 |
| `vllm/v1/core/kv_cache_coordinator.py` | `UnitaryKVCacheCoordinator`, `HybridKVCacheCoordinator` | 多 KV cache 组协调 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `FullAttentionManager`, `SlidingWindowManager`, `ChunkedLocalAttentionManager`, `MambaManager`, `SinkFullAttentionManager` | 注意力类型特定的缓存逻辑 |
| `vllm/v1/core/kv_cache_utils.py` | `hash_block_tokens()`, `BlockHash`, `BlockHashWithGroupId`, `KVCacheBlock`, `FreeKVCacheBlockQueue` | Block 哈希、元数据结构、工具函数 |
| `vllm/v1/core/sched/scheduler.py` | `Scheduler.get_computed_blocks()`, `PrefixCacheStats`, `CachingMetrics` | 请求调度和缓存命中检测 |
| `vllm/v1/worker/block_table.py` | `BlockTable`, `MultiGroupBlockTable` | Worker 端 block 表管理 |
| `vllm/v1/engine/llm_engine.py` | `LLMEngine` (enable_prefix_caching 参数) | 引擎配置和初始化 |

### 关键代码位置

#### 1. 前缀缓存启用检查

**文件**: `vllm/v1/core/kv_cache_manager.py`

```python
def get_computed_blocks(self, req: SchedulerRequest) -> list[int]:
    """Find cached blocks for a request (prefix cache hit)."""
    # Skip if caching disabled or request-specific skip flag set
    if not self.enable_caching or req.skip_reading_prefix_cache:
        return []

    # Find longest cache hit via coordinator
    computed_blocks = self.kv_cache_coordinator.find_longest_cache_hit(
        req.block_hashes,
        req.num_tokens,
        req.kv_cache_group_ids,
        self.block_pool,
        self.kv_cache_spec,
    )

    # Update request metrics
    req.num_cached_tokens = len(computed_blocks) * self.block_size
    ...
```

#### 2. Block 哈希计算

**文件**: `vllm/v1/core/kv_cache_utils.py`

```python
def hash_block_tokens(
    block_hash: BlockHash,
    token_ids: list[int],
    extra_keys: tuple,
) -> BlockHash:
    """Compute hash for a block of tokens.

    Args:
        block_hash: Hash of parent block (for chain)
        token_ids: Token IDs in current block
        extra_keys: Extra hash components (mm_hash, lora_name, cache_salt)

    Returns:
        Hash of current block (includes parent hash for chain)
    """
    # Combine parent hash + tokens + extra keys
    hash_data = (block_hash, tuple(token_ids), extra_keys)
    # Use xxhash for speed or sha256 for reproducibility
    return xxhash_cbor(hash_data)
```

#### 3. LRU 驱逐逻辑

**文件**: `vllm/v1/core/block_pool.py`

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    """Allocate new blocks from free pool, evicting if necessary."""
    # Check if enough free blocks
    if len(self.free_block_queue) < num_blocks:
        # Need to evict blocks
        num_to_evict = num_blocks - len(self.free_block_queue)
        for _ in range(num_to_evict):
            # Pop from head (oldest / least recently used)
            block = self.free_block_queue.popleft()
            # Remove from hash table
            self.cached_block_hash_to_block.pop(block.block_hash, None)
            # Reset block metadata
            block.ref_cnt = 0
            block._block_hash = None

    # Allocate from free pool
    blocks = [self.free_block_queue.popleft() for _ in range(num_blocks)]
    return blocks
```

#### 4. 缓存命中时 Touch Block

**文件**: `vllm/v1/core/block_pool.py`

```python
def touch(self, blocks: list[KVCacheBlock]) -> None:
    """Increase block reference count (cache hit).

    Removes blocks from free queue so they won't be evicted.
    """
    for block in blocks:
        block.ref_cnt += 1
        # Remove from free queue (if present)
        if block in self.free_block_queue:
            self.free_block_queue.remove(block)
```

#### 5. 不同注意力类型的缓存查找

**文件**: `vllm/v1/core/single_type_kv_cache_manager.py`

```python
class FullAttentionManager:
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: tuple[int],
        block_pool: BlockPool,
        ...
    ) -> list[KVCacheBlock]:
        """Find longest prefix match for full attention.

        Left-to-right search: find the longest chain of matching hashes.
        """
        computed_blocks = []
        for block_hash in block_hashes[:max_length]:
            cached_block = block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            )
            if cached_block is None:
                break  # Chain broken
            computed_blocks.append(cached_block)
        return computed_blocks


class SlidingWindowManager:
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: tuple[int],
        block_pool: BlockPool,
        ...
    ) -> list[KVCacheBlock]:
        """Find longest prefix match for sliding window attention.

        Right-to-left search: find contiguous blocks within window.
        Returns [NULL, NULL, cached_block, ...] pattern for tokens
        outside sliding window.
        """
        computed_blocks = []
        window_size = kv_cache_spec.window_size

        # Start from end, search backwards
        for i in range(len(block_hashes) - 1, -1, -1):
            block_hash = block_hashes[i]
            cached_block = block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            )
            if cached_block is None:
                break  # Chain broken

            # Prepend (building from right to left)
            computed_blocks.insert(0, cached_block)

            # Check if we have enough blocks for window
            if len(computed_blocks) >= window_size:
                break

        # Pad with NULL blocks for tokens outside window
        num_null_blocks = max_length - len(computed_blocks)
        return [NULL_BLOCK] * num_null_blocks + computed_blocks
```

#### 6. 混合注意力协调

**文件**: `vllm/v1/core/kv_cache_coordinator.py`

```python
class HybridKVCacheCoordinator:
    def find_longest_cache_hit(
        self,
        block_hashes: dict[str, list[BlockHash]],  # Per attention type
        max_length: int,
        kv_cache_group_ids: dict[str, tuple[int]],  # Per attention type
        ...
    ) -> dict[str, list[KVCacheBlock]]:
        """Find longest cache hit for hybrid attention.

        Uses iterative fixed-point algorithm to ensure all
        attention types agree on prefix length.
        """
        # Start with full attention bound (monotonic property)
        full_blocks = self.managers["full"].find_longest_cache_hit(...)
        candidate_length = len(full_blocks)

        # Iteratively reduce until all types agree
        for _ in range(len(self.managers)):
            for attn_type, manager in self.managers.items():
                blocks = manager.find_longest_cache_hit(
                    block_hashes[attn_type][:candidate_length],
                    candidate_length,
                    ...
                )
                candidate_length = min(candidate_length, len(blocks))

        # Return blocks for all types at agreed length
        return {
            attn_type: manager.find_longest_cache_hit(...)[
                :candidate_length
            ]
            for attn_type, manager in self.managers.items()
        }
```

## 使用说明

### 启用前缀缓存

在 vLLM 引擎中设置 `enable_prefix_caching=True` 即可启用 APC:

#### Python API

```python
from vllm import LLM, SamplingParams

# Initialize LLM with prefix caching enabled
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_prefix_caching=True,  # Enable APC
    gpu_memory_utilization=0.9,
)

# Generate with shared prefix
prompts = [
    "You are a helpful assistant. " + user_query_1,
    "You are a helpful assistant. " + user_query_2,
    # The system prompt prefix will be cached after first request
    # and reused for all subsequent requests
]

outputs = llm.generate(prompts, SamplingParams(temperature=0.8))
```

#### OpenAI Compatible API Server

```bash
# Start server with prefix caching enabled
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9
```

#### CLI

```bash
# Run with prefix caching
vllm serve meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9
```

### 配置参数

**相关配置参数**:

- **`enable_prefix_caching`** (bool): 启用/禁用前缀缓存 (默认: `False`)
- **`gpu_memory_utilization`** (float): GPU 内存利用率 (0.0-1.0), 影响 KV cache 大小
- **`block_size`** (int): 每个 block 的 token 数量 (默认: 16)
- **`max_num_batched_tokens`** (int): 最大批处理 token 数,影响内存分配

### 监控和指标

vLLM 提供前缀缓存的性能指标:

```python
from vllm import LLM

llm = LLM(model="...", enable_prefix_caching=True)

# After serving requests, check metrics
metrics = llm.get_prefix_cache_stats()
print(f"Prefix cache hit rate: {metrics.hit_rate:.2%}")
print(f"Total cached tokens: {metrics.total_cached_tokens}")
print(f"Cache utilization: {metrics.cache_utilization:.2%}")
```

### 使用示例

#### 示例 1: 长文档问答

```python
from vllm import LLM, SamplingParams

# Initialize with prefix caching
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_prefix_caching=True,
    max_model_len=4096,
)

# Long document (will be cached after first query)
document = """
[Long document content, e.g., 2024 Annual Report...
 approximately 3000 tokens...]
"""

# Multiple queries on the same document
queries = [
    document + "\n\nQuestion: What was the revenue in Q4?",
    document + "\n\nQuestion: Who is the CEO?",
    document + "\n\nQuestion: What are the risk factors?",
]

# First request computes and caches the document
# Subsequent requests reuse the cached document KV cache
outputs = llm.generate(queries, SamplingParams(temperature=0.1))
```

**性能收益**:
- 第一个请求: 处理完整文档 (3000 tokens)
- 后续请求: 仅处理问题部分 (~20 tokens),节省 ~99% 的 prefill 时间

#### 示例 2: 多轮对话

```python
from vllm import LLM, SamplingParams

llm = LLM(model="...", enable_prefix_caching=True)

conversation_history = []

# Round 1
user_msg_1 = "Hello, how are you?"
conversation_history.append(f"User: {user_msg_1}")
prompt_1 = "\n".join(conversation_history)
output_1 = llm.generate([prompt_1], SamplingParams(...))[0]
conversation_history.append(f"Assistant: {output_1.outputs[0].text}")

# Round 2
user_msg_2 = "Tell me a joke"
conversation_history.append(f"User: {user_msg_2}")
prompt_2 = "\n".join(conversation_history)
output_2 = llm.generate([prompt_2], SamplingParams(...))[0]
# The conversation history from Round 1 is automatically cached
# and reused for Round 2, avoiding recomputation
```

**性能收益**:
- 每轮对话自动复用之前轮次的 KV cache
- 避免 recompute 整个对话历史
- 显著降低 multi-turn 应用的延迟

#### 示例 3: RAG 应用

```python
from vllm import LLM, SamplingParams

llm = LLM(model="...", enable_prefix_caching=True)

# Fixed document context (cached after first request)
document_context = """
Context: [Retrieved document chunks from vector database...
 approximately 1000 tokens...]
"""

# Multiple user queries with the same context
queries = [
    document_context + f"\n\nQuestion: {q1}",
    document_context + f"\n\nQuestion: {q2}",
    document_context + f"\n\nQuestion: {q3}",
]

# Document context is cached and reused across all queries
outputs = llm.generate(queries, SamplingParams(temperature=0.1))
```

**性能收益**:
- 固定的文档上下文仅计算一次
- 所有查询共享缓存的 document KV cache
- 特别适合 high-QPS RAG 服务

### 性能调优建议

1. **内存分配**: 前缀缓存需要额外的 GPU 内存来维护 cached blocks。如果 OOM,可以降低 `gpu_memory_utilization` 或减少 `max_num_batched_tokens`。

2. **Block 大小**: 较小的 block size (如 16) 提供更细粒度的缓存,但增加哈希开销。较大的 block size (如 32) 减少开销但降低缓存命中率。

3. **缓存命中率监控**: 监控 `prefix_cache_hit_rate` 指标。如果命中率很低 (<30%),前缀缓存可能不适合当前 workload。

4. **适用场景**:
   - **适合**: 高 prefill/decode 比、高前缀复用率 (multi-turn chat, RAG, prompt engineering)
   - **不适合**: 低前缀复用率、长生成 (答案 >> 问题)、随机 prompts

### 已知限制

1. **内存开销**: 维护 prefix cache 需要额外的 GPU 内存,可能减少可服务的并发请求数。

2. **仅加速 Prefill**: APC 不加速 decode 阶段。对于长生成场景,收益有限。

3. **哈希计算开销**: 计算 block hashes 需要额外 CPU 时间,但通常远小于 prefill 时间节省。

4. **缓存污染**: 如果请求前缀差异很大,低命中率会导致频繁驱逐,反而降低性能。

5. **动态 LoRA/多模态**: 不同 LoRA 适配器或多模态输入会导致不同的 hash,无法共享 cache。

---

**参考资源**:

- [Automatic Prefix Caching - vLLM 官方文档](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [Prefix Caching Design - vLLM 设计文档](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [Understanding vLLM KV Cache - Community Discussion](https://discuss.vllm.ai/t/understanding-vllm-kv-cache/2061)
- [vLLM Automatic Prefix Cache 原理&图解 - 知乎](https://zhuanlan.zhihu.com/p/693556044)
- [Prefix Caching 详解 - AI Infra 教程](https://cr7258.github.io/courses/ai-infra/AI%20Infra%20教程/03-prefix-caching)
