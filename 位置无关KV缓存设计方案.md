# 位置无关 KV Cache 缓存和复用 - 昇腾 NPU 实现方案

**版本**: v1.0  
**创建日期**: 2026-01-16  
**目标平台**: 昇腾 NPU (Atlas 910B/910C/310P)  
**软件栈**: vLLM v1 + vLLM-Ascend + CANN 8.3.rc2 + torch-npu 2.8.0

---

## 目录

1. [需求概述](#1-需求概述)
2. [技术背景](#2-技术背景)
3. [系统架构](#3-系统架构)
4. [数据结构设计](#4-数据结构设计)
5. [核心算法](#5-核心算法)
6. [模块设计](#6-模块设计)
7. [昇腾 NPU 特定实现](#7-昇腾-npu-特定实现)
8. [集成到 vLLM Engine](#8-集成到-vllm-engine)
9. [测试策略](#9-测试策略)
10. [实施计划](#10-实施计划)

---

## 1. 需求概述

### 1.1 业务场景

在 RAG (检索增强生成) 应用中，经常需要处理多个文档片段的场景。传统方法存在以下问题：

- 位置相关性导致相同内容在不同位置无法复用,实际应用中Prefix Caching命中率不高

**本方案的优势**:
- Chunk 基于内容哈希缓存，位置无关
- 相同内容跨请求复用，大幅减少计算
- Chunk 隔离注意力，避免 chunk 之间相互干扰
- 高效KV拷贝重映射，利用NPU加速，开销仅~10%

### 1.2 核心功能需求

#### 分块处理

**Prompt 格式**:
```
sys_prompt + "##" + chunk1 + "##" + chunk2 + "##" + chunk3 + "##" + user_question
```

**分隔符**: 默认 "##"，可配置

#### Chunk 隔离注意力

**注意力规则**:
- ✅ Chunk token 可以关注 sys_prompt 的所有 token
- ✅ Chunk token 可以关注同 chunk 内的前序 token (causal attention)
- ❌ Chunk token **不能**关注其他 chunk 的 token
- ✅ User question 可以关注所有内容 (sys_prompt + 所有 chunks)

#### 位置无关缓存

**核心思想**: 相同内容的 chunk，无论出现在 prompt 的哪个位置，都复用同一个 KV cache

**实现方法**:
- Chunk 在虚拟位置 [VIRTUAL_POS_START, VIRTUAL_POS_START + max_chunk_len) 计算 KV cache
- 缓存 key 基于内容哈希 (不包含位置信息)
- 使用时**拷贝**KV数据并重映射到实际位置，应用新位置的RoPE编码
- 利用 NPU 加速拷贝和 RoPE，开销仅 ~2-5ms (4K tokens)

#### 位置编码处理

**Chunk 位置编码**:
- 所有 chunk 共享相同的位置范围
- 虚拟位置: [sys_prompt_end, sys_prompt_end + max_chunk_len)
- 实际位置: 通过重映射调整

**User Question 位置编码**:
- 从所有 chunk 中最大的位置之后开始

#### 显存管理

**独立显存池**:
- 与主 KV Cache 分离
- 使用 vLLM-Ascend 的 CaMemAllocator 标签化内存池
- Tag: "chunk_cache"
- LRU (Least Recently Used) 淘汰策略

### 1.3 非功能性需求

#### 性能要求

- **缓存命中率**: > 80% (典型 RAG 场景)
- **TTFT 减少**: 50-80% (Time To First Token)
- **端到端加速**: 2-5x (缓存命中场景)
- **重映射延迟**: < 5ms (4K tokens)

#### 兼容性要求

- **向后兼容**: 不影响现有 vLLM 功能
- **可选功能**: 默认禁用，需要显式启用
- **API 兼容**: 与现有 vLLM API 一致

---

## 2. 技术背景

### 2.1 vLLM v1 Engine 架构

vLLM v1 Engine 是当前默认引擎，包含以下核心组件：

- **LLMEngine**: 主引擎，负责调度和管理
- **KVCacheManager**: KV cache 管理
- **Scheduler**: 请求调度
- **GPUModelRunner**: 单卡推理执行器

### 2.2 vLLM-Ascend 插件架构

vLLM-Ascend 通过插件架构扩展 vLLM 支持昇腾 NPU：

- **NPUPlatform**: 扩展 Platform 类
- **AscendSFABackend**: Sparse Flash Attention 后端
- **CaMemAllocator**: 标签化内存池
- **HCCL**: 替代 NCCL 的通信库

### 2.3 Prefix Caching 机制

**现有实现**:
```python
block_hash = hash((
    parent_hash,      # 前一个 block 的 hash
    tuple(block_tokens),
    tuple(extra_hashes)
))
```

**限制**:
- 位置相关 (parent_hash 包含位置信息)
- 跨请求复用时需要严格的前缀匹配 (位置不同则 hash 不同)

### 2.4 本方案的创新点

1. **位置无关缓存**: Chunk hash 不包含位置信息，支持跨位置复用
2. **虚拟位置计算**: 所有 chunk 在统一虚拟位置计算KV，便于复用
3. **KV拷贝重映射**: 明确拷贝KV数据并应用新位置编码，实现简单可靠
4. **统一chunk处理**: sys_prompt 也作为特殊chunk处理，不依赖prefix caching
5. **Chunk 隔离注意力**: 自定义 attention mask 实现chunk间隔离
6. **独立内存池**: 与主 KV cache 分离，使用CaMemAllocator标签化管理

---

## 3. 系统架构

### 3.1 整体架构图

```
用户请求 (Prompt with "##")
    ↓
InputProcessor 解析 chunk
    ↓
LLMEngine 执行
    ├─→ ChunkCacheManager: 处理所有 chunks (包括 sys_prompt)
    │   ├─→ ChunkHashIndex: 内容哈希索引
    │   ├─→ ChunkBlockPool: 独立显存池
    │   └─→ PositionRemapper: 位置重映射 (KV拷贝)
    ├─→ GPUModelRunner: 模型执行
    └─→ AscendSFABackend: SFA attention (chunk-aware mask)
```

**关键变化**:
- **不使用标准 prefix caching**: 所有内容 (包括 sys_prompt) 都作为 chunk 处理
- **统一缓存机制**: 基于内容哈希的位置无关缓存
- **KV拷贝重映射**: 明确需要拷贝KV数据并应用新位置编码

### 3.2 模块依赖关系

```
ChunkCacheManager
    ├── ChunkHashIndex (内容哈希索引)
    ├── ChunkBlockPool (独立显存池)
    │   └── CaMemAllocator (Ascend 内存分配器)
    ├── PositionRemapper (位置重映射器 - KV拷贝)
    └── GPUModelRunner
            └── AscendSFABackend
                    └── AttentionMaskBuilder (chunk-aware mask)
```

### 3.3 分层架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (vllm/entrypoints/)          │
│  - OpenAI API Server                                         │
│  - Python API (LLM, SamplingParams)                         │
│  - 接收用户请求，解析 chunk cache 配置                       │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   ENGINE LAYER (vllm/v1/engine/)            │
│                                                              │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │InputProcessor │  │ LLMEngine    │  │OutputProcessor  │ │
│  │               │  │              │  │                 │ │
│  │ 解析 chunked  │→ │ 调度请求     │→ │ 格式化输出      │ │
│  │ prompt        │  │ 管理缓存     │  │                 │ │
│  └───────────────┘  └──────┬───────┘  └─────────────────┘ │
│                            ↓                                 │
│                   ┌──────────────────┐                      │
│                   │ChunkCacheManager │ ◄─── 新增模块        │
│                   │                  │                      │
│                   │ - 查找/计算chunk │                      │
│                   │ - LRU淘汰       │                      │
│                   │ - 统计信息      │                      │
│                   └──────────────────┘                      │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  WORKER LAYER (vllm/v1/worker/)             │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           GPUModelRunner (修改)                       │  │
│  │                                                       │  │
│  │  新增功能:                                            │  │
│  │  - parse_chunked_prompt()     解析分块提示词          │  │
│  │  - get_or_compute_chunks()    获取/计算chunks         │  │
│  │  - compute_question_kv()      计算question的KV        │  │
│  │  - merge_chunk_kvs()          合并chunk KV            │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        PositionRemapper (新增)                        │  │
│  │                                                       │  │
│  │  职责:                                                 │  │
│  │  - remap_and_copy()         拷贝KV并重映射            │  │
│  │  - 分配物理块                                         │  │
│  │  - 应用RoPE编码                                       │  │
│  │  - 释放重映射块                                       │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                CORE LAYER (vllm/v1/core/)                   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ChunkHashIndex│  │ChunkBlockPool│  │Chunk Structures  │  │
│  │    (新增)     │  │    (新增)     │  │    (新增)         │  │
│  │              │  │              │  │                  │  │
│  │- compute_hash│  │- allocate    │  │- ChunkHash       │  │
│  │- lookup      │  │- free        │  │- ChunkKVCache    │  │
│  │- insert      │  │- LRU evict   │  │- RemappedChunkKV │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│            ATTENTION LAYER (vllm-ascend/attention/)         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           AscendSFABackend (修改)                     │  │
│  │                                                       │  │
│  │  新增:                                                 │  │
│  │  - 集成 chunk-aware attention mask                    │  │
│  │  - 使用重映射后的 KV cache                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │       AttentionMaskBuilder (新增)                     │  │
│  │                                                       │  │
│  │  职责:                                                 │  │
│  │  - get_chunk_aware_mask()    构建chunk隔离mask       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│            PLATFORM LAYER (vllm-ascend/platform/)           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           CaMemAllocator (使用)                       │  │
│  │                                                       │  │
│  │  功能:                                                 │  │
│  │  - use_memory_pool("chunk_cache") 独立内存池          │  │
│  │  - 标签化管理                                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 模块详细设计

#### 3.4.1 新增模块清单

| 模块名称 | 文件路径 | 职责 | 关键接口 |
|---------|---------|------|----------|
| **ChunkHashIndex** | `vllm/v1/core/chunk_hash_index.py` | 内容哈希索引管理 | `compute_hash()`, `lookup()`, `insert()` |
| **ChunkBlockPool** | `vllm/v1/core/chunk_block_pool.py` | 独立块池管理 | `allocate_blocks()`, `free_blocks()`, `evict_lru()` |
| **ChunkCacheManager** | `vllm/v1/core/chunk_cache_manager.py` | Chunk缓存协调器 | `get_or_compute_chunk()`, `get_stats()` |
| **PositionRemapper** | `vllm/v1/worker/position_remapper.py` | 位置重映射和KV拷贝 | `remap_and_copy()`, `release_remapped_blocks()` |
| **AttentionMaskBuilder** | `vllm-ascend/attention/attention_mask.py` | Chunk-aware mask构建 | `get_chunk_aware_mask()` |
| **Chunk Structures** | `vllm/v1/core/chunk_structures.py` | 数据结构定义 | `ChunkHash`, `ChunkKVCache`, `RemappedChunkKV` |

#### 3.4.2 修改模块清单

| 模块名称 | 文件路径 | 修改内容 | 新增/修改方法 |
|---------|---------|----------|--------------|
| **VllmConfig** | `vllm/config/vllm_config.py` | 添加chunk cache配置字段 | 新增 `chunk_cache_config: ChunkCacheConfig` |
| **ChunkCacheConfig** | `vllm/config/chunk_cache.py` | 新增配置类 | 新增配置参数定义 |
| **LLMEngine** | `vllm/v1/engine/llm_engine.py` | 集成ChunkCacheManager | `__init__()` 中初始化, `_execute_request()` 中调用 |
| **InputProcessor** | `vllm/v1/engine/input_processor.py` | 解析chunked prompt | 新增 `parse_chunked_prompt()` |
| **GPUModelRunner** | `vllm/v1/worker/gpu_model_runner.py` | Chunk KV获取和合并 | 新增 `get_or_compute_chunks()`, `compute_question_kv()`, `merge_chunk_kvs()` |
| **AscendCommonAttentionMetadata** | `vllm-ascend/attention/utils.py` | 扩展attention元数据 | 新增 `chunk_ids`, `chunk_boundaries`, `chunk_attn_mask` |
| **AscendSFABackend** | `vllm-ascend/attention/sfa_v1.py` | 集成chunk-aware mask | `forward()` 中检查并应用chunk mask |

### 3.5 核心流程设计

#### 3.5.1 Chunk缓存查找流程

```
┌────────────────────────────────────────────────────────────┐
│                    请求: 获取 Chunk KV                      │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  Step 1: ChunkHashIndex.compute_hash(tokens)               │
│                                                             │
│  输入: chunk_tokens (list[int])                            │
│  处理: XXHash128(tokens)                                    │
│  输出: ChunkHash {hash_bytes, token_count, num_blocks}     │
│  时间: ~0.01ms (4K tokens)                                 │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  Step 2: ChunkHashIndex.lookup(chunk_hash)                 │
│                                                             │
│  处理: 哈希表查找 O(1)                                      │
│  输出: ChunkKVCache | None                                 │
└───────────────┬─────────────────┬──────────────────────────┘
                ↓                 ↓
        [缓存命中]          [缓存未命中]
                ↓                 ↓
┌──────────────────────┐  ┌──────────────────────────────────┐
│ Step 3a: 拷贝重映射   │  │ Step 3b: 计算并缓存                │
│                      │  │                                    │
│ PositionRemapper:    │  │ 1. ChunkBlockPool.allocate_blocks │
│ remap_and_copy()     │  │ 2. GPUModelRunner.compute_chunk_kv│
│                      │  │    - 在虚拟位置计算                │
│ - 分配物理块          │  │ 3. 存储到 ChunkHashIndex         │
│ - 拷贝KV数据          │  │ 4. 调用 Step 3a 拷贝重映射         │
│ - 应用RoPE编码        │  │                                    │
│ 时间: ~3-5ms         │  │ 时间: ~50-60ms                    │
└───────────────┬──────┘  └──────────────┬───────────────────┘
                ↓                       ↓
┌────────────────────────────────────────────────────────────┐
│                   Step 4: 返回 RemappedChunkKV              │
│                                                             │
│ 包含:                                                       │
│ - block_ids: 新分配的物理块                                 │
│ - key_cache, value_cache: 重映射后的KV (已应用RoPE)        │
│ - position_start, position_end: 实际位置                   │
└────────────────────────────────────────────────────────────┘
```

#### 3.5.2 完整请求处理流程

```
┌────────────────────────────────────────────────────────────┐
│  用户请求: "You are helpful.##Doc1##Doc2##What is this?"   │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  InputProcessor.parse_chunked_prompt()                     │
│                                                             │
│  输入: prompt_tokens, separator="##"                        │
│  输出: ChunkedPrompt {                                      │
│    sys_prompt: [1,2,3,...],     # 作为chunk处理            │
│    chunks: [[10,11,...], [20,21,...]],                    │
│    user_question: [100,101,...]                            │
│  }                                                          │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  GPUModelRunner.get_or_compute_chunks(chunked_prompt)      │
│                                                             │
│  for chunk in chunked_prompt.get_all_chunks():             │
│    → ChunkCacheManager.get_or_compute_chunk(chunk)         │
│      → [查找缓存] → [命中/未命中] → [返回RemappedChunkKV]   │
│                                                             │
│  返回: [sys_chunk_kv, chunk1_kv, chunk2_kv, ...]           │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  GPUModelRunner.compute_question_kv()                      │
│                                                             │
│  处理: user_question 需要关注所有之前的内容                 │
│  输入: question_tokens, chunk_kvs                          │
│  输出: question_kv (在最后位置计算)                        │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  GPUModelRunner.merge_chunk_kvs_for_attention()            │
│                                                             │
│  合并:                                                      │
│  - 所有 remapped chunk KV                                  │
│  - question KV                                             │
│                                                             │
│  输出:                                                      │
│  - merged_key, merged_value                                │
│  - merged_block_ids                                        │
│  - chunk_ids (用于mask构建)                                │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  AttentionMaskBuilder.get_chunk_aware_mask()               │
│                                                             │
│  根据 chunk_ids 构建 attention mask:                       │
│  - sys_prompt: 标准 causal                                 │
│  - chunk tokens: 可看 sys_prompt + 同chunk (causal)        │
│  - question: 可看所有 (causal)                             │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  AscendSFABackend.forward()                                │
│                                                             │
│  执行 SFA attention:                                       │
│  - 使用 merged KV cache (已应用正确的RoPE)                 │
│  - 应用 chunk-aware mask                                   │
│  - 生成 output                                             │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  OutputProcessor.format_output()                           │
│  ↓                                                         │
│  返回生成结果给用户                                         │
└────────────────────────────────────────────────────────────┘
```

#### 3.5.3 KV拷贝和重映射详细流程

```
┌────────────────────────────────────────────────────────────┐
│  PositionRemapper.remap_and_copy(cached_kv, new_position)  │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  Step 1: 分配物理块                                         │
│                                                             │
│  num_blocks = len(cached_kv.block_ids)                     │
│  new_block_ids = ChunkBlockPool.allocate_blocks(num_blocks)│
│                                                             │
│  示例: cached_kv.block_ids = [100, 101]                    │
│        new_block_ids = [500, 501]  (新分配)                │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  Step 2: 计算新位置序列                                      │
│                                                             │
│  new_positions = torch.arange(                              │
│      new_position,                                          │
│      new_position + num_tokens,                             │
│      device="npu"                                           │
│  )                                                          │
│                                                             │
│  示例: new_position = 1000, num_tokens = 200               │
│        new_positions = [1000, 1001, ..., 1199]             │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  Step 3: 拷贝KV数据并应用RoPE                                │
│                                                             │
│  _copy_and_apply_rope():                                   │
│                                                             │
│  for src_bid, dst_bid in zip(src_blocks, dst_blocks):      │
│    # 使用NPU加速拷贝+RoPE                                  │
│    dst_k[dst_bid], dst_v[dst_bid] = npu_kv_copy_and_rope( │
│        src_k=src_k[src_bid],                                │
│        src_v=src_v[src_bid],                                │
│        src_position=cached_kv.virtual_position_start,      │
│        dst_position=new_position,                           │
│        cos=cos, sin=sin                                     │
│    )                                                        │
│                                                             │
│  NPU Kernel: torch_npu.npu_kv_rmsnorm_rope_cache()          │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  Step 4: 创建RemappedChunkKV                                │
│                                                             │
│  return RemappedChunkKV(                                   │
│      block_ids=new_block_ids,         # [500, 501]         │
│      key_cache=dst_k,                  # 已应用新位置RoPE  │
│      value_cache=dst_v,                # 已应用新位置RoPE  │
│      position_start=1000,                                     │
│      position_end=1200,                                       │
│      num_tokens=200,                                         │
│      chunk_hash=cached_kv.chunk_hash                         │
│  )                                                          │
└────────────────────────────────────────────────────────────┘
```

### 3.6 数据流图

```
┌─────────────┐
│ User Prompt │ "You are helpful.##Doc1##What is this?"
└──────┬──────┘
       ↓
┌─────────────────────────┐
│  InputProcessor         │
│  - tokenize             │ → [1,2,3, 99, 10,11, 99, 100,101]
│  - parse_chunked_prompt │
└──────┬──────────────────┘
       ↓
┌──────────────────────────────────────────────────────────┐
│  ChunkedPrompt                                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ sys_prompt │  │ chunks     │  │ question   │        │
│  │ [1,2,3]    │  │ [[10,11]]  │  │ [100,101]  │        │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘        │
└────────┼───────────────┼─────────────────┼────────────────┘
         ↓               ↓                 ↓
    ┌─────────────────────────────────────────────────┐
    │  ChunkCacheManager.get_or_compute_chunk()       │
    │  ┌─────────────────────────────────────────┐   │
    │  │ ChunkHashIndex.compute_hash()            │   │
    │  │ sys: hash_abc → [MISS]                   │   │
    │  │ chunk1: hash_def → [HIT/MISS]            │   │
    │  └────────────────┬────────────────────────┘   │
    │                   ↓                             │
    │  ┌─────────────────────────────────────────┐   │
    │  │ [HIT] → PositionRemapper.remap_and_copy│   │
    │  │ [MISS] → compute + cache + remap       │   │
    │  └────────────────┬────────────────────────┘   │
    └───────────────────┼─────────────────────────────┘
                        ↓
    ┌─────────────────────────────────────────────────┐
    │  RemappedChunkKV (每个chunk一个)                 │
    │  ┌────────────┐  ┌────────────┐                 │
    │  │ sys_kv     │  │ chunk1_kv  │                 │
    │  │ blocks:0-1 │  │ blocks:2-3 │                 │
    │  │ pos:0-2    │  │ pos:3-4    │                 │
    │  └─────┬──────┘  └─────┬──────┘                 │
    └────────┼───────────────┼─────────────────────────┘
             ↓               ↓
    ┌─────────────────────────────────────────────────┐
    │  GPUModelRunner.compute_question_kv()           │
    │  - 需要关注所有之前的内容                         │
    │  - blocks:4-5, pos:5-6                          │
    └────────────────────┬────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │  合并KV用于attention                             │
    │  - sys_kv (blocks 0-1)                          │
    │  - chunk1_kv (blocks 2-3)                       │
    │  - question_kv (blocks 4-5)                     │
    │  → merged_blocks: [0,1,2,3,4,5]                 │
    └────────────────────┬────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │  AttentionMaskBuilder.get_chunk_aware_mask()    │
    │  - 构建 chunk 隔离的 attention mask             │
    └────────────────────┬────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │  AscendSFABackend.forward()                     │
    │  - SFA attention with chunk-aware mask          │
    │  - 生成 output                                  │
    └────────────────────┬────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │  Generated Tokens                               │
    │  [102, 103, 104, ...]                           │
    └─────────────────────────────────────────────────┘
```

### 3.7 内存管理设计

#### 3.7.1 双内存池架构

```
┌─────────────────────────────────────────────────────────────┐
│                      KV Cache Memory Layout                  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │         主 KV Cache Pool (vLLM原生)                   │  │
│  │  - 用于存储请求处理时的活跃KV                        │  │
│  │  - 包括: sys_prompt, chunks, question, decode KV     │  │
│  │  - 管理: vLLM KVCacheManager                         │  │
│  │  - 大小: gpu_memory_utilization * 0.7                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      Chunk Cache Pool (新增,独立管理)                 │  │
│  │  - 用于缓存位置无关的chunk KV                        │  │
│  │  - 标签: "chunk_cache" (CaMemAllocator)              │  │
│  │  - 管理: ChunkBlockPool                              │  │
│  │  - 大小: chunk_cache_gpu_memory_utilization          │  │
│  │  - 淘汰: LRU策略                                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**内存分配示例**:
```
配置:
- GPU总内存: 32GB
- gpu_memory_utilization: 0.9 (28.8GB)
- 主KV池: 0.7 * 28.8 = 20GB
- Chunk缓存池: 0.15 * 28.8 = 4.3GB
- 预留: 0.05 * 28.8 = 1.5GB

Chunk Pool容量 (block_size=16, head_dim=128, num_heads=32):
- 单block: 16 * 128 * 32 * 2 * 2 bytes = 256KB
- 可缓存blocks: 4.3GB / 256KB ≈ 17,000 blocks
- 可缓存tokens: 17,000 * 16 ≈ 272K tokens
```

#### 3.7.2 重映射内存管理

```
┌────────────────────────────────────────────────────────────┐
│           重映射时的内存分配流程                            │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  Step 1: 从主KV池分配 (用于重映射后的KV)                    │
│                                                             │
│  block_ids = main_kv_pool.allocate(num_blocks)             │
│  示例: [5000, 5001, 5002, 5003]  (主池)                    │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  Step 2: 从Chunk缓存池读取源KV                              │
│                                                             │
│  cached_kv = chunk_hash_index.lookup(chunk_hash)           │
│  cached_kv.block_ids = [100, 101]  (chunk池)              │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  Step 3: 拷贝并重映射 (chunk池 → 主池)                      │
│                                                             │
│  从 block 100,101 (chunk池)                                │
│  拷贝到 block 5000-5003 (主池)                             │
│  同时应用新位置的RoPE编码                                  │
└───────────────────────────┬────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│  Step 4: 请求结束后释放主池中的重映射块                     │
│                                                             │
│  main_kv_pool.free([5000, 5001, 5002, 5003])               │
│  注意: chunk池中的缓存块保留                               │
└────────────────────────────────────────────────────────────┘
```

#### 3.7.3 LRU淘汰策略

```python
class ChunkBlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.lru_cache = LRUCache()  # chunk_hash → ChunkKVCache
        self.block_to_chunk = {}     # block_id → chunk_hash

    def evict_lru_until_enough(self, required_blocks: int):
        """
        淘汰LRU chunks直到有足够空间

        策略:
        1. 找到最久未访问的chunks
        2. 检查ref_count (确保没有被使用)
        3. 释放blocks到free池
        4. 从hash_index中移除
        """
        freed = 0
        while freed < required_blocks:
            # 获取LRU chunk
            chunk_hash, cached_kv = self.lru_cache.get_lru()

            # 检查是否可以淘汰
            if cached_kv.ref_count == 0:
                # 释放blocks
                self.free_blocks.extend(cached_kv.block_ids)
                freed += len(cached_kv.block_ids)

                # 清理
                for bid in cached_kv.block_ids:
                    del self.block_to_chunk[bid]
                del self.lru_cache[chunk_hash]
            else:
                # 跳过正在使用的chunk
                self.lru_cache.move_to_tail(chunk_hash)
```

### 3.8 关键接口详细定义

#### 3.8.1 ChunkCacheManager接口

```python
class ChunkCacheManager:
    """
    Chunk缓存管理器 - 核心接口
    """

    def __init__(
        self,
        config: ChunkCacheConfig,
        kv_cache_spec: KVCacheSpec,
        block_size: int,
        ascend_allocator: Optional[CaMemAllocator] = None,
    ):
        """
        初始化

        Args:
            config: chunk缓存配置
            kv_cache_spec: KV cache规格 (num_kv_heads, head_dim)
            block_size: 块大小 (tokens per block)
            ascend_allocator: 昇腾内存分配器 (可选)
        """
        self.chunk_hash_index = ChunkHashIndex(
            hash_algo=config.chunk_hash_algo
        )
        self.chunk_block_pool = ChunkBlockPool(
            num_blocks=config.max_chunks,
            block_size=block_size,
            ascend_allocator=ascend_allocator,
        )
        self.position_remapper = PositionRemapper(
            block_pool=self.chunk_block_pool,
            rope_config=...,  # 从模型配置获取
            device="npu",
        )
        self.metrics = ChunkCacheMetrics()

    def get_or_compute_chunk(
        self,
        chunk_tokens: list[int],
        position_offset: int,
        model_runner: GPUModelRunner,
    ) -> RemappedChunkKV:
        """
        获取或计算chunk KV cache

        Args:
            chunk_tokens: chunk的token序列
            position_offset: 目标位置偏移
            model_runner: 模型运行器

        Returns:
            RemappedChunkKV: 重映射后的chunk KV (在主KV池中)

        Raises:
            MemoryError: 内存不足 (即使LRU淘汰后)
        """
        # Step 1: 计算hash
        chunk_hash = self.chunk_hash_index.compute_hash(
            chunk_tokens,
            self.chunk_block_pool.block_size,
        )

        # Step 2: 查找缓存
        cached_chunk = self.chunk_hash_index.lookup(chunk_hash)

        if cached_chunk is not None:
            self.metrics.cache_hit()
            cached_chunk.touch()

            # Step 3a: 拷贝重映射
            return self.position_remapper.remap_and_copy(
                cached_chunk, position_offset
            )

        # Step 3b: 缓存未命中
        self.metrics.cache_miss()

        # 检查并淘汰LRU
        required_blocks = (len(chunk_tokens) + self.chunk_block_pool.block_size - 1) // self.chunk_block_pool.block_size
        self.chunk_block_pool.evict_lru_until_enough(required_blocks)

        # 在虚拟位置计算KV
        virtual_position = self._get_virtual_position(len(chunk_tokens))
        computed_kv = model_runner.compute_chunk_kv(
            tokens=chunk_tokens,
            position=virtual_position,
        )

        # 存储到chunk池
        self.chunk_hash_index.insert(chunk_hash, computed_kv)

        # 拷贝重映射到目标位置
        return self.position_remapper.remap_and_copy(
            computed_kv, position_offset
        )

    def get_stats(self) -> ChunkCacheMetrics:
        """获取缓存统计信息"""
        return self.metrics.as_dict()

    def clear(self):
        """清空所有缓存 (测试用)"""
        self.chunk_hash_index.clear()
        self.chunk_block_pool.reset()
        self.metrics.reset()
```

#### 3.8.2 PositionRemapper接口

```python
class PositionRemapper:
    """
    位置重映射器 - KV拷贝和RoPE重新编码
    """

    def __init__(
        self,
        block_pool: ChunkBlockPool,
        rope_config: RoPEConfig,
        device: torch.device,
    ):
        self.block_pool = block_pool
        self.rope_config = rope_config
        self.device = device

    def remap_and_copy(
        self,
        chunk_kv_cache: ChunkKVCache,
        new_position_offset: int,
    ) -> RemappedChunkKV:
        """
        拷贝并重映射chunk KV cache

        Args:
            chunk_kv_cache: 缓存的chunk KV (在chunk池中)
            new_position_offset: 新的位置偏移

        Returns:
            RemappedChunkKV: 重映射后的KV (在主KV池中)

        时间复杂度: O(num_tokens)
        空间复杂度: O(num_tokens) (新分配blocks)
        """
        num_tokens = chunk_kv_cache.num_tokens
        num_blocks = len(chunk_kv_cache.block_ids)

        # Step 1: 从主KV池分配 (不是chunk池!)
        # 注意: 这里应该从主KV pool分配，而不是chunk pool
        # 主KV pool由GPUModelRunner管理
        new_block_ids = self._allocate_from_main_pool(num_blocks)

        # Step 2: 计算新位置
        new_positions = torch.arange(
            new_position_offset,
            new_position_offset + num_tokens,
            dtype=torch.long,
            device=self.device,
        )

        # Step 3: 拷贝并应用RoPE
        remapped_k, remapped_v = self._copy_and_apply_rope(
            src_k=chunk_kv_cache.key_cache,
            src_v=chunk_kv_cache.value_cache,
            src_blocks=chunk_kv_cache.block_ids,
            dst_blocks=new_block_ids,
            new_positions=new_positions,
            num_tokens=num_tokens,
        )

        # Step 4: 返回重映射后的KV
        return RemappedChunkKV(
            block_ids=new_block_ids,
            key_cache=remapped_k,
            value_cache=remapped_v,
            position_start=new_position_offset,
            position_end=new_position_offset + num_tokens,
            positions=new_positions,
            num_tokens=num_tokens,
            chunk_hash=chunk_kv_cache.chunk_hash,
            source_cache_ref=chunk_kv_cache,
        )

    def release_remapped_blocks(self, block_ids: list[int]):
        """
        释放重映射使用的物理块 (请求结束后)

        Args:
            block_ids: 要释放的block IDs
        """
        # 归还到主KV pool
        self._free_to_main_pool(block_ids)

    def _copy_and_apply_rope(
        self,
        src_k: torch.Tensor,
        src_v: torch.Tensor,
        src_blocks: list[int],
        dst_blocks: list[int],
        new_positions: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        拷贝KV数据并应用新的RoPE编码

        使用昇腾NPU优化: torch_npu.npu_kv_rmsnorm_rope_cache()
        """
        import torch_npu

        # 计算RoPE编码
        cos, sin = self._compute_rope_for_positions(new_positions)

        # 准备目标KV
        dst_k = torch.empty_like(src_k)
        dst_v = torch.empty_like(src_v)

        # 使用NPU kernel拷贝并应用RoPE
        for src_bid, dst_bid in zip(src_blocks, dst_blocks):
            dst_k[dst_bid], dst_v[dst_bid] = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv_input=None,  # 直接拷贝KV
                gamma=None,     # 无需rmsnorm (已应用)
                cos=cos,
                sin=sin,
                positions=new_positions,
                key_cache=src_k,
                value_cache=src_v,
                block_table=torch.tensor([src_bid]),
                cache_mode="PA",
            )

        return dst_k, dst_v
```

#### 3.8.3 GPUModelRunner扩展接口

```python
class GPUModelRunner:
    """
    GPU模型运行器 - Chunk缓存扩展
    """

    # ==================== 新增方法 ====================

    def parse_chunked_prompt(
        self,
        prompt_tokens: list[int],
        separator: str = "##",
    ) -> ChunkedPrompt:
        """
        解析分块提示词

        格式: sys_prompt + "##" + chunk1 + "##" + chunk2 + "##" + ... + question

        Args:
            prompt_tokens: tokenized prompt
            separator: 分隔符token (默认"##")

        Returns:
            ChunkedPrompt: 解析后的分块结构

        Raises:
            ValueError: 格式错误
        """
        # 将分隔符字符串转为token
        separator_tokens = self.tokenizer.encode(separator, add_special_tokens=False)

        # 查找分隔符位置
        sep_positions = []
        for i in range(len(prompt_tokens) - len(separator_tokens) + 1):
            if prompt_tokens[i:i+len(separator_tokens)] == separator_tokens:
                sep_positions.append(i)

        if len(sep_positions) < 1:
            raise ValueError(f"No separator '{separator}' found in prompt")

        # 解析: sys_prompt (第一个分隔符之前)
        sys_prompt = prompt_tokens[:sep_positions[0]]

        # 解析: chunks (分隔符之间)
        chunks = []
        for i in range(len(sep_positions) - 1):
            start = sep_positions[i] + len(separator_tokens)
            end = sep_positions[i + 1]
            chunks.append(prompt_tokens[start:end])

        # 解析: question (最后一个分隔符之后)
        last_sep_end = sep_positions[-1] + len(separator_tokens)
        user_question = prompt_tokens[last_sep_end:]

        return ChunkedPrompt(
            sys_prompt=sys_prompt,
            chunks=chunks,
            user_question=user_question,
            separator=separator,
        )

    def get_or_compute_chunks(
        self,
        chunked_prompt: ChunkedPrompt,
    ) -> list[RemappedChunkKV]:
        """
        获取或计算所有chunks的KV cache (包括sys_prompt)

        Args:
            chunked_prompt: 解析后的分块提示词

        Returns:
            list[RemappedChunkKV]: 所有chunk的重映射KV

        注意: user_question不在这里处理,需要单独调用compute_question_kv()
        """
        remapped_kvs = []
        current_position = 0

        # 处理sys_prompt和所有chunks
        for chunk_tokens in chunked_prompt.get_all_chunks():
            remapped_kv = self.chunk_cache_manager.get_or_compute_chunk(
                chunk_tokens=chunk_tokens,
                position_offset=current_position,
                model_runner=self,
            )
            remapped_kvs.append(remapped_kv)
            current_position += len(chunk_tokens)

        return remapped_kvs

    def compute_question_kv(
        self,
        question_tokens: list[int],
        chunk_kvs: list[RemappedChunkKV],
    ) -> KVCache:
        """
        计算user_question的KV cache

        注意: question需要关注所有之前的内容,所以需要完整上下文

        Args:
            question_tokens: question的token序列
            chunk_kvs: 所有之前chunks的重映射KV

        Returns:
            KVCache: question的KV cache
        """
        # 计算question的起始位置
        question_position = sum(kv.num_tokens for kv in chunk_kvs)

        # 准备完整上下文
        all_tokens = []
        for kv in chunk_kvs:
            all_tokens.extend(...)  # 获取chunk的tokens
        all_tokens.extend(question_tokens)

        # 在完整上下文后计算question的KV
        question_kv = self._execute_model(
            tokens=all_tokens,
            start_pos=question_position,
        )

        return question_kv

    def merge_chunk_kvs_for_attention(
        self,
        chunk_kvs: list[RemappedChunkKV],
        question_kv: KVCache,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        """
        合并所有KV cache用于attention计算

        Args:
            chunk_kvs: 所有chunk的重映射KV
            question_kv: question的KV

        Returns:
            tuple: (merged_key, merged_value, merged_block_ids, chunk_ids)
        """
        # 收集所有block IDs
        all_block_ids = []
        for kv in chunk_kvs:
            all_block_ids.extend(kv.block_ids)
        all_block_ids.extend(question_kv.block_ids)

        # 构建chunk_ids (用于attention mask)
        chunk_ids = self._build_chunk_ids(chunk_kvs, question_kv)

        # 合并KV (逻辑合并,物理KV已在各自的blocks中)
        merged_key = ...  # 逻辑索引
        merged_value = ...

        return merged_key, merged_value, all_block_ids, chunk_ids
```

---

## 4. 数据结构设计

### 4.1 ChunkHash (内容哈希)

```python
@dataclass(frozen=True)
class ChunkHash:
    """
    Chunk 内容哈希（位置无关）

    特性:
    - frozen=True: 不可变，可作为字典键
    - 仅包含内容信息，不包含位置信息
    """
    hash_bytes: bytes          # XXHash128(16 bytes) 或 SHA256(32 bytes)
    token_count: int           # Token 数量
    num_blocks: int            # 占用的 block 数量

    def __hash__(self) -> int:
        return int.from_bytes(self.hash_bytes[:16], byteorder='big')

    def __eq__(self, other) -> bool:
        return self.hash_bytes == other.hash_bytes
```

### 4.2 ChunkKVCache (缓存的 Chunk KV)

```python
@dataclass
class ChunkKVCache:
    """
    Chunk KV Cache 数据结构 - 位置无关缓存

    设计理念:
    1. **虚拟位置计算**: 所有chunk在统一虚拟位置 [VIRTUAL_POS_START, VIRTUAL_POS_START + max_chunk_len) 计算KV
    2. **位置无关复用**: 缓存key仅包含内容哈希，支持跨位置复用
    3. **使用时重映射**: 将缓存的KV数据拷贝并重映射到实际位置

    重映射开销:
    - KV数据拷贝: O(num_tokens)
    - RoPE重新编码: O(num_tokens)
    - Position更新: O(num_tokens)
    """
    chunk_hash: ChunkHash
    num_tokens: int
    block_ids: list[int]              # 物理块 ID 列表 (连续或不连续)
    block_size: int

    # KV数据 (在虚拟位置计算，未应用RoPE或应用了虚拟位置的RoPE)
    # Shape: [num_blocks, block_size, 2, num_kv_heads, head_dim]
    # 2 表示 [K, V]
    key_cache: torch.Tensor
    value_cache: torch.Tensor

    # 虚拟位置信息
    virtual_position_start: int       # 虚拟起始位置
    virtual_position_end: int         # 虚拟结束位置

    # LRU 元数据
    created_at: float
    last_accessed: float
    access_count: int
    ref_count: int = 0
```

**关键设计**: KV数据缓存在虚拟位置，使用时需要**拷贝并重映射**到实际位置。

### 4.3 RemappedChunkKV (重映射后的Chunk) - 包含KV拷贝

```python
@dataclass
class RemappedChunkKV:
    """
    重映射后的 Chunk KV Cache - 包含KV数据拷贝

    设计理念:
    - **显式拷贝**: 明确KV数据需要从缓存拷贝到目标位置
    - **位置编码更新**: 拷贝时应用新的位置编码
    - **独立生命周期**: 重映射后的KV独立于原始缓存

    拷贝操作:
    1. 分配新的物理块
    2. 拷贝KV数据
    3. 应用新的RoPE编码
    4. 更新位置元数据

    性能考虑:
    - 拷贝开销: ~1-3ms (4K tokens, NPU)
    - RoPE编码: ~0.5-1ms (4K tokens)
    - 总开销: ~2-5ms (相比计算KV的50ms，仍可节省90%+)
    """
    # 重映射后的数据
    block_ids: list[int]              # 新分配的物理块 ID
    key_cache: torch.Tensor           # 重映射后的K (应用了新位置的RoPE)
    value_cache: torch.Tensor         # 重映射后的V

    # 位置信息
    position_start: int               # 实际位置
    position_end: int                 # 实际位置 + num_tokens
    positions: torch.Tensor           # 实际位置序列 [num_tokens]

    # 元数据
    num_tokens: int
    chunk_hash: ChunkHash             # 保留原始hash (用于调试)
    source_cache_ref: ChunkKVCache | None  # 源缓存引用 (可选，用于统计)
```

**有拷贝设计**: `RemappedChunkKV` 包含实际拷贝的KV数据，位置编码已更新。

### 4.4 ChunkedPrompt (分块提示词)

```python
@dataclass
class ChunkedPrompt:
    """
    分块提示词结构

    设计理念:
    - **统一chunk处理**: sys_prompt也作为一个特殊chunk处理
    - **位置无关**: 所有chunk (包括sys_prompt) 都基于内容哈希缓存
    - **chunk隔离**: user_question是唯一可以关注所有内容的部分
    """
    sys_prompt: list[int]           # 系统提示词 (作为特殊chunk处理)
    chunks: list[list[int]]         # 文档chunks (每个独立缓存)
    user_question: list[int]        # 用户问题 (不缓存，每次计算)
    separator: str = "##"           # 分隔符

    def get_all_chunks(self) -> list[list[int]]:
        """
        获取所有需要缓存的chunks (包括sys_prompt)

        返回: [sys_prompt, chunk1, chunk2, ...]
        """
        return [self.sys_prompt] + self.chunks

    def get_total_cached_tokens(self) -> int:
        """计算所有缓存chunk的总token数"""
        return len(self.sys_prompt) + sum(len(c) for c in self.chunks)
```

---

## 5. 核心算法

### 5.1 Chunk Hash 计算算法

```python
import xxhash

def compute_chunk_hash(
    tokens: list[int],
    block_size: int,
    hash_algo: str = "xxhash",
) -> ChunkHash:
    """
    计算 chunk 内容哈希（位置无关）

    时间复杂度: O(n)，n = len(tokens)
    """
    token_count = len(tokens)
    num_blocks = (token_count + block_size - 1) // block_size

    if hash_algo == "xxhash":
        hash_bytes = xxhash.xxh128(bytes(tokens)).digest()
    elif hash_algo == "sha256":
        import hashlib
        token_bytes = tuple(tokens).__repr__().encode()
        hash_bytes = hashlib.sha256(token_bytes).digest()

    return ChunkHash(hash_bytes, token_count, num_blocks)
```

**性能**:
- XXHash128: ~0.01ms (4K tokens)
- SHA256: ~0.05ms (4K tokens)

### 5.2 Chunk Cache 查找算法

```python
def get_or_compute_chunk(
    chunk_tokens: list[int],
    position_offset: int,
    model_runner,
    chunk_cache_manager,
) -> RemappedChunkKV:
    """
    Chunk Cache 查找算法

    时间复杂度:
    - 缓存命中: O(1) + O(n) 拷贝重映射
    - 缓存未命中: O(n) 计算 + O(n) 拷贝重映射

    空间复杂度:
    - 缓存命中: O(n) 额外空间 (重映射后的KV)
    - 缓存未命中: O(n) 缓存空间 + O(n) 重映射空间
    """
    # Step 1: 计算内容哈希
    chunk_hash = chunk_cache_manager.chunk_hash_index.compute_hash(
        chunk_tokens,
        chunk_cache_manager.block_size,
    )

    # Step 2: 查找缓存
    cached_chunk = chunk_cache_manager.chunk_hash_index.lookup(chunk_hash)

    if cached_chunk is not None:
        # 缓存命中: 拷贝并重映射到目标位置
        cached_chunk.touch()
        return chunk_cache_manager.position_remapper.remap_and_copy(
            cached_chunk, position_offset
        )

    # 缓存未命中: 计算并缓存
    # 1. 检查内存，必要时淘汰 LRU
    # 2. 在虚拟位置 [VIRTUAL_POS_START, VIRTUAL_POS_START + len(chunk)) 计算 KV cache
    # 3. 存储 chunk 到缓存池
    # 4. 拷贝并重映射到目标位置
    # 5. 返回重映射后的KV
```

**性能分析**:

| 场景 | 时间 (4K tokens) | 主要开销 |
|------|------------------|----------|
| 缓存命中 | 3-8ms | Hash(0.01ms) + Lookup(O(1)) + 拷贝重映射(3-5ms) |
| 缓存未命中 | 55-65ms | Hash + Compute(50ms) + 拷贝重映射(3-5ms) |

**加速比**: 60ms / 5ms ≈ **12x** (考虑拷贝开销)

**拷贝开销分析**:
- KV数据拷贝 (NPU): ~1-3ms (4K tokens)
- RoPE重新编码: ~0.5-1ms (4K tokens)
- 相比完整计算的50ms，拷贝开销仅占~10%
- **净收益**: 仍可节省 ~90% 计算时间

### 5.3 位置重映射算法 (包含KV拷贝)

```python
class PositionRemapper:
    def __init__(self, block_pool, rope_config, device):
        self.block_pool = block_pool
        self.rope_config = rope_config
        self.device = device

    def remap_and_copy(
        self,
        chunk_kv_cache: ChunkKVCache,
        new_position_offset: int,
    ) -> RemappedChunkKV:
        """
        Chunk KV Cache 位置重映射算法 - 包含KV数据拷贝

        核心步骤:
        1. 分配新的物理块
        2. 拷贝KV数据到新位置
        3. 应用新的RoPE位置编码
        4. 更新位置元数据

        时间复杂度: O(n)，n = chunk_kv_cache.num_tokens
        空间复杂度: O(n) (新分配的块)
        """
        num_tokens = chunk_kv_cache.num_tokens
        num_blocks = len(chunk_kv_cache.block_ids)

        # Step 1: 分配新的物理块
        new_block_ids = self.block_pool.allocate_blocks(num_blocks)

        # Step 2: 计算新的位置序列
        new_positions = torch.arange(
            new_position_offset,
            new_position_offset + num_tokens,
            dtype=torch.long,
            device=self.device,
        )

        # Step 3: 拷贝KV数据并应用新的RoPE编码
        # 从缓存的虚拟位置KV拷贝，并应用新位置的RoPE
        remapped_k, remapped_v = self._copy_and_apply_rope(
            src_k=chunk_kv_cache.key_cache,      # 源K (虚拟位置)
            src_v=chunk_kv_cache.value_cache,    # 源V (虚拟位置)
            src_blocks=chunk_kv_cache.block_ids, # 源块ID
            dst_blocks=new_block_ids,            # 目标块ID
            new_positions=new_positions,         # 新位置
            num_tokens=num_tokens,
        )

        # Step 4: 创建重映射后的 chunk KV (包含实际拷贝的数据)
        return RemappedChunkKV(
            block_ids=new_block_ids,
            key_cache=remapped_k,
            value_cache=remapped_v,
            position_start=new_position_offset,
            position_end=new_position_offset + num_tokens,
            positions=new_positions,
            num_tokens=num_tokens,
            chunk_hash=chunk_kv_cache.chunk_hash,
            source_cache_ref=chunk_kv_cache,
        )

    def _copy_and_apply_rope(
        self,
        src_k: torch.Tensor,
        src_v: torch.Tensor,
        src_blocks: list[int],
        dst_blocks: list[int],
        new_positions: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        拷贝KV数据并应用新的RoPE编码

        实现策略:
        1. 从源块拷贝原始KV (未应用RoPE或应用了虚拟位置的RoPE)
        2. 应用新位置的RoPE编码

        昇腾NPU优化:
        - 使用 torch_npu.npu_kv_rmsnorm_rope_cache 进行高效拷贝和RoPE
        - 利用NPU的tensor copy加速
        """
        # 准备目标KV缓存
        dst_k = torch.empty_like(src_k)
        dst_v = torch.empty_like(src_v)

        # 方案A: 如果缓存的是未应用RoPE的原始KV
        if self._cache_raw_kv():
            # 1. 拷贝原始KV
            for src_bid, dst_bid in zip(src_blocks, dst_blocks):
                dst_k[dst_bid] = src_k[src_bid].clone()
                dst_v[dst_bid] = src_v[src_bid].clone()

            # 2. 应用新位置的RoPE
            import torch_npu
            cos, sin = self._compute_rope_for_positions(new_positions)
            dst_k, dst_v = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv_input=None,  # 已有KV数据
                gamma=None,     # 无需rmsnorm
                cos=cos,
                sin=sin,
                positions=new_positions,
                key_cache=dst_k,
                value_cache=dst_v,
                block_ids=dst_blocks,
            )

        # 方案B: 如果缓存的是应用了虚拟位置RoPE的KV
        else:
            # 1. 拷贝并移除旧的RoPE，应用新的RoPE
            # 使用NPU的高效copy + rope kernel
            import torch_npu
            cos, sin = self._compute_rope_for_positions(new_positions)

            for src_bid, dst_bid in zip(src_blocks, dst_blocks):
                # 使用NPU加速的拷贝+RoPE
                dst_k[dst_bid], dst_v[dst_bid] = torch_npu.npu_kv_copy_and_apply_rope(
                    src_k=src_k[src_bid],
                    src_v=src_v[src_bid],
                    src_position=chunk_kv_cache.virtual_position_start,
                    dst_position=new_position_offset,
                    cos=cos,
                    sin=sin,
                )

        return dst_k, dst_v
```

**拷贝操作验证**:
```python
remapped_kv = remapper.remap_and_copy(cached_kv, new_position=1000)

# 验证拷贝: 是不同的对象
assert remapped_kv.block_ids != cached_kv.block_ids
assert remapped_kv.key_cache is not cached_kv.key_cache
assert remapped_kv.value_cache is not cached_kv.value_cache

# 验证数据正确性: 数值已应用新位置的RoPE
assert remapped_kv.position_start == 1000
```

**性能优化策略**:
1. **NPU异步拷贝**: 使用NPU的异步拷贝API，与CPU并行
2. **批量处理**: 一次拷贝多个chunks，减少kernel启动开销
3. **内存池复用**: 重映射后的块在请求结束后归还给池，而非立即释放

### 5.4 Chunk-Aware Attention Mask 构建

```python
def get_chunk_aware_mask(
    num_tokens: int,
    chunk_ids: torch.Tensor,  # [-1=sys, 0,1,2=chunks, -2=question]
    chunk_boundaries: list[tuple[int, int]],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    构建 chunk 隔离的 causal attention mask

    规则:
    - sys_prompt tokens: 标准 causal attention
    - chunk tokens: 可以关注 sys_prompt + 同 chunk 内的前序 token (causal)
    - question tokens: 可以关注所有内容 (causal)
    """
    mask_value = torch.finfo(torch.float32).min if dtype == torch.float16 else 1.0
    mask = torch.zeros(num_tokens, num_tokens, dtype=dtype, device=device)

    for query_pos in range(num_tokens):
        query_chunk_id = chunk_ids[query_pos].item()

        if query_chunk_id >= 0:  # Chunk token
            # 可以关注 sys_prompt
            sys_end = chunk_boundaries[0][0] if chunk_boundaries else num_tokens
            mask[query_pos, :sys_end] = 0

            # 可以关注同 chunk 内的前序 token (causal)
            chunk_start, chunk_end = chunk_boundaries[query_chunk_id]
            causal_end = min(query_pos, chunk_end - 1) + 1
            mask[query_pos, chunk_start:causal_end] = 0

            # 不能关注其他 chunk 或未来位置
            mask[query_pos, causal_end:] = mask_value

        elif query_chunk_id == -2:  # Question token
            mask[query_pos, :query_pos] = 0
            mask[query_pos, query_pos:] = mask_value

        else:  # sys_prompt token
            mask[query_pos, :query_pos] = 0
            mask[query_pos, query_pos:] = mask_value

    return mask
```

---

## 6. 模块设计

### 6.1 ChunkHashIndex (哈希索引模块)

**文件**: `vllm/v1/core/chunk_hash_index.py`

**职责**:
- 计算内容哈希
- 管理 chunk hash → chunk KV cache 映射
- 提供 O(1) 查找

**接口**:
```python
class ChunkHashIndex:
    def __init__(self, hash_algo: str = "xxhash"):
        self.index: Dict[ChunkHash, ChunkKVCache] = {}
        self.hash_algo = hash_algo

    def compute_hash(self, tokens: list[int], block_size: int) -> ChunkHash:
        """计算内容哈希"""
        pass

    def lookup(self, chunk_hash: ChunkHash) -> ChunkKVCache | None:
        """查找 chunk"""
        pass

    def insert(self, chunk_hash: ChunkHash, chunk_kv: ChunkKVCache):
        """插入 chunk"""
        pass
```

### 6.2 ChunkBlockPool (Block 池管理)

**文件**: `vllm/v1/core/chunk_block_pool.py`

**职责**:
- 管理独立的 block 池
- 分配和释放 blocks
- LRU 淘汰策略
- 与 CaMemAllocator 集成 (Ascend)

**关键实现**:
```python
class ChunkBlockPool:
    def __init__(self, num_blocks: int, block_size: int, ascend_allocator=None):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.lru_cache = LRUCache()

        if ascend_allocator:
            # 使用 "chunk_cache" tag 创建独立内存池
            with ascend_allocator.use_memory_pool(tag="chunk_cache"):
                # 预分配内存
                pass

    def allocate_blocks(self, num_blocks: int) -> list[int]:
        """分配 blocks"""
        if len(self.free_blocks) < num_blocks:
            raise MemoryError("Insufficient free blocks")
        allocated = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        return allocated

    def free_blocks(self, block_ids: list[int]):
        """释放 blocks"""
        self.free_blocks.extend(block_ids)
```

### 6.3 ChunkCacheManager (核心管理器)

**文件**: `vllm/v1/core/chunk_cache_manager.py`

**职责**:
- 协调 ChunkHashIndex 和 ChunkBlockPool
- 提供统一的 chunk 获取/计算接口
- 管理生命周期 (创建、缓存、淘汰)
- 统计信息收集

**接口**:
```python
class ChunkCacheManager:
    def __init__(self, config, kv_cache_spec, block_size, ascend_allocator=None):
        self.chunk_hash_index = ChunkHashIndex(hash_algo=config.chunk_hash_algo)
        self.chunk_block_pool = ChunkBlockPool(
            num_blocks=config.max_chunks,
            block_size=block_size,
            ascend_allocator=ascend_allocator,
        )
        self.position_remapper = None
        self.metrics = ChunkCacheMetrics()

    def get_or_compute_chunk(
        self,
        chunk_tokens: list[int],
        position_offset: int,
        model_runner,
    ) -> RemappedChunkKV:
        """
        获取或计算 chunk KV cache

        缓存命中: 从缓存拷贝KV并重映射到目标位置
        缓存未命中: 计算KV，缓存，然后拷贝重映射到目标位置
        """
        pass

    def evict_lru_until_enough(self, required_blocks: int):
        """淘汰 LRU chunks 直到有足够空间"""
        pass
```

### 6.4 PositionRemapper (位置重映射器 - KV拷贝)

**文件**: `vllm/v1/worker/position_remapper.py`

**职责**:
- KV数据拷贝和位置编码重映射
- RoPE cos/sin 计算
- 管理重映射后的物理块分配

**接口**:
```python
class PositionRemapper:
    def __init__(self, block_pool, rope_config, device):
        self.block_pool = block_pool
        self.rope_config = rope_config
        self.device = device
        self.cos_cache = {}  # 位置 -> cos
        self.sin_cache = {}  # 位置 -> sin

    def remap_and_copy(
        self,
        chunk_kv_cache: ChunkKVCache,
        new_position_offset: int,
    ) -> RemappedChunkKV:
        """
        拷贝并重映射 chunk KV cache

        步骤:
        1. 分配新的物理块
        2. 拷贝KV数据
        3. 应用新位置的RoPE编码
        4. 返回重映射后的KV
        """
        pass

    def release_remapped_blocks(self, block_ids: list[int]):
        """释放重映射使用的物理块 (请求结束后)"""
        self.block_pool.free_blocks(block_ids)
```

### 6.5 GPUModelRunner 扩展

**文件**: `vllm/v1/worker/gpu_model_runner.py` (修改)

**新增方法**:
```python
class GPUModelRunner:
    def parse_chunked_prompt(
        self,
        prompt_tokens: list[int],
        separator: str = "##",
    ) -> ChunkedPrompt:
        """
        解析分块提示词

        格式: sys_prompt + "##" + chunk1 + "##" + chunk2 + "##" + ... + user_question
        """
        pass

    def get_or_compute_chunks(
        self,
        chunked_prompt: ChunkedPrompt,
    ) -> list[RemappedChunkKV]:
        """
        获取或计算所有 chunks (包括 sys_prompt)

        返回: [sys_prompt_chunk, chunk1, chunk2, ...]
        注意: user_question 不缓存，需要单独计算
        """
        pass

    def compute_question_kv(
        self,
        question_tokens: list[int],
        chunk_kvs: list[RemappedChunkKV],
    ) -> KVCache:
        """
        计算 user_question 的 KV cache

        注意: question 需要关注所有之前的内容，所以需要完整的上下文
        """
        pass

    def merge_chunk_kvs_for_attention(
        self,
        chunk_kvs: list[RemappedChunkKV],
        question_kv: KVCache,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        合并所有 KV cache 用于 attention 计算

        返回: (merged_key, merged_value, merged_block_ids)
        """
        pass
```

### 6.6 AscendSFABackend 扩展

**文件**: `vllm-ascend/vllm_ascend/attention/sfa_v1.py` (修改)

**集成 chunk-aware mask 和重映射后的 KV**:
```python
class AscendSFABackend:
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, **kwargs):
        if attn_metadata.chunk_ids is not None:
            # 1. 构建 chunk-aware mask
            if attn_metadata.chunk_attn_mask is None:
                mask_builder = AttentionMaskBuilder(query.device)
                attn_metadata.chunk_attn_mask = mask_builder.get_chunk_aware_mask(
                    num_tokens=query.shape[0],
                    chunk_ids=attn_metadata.chunk_ids,
                    chunk_boundaries=attn_metadata.chunk_boundaries,
                    dtype=query.dtype,
                )

            # 2. 使用重映射后的 KV cache (已在主 KV pool 中)
            # 重映射后的 KV 已经应用了正确的位置编码
            # 直接使用标准的 SFA attention 流程即可

        return output
```

**关键点**:
- 重映射后的 KV 已经在主 KV cache 的物理块中
- 位置编码已在重映射时更新
- SFA attention 无需特殊处理，只需要 chunk-aware mask

---

## 7. 昇腾 NPU 特定实现

### 7.1 CaMemAllocator 集成

**文件**: `vllm-ascend/device_allocator/camem.py` (使用，无需修改)

**使用方法**:
```python
from vllm_ascend.device_allocator.camem import CaMemAllocator

allocator = CaMemAllocator()

# 使用 "chunk_cache" 标签创建独立内存池
with allocator.use_memory_pool(tag="chunk_cache"):
    chunk_kv_memory = torch.empty(
        (num_blocks, block_size, num_kv_heads, head_dim),
        dtype=dtype,
        device="npu",
    )
```

**优势**:
- 物理内存隔离
- Sleep/wake 生命周期管理
- NPU 优化的内存分配

### 7.2 SFA (Sparse Flash Attention) 适配

**文件**: `vllm-ascend/vllm_ascend/attention/sfa_v1.py` (扩展)

**关键集成点**:
```python
class AscendSFAImpl:
    def forward_with_remapped_chunks(
        self,
        query: torch.Tensor,
        remapped_chunk_kvs: list[RemappedChunkKV],
        positions: torch.Tensor,
        chunk_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        SFA attention with remapped chunk KV cache support

        关键点:
        1. 使用重映射后的 KV cache (已在新位置)
        2. KV 已经应用了正确的 RoPE 编码
        3. 仅需应用 chunk-aware attention mask
        """
        # 收集所有 block IDs (这些是重映射后的块)
        all_block_ids = []
        for chunk_kv in remapped_chunk_kvs:
            all_block_ids.extend(chunk_kv.block_ids)

        # 准备 KV cache (重映射后的数据)
        # 注意: 这些 KV 已经在正确的位置，且 RoPE 已应用
        # 直接使用标准的 SFA attention 即可

        # 应用 chunk-aware mask
        output = self.sfa_attention_with_mask(
            query=query,
            key=self.key_cache,
            value=self.value_cache,
            block_ids=all_block_ids,
            attn_mask=chunk_attn_mask,
        )

        return output
```

**简化设计**:
- 重映射后的 KV 直接放入主 KV cache pool
- SFA 无需特殊处理，只需 chunk-aware mask
- 避免了复杂的位置映射逻辑

---

## 8. 集成到 vLLM Engine

### 8.1 配置扩展

**文件**: `vllm/config/chunk_cache.py` (新建)

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ChunkCacheConfig:
    """Chunk Cache 配置"""
    enable_chunk_cache: bool = False
    chunk_separator: str = "##"
    max_chunks: int = 500
    max_chunk_tokens: int = 4096
    chunk_cache_gpu_memory_utilization: float = 0.15
    chunk_hash_algo: Literal["xxhash", "sha256"] = "xxhash"
    cache_policy: Literal["lru", "fifo"] = "lru"
    log_chunk_cache_stats: bool = False
    enable_ascend_optimizations: bool = True
    ascend_sleep_wake_enabled: bool = False
```

**文件**: `vllm/config/vllm.py` (修改)

```python
@dataclass
class VllmConfig:
    # ... 现有字段 ...
    chunk_cache_config: ChunkCacheConfig = field(default_factory=ChunkCacheConfig)
```

### 8.2 LLMEngine 初始化

**文件**: `vllm/v1/engine/llm_engine.py` (修改)

```python
class LLMEngine:
    def __init__(self, vllm_config: VllmConfig):
        # ... 现有初始化 ...

        if vllm_config.chunk_cache_config.enable_chunk_cache:
            from vllm.v1.core.chunk_cache_manager import ChunkCacheManager

            ascend_allocator = None
            if vllm_config.chunk_cache_config.enable_ascend_optimizations:
                try:
                    from vllm_ascend.device_allocator.camem import CaMemAllocator
                    ascend_allocator = CaMemAllocator()
                except ImportError:
                    logger.warning("vllm-ascend not available, using CUDA fallback")

            self.chunk_cache_manager = ChunkCacheManager(
                config=vllm_config.chunk_cache_config,
                kv_cache_spec=self.model_config.kv_cache_spec,
                block_size=self.cache_config.block_size,
                ascend_allocator=ascend_allocator,
            )
```

---

## 9. 测试策略

### 9.1 单元测试

**文件**: `vllm/v1/tests/test_chunk_cache.py`

```python
def test_chunk_hash_computation():
    """测试 chunk hash 计算"""
    hash_index = ChunkHashIndex()
    hash1 = hash_index.compute_hash([1, 2, 3, 4, 5], block_size=16)
    hash2 = hash_index.compute_hash([1, 2, 3, 4, 5], block_size=16)
    hash3 = hash_index.compute_hash([1, 2, 3, 4, 6], block_size=16)

    assert hash1 == hash2  # 相同内容，相同 hash
    assert hash1 != hash3  # 不同内容，不同 hash

def test_position_reindexing():
    """测试位置重映射"""
    reindexer = PositionReindexer(...)
    chunk_kv = create_mock_chunk_kv(num_tokens=100)

    reindexed1 = reindexer.reindex_chunk_kv(chunk_kv, 0)
    reindexed2 = reindexer.reindex_chunk_kv(chunk_kv, 1000)

    assert reindexed1.original_cache is chunk_kv  # 零拷贝
    assert reindexed2.original_cache is chunk_kv  # 零拷贝

def test_chunk_aware_mask():
    """测试 chunk-aware attention mask"""
    mask_builder = AttentionMaskBuilder(device="cpu")

    chunk_ids = torch.tensor([-1]*10 + [0]*10 + [-2]*5)
    chunk_boundaries = [(10, 20)]

    mask = mask_builder.get_chunk_aware_mask(
        num_tokens=25,
        chunk_ids=chunk_ids,
        chunk_boundaries=chunk_boundaries,
        dtype=torch.float16,
    )

    assert mask.shape == (25, 25)
    # 验证 sys_prompt causal
    # 验证 chunk 可以关注 sys_prompt
    # 验证 question 可以关注所有
```

### 9.2 集成测试

```python
def test_end_to_end_chunk_cache():
    """端到端测试"""
    from vllm import LLM

    llm = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        enable_chunk_cache=True,
        chunk_cache_config=ChunkCacheConfig(max_chunks=100),
    )

    prompt1 = "You are helpful.##Doc1##What is this?"
    outputs1 = llm.generate([prompt1])

    stats = llm.get_chunk_cache_stats()
    assert stats.cached_chunks == 1
    assert stats.miss_count == 1

    prompt2 = "Different system.##Doc1##Tell me more"
    outputs2 = llm.generate([prompt2])

    stats = llm.get_chunk_cache_stats()
    assert stats.hit_count >= 1

def test_output_correctness():
    """验证输出一致性"""
    llm_baseline = LLM(model=..., enable_chunk_cache=False)
    output_baseline = llm_baseline.generate([test_prompt])

    llm_cached = LLM(model=..., enable_chunk_cache=True)
    output_cached = llm_cached.generate([test_prompt])

    assert output_baseline.outputs[0].token_ids == output_cached.outputs[0].token_ids
```

---

## 10. 实施计划

### Phase 1: 核心基础设施 (2-3 周)

**目标**: 实现基本的 chunk 解析、位置重映射和 KV cache 管理

**Week 1: 数据结构和基础索引**
- 创建 `vllm/v1/core/chunk_structures.py`
  - `ChunkHash`, `ChunkKVCache`, `ReindexedChunkKV`, `ChunkedPrompt`, `ChunkCacheMetrics`
- 创建 `vllm/v1/core/chunk_hash_index.py`
  - `ChunkHashIndex.compute_hash()`, `lookup()`, `insert()`
- 单元测试

**Week 2: Block Pool 和 Cache Manager**
- 创建 `vllm/v1/core/chunk_block_pool.py`
  - `ChunkBlockPool.allocate_blocks()`, `free_blocks()`
  - LRU 实现
  - CaMemAllocator 集成
- 创建 `vllm/v1/core/chunk_cache_manager.py`
  - `ChunkCacheManager.get_or_compute_chunk()`, `evict_lru()`
- 单元测试

**Week 3: Position Reindexer**
- 创建 `vllm/v1/worker/position_reindexer.py`
  - `PositionReindexer.reindex_chunk_kv()`
  - RoPE cos/sin 复用和重计算
  - slot_mapping 重计算
- 单元测试和零拷贝验证

**验收标准**:
- Chunk cache 可以存储和检索
- 位置重映射基本工作
- 所有单元测试通过

### Phase 2: 位置编码与注意力隔离 (2 周)

**目标**: 实现位置重映射和 chunk 隔离注意力

**Week 4: 位置重映射优化**
- 完善 `PositionReindexer`
  - 标准RoPE 优化 (cos/sin 缓存复用)
  - NTK-aware RoPE 支持
  - 性能测试和优化

**Week 5: Chunk-Aware Attention**
- 修改 `vllm-ascend/vllm_ascend/attention/attention_mask.py`
  - `AttentionMaskBuilder.get_chunk_aware_mask()`
- 修改 `vllm-ascend/vllm_ascend/attention/utils.py`
  - 扩展 `AscendCommonAttentionMetadata`
- 修改 `vllm-ascend/vllm_ascend/attention/sfa_v1.py`
  - 集成 chunk-aware mask 到 SFA
- 单元测试

**验收标准**:
- Chunk 在不同位置复用时输出一致
- RoPE 计算正确
- Chunk 隔离注意力正确工作

### Phase 3: Engine 集成 (1-2 周)

**目标**: 将 chunk cache 集成到 vLLM v1 Engine

**Week 6: 配置和初始化**
- 创建 `vllm/config/chunk_cache.py`
- 修改 `vllm/config/vllm.py`
- 修改 `vllm/v1/engine/llm_engine.py`
  - 初始化 `ChunkCacheManager` 和 `PositionReindexer`
- 集成测试: LLM 初始化

**Week 7: Prompt 解析和执行流程**
- 修改 `vllm/v1/engine/input_processor.py`
  - `InputProcessor.parse_chunked_prompt()`
- 修改 `vllm/v1/worker/gpu_model_runner.py`
  - `GPUModelRunner.parse_chunked_prompt()`
  - `GPUModelRunner.get_or_compute_chunks()`
  - `GPUModelRunner.merge_kv_caches()`
- 修改 `vllm/v1/engine/core_client.py`
  - `_execute_chunked_request()`
- 集成测试: 端到端流程

**验收标准**:
- 端到端流程工作
- API 兼容
- 向后兼容 (不影响非 chunk cache 请求)

### Phase 4: 测试与优化 (1-2 周)

**目标**: 验证正确性和性能

**Week 8: 正确性测试**
- 单元测试覆盖率 >80%
- 集成测试 (真实 RAG workload)
- 正确性验证 (有/无 chunk cache 输出一致)
- 边界情况测试

**Week 9: 性能测试**
- 性能 benchmark
  - 缓存命中率测试
  - 性能加速测试
  - 内存占用测试
- 优化迭代
- 文档完善

**验收标准**:
- 所有测试通过
- 性能达标 (>2x 加速)
- 无严重 bug
- 文档完整

### 关键文件清单

**新增文件**:
1. `vllm/v1/core/chunk_structures.py` - 数据结构定义 (ChunkHash, ChunkKVCache, RemappedChunkKV)
2. `vllm/v1/core/chunk_block_pool.py` - Block pool 管理
3. `vllm/v1/core/chunk_hash_index.py` - 哈希索引
4. `vllm/v1/core/chunk_cache_manager.py` - 核心管理器
5. `vllm/v1/worker/position_remapper.py` - 位置重映射器 (KV拷贝)
6. `vllm/config/chunk_cache.py` - 配置类
7. `vllm/v1/tests/test_chunk_cache.py` - 单元测试

**修改文件**:
1. `vllm/config/vllm.py` - 添加 ChunkCacheConfig
2. `vllm/v1/engine/llm_engine.py` - 集成 ChunkCacheManager
3. `vllm/v1/engine/input_processor.py` - 添加 parse_chunked_prompt
4. `vllm/v1/worker/gpu_model_runner.py` - 扩展支持 chunk
5. `vllm-ascend/vllm_ascend/attention/attention_mask.py` - 添加 get_chunk_aware_mask
6. `vllm-ascend/vllm_ascend/attention/utils.py` - 扩展 AscendCommonAttentionMetadata
7. `vllm-ascend/vllm_ascend/attention/sfa_v1.py` - 集成 chunk-aware mask

### 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 位置重映射不正确 | 生成质量下降 | 严格单元测试，逐步验证，对比 baseline |
| 内存超限 | OOM | 可配置上限，LRU 淘汰，监控告警 |
| Hash 冲突 | 错误缓存复用 | SHA256 默认，可选 token 验证 |
| 重映射开销过大 | 性能提升不明显 | 虚拟位置策略，RoPE 缓存复用，profiling |
| SFA 兼容性 | 性能下降 | 验证 mask 模式与 SFA kernel 兼容性 |
| 与 prefix caching 冲突 | 系统不稳定 | 清晰职责划分，优先级设计，充分测试 |

### 预期收益

**性能提升**:
- 典型 RAG workload (3-5 chunks): **2-5x 加速**
- Chunk 缓存命中率 80%+ 场景: **3-10x 加速**
- TTFT 减少: **50-80%**

**适用场景**:
- RAG 应用 (多文档检索)
- 知识库问答
- 多文档摘要
- 系统提示词 + 多个独立任务

---

## 总结

本方案设计了一个位置无关的 KV Cache 缓存和复用系统，基于 vLLM + vLLM-Ascend 实现，支持 chunk 隔离注意力和跨请求缓存复用。

### 核心创新

1. **位置无关缓存**: Chunk hash 不包含位置信息，支持跨位置复用
2. **虚拟位置策略**: 统一在虚拟位置计算，复用时拷贝并重映射到实际位置
3. **高效KV拷贝**: 利用 NPU 加速拷贝和 RoPE 编码，开销仅 ~10%
4. **统一chunk处理**: sys_prompt 也作为特殊chunk处理，不依赖prefix caching
5. **Chunk 隔离注意力**: 通过自定义 attention mask 实现 chunk 之间隔离
6. **独立显存管理**: 使用 CaMemAllocator 标签化内存池
7. **昇腾优化**: 深度集成 SFA/MLA attention 和 HCCL 通信

### 技术亮点

- **高效拷贝重映射**: KV 数据拷贝 + RoPE 编码 ~2-5ms (4K tokens)，相比完整计算节省 90%+ 时间
- **虚拟位置计算**: 所有 chunk 在统一虚拟位置计算，简化缓存逻辑
- **高效哈希索引**: O(1) 查找，支持 XXHash128 和 SHA256
- **灵活淘汰策略**: LRU 默认
- **NPU 优化**: 利用 NPU 的异步拷贝和高效 RoPE kernel

### 性能预期

| 场景 | 加速比 | 说明 |
|------|--------|------|
| Chunk 缓存命中 (1个) | ~12x | 考虑拷贝开销 (5ms vs 60ms) |
| Chunk 缓存命中 (3-5个) | ~30-50x | 多个 chunks 累积收益 |
| TTFT 减少 | 50-80% | RAG 场景典型性能提升 |
| 端到端加速 | 2-5x | 缓存命中率 80%+ 场景 |

### 实施策略

**快速原型优先**: 先实现基本功能 (MVP)，验证可行性，再进行优化

**总时间**: 6-9 周 (MVP 可用)

**后续优化**: ACL 图编译、HCCL 多卡同步、Sleep/wake 生命周期

---

**文档结束**
