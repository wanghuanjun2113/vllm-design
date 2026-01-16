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

### 2.1 整体架构 (参考LMCache分层设计)

```
┌───────────────────────────────────────────────────────────────┐
│                    LLMEngine (vllm/v1/engine/)                │
│                                                                │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │InputProcessor│→ │ ChunkCacheEngine │→ │EngineCoreClient│  │
│  │              │  │    (新增)        │  │                │  │
│  │ chunk解析    │  │ 缓存管理层       │  │ 调度执行       │  │
│  └──────────────┘  └────────┬─────────┘  └────────────────┘  │
│                             ↓                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │          Worker (vllm/v1/worker/)                      │   │
│  │                                                         │   │
│  │  GPUModelRunner (修改)                                 │   │
│  │  - 新增: parse_chunked_prompt()                        │   │
│  │  - 新增: get_or_compute_chunks()                       │   │
│  │  - 调用: ChunkCacheEngine.lookup()/store()             │   │
│  └────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

**分层架构设计**:

```
┌─────────────────────────────────────────────────────────────┐
│                    ENGINE LAYER                             │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │ ChunkCacheEngine  │ ───→ │ ChunkTokenParser │            │
│  │  (缓存引擎)       │      │  (分词解析)       │            │
│  │ - lookup()        │      │ - parse()        │            │
│  │ - store()         │      │ - validate()     │            │
│  │ - clear()         │      │                  │            │
│  └────────┬─────────┘      └──────────────────┘            │
│           ↓                                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          STORAGE LAYER (存储层)                      │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │    │
│  │  │ ChunkHash    │  │ ChunkStorage │  │ ChunkLRU │  │    │
│  │  │ Index        │  │ Manager      │  │ Policy   │  │    │
│  │  │ (哈希索引)   │  │ (存储管理)    │  │ (淘汰策略)│  │    │
│  │  │ - lookup()   │  │ - allocate() │  │ - evict() │  │    │
│  │  │ - insert()   │  │ - free()     │  │           │  │    │
│  │  └──────────────┘  └──────────────┘  └───────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          ADAPTER LAYER (适配层)                      │    │
│  │  ┌──────────────┐  ┌──────────────┐                 │    │
│  │  │ ChunkKV      │  │ Position     │                 │    │
│  │  │ Connector    │  │ Remapper     │                 │    │
│  │  │ (GPU↔CPU)    │  │ (位置重映射)   │                 │    │
│  │  │ - to_cpu()   │  │ - remap()    │                 │    │
│  │  │ - to_gpu()   │  │ - copy()     │                 │    │
│  │  └──────────────┘  └──────────────┘                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件职责

**设计原则**: 参考 LMCache 的分层架构，实现职责分离

| 层级 | 组件 | 职责 | 参考LMCache |
|------|------|------|-------------|
| **Engine层** | ChunkCacheEngine | 协调各组件，提供统一API | LMCacheEngine |
| **Token层** | ChunkTokenParser | 解析chunked prompt，识别chunks | TokenDatabase |
| **存储层** | ChunkStorageManager | 管理存储后端，分配/释放内存 | StorageManager |
| **索引层** | ChunkHashIndex | 内容哈希 → chunk 映射 | 内嵌在StorageManager |
| **淘汰层** | ChunkLRUPolicy | LRU淘汰策略 | cache_policy/lru.py |
| **适配层** | ChunkKVConnector | GPU↔CPU数据传输 | GPUConnector |
| **重映射层** | PositionRemapper | KV拷贝+RoPE重新编码 | 特有功能 |

### 2.3 新增/修改模块清单

**新增模块** (7个):
- `vllm/v1/core/chunk_cache_engine.py` - 缓存引擎 (协调层)
- `vllm/v1/core/chunk_token_parser.py` - Token解析器
- `vllm/v1/core/chunk_storage_manager.py` - 存储管理器
- `vllm/v1/core/chunk_hash_index.py` - 哈希索引
- `vllm/v1/core/chunk_kv_connector.py` - GPU↔CPU适配器
- `vllm/v1/worker/position_remapper.py` - 位置重映射
- `vllm/v1/core/chunk_lru_policy.py` - LRU淘汰策略

**修改模块** (7个):
- `vllm/config/vllm_config.py` - 添加 ChunkCacheConfig
- `vllm/v1/engine/llm_engine.py` - 集成 ChunkCacheEngine
- `vllm/v1/engine/input_processor.py` - 解析 chunked prompt
- `vllm/v1/worker/gpu_model_runner.py` - 新增方法
- `vllm-ascend/attention/utils.py` - 扩展元数据
- `vllm-ascend/attention/sfa_v1.py` - 集成 chunk mask
- `vllm/v1/core/kv_cache_spec.py` - 扩展KV规格定义

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

### 3.1 数据结构 (参考LMCache CacheEngineKey)

**设计原则**:
- 使用统一的数据结构表示chunk key
- 包含元数据但不包含实际KV数据
- 支持可扩展的配置字段

**ChunkKey** - Chunk的唯一标识 (位置无关):
```python
@dataclass(frozen=True)
class ChunkKey:
    """
    Chunk唯一标识 (参考LMCache CacheEngineKey)

    Attributes:
        chunk_hash: 内容哈希 (XXHash128/SHA256)
        token_count: chunk的token数量
        num_blocks: 占用的块数量
        fmt: KV格式 (标准/MLA)
        model_name: 模型名称
        kv_dtype: KV数据类型
    """
    chunk_hash: bytes              # 16 bytes (XXHash128)
    token_count: int
    num_blocks: int
    fmt: str                       # "standard" or "mla"
    model_name: str
    kv_dtype: torch.dtype
```

**ChunkKV** - Chunk的KV数据封装:
```python
@dataclass
class ChunkKV:
    """
    Chunk KV数据封装 (参考LMCache MemoryObj)

    Attributes:
        chunk_key: Chunk唯一标识
        block_ids: 物理块ID列表
        key_cache: K cache [num_layers, num_tokens, num_kv_heads, head_dim]
        value_cache: V cache
        virtual_position: 虚拟位置 (用于统一缓存)
        metadata: 扩展元数据
    """
    chunk_key: ChunkKey
    block_ids: list[int]
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    virtual_position_start: int
    virtual_position_end: int
    metadata: dict                # 扩展字段
```

**RemappedChunkKV** - 重映射后的Chunk KV:
```python
@dataclass
class RemappedChunkKV:
    """
    重映射后的Chunk KV (包含KV数据拷贝)

    Attributes:
        chunk_key: 原始chunk标识
        block_ids: 主KV池的新块ID
        key_cache: 已应用新位置RoPE的K
        value_cache: 已应用新位置RoPE的V
        position_start: 实际起始位置
        position_end: 实际结束位置
        positions: 位置张量
    """
    chunk_key: ChunkKey
    block_ids: list[int]
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    position_start: int
    position_end: int
    positions: torch.Tensor
```

**ChunkedPrompt** - 分块提示词:
```python
@dataclass
class ChunkedPrompt:
    """
    分块提示词结构

    Attributes:
        sys_prompt: 系统提示词 (作为第一个chunk)
        chunks: 文档chunks列表
        user_question: 用户问题
        separator: 分隔符 (默认 "##")
        chunk_boundaries: chunk边界 [(start, end), ...]
    """
    sys_prompt: list[int]
    chunks: list[list[int]]
    user_question: list[int]
    separator: str = "##"
    chunk_boundaries: list[tuple[int, int]]

    def get_all_chunks(self) -> list[list[int]]:
        """获取所有需要缓存的chunks (包括sys_prompt)"""
```

### 3.2 ChunkCacheEngine 核心流程 (参考LMCache LMCacheEngine)

**设计原则**:
- Engine作为协调层，不直接处理数据
- 提供统一的lookup/store/clear API
- 参考LMCache的异步设计模式

**核心API**:
```python
class ChunkCacheEngine:
    """
    Chunk缓存引擎 (参考LMCache LMCacheEngine)

    职责:
    - 协调TokenParser、StorageManager、KVConnector
    - 提供统一的lookup/store/clear API
    - 管理缓存生命周期
    """

    def lookup(
        self,
        tokens: list[int],
        position_offset: int,
        **kwargs
    ) -> tuple[list[RemappedChunkKV], int]:
        """
        查找并获取chunks (参考LMCache.retrieve)

        流程:
        1. ChunkTokenParser.parse() → ChunkedPrompt
        2. 对每个chunk:
           - ChunkHashIndex.lookup(chunk_key)
           - [HIT] → PositionRemapper.remap()
           - [MISS] → break
        3. 返回命中的RemappedChunkKV列表

        Returns:
            (remapped_chunks, num_hit_tokens)
        """

    def store(
        self,
        tokens: list[int],
        chunk_kv: ChunkKV,
        **kwargs
    ) -> None:
        """
        存储chunk (参考LMCache.store)

        流程:
        1. 计算chunk_key
        2. ChunkStorageManager.allocate()
        3. ChunkKVConnector.to_cpu()
        4. ChunkHashIndex.insert()
        """

    def clear(
        self,
        tokens: Optional[list[int]] = None
    ) -> int:
        """清空缓存 (参考LMCache.clear)"""

    def get_stats(self) -> dict:
        """获取统计信息"""
```

### 3.3 ChunkTokenParser (参考LMCache TokenDatabase)

**设计原则**:
- 支持多种解析策略 (fixed chunk / separator)
- 提供validate和parse两个阶段
- 返回ChunkedPrompt结构

**主要接口**:
```python
class ChunkTokenParser:
    """
    Token解析器 (参考LMCache TokenDatabase)

    职责:
    - 解析chunked prompt
    - 识别chunk边界
    - 验证格式
    """

    def parse(
        self,
        tokens: list[int],
        separator: str = "##",
    ) -> ChunkedPrompt:
        """
        解析chunked prompt (参考TokenDatabase.process_tokens)

        功能:
        1. 使用滑动窗口识别分隔符 (torch.unfold)
        2. 分割: sys_prompt, chunks, user_question
        3. 计算chunk_boundaries

        Args:
            tokens: token序列
            separator: 分隔符

        Returns:
            ChunkedPrompt结构
        """

    def validate(
        self,
        prompt_str: str,
        separator: str = "##",
    ) -> bool:
        """验证prompt格式"""

    def compute_hash(
        self,
        tokens: list[int]
    ) -> bytes:
        """
        计算内容哈希 (参考TokenDatabase._hash_tokens)

        Args:
            tokens: token序列

        Returns:
            16 bytes hash (XXHash128)
        """
```

### 3.4 ChunkStorageManager (参考LMCache StorageManager)

**设计原则**:
- 管理存储后端 (CPU/Disk)
- 分配/释放物理块
- 实现LRU淘汰策略

**主要接口**:
```python
class ChunkStorageManager:
    """
    存储管理器 (参考LMCache StorageManager)

    职责:
    - 管理CPU/Disk存储后端
    - 分配/释放物理块
    - LRU淘汰策略
    """

    def allocate(
        self,
        chunk_key: ChunkKey,
        kv_shape: tuple,
        kv_dtype: torch.dtype,
    ) -> Optional[list[int]]:
        """
        分配存储块

        流程:
        1. 检查容量
        2. [不足] → ChunkLRUPolicy.evict()
        3. 分配块
        4. 更新LRU链表

        Returns:
            block_ids列表，失败返回None
        """

    def free(
        self,
        block_ids: list[int]
    ) -> None:
        """释放存储块"""

    def contains(
        self,
        chunk_key: ChunkKey
    ) -> bool:
        """检查chunk是否存在"""

    def get(
        self,
        chunk_key: ChunkKey
    ) -> Optional[ChunkKV]:
        """获取chunk数据"""

    def put(
        self,
        chunk_key: ChunkKey,
        chunk_kv: ChunkKV
    ) -> None:
        """存储chunk数据"""

    def get_memory_usage(self) -> dict:
        """获取内存使用情况"""
```

### 3.5 ChunkKVConnector (参考LMCache GPUConnector)

**设计原则**:
- 封装GPU↔CPU数据传输
- 支持异步传输
- 零拷贝优化 (pinned memory)

**主要接口**:
```python
class ChunkKVConnector:
    """
    GPU↔CPU数据适配器 (参考LMCache GPUConnector)

    职责:
    - GPU KV → CPU MemoryObj
    - CPU MemoryObj → GPU KV
    - 异步传输优化
    """

    def to_cpu(
        self,
        gpu_kv: tuple[torch.Tensor, torch.Tensor],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> ChunkKV:
        """
        GPU KV → CPU (参考GPUConnector.batched_from_gpu)

        功能:
        1. 分配pinned memory
        2. 异步拷贝 (cudaMemcpyAsync)
        3. 返回ChunkKV
        """

    def to_gpu(
        self,
        chunk_kv: ChunkKV,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CPU → GPU KV (参考GPUConnector.batched_to_gpu)

        功能:
        1. 分配GPU memory
        2. 异步拷贝
        3. 返回 (key_cache, value_cache)
        """
```

### 3.6 PositionRemapper (特有功能)

**主要接口**:
```python
class PositionRemapper:
    """
    位置重映射器 (特有功能)

    职责:
    - 拷贝KV数据
    - 应用新位置的RoPE
    - NPU加速 (torch_npu.npu_kv_rmsnorm_rope_cache)
    """

    def remap(
        self,
        src_kv: ChunkKV,
        dst_position: int,
    ) -> RemappedChunkKV:
        """
        重映射chunk KV到新位置

        功能:
        1. 从主KV池分配块
        2. 计算新位置的RoPE (带缓存)
        3. 拷贝KV并应用RoPE (NPU加速)
        4. 返回RemappedChunkKV

        时间复杂度: O(num_tokens)
        预期开销: ~2-5ms (4K tokens)
        """

    def release(
        self,
        block_ids: list[int]
    ) -> None:
        """释放重映射的块"""
```

### 3.7 ChunkLRUPolicy (参考LMCache cache_policy/lru.py)

**主要接口**:
```python
class ChunkLRUPolicy:
    """
    LRU淘汰策略 (参考LMCache cache_policy/lru.py)

    职责:
    - 维护LRU链表
    - 触发淘汰
    - 更新访问顺序
    """

    def evict(
        self,
        num_blocks: int
    ) -> list[ChunkKey]:
        """
        淘汰chunks

        Returns:
            被淘汰的ChunkKey列表
        """

    def touch(
        self,
        chunk_key: ChunkKey
    ) -> None:
        """更新访问顺序"""

    def get_lru_list(self) -> list[ChunkKey]:
        """获取LRU列表"""
```

### 3.8 配置系统 (参考LMCache config.py)

**设计原则**:
- 支持YAML/环境变量/命令行
- 类型安全的配置定义
- 配置别名和废弃警告

**ChunkCacheConfig**:
```python
@dataclass
class ChunkCacheConfig:
    """
    Chunk缓存配置 (参考LMCache LMCacheEngineConfig)

    支持的配置源:
    - YAML配置文件
    - 环境变量 (LMCACHE_CHUNK_*)
    - 命令行参数
    """
    # 基础配置
    enable_chunk_cache: bool = False
    chunk_separator: str = "##"
    chunk_size: int = 256

    # 存储配置
    max_local_cpu_size: float = 5.0        # GB
    chunk_cache_gpu_memory_utilization: float = 0.15
    enable_disk: bool = False
    max_local_disk_size: float = 0.0        # GB
    local_disk_path: Optional[str] = None

    # Hash配置
    chunk_hash_algo: str = "xxhash"        # "xxhash" or "sha256"

    # 性能配置
    enable_async_loading: bool = False
    enable_freeze_mode: bool = False
```

**VllmConfig扩展**:
```python
@dataclass
class VllmConfig:
    # ... 现有字段 ...

    # 新增: Chunk缓存配置
    chunk_cache: ChunkCacheConfig = field(
        default_factory=ChunkCacheConfig
    )
```

### 3.9 集成流程 (参考LMCache集成方式)

**LLMEngine初始化**:
```python
class LLMEngine:
    def __init__(self, vllm_config: VllmConfig, ...):
        # ... 现有初始化 ...

        # 新增: 初始化ChunkCacheEngine (如果启用)
        if vllm_config.chunk_cache.enable_chunk_cache:
            self.chunk_cache_engine = ChunkCacheEngine(
                config=vllm_config.chunk_cache,
                metadata=self._create_chunk_metadata(),
            )
        else:
            self.chunk_cache_engine = None
```

**InputProcessor扩展**:
```python
class InputProcessor:
    def process_inputs(self, prompts, ...):
        # 检测chunked prompt
        if "##" in prompt_str:
            chunked_prompt = self.chunk_token_parser.parse(
                prompt_tokens, separator="##"
            )
            request_metadata["use_chunk_cache"] = True
            request_metadata["chunked_prompt"] = chunked_prompt
```

**GPUModelRunner扩展**:
```python
class GPUModelRunner:
    def get_or_compute_chunks(
        self,
        chunked_prompt: ChunkedPrompt,
    ) -> list[RemappedChunkKV]:
        """
        获取或计算chunks (参考LMCache流程)

        流程:
        1. 调用 ChunkCacheEngine.lookup()
        2. 对未命中的chunks:
           - compute_chunk_kv()
           - 调用 ChunkCacheEngine.store()
        3. 返回所有RemappedChunkKV
        """
```

---

## 4. 关键接口 (参考LMCache接口设计)

### 4.1 ChunkCacheEngine (参考LMCache LMCacheEngine)

**职责**: Chunk缓存引擎，作为协调层提供统一API

**位置**: `vllm/v1/core/chunk_cache_engine.py`

**设计参考**: LMCache LMCacheEngine的分层架构

**主要接口**:

```python
class ChunkCacheEngine:
    """
    Chunk缓存引擎 (参考LMCache LMCacheEngine)

    职责:
    - 协调TokenParser、StorageManager、KVConnector、PositionRemapper
    - 提供统一的lookup/store/clear API
    - 管理缓存生命周期
    - 统计信息收集

    组成部分:
    - token_parser: ChunkTokenParser实例
    - storage_manager: ChunkStorageManager实例
    - kv_connector: ChunkKVConnector实例
    - position_remapper: PositionRemapper实例
    - lru_policy: ChunkLRUPolicy实例
    """

    def __init__(
        self,
        config: ChunkCacheConfig,
        metadata: "ChunkMetadata",
    ):
        """
        初始化Chunk缓存引擎

        功能:
            1. 创建TokenParser
            2. 创建StorageManager (CPU/Disk后端)
            3. 创建KVConnector (GPU↔CPU)
            4. 创建PositionRemapper (NPU加速)
            5. 创建LRU Policy

        Args:
            config: Chunk缓存配置
            metadata: 元数据 (模型名称、KV规格等)

        Raises:
            ValueError: 配置参数无效
        """

    def lookup(
        self,
        tokens: list[int],
        position_offset: int,
        **kwargs
    ) -> tuple[list[RemappedChunkKV], int]:
        """
        查找并获取chunks (参考LMCache.retrieve)

        流程:
        1. token_parser.parse() → ChunkedPrompt
        2. 对每个chunk:
           a. storage_manager.contains(chunk_key)
           b. [HIT]
              - storage_manager.get() → ChunkKV
              - position_remapper.remap() → RemappedChunkKV
           c. [MISS] → break (prefix匹配)
        3. 返回 (remapped_chunks, num_hit_tokens)

        Args:
            tokens: token序列
            position_offset: 目标起始位置
            **kwargs: 扩展参数

        Returns:
            (remapped_chunks, num_hit_tokens)
            - remapped_chunks: 重映射后的chunk KV列表
            - num_hit_tokens: 命中的token总数

        时间复杂度:
            - 最佳 (全部未命中): O(1)
            - 最差 (全部命中): O(n * m)
              n = num_chunks, m = remap时间
        """

    def store(
        self,
        tokens: list[int],
        chunk_kv: ChunkKV,
        **kwargs
    ) -> None:
        """
        存储chunk (参考LMCache.store)

        流程:
        1. 计算chunk_key
        2. storage_manager.allocate()
        3. kv_connector.to_cpu() (如果需要)
        4. storage_manager.put()

        Args:
            tokens: token序列
            chunk_kv: 要存储的chunk KV
            **kwargs: 扩展参数

        Raises:
            MemoryError: 存储不足 (即使LRU淘汰后)
            RuntimeError: 存储失败
        """

    def clear(
        self,
        tokens: Optional[list[int]] = None,
        locations: Optional[list[str]] = None,
    ) -> int:
        """
        清空缓存 (参考LMCache.clear)

        Args:
            tokens: 如果指定，只清除这些tokens对应的chunk
                   如果为None，清空所有缓存
            locations: 存储位置列表 (如 ["cpu", "disk"])

        Returns:
            清除的chunk数量
        """

    def get_stats(self) -> dict:
        """
        获取缓存统计信息 (参考LMCache stats_monitor)

        Returns:
            {
                "total_chunks": int,
                "cache_hits": int,
                "cache_misses": int,
                "hit_rate": float,
                "total_tokens_cached": int,
                "evicted_chunks": int,
                "current_memory_mb": {
                    "cpu": float,
                    "disk": float,
                },
            }
        """

    def pin(
        self,
        chunk_key: ChunkKey
    ) -> None:
        """
        Pin chunk，防止被淘汰 (参考LMCache pin机制)

        Args:
            chunk_key: 要pin的chunk
        """

    def unpin(
        self,
        chunk_key: ChunkKey
    ) -> None:
        """Unpin chunk"""

    def close(self) -> None:
        """
        关闭引擎，释放资源 (参考LMCache.close)
        """
```

### 4.2 ChunkTokenParser (参考LMCache TokenDatabase)

**职责**: Token解析器，识别chunk边界

**位置**: `vllm/v1/core/chunk_token_parser.py`

**设计参考**: LMCache TokenDatabase + SegmentTokenDatabase

**主要接口**:

```python
class ChunkTokenParser:
    """
    Token解析器 (参考LMCache TokenDatabase)

    职责:
    - 解析chunked prompt
    - 识别chunk边界 (滑动窗口)
    - 计算内容哈希
    - 验证格式
    """

    def __init__(
        self,
        config: ChunkCacheConfig,
        tokenizer,
    ):
        """
        初始化

        功能:
            - 编码分隔符为tokens
            - 准备滑动窗口匹配的tensor

        Args:
            config: Chunk缓存配置
            tokenizer: 分词器
        """

    def parse(
        self,
        tokens: list[int],
        separator: str = "##",
    ) -> ChunkedPrompt:
        """
        解析chunked prompt (参考TokenDatabase.process_tokens)

        功能:
        1. 转换为torch.Tensor
        2. 使用torch.unfold()滑动窗口识别分隔符
        3. 分割: sys_prompt, chunks, user_question
        4. 计算chunk_boundaries

        Args:
            tokens: token序列
            separator: 分隔符

        Returns:
            ChunkedPrompt结构

        Raises:
            ValueError: 格式错误、分隔符不存在、空chunk

        性能:
            - O((n-m+1) * m) where n=len(tokens), m=len(separator)
            - 使用torch.unfold()向量化操作
        """

    def validate(
        self,
        prompt_str: str,
        separator: str = "##",
    ) -> bool:
        """
        验证prompt格式

        检查:
        1. 分隔符存在
        2. 分隔符数量 >= 1
        3. 不存在空chunk

        Args:
            prompt_str: 原始提示词字符串
            separator: 分隔符

        Returns:
            True if valid, False otherwise
        """

    def compute_hash(
        self,
        tokens: list[int]
    ) -> bytes:
        """
        计算内容哈希 (参考TokenDatabase._hash_tokens)

        Args:
            tokens: token序列

        Returns:
            16 bytes hash (XXHash128)

        注意:
            - 不包含位置信息 (位置无关)
            - 使用vLLM的hash函数 (兼容性)
        """
```

### 4.3 ChunkStorageManager (参考LMCache StorageManager)

**职责**: 存储管理器，管理CPU/Disk存储后端

**位置**: `vllm/v1/core/chunk_storage_manager.py`

**设计参考**: LMCache StorageManager

**主要接口**:

```python
class ChunkStorageManager:
    """
    存储管理器 (参考LMCache StorageManager)

    职责:
    - 管理CPU/Disk存储后端
    - 分配/释放物理块
    - 实现LRU淘汰策略
    - Pin/unpin机制

    存储层次:
    - LocalCPUBackend: CPU pinned memory (热缓存)
    - LocalDiskBackend: 本地磁盘 (可选)
    """

    def __init__(
        self,
        config: ChunkCacheConfig,
        metadata: "ChunkMetadata",
    ):
        """
        初始化

        功能:
        1. 创建CPU后端 (pinned memory)
        2. [可选] 创建Disk后端
        3. 创建LRU Policy
        4. 初始化块分配器

        Args:
            config: Chunk缓存配置
            metadata: 元数据
        """

    def allocate(
        self,
        chunk_key: ChunkKey,
        kv_shape: tuple,
        kv_dtype: torch.dtype,
    ) -> Optional[list[int]]:
        """
        分配存储块 (参考StorageManager.allocate)

        流程:
        1. 检查容量
        2. [不足] → lru_policy.evict() → 淘汰chunks
        3. 分配块
        4. 更新LRU链表

        Args:
            chunk_key: Chunk标识
            kv_shape: KV shape [num_layers, num_tokens, num_kv_heads, head_dim]
            kv_dtype: KV数据类型

        Returns:
            block_ids列表，失败返回None

        时间复杂度:
            - O(1) 分配
            - O(k) 淘汰 (k = 淘汰的chunk数)
        """

    def free(
        self,
        block_ids: list[int]
    ) -> None:
        """
        释放存储块

        Args:
            block_ids: 要释放的块ID列表
        """

    def contains(
        self,
        chunk_key: ChunkKey
    ) -> bool:
        """
        检查chunk是否存在 (参考StorageManager.contains)

        Args:
            chunk_key: Chunk标识

        Returns:
            True if exists, False otherwise
        """

    def get(
        self,
        chunk_key: ChunkKey
    ) -> Optional[ChunkKV]:
        """
        获取chunk数据 (参考StorageManager.get)

        功能:
        1. 查找chunk
        2. 更新LRU (touch)
        3. 返回ChunkKV

        Args:
            chunk_key: Chunk标识

        Returns:
            ChunkKV，不存在返回None
        """

    def put(
        self,
        chunk_key: ChunkKey,
        chunk_kv: ChunkKV
    ) -> None:
        """
        存储chunk数据 (参考StorageManager.put)

        Args:
            chunk_key: Chunk标识
            chunk_kv: Chunk KV数据
        """

    def batched_contains(
        self,
        chunk_keys: list[ChunkKey],
    ) -> tuple[int, dict[str, list[ChunkKey]]]:
        """
        批量检查chunks (参考StorageManager.batched_contains)

        Args:
            chunk_keys: Chunk标识列表

        Returns:
            (num_hit, block_mapping)
            - num_hit: 命中的chunk数量
            - block_mapping: location → [ChunkKey]
        """

    def get_memory_usage(self) -> dict:
        """
        获取内存使用情况

        Returns:
            {
                "cpu_used_mb": float,
                "cpu_total_mb": float,
                "cpu_utilization": float,
                "disk_used_mb": float,  # 如果启用
                "disk_total_mb": float,
                "num_cached_chunks": int,
            }
        """

    def touch(
        self,
        chunk_key: ChunkKey
    ) -> None:
        """
        更新LRU访问顺序 (参考StorageManager.touch)

        Args:
            chunk_key: Chunk标识
        """

    def evict(
        self,
        num_blocks: int
    ) -> list[ChunkKey]:
        """
        淘汰chunks (参考LRU policy)

        Args:
            num_blocks: 需要释放的块数

        Returns:
            被淘汰的ChunkKey列表
        """

    def pin(
        self,
        chunk_key: ChunkKey
    ) -> None:
        """
        Pin chunk，防止被淘汰 (参考LMCache pin)

        Args:
            chunk_key: Chunk标识
        """

    def unpin(
        self,
        chunk_key: ChunkKey
    ) -> None:
        """Unpin chunk"""

    def close(self) -> None:
        """关闭存储管理器，释放资源"""
```

### 4.4 ChunkKVConnector (参考LMCache GPUConnector)

**职责**: GPU↔CPU数据传输适配器

**位置**: `vllm/v1/core/chunk_kv_connector.py`

**设计参考**: LMCache GPUConnector

**主要接口**:

```python
class ChunkKVConnector:
    """
    GPU↔CPU数据适配器 (参考LMCache GPUConnector)

    职责:
    - GPU KV → CPU MemoryObj
    - CPU MemoryObj → GPU KV
    - 异步传输优化
    - 零拷贝优化 (pinned memory)

    优化:
    - 使用pinned memory加速传输
    - 异步拷贝 (cudaMemcpyAsync)
    - 批量操作
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        初始化

        Args:
            device: 设备 (cuda/npu)
            dtype: KV数据类型
        """

    def to_cpu(
        self,
        gpu_kv: tuple[torch.Tensor, torch.Tensor],
        chunk_key: ChunkKey,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> ChunkKV:
        """
        GPU KV → CPU (参考GPUConnector.batched_from_gpu)

        功能:
        1. 分配pinned memory
        2. 异步拷贝 (cudaMemcpyAsync / npu_memcpy)
        3. 封装为ChunkKV

        Args:
            gpu_kv: (key_cache, value_cache) GPU tensors
            chunk_key: Chunk标识
            stream: CUDA stream (可选)

        Returns:
            ChunkKV (CPU)

        性能:
            - 使用pinned memory: ~6-10 GB/s
            - 异步拷贝，不阻塞
        """

    def to_gpu(
        self,
        chunk_kv: ChunkKV,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CPU → GPU KV (参考GPUConnector.batched_to_gpu)

        功能:
        1. 分配GPU memory
        2. 异步拷贝
        3. 返回 (key_cache, value_cache)

        Args:
            chunk_kv: Chunk KV (CPU)
            stream: CUDA stream (可选)

        Returns:
            (key_cache, value_cache) GPU tensors

        性能:
            - 使用pinned memory: ~6-10 GB/s
            - 异步拷贝，不阻塞
        """

    def batched_to_cpu(
        self,
        gpu_kvs: list[tuple[torch.Tensor, torch.Tensor]],
        chunk_keys: list[ChunkKey],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> list[ChunkKV]:
        """
        批量GPU → CPU (参考GPUConnector.batched_from_gpu)

        功能:
        1. 批量分配pinned memory
        2. 批量异步拷贝
        3. 返回ChunkKV列表

        Args:
            gpu_kvs: GPU KV列表
            chunk_keys: Chunk标识列表
            stream: CUDA stream (可选)

        Returns:
            ChunkKV列表

        性能:
            - 批量操作，减少kernel launch overhead
        """

    def batched_to_gpu(
        self,
        chunk_kvs: list[ChunkKV],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        批量CPU → GPU (参考GPUConnector.batched_to_gpu)

        功能:
        1. 批量分配GPU memory
        2. 批量异步拷贝
        3. 返回GPU KV列表

        Args:
            chunk_kvs: Chunk KV列表 (CPU)
            stream: CUDA stream (可选)

        Returns:
            GPU KV列表
        """
```

### 4.5 PositionRemapper (特有功能)

**职责**: 位置重映射器，KV拷贝+RoPE重新编码

**位置**: `vllm/v1/worker/position_remapper.py`

**主要接口**:

```python
class PositionRemapper:
    """
    位置重映射器 (特有功能，LMCache无对应)

    职责:
    - 拷贝KV数据
    - 应用新位置的RoPE
    - NPU加速 (torch_npu.npu_kv_rmsnorm_rope_cache)
    - RoPE缓存优化

    NPU优化:
    - 使用 torch_npu.npu_kv_rmsnorm_rope_cache()
    - 一次调用完成KV拷贝+RoPE编码
    - 预期开销: ~2-5ms (4K tokens)
    """

    def __init__(
        self,
        kv_cache_spec: "KVCacheSpec",
        device: torch.device,
    ):
        """
        初始化

        功能:
            - 保存KV规格
            - 初始化RoPE缓存

        Args:
            kv_cache_spec: KV缓存规格
            device: 设备 (npu/cuda)
        """

    def remap(
        self,
        src_kv: ChunkKV,
        dst_position: int,
    ) -> RemappedChunkKV:
        """
        重映射chunk KV到新位置 (核心方法)

        功能:
        1. 从主KV池分配块
        2. 计算新位置序列
        3. 获取/计算RoPE编码 (带缓存)
        4. 拷贝KV并应用RoPE (NPU加速)
        5. 创建RemappedChunkKV

        Args:
            src_kv: 源Chunk KV (在虚拟位置)
            dst_position: 目标位置偏移

        Returns:
            RemappedChunkKV (在主KV池，应用了新位置RoPE)

        时间复杂度: O(num_tokens)
        预期开销: ~2-5ms (4K tokens)

        NPU优化:
            - torch_npu.npu_kv_rmsnorm_rope_cache()
            - 一次kernel调用完成拷贝+RoPE
        """

    def release(
        self,
        block_ids: list[int]
    ) -> None:
        """
        释放重映射的块 (请求结束后)

        Args:
            block_ids: 要释放的块ID列表
        """
```

### 4.6 ChunkLRUPolicy (参考LMCache cache_policy/lru.py)

**职责**: LRU淘汰策略

**位置**: `vllm/v1/core/chunk_lru_policy.py`

**设计参考**: LMCache cache_policy/lru.py

**主要接口**:

```python
class ChunkLRUPolicy:
    """
    LRU淘汰策略 (参考LMCache cache_policy/lru.py)

    职责:
    - 维护LRU链表
    - 触发淘汰
    - 更新访问顺序
    - Pin机制
    """

    def __init__(self, capacity_blocks: int):
        """
        初始化

        Args:
            capacity_blocks: 总容量 (块数)
        """

    def evict(
        self,
        num_blocks: int,
        skip_pinned: bool = True,
    ) -> list[ChunkKey]:
        """
        淘汰chunks

        功能:
        1. 从LRU链表尾部开始
        2. 跳过pinned chunks (可选)
        3. 淘汰直到释放足够的块

        Args:
            num_blocks: 需要释放的块数
            skip_pinned: 是否跳过pinned chunks

        Returns:
            被淘汰的ChunkKey列表

        Raises:
            RuntimeError: 无法释放足够的块 (所有chunks都被pin)
        """

    def touch(
        self,
        chunk_key: ChunkKey
    ) -> None:
        """
        更新访问顺序 (访问时调用)

        功能:
        - 将chunk移到LRU链表头部

        Args:
            chunk_key: Chunk标识
        """

    def pin(
        self,
        chunk_key: ChunkKey
    ) -> None:
        """
        Pin chunk，防止被淘汰

        Args:
            chunk_key: Chunk标识
        """

    def unpin(
        self,
        chunk_key: ChunkKey
    ) -> None:
        """Unpin chunk"""

    def get_lru_list(self) -> list[ChunkKey]:
        """
        获取LRU列表 (最不常用 → 最常用)

        Returns:
            ChunkKey列表 (按LRU顺序)
        """

    def get_stats(self) -> dict:
        """
        获取统计信息

        Returns:
            {
                "total_chunks": int,
                "pinned_chunks": int,
                "lru_list_size": int,
                "capacity_utilization": float,
            }
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
