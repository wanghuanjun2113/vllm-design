# 位置无关Chunk缓存设计方案

## 概述

本方案设计了一个**位置无关的KVCache缓存和复用功能**，用于vLLM-Ascend平台。与现有的Prefix Caching机制（使用严格的位置匹配）不同，本方案通过固定分隔符(`# #`)将prompt分割成独立chunk，每个chunk基于内容哈希（而非位置）进行缓存和匹配，从而实现跨请求的chunk复用。

### 背景与动机

现有的Prefix Caching采用**严格的位置匹配**策略：
- 使用分层哈希结构：`parent_hash + curr_block_token_ids + extra_keys`
- 位置依赖性强：由于包含`parent_hash`，只有前面所有块都匹配时，当前块才能匹配
- 适用场景有限：仅适用于相同内容在相同位置出现的情况

**问题场景：**
```
请求1: "You are helpful. # # Document A # # Question?"  → Document A可以缓存
请求2: "Different role. # # Document A # # Question?"   → Document A无法复用（位置不同）
```

在RAG（检索增强生成）场景中，同一份文档经常在不同的系统提示下被查询，导致Prefix Caching命中率低。本方案通过**位置无关的chunk缓存**解决这个问题。

### 核心创新点

1. **位置无关缓存**
   - Chunk hash仅基于token内容，不包含位置信息
   - 使用XXHash128算法计算内容哈希
   - 实现真正的跨请求chunk复用

2. **虚拟位置计算**
   - 所有chunk在统一虚拟位置`[VIRTUAL_POS_START, VIRTUAL_POS_START + max_chunk_len)`计算KV
   - 使用时通过**PositionRemapper**重映射到实际位置
   - 避免位置编码冲突

3. **Chunk隔离注意力**
   - 通过自定义attention mask实现chunk间的隔离
   - Chunk token可看sys_prompt + 同chunk内前序token
   - Chunk token**不能**看其他chunk的token
   - User question可看所有内容

4. **独立内存池**
   - 与主KV cache分离的chunk缓存池
   - 默认2GB显存，可配置
   - 不影响主推理流程的内存管理

5. **精确匹配**
   - Chunk必须完全匹配才能复用
   - 不支持部分匹配或chunk组合

### 适用场景

| 场景 | 说明 | 性能提升 |
|------|------|---------|
| **RAG应用** | 相同文档内容在不同系统提示下查询 | 10-15x |
| **多轮对话** | 对话历史中共享知识库 | 5-10x |
| **文档问答** | 多个问题基于相同文档 | 8-12x |
| **长文档处理** | 文档片段在不同上下文中复用 | 6-10x |

### 性能预期

- **单个chunk缓存命中**：约12x加速（5ms vs 60ms）
- **3-5个chunk缓存命中**：约30-50x加速（累积收益）
- **TTFT（Time To First Token）减少**：50-80%（RAG场景）
- **端到端加速**：2-5x（80%+命中率）

---

## 架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    vLLM-Ascend Engine                           │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ InputProcessor   │→ │ ChunkCacheEngine │→ │ ModelRunner  │ │
│  │ (修改)           │  │  (新增)          │  │  (修改)       │ │
│  │ - 检测分隔符     │  │ - lookup/store   │  │ - 计算KV      │ │
│  │ - 解析chunked    │  │ - 协调组件       │  │ - 拷贝KV      │ │
│  └──────────────────┘  └────────┬─────────┘  └───────────────┘ │
└─────────────────────────────────┼───────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ChunkCache 分层架构                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    ENGINE LAYER                          │  │
│  │  ChunkCacheEngine - 协调层，提供统一API                   │  │
│  │  ├── ChunkTokenParser - 解析chunked prompt               │  │
│  │  └── ChunkMetadataBuilder - 构建chunk元数据              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   STORAGE LAYER                           │  │
│  │  ├── ChunkHashIndex - 内容哈希→chunk映射                  │  │
│  │  ├── ChunkStorageManager - 管理chunk存储                 │  │
│  │  └── ChunkKVPool - 独立的显存池（默认2GB）                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  ADAPTER LAYER                            │  │
│  │  ├── ChunkKVConnector - GPU↔CPU数据传输                  │  │
│  │  └── PositionRemapper - KV拷贝+RoPE重新编码（核心）      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              NPU-SPECIFIC LAYER                           │  │
│  │  ├── ChunkAwareAttentionMask - chunk隔离注意力mask       │  │
│  │  └── NPU-Accelerated KV Copy - NPU加速KV拷贝             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 模块职责

#### 新增模块（7个）

| 模块 | 路径 | 职责 | 主要方法 |
|------|------|------|---------|
| **ChunkCacheEngine** | `vllm/v1/core/chunk_cache_engine.py` | 协调层，提供统一的lookup/store/clear API | `lookup()`, `store()`, `clear()` |
| **ChunkTokenParser** | `vllm/v1/core/chunk_token_parser.py` | 解析chunked prompt，识别chunk边界，计算内容哈希 | `parse()`, `compute_hash()` |
| **ChunkStorageManager** | `vllm/v1/core/chunk_storage_manager.py` | 管理chunk存储，分配/释放存储块 | `allocate()`, `free()`, `get()`, `put()` |
| **ChunkHashIndex** | `vllm/v1/core/chunk_hash_index.py` | 内容哈希→chunk映射索引，O(1)查找 | `lookup()`, `insert()` |
| **ChunkKVConnector** | `vllm/v1/core/chunk_kv_connector.py` | GPU↔CPU数据传输 | `to_cpu()`, `to_gpu()` |
| **PositionRemapper** | `vllm/v1/worker/position_remapper.py` | KV拷贝+RoPE重新编码（核心方法） | `remap()` |
| **ChunkKVPool** | `vllm/v1/worker/chunk_kv_pool.py` | 独立的显存池管理（默认2GB） | `allocate()`, `free()` |

#### 修改模块（6个）

| 模块 | 路径 | 修改内容 | 新增方法 |
|------|------|---------|---------|
| **VllmConfig** | `vllm/config/vllm_config.py` | 添加ChunkCacheConfig配置类 | `chunk_cache_config: ChunkCacheConfig` |
| **InputProcessor** | `vllm/v1/engine/input_processor.py` | 检测`# #`分隔符，解析chunked prompt | `_detect_chunked_prompt()`, `parse_chunked_prompt()` |
| **GPUModelRunner** | `vllm/v1/worker/gpu_model_runner.py` | 新增chunk处理方法 | `get_or_compute_chunks()`, `compute_chunk_kv()` |
| **AscendSFABend** | `vllm-ascend/attention/sfa_v1.py` | 集成chunk-aware attention mask | 修改`forward()` |
| **AttentionMaskBuilder** | `vllm-ascend/attention/attention_mask.py` | 新增chunk-aware attention mask生成 | `get_chunk_aware_mask()` |
| **AscendStoreConnector** | `vllm-ascend/distributed/kvpool/ascend_store_connector.py` | 扩展支持chunk KV传输 | 新增chunk传输方法 |

### 关键数据结构

```python
@dataclass
class ChunkKey:
    """Chunk的唯一标识（位置无关）"""
    content_hash: bytes      # 内容哈希（XXHash128）
    num_tokens: int          # token数量
    model_name: str          # 模型名称

@dataclass
class ChunkKV:
    """Chunk的KV Cache数据"""
    chunk_key: ChunkKey
    block_ids: list[int]     # 在Chunk Pool中的块ID
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    virtual_position: range  # 虚拟位置范围

@dataclass
class ChunkedPrompt:
    """解析后的chunked prompt"""
    sys_prompt: list[int]
    chunks: list[list[int]]   # 各个chunk的token序列
    user_question: list[int]
    separator: str
    chunk_boundaries: list[tuple[int, int]]  # [(start, end), ...]

@dataclass
class RemappedChunkKV:
    """重映射后的Chunk KV（用于推理）"""
    block_ids: list[int]     # 在主KV Pool中的块ID
    position: range          # 实际位置范围
```

---

## 主要流程

### 流程1：缓存未命中 - 生成Chunk KV并缓存

```
User Request: "You are helpful. # # Doc1 # # What?"
    ↓
[1] Prompt预处理 - InputProcessor
    • 检测到" # # "分隔符
    • 调用 ChunkTokenParser.parse()
    • 返回 ChunkedPrompt {
        sys_prompt: [1,2,3],
        chunks: [[10,11,...,200]],
        user_question: [100,101],
        chunk_boundaries: [(0,3), (3,203), (203,205)]
      }
    ↓
[2] Chunk哈希查找 - ChunkCacheEngine.lookup()
    • 对每个chunk调用 ChunkHashIndex.lookup()
    • sys_prompt: hash_abc → [MISS]
    • Doc1: hash_def → [MISS]
    ↓
[3] 生成Chunk KV (虚拟位置) - GPUModelRunner.compute_chunk_kv()
    • 分配 Chunk Pool 块 [100, 101]
    • 在虚拟位置 [0, max_chunk_len) 计算KV
    • 应用虚拟位置的RoPE编码
    • ChunkKV {
        chunk_key: ChunkKey(hash_abc, ...),
        block_ids: [100, 101],
        key_cache, value_cache,
        virtual_position: [0, 3)
      }
    ↓
[4] 缓存Chunk KV - ChunkCacheEngine.store()
    • ChunkKVConnector.to_cpu()  # GPU→CPU
    • ChunkHashIndex.insert(hash_abc, chunk_kv)
    ↓
[5] 重复[3-4]处理Doc1 chunk
    compute_chunk_kv(tokens=Doc1, position=0)
    → ChunkKV(hash_def, block_ids=[102,103,...], ...)
    → store(Doc1, chunk_kv)
    ↓
[6] 拷贝并重映射到实际位置 - PositionRemapper.remap()
    • 分配主KV Pool块 [5000, 5001]
    • 拷贝KV数据 (100→5000, 101→5001)
    • 应用位置0的RoPE编码
    • RemappedChunkKV {
        block_ids: [5000, 5001],
        position: [0, 3)
      }
    ↓
[7] 重复[6]处理Doc1 chunk
    remap(src_kv=Doc1_chunk, dst_position=3)
    → 分配主KV Pool块 [5002, 5003,...]
    → 拷贝KV+应用位置3的RoPE
    ↓
[8] 构建Chunk-Aware Attention Mask
    AttentionMaskBuilder.get_chunk_aware_mask(
        chunk_boundaries=[(0,3), (3,203), (203,205)]
    )
    → sys_prompt tokens: 标准causal attention
    → Doc1 tokens: 可看sys_prompt + 同chunk内causal
    → question tokens: 可看所有内容
    ↓
[9] 执行推理
    AscendSFABackend.forward(
        query, key, value,
        chunk_attn_mask,
        remapped_kv_blocks=[5000,5001,5002,5003,...]
    )
    → Generated Output
```

### 流程2：缓存命中 - 拷贝拼接KV并推理

```
User Request: "Different sys. # # Doc1 # # Tell me more"
    ↓
[1] Prompt解析
    InputProcessor.parse_chunked_prompt()
    → ChunkedPrompt {
        sys_prompt: [5,6,7],  # 不同!
        chunks: [[10,11,...,200]],  # 相同Doc1
        user_question: [200,201]
      }
    ↓
[2] Chunk哈希查找
    ChunkCacheEngine.lookup()
    → sys_prompt: hash_xyz → [MISS]
    → Doc1: hash_def → [HIT]  ✓ (来自之前请求)
    ↓
[3] 计算未命中的sys_prompt
    compute_chunk_kv(tokens=[5,6,7], position=0)
    → store(hash_xyz, sys_prompt_kv)
    ↓
[4] 拷贝sys_prompt并重映射
    PositionRemapper.remap(sys_prompt_kv, dst_position=0)
    → 分配主KV Pool块 [6000, 6001]
    → RemappedChunkKV {
        block_ids: [6000, 6001],
        position: [0, 3)
      }
    ↓
[5] 拷贝缓存的Doc1并重映射 (跳过计算)
    PositionRemapper.remap(
        src_kv=Doc1_chunk,  # 从缓存读取
        dst_position=3  # 新sys_prompt在位置0-2
    )
    → 分配主KV Pool块 [6002, 6003,...]
    → 拷贝KV+应用位置3的RoPE
    → RemappedChunkKV {
        block_ids: [6002, 6003,...],
        position: [3, 203)
      }
    ↓
[6] 拼接所有KV
    合并:
    - sys_prompt KV (blocks [6000,6001], pos [0,3))
    - Doc1 KV (blocks [6002,6003,...], pos [3,203))  # 来自缓存
    - question KV (blocks [6100,6101], pos [203,205))
    → merged_blocks: [6000,6001,6002,6003,...,6100,6101]
    ↓
[7] 构建Chunk-Aware Attention Mask
    根据新的chunk边界构建mask
    → chunk_ids: [-1,-1,-1, 0,0,..., -2,-2]
    → chunk隔离: Doc1不能看新sys_prompt的token
    ↓
[8] 执行推理
    AscendSFABend.forward(
        query, key, value,
        chunk_attn_mask,
        merged_kv_blocks
    )
    → Generated Output
```

**性能对比：**

| 步骤 | 缓存未命中 | 缓存命中 |
|------|-----------|---------|
| 哈希查找 | MISS | HIT |
| KV计算 | ✓ (50ms) | ✗ |
| 缓存存储 | ✓ | ✗ |
| KV拷贝 | ✓ (3-5ms) | ✓ (3-5ms) |
| RoPE处理 | 虚拟→实际 | 虚拟→实际 |
| 总时间 | ~55ms | ~5ms |
| 加速比 | 1x | **12x** |

### 流程3：位置编码处理（关键流程）

```
虚拟位置计算：
┌─────────────────────────────────────────────────┐
│ Chunk计算时（所有chunk在同一虚拟位置）          │
│ sys_prompt:  [0, 1, 2]                          │
│ chunk1:      [0, 1, 2, ..., 199]                │
│ chunk2:      [0, 1, 2, ..., 199]                │
└─────────────────────────────────────────────────┘
                    ↓ 拷贝+重映射
┌─────────────────────────────────────────────────┐
│ 实际推理位置（根据prompt中的实际位置）          │
│ sys_prompt:  [0, 1, 2]                          │
│ chunk1:      [3, 4, 5, ..., 202]                │
│ chunk2:      [203, 204, ..., 402]               │
│ question:    [403, 404, ...]                    │
└─────────────────────────────────────────────────┘

PositionRemapper.remap()实现：
1. 从Chunk Pool读取KV（虚拟位置编码）
2. 拷贝到主KV Pool
3. 应用新位置的RoPE编码（覆盖虚拟位置编码）
4. 返回重映射后的KV

伪代码：
def remap(src_kv: ChunkKV, dst_position: int) -> RemappedChunkKV:
    # 1. 分配主KV Pool块
    dst_block_ids = main_kv_pool.allocate(src_kv.num_tokens)

    # 2. 拷贝KV数据
    main_kv_pool.copy(src_kv.block_ids, dst_block_ids)

    # 3. 应用新位置的RoPE编码
    positions = torch.arange(dst_position, dst_position + src_kv.num_tokens)
    apply_rope_inplace(dst_block_ids, positions)

    return RemappedChunkKV(dst_block_ids, range(dst_position, ...))
```

### 流程4：Chunk-Aware Attention Mask构建

```
输入：chunk_boundaries = [(0,3), (3,203), (203,205)]
      seq_len = 205

步骤：
1. 为每个token分配chunk_id
   tokens [0,1,2]   → chunk_id = -1 (sys_prompt)
   tokens [3..202]  → chunk_id = 0 (chunk1)
   tokens [203..204]→ chunk_id = -2 (user_question)

2. 构建attention mask矩阵 (205x205)
   mask[i,j] = 可见性

   规则：
   - chunk_id = -1 (sys_prompt): 标准causal attention
   - chunk_id = 0 (chunk): 可看chunk_id=-1 + 同chunk内causal
   - chunk_id = -2 (question): 可看所有

3. 生成最终的attention mask张量

示例 (10x10子矩阵):
     j=0 1 2 3 4 5 6 7 8 9
i=0  1 1 1 1 1 1 1 1 1 1  ← sys_prompt (标准causal)
i=1  0 1 1 1 1 1 1 1 1 1  ← sys_prompt
i=2  0 0 1 1 1 1 1 1 1 1  ← sys_prompt
i=3  1 1 1 1 1 1 0 0 0 0  ← chunk1 (可看sys_prompt+同chunk)
i=4  1 1 1 0 1 1 0 0 0 0  ← chunk1
i=5  1 1 1 0 0 1 0 0 0 0  ← chunk1
i=6  1 1 1 1 1 1 1 1 1 1  ← question (可看所有)
i=7  1 1 1 1 1 1 0 1 1 1  ← question
i=8  1 1 1 1 1 1 0 0 1 1  ← question
i=9  1 1 1 1 1 1 0 0 0 1  ← question
```

---

## 相关代码

### 需要新增的文件

| 文件路径 | 主要类/方法 | 说明 |
|---------|------------|------|
| `vllm/v1/core/chunk_datastructures.py` | `ChunkKey`, `ChunkKV`, `ChunkedPrompt` | 核心数据结构定义 |
| `vllm/v1/core/chunk_token_parser.py` | `ChunkTokenParser` | 解析`# #`分隔符，计算内容哈希 |
| `vllm/v1/core/chunk_hash_index.py` | `ChunkHashIndex` | 哈希→chunk映射，O(1)查找 |
| `vllm/v1/core/chunk_storage_manager.py` | `ChunkStorageManager` | 管理chunk存储分配 |
| `vllm/v1/core/chunk_kv_connector.py` | `ChunkKVConnector` | GPU↔CPU数据传输 |
| `vllm/v1/core/chunk_cache_engine.py` | `ChunkCacheEngine` | 协调层，提供lookup/store API |
| `vllm/v1/worker/chunk_kv_pool.py` | `ChunkKVPool` | 独立显存池管理 |
| `vllm/v1/worker/position_remapper.py` | `PositionRemapper` | 核心拷贝+RoPE方法 |
| `vllm-ascend/attention/chunk_aware_mask.py` | `ChunkAwareMaskBuilder` | Chunk隔离注意力mask |

### 需要修改的文件

| 文件路径 | 修改内容 | 新增方法 |
|---------|---------|---------|
| `vllm/config/vllm_config.py` | 添加ChunkCacheConfig配置类 | `chunk_cache_config: ChunkCacheConfig` |
| `vllm/v1/engine/input_processor.py` | 检测`# #`，解析chunked prompt | `_detect_chunked_prompt()`, `parse_chunked_prompt()` |
| `vllm/v1/worker/gpu_model_runner.py` | 新增chunk处理方法 | `get_or_compute_chunks()`, `compute_chunk_kv()` |
| `vllm-ascend/attention/sfa_v1.py` | 集成chunk-aware attention mask | 修改`forward()` |

### 核心方法签名

```python
# ChunkTokenParser
class ChunkTokenParser:
    def parse(
        self,
        tokens: list[int],
        separator: str = " # # "
    ) -> ChunkedPrompt:
        """解析chunked prompt"""
        pass

    def compute_hash(
        self,
        tokens: list[int]
    ) -> bytes:
        """计算内容哈希（XXHash128）"""
        pass

# ChunkCacheEngine
class ChunkCacheEngine:
    def lookup(
        self,
        chunk_hash: bytes
    ) -> ChunkKV | None:
        """查找chunk缓存"""
        pass

    def store(
        self,
        chunk_hash: bytes,
        chunk_kv: ChunkKV
    ) -> None:
        """存储chunk"""
        pass

    def clear(self) -> None:
        """清空缓存"""
        pass

# PositionRemapper
class PositionRemapper:
    def remap(
        self,
        src_kv: ChunkKV,
        dst_position: int
    ) -> RemappedChunkKV:
        """拷贝KV并应用新位置的RoPE编码"""
        pass
```

### 关键文件路径

```
vllm/
├── vllm/
│   ├── config/
│   │   └── vllm_config.py              # 修改：添加ChunkCacheConfig
│   ├── v1/
│   │   ├── core/
│   │   │   ├── chunk_datastructures.py   # 新增：数据结构
│   │   │   ├── chunk_token_parser.py     # 新增：解析器
│   │   │   ├── chunk_hash_index.py       # 新增：哈希索引
│   │   │   ├── chunk_storage_manager.py  # 新增：存储管理
│   │   │   ├── chunk_kv_connector.py     # 新增：GPU↔CPU传输
│   │   │   └── chunk_cache_engine.py     # 新增：协调层
│   │   ├── engine/
│   │   │   └── input_processor.py        # 修改：检测和解析chunked
│   │   └── worker/
│   │       ├── chunk_kv_pool.py          # 新增：显存池
│   │       ├── position_remapper.py      # 新增：位置重映射
│   │       └── gpu_model_runner.py       # 修改：集成chunk处理
│   └── model_executor/
│       └── layers/
│           └── rotary_embedding/         # 现有：RoPE实现

vllm-ascend/
└── vllm_ascend/
    ├── attention/
    │   ├── sfa_v1.py                    # 修改：集成chunk mask
    │   └── chunk_aware_mask.py          # 新增：chunk-aware mask
    └── distributed/
        └── kvpool/
            └── ascend_store_connector.py # 修改：支持chunk传输
```

---

## 使用说明

### 配置参数

```bash
vllm serve <model_name> \
  --enable-chunk-cache \
  --chunk-separator " # # " \
  --chunk-cache-gpu-memory-fraction 0.15 \
  --chunk-hash-algorithm xxhash
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_chunk_cache` | `False` | 是否启用chunk缓存 |
| `chunk_separator` | `" # # "` | chunk分隔符 |
| `chunk_cache_gpu_memory_fraction` | `0.15` | Chunk池占用GPU比例（默认2GB） |
| `chunk_hash_algorithm` | `"xxhash"` | 哈希算法（xxhash/sha256） |

### Prompt格式

使用`# #`分隔符分割prompt：

```python
sys_prompt = "You are a helpful assistant."
doc1 = "This is document 1 content..."
doc2 = "This is document 2 content..."
question = "What are the key points?"

prompt = f"{sys_prompt} # # {doc1} # # {doc2} # # {question}"
```

**格式说明：**
1. 第一部分：系统提示（sys_prompt）
2. 中间部分：一个或多个chunk，用`# #`分隔
3. 最后部分：用户问题（user_question）

### Python API示例

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

# 初始化模型（启用chunk缓存）
llm = LLM(
    model="Qwen/Qwen2.5-7B",
    enable_chunk_cache=True,
    chunk_separator=" # # ",
    chunk_cache_gpu_memory_fraction=0.15,
)

# 第一次请求（缓存未命中）
prompt1 = "You are helpful. # # Doc1 content here. # # What?"
outputs = llm.generate([prompt1], SamplingParams(max_tokens=100))
print(outputs[0].outputs[0].text)

# 第二次请求（Doc1缓存命中）
prompt2 = "Different role. # # Doc1 content here. # # Tell me more"
outputs = llm.generate([prompt2], SamplingParams(max_tokens=100))
print(outputs[0].outputs[0].text)  # Doc1部分从缓存读取，加速约12x
```

### 监控统计

```python
# 获取缓存统计
stats = llm.llm_engine.chunk_cache_engine.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Total chunks cached: {stats['total_chunks']}")
print(f"Memory usage: {stats['current_memory_mb']:.2f} MB")
```

**统计指标：**
- `hit_rate`: 缓存命中率
- `total_chunks`: 总chunk数量
- `current_memory_mb`: 当前内存使用量（MB）
- `cache_hits`: 缓存命中次数
- `cache_misses`: 缓存未命中次数

---

## 实施步骤

### Phase 1：核心数据结构（1-2天）

**目标：** 建立基础的数据结构和接口定义

**任务：**
1. 创建 `vllm/v1/core/chunk_datastructures.py`
   - 定义 `ChunkKey`, `ChunkKV`, `ChunkedPrompt`, `RemappedChunkKV`
   - 添加类型注解和文档字符串

2. 创建 `vllm/v1/core/chunk_token_parser.py`
   - 实现 `ChunkTokenParser.parse()`: 解析`# #`分隔符
   - 实现 `ChunkTokenParser.compute_hash()`: 计算XXHash128
   - 添加单元测试

3. 创建 `vllm/v1/core/chunk_hash_index.py`
   - 实现 `ChunkHashIndex`: 字典映射
   - 实现 `lookup()`, `insert()` 方法
   - 添加单元测试

**验收标准：**
- 所有数据结构定义完整
- 单元测试通过
- 代码覆盖率 > 80%

### Phase 2：存储层（2-3天）

**目标：** 实现chunk的存储和管理

**任务：**
1. 创建 `vllm/v1/core/chunk_storage_manager.py`
   - 实现 `ChunkStorageManager`: 管理chunk存储
   - 实现 `allocate()`, `free()`, `get()`, `put()` 方法
   - 添加内存统计功能

2. 创建 `vllm/v1/worker/chunk_kv_pool.py`
   - 实现 `ChunkKVPool`: 独立显存池
   - 集成到vLLM的内存管理系统
   - 默认分配2GB显存

3. 创建 `vllm/v1/core/chunk_kv_connector.py`
   - 实现 `ChunkKVConnector`: GPU↔CPU传输
   - 实现 `to_cpu()`, `to_gpu()` 方法
   - 优化传输性能

**验收标准：**
- 存储管理功能正常
- 内存分配和释放无泄漏
- GPU↔CPU传输性能测试通过

### Phase 3：协调层（1-2天）

**目标：** 实现统一的缓存API

**任务：**
1. 创建 `vllm/v1/core/chunk_cache_engine.py`
   - 实现 `ChunkCacheEngine`: 协调层
   - 实现 `lookup()`, `store()`, `clear()` API
   - 集成 ChunkTokenParser, ChunkHashIndex, ChunkStorageManager

2. 添加缓存统计功能
   - 命中率统计
   - 内存使用统计
   - 性能指标收集

**验收标准：**
- API功能完整
- 统计功能正常
- 集成测试通过

### Phase 4：集成到vLLM（2-3天）

**目标：** 将chunk缓存集成到vLLM核心流程

**任务：**
1. 修改 `vllm/config/vllm_config.py`
   - 添加 `ChunkCacheConfig` 配置类
   - 集成到 `VllmConfig`

2. 修改 `vllm/v1/engine/input_processor.py`
   - 添加 `_detect_chunked_prompt()`: 检测`# #`分隔符
   - 添加 `parse_chunked_prompt()`: 解析chunked prompt
   - 集成到 `process_inputs()` 流程

3. 修改 `vllm/v1/worker/gpu_model_runner.py`
   - 添加 `get_or_compute_chunks()`: 获取或计算chunks
   - 添加 `compute_chunk_kv()`: 计算chunk KV
   - 集成 ChunkCacheEngine

**验收标准：**
- 配置系统正常工作
- Prompt解析功能正常
- 与vLLM核心流程集成成功

### Phase 5：NPU优化（2-3天）

**目标：** 实现NPU特定的优化

**任务：**
1. 创建 `vllm/v1/worker/position_remapper.py`
   - 实现 `PositionRemapper.remap()`: 核心拷贝+RoPE方法
   - 优化NPU性能
   - 添加性能测试

2. 创建 `vllm-ascend/attention/chunk_aware_mask.py`
   - 实现 `ChunkAwareMaskBuilder`
   - 实现 `get_chunk_aware_mask()` 方法
   - 添加单元测试

3. 修改 `vllm-ascend/attention/sfa_v1.py`
   - 集成chunk-aware attention mask
   - 修改 `forward()` 方法
   - 添加集成测试

**验收标准：**
- KV拷贝+RoPE性能测试通过
- Attention mask生成正确
- NPU性能优化生效

### Phase 6：测试和优化（2-3天）

**目标：** 全面测试和性能优化

**任务：**
1. 单元测试
   - 所有模块的单元测试
   - 边界条件测试
   - 错误处理测试

2. 集成测试
   - 端到端流程测试
   - 与Prefix Caching共存测试
   - 性能回归测试

3. 性能测试和优化
   - RAG场景性能测试
   - 缓存命中率测试
   - 内存使用优化
   - NPU性能调优

4. 文档完善
   - API文档
   - 使用示例
   - 性能指南

**验收标准：**
- 所有测试通过
- 性能达标（12x加速）
- 文档完整

**总计：10-17天**

---

## 注意事项

### 1. 与Prefix Caching共存

- ChunkCache与Prefix Caching独立运行
- ChunkCache优先级更高（检测到`# #`时启用）
- 无`# #`时回退到标准Prefix Caching

**检测逻辑：**
```python
if " # # " in prompt_str:
    use_chunk_cache = True
else:
    use_chunk_cache = False  # 回退到Prefix Caching
```

### 2. 内存管理

双内存池架构：
```
GPU内存分布：
┌─────────────────────────────────┐
│ 主KV Pool (70%)                 │
│ - 活跃请求的KV cache            │
│ - 动态分配                      │
├─────────────────────────────────┤
│ Chunk Pool (15%)                │
│ - 位置无关的chunk缓存           │
│ - 默认2GB                       │
├─────────────────────────────────┤
│ 预留 (15%)                      │
│ - 模型权重                      │
│ - 其他用途                      │
└─────────────────────────────────┘
```

### 3. 位置编码处理

**虚拟位置计算：**
- 所有chunk在`[VIRTUAL_POS_START, VIRTUAL_POS_START + max_chunk_len)`计算
- 虚拟位置范围：`[0, 8192)`（假设max_chunk_len=8192）

**重映射：**
- 拷贝时应用新位置的RoPE编码
- 覆盖虚拟位置编码

**NPU优化：**
- 使用NPU加速的KV拷贝
- 使用NPU优化的RoPE计算

### 4. Chunk隔离规则

| Token类型 | 可见范围 |
|-----------|---------|
| **sys_prompt** | 标准causal attention（可看前面所有sys_prompt token） |
| **chunk token** | 可看所有sys_prompt + 同chunk内前序token |
| **user_question** | 可看所有内容（sys_prompt + 所有chunks + 前序question token） |

**示例：**
```
sys_prompt: [A, B, C]
chunk1:     [D, E, F]
chunk2:     [G, H, I]
question:   [J, K]

Attention可见性：
- A: 可看 {}
- B: 可看 {A}
- C: 可看 {A, B}
- D: 可看 {A, B, C}  (sys_prompt)
- E: 可看 {A, B, C, D}  (sys_prompt + 同chunk)
- F: 可看 {A, B, C, D, E}  (sys_prompt + 同chunk)
- G: 可看 {A, B, C}  (sys_prompt，不能看chunk1)
- H: 可看 {A, B, C, G}  (sys_prompt + 同chunk)
- I: 可看 {A, B, C, G, H}  (sys_prompt + 同chunk)
- J: 可看 {A, B, C, D, E, F, G, H, I}  (所有)
- K: 可看 {A, B, C, D, E, F, G, H, I, J}  (所有)
```

### 5. 兼容性保证

- 默认禁用，检测到`# #`时启用
- 不破坏现有Prefix Caching功能
- 向后兼容现有代码
- 不影响现有API

**配置开关：**
```python
# 默认禁用
enable_chunk_cache = False

# 检测到分隔符时启用
if " # # " in prompt:
    enable_chunk_cache = True
```

### 6. 错误处理

**错误场景：**
1. **Chunk Pool内存不足**
   - 策略：报错并回退到Prefix Caching
   - 日志：记录内存不足事件

2. **哈希冲突**
   - 概率：XXHash128冲突概率极低（2^-128）
   - 策略：忽略（可接受）

3. **格式错误**
   - 示例：`" # # " # # " # # "`（空chunk）
   - 策略：跳过空chunk，记录警告

4. **位置编码超出范围**
   - 策略：报错并回退到Prefix Caching
   - 日志：记录错误详情

---

## 总结

本方案设计了一个完整的位置无关KVCache缓存和复用功能，通过以下创新点实现了高效的chunk复用：

1. **位置无关缓存**：基于内容哈希，实现真正的跨请求chunk复用
2. **虚拟位置计算**：避免位置编码冲突，支持chunk重用
3. **Chunk隔离注意力**：确保chunk间的正确隔离
4. **独立内存池**：不影响主推理流程
5. **NPU优化**：充分利用昇腾NPU的加速能力

**性能提升：**
- 单个chunk缓存命中：12x加速
- RAG场景：50-80% TTFT减少
- 端到端：2-5x加速（80%+命中率）

**实施计划：**
- 分6个阶段，预计10-17天完成
- 从数据结构开始，到集成和NPU优化
- 最后进行全面测试和文档完善

本方案在vLLM-Ascend平台上实现，完全兼容vLLM主项目，不影响现有功能，为RAG等场景提供了强大的性能提升。
