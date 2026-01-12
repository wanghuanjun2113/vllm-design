# 多文档Chunk处理与受限注意力机制 - 设计文档

## 1. 需求概述

### 1.1 应用场景
多文档处理场景，需要让多个文档chunk独立关注系统提示词，同时避免chunk之间的相互干扰。

### 1.2 Prompt格式
```
sys_prompt + " # # " + chunk1 + " # # " + chunk2 + " # # " + chunk3 + " # # " user question
```

### 1.3 核心约束

**注意力约束**:
- ✅ 每个chunk的token可以关注sys_prompt的token
- ✅ chunk内的token之间**可以相互关注** (causal attention，仅限同chunk内的前序token)
- ❌ chunk之间**不能相互关注**
- ✅ user question的token可以关注所有内容（sys_prompt + 所有chunks）

**位置编码约束**:
- 每个chunk的位置编码都从sys_prompt之后重新开始（共享相同的位置范围）
- user question的位置编码从sys_prompt + 单个chunk长度之后开始
- 采用相对位置编码

**用户澄清**:
- chunk之间**无重叠**（独立文档）
- chunk内的token**可以关注sys_prompt和同chunk内的前序token**（causal attention，但不能关注其他chunk）
- 位置编码采用**相对位置编码**策略

---

## 2. 技术挑战分析

### 2.1 挑战1: KV Cache位置冲突
**问题**: KV cache存储的是**已包含位置编码**的K/V。如果多个chunk共享相同的位置范围，它们在位置编码空间中会冲突。

**影响**:
- chunk1的token和chunk2的token有相同的位置ID
- 它们的K在位置编码后是相同的
- 无法区分它们应该关注哪些内容

### 2.2 挑战2: 自定义Attention Pattern
**问题**: 需要实现特殊的注意力模式：
- 标准causal attention: token i 只能关注 [0, i]
- 需求的pattern: chunk token可以关注sys_prompt + 同chunk内的前序token（chunk-isolated causal attention）
- chunk i的token CANNOT关注 chunk j（j != i）的token

**影响**:
- 不能简单使用标准causal mask
- 需要自定义attention mask来支持"chunk隔离的causal attention"

### 2.3 挑战3: 位置编码的语义一致性
**问题**: RoPE通过旋转角度编码位置信息。如果两个token有相同的位置ID，它们会得到相同的旋转。

**影响**:
- chunk1的位置0和chunk2的位置0会得到相同的RoPE编码
- 这是**期望的行为**（让它们都能关注sys_prompt）
- 但需要确保attention mask正确限制它们的关注范围

### 2.4 挑战4: 性能与内存效率
**问题**:
- 多个chunk意味着序列更长
- 不能简单将多个chunk拼接成长序列（会违反约束）
- 需要考虑KV cache的复用

---

## 3. 整体设计架构

### 3.1 设计思路

采用**分层设计**，在vLLM现有架构基础上通过扩展点实现：

```
┌─────────────────────────────────────────────────────┐
│                   User API Layer                    │
│  LLM(enable_chunk_aware=True, chunk_separator=" # # ")│
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              Input Processing Layer                 │
│  - Parse chunks from prompt                         │
│  - Identify chunk boundaries                        │
│  - Compute remapped positions                       │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│          Attention Metadata Layer                   │
│  ChunkAwareAttentionMetadata:                       │
│    - chunk_ids: [-1, -1, ..., 0,0,0, ..., 1,1,1, ...]│
│    - remapped_positions: [0,1,2,...,10,11,12,...]   │
│    - chunk_attn_mask: custom attention pattern       │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼─────────┐    ┌────────▼────────────┐
│ Rotary Embedding │    │ Attention Backend   │
│ ChunkAwareRoPE   │    │ ChunkAwareAttn      │
│                  │    │                     │
│ - Remap positions│    │ - Custom mask       │
│ - Shared range   │    │ - Enforce constraints│
└──────────────────┘    └──────────────────────┘
```

### 3.2 核心设计原则

1. **插件化架构**: 通过backend和metadata扩展，最小化对核心代码的修改
2. **向后兼容**: 通过配置开关控制，默认禁用chunk-aware模式
3. **性能优先**: 利用FlashAttention、prefix caching等现有优化
4. **用户友好**: 自动检测chunk边界，无需手动指定

---

## 4. 详细技术设计

### 4.1 Chunk-Aware Rotary Embedding

#### 4.1.1 设计原理

**关键洞察**: RoPE编码的是**相对位置关系**。如果两个token有相同的位置ID，它们会得到相同的旋转。

**利用这个特性**:
- 让所有chunk的token使用相同的位置范围 [sys_prompt_end, sys_prompt_end + chunk_len)
- 这样它们的K/V在RoPE编码空间中是"等价"的
- 通过attention mask限制它们只能关注sys_prompt

#### 4.1.2 位置重映射算法

```python
# 位置重映射示例
# 原始序列: [sys:0-9] [sep:10-12] [chunk1:13-22] [sep:23-25] [chunk2:26-35] [sep:36-38] [chunk3:39-48] [sep:49-51] [question:52-60]

# chunk_ids: [-1,-1,...,-1, 0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2, -2,-2,-2,-2,-2,-2,-2,-2,-2]
#              sys_prompt(10)  chunk1(10)               chunk2(10)               chunk3(10)    question(9)

# remapped_positions:
#   sys_prompt: [0,1,2,3,4,5,6,7,8,9]
#   chunk1:     [10,11,12,13,14,15,16,17,18,19]  # 从10开始
#   chunk2:     [10,11,12,13,14,15,16,17,18,19]  # 相同范围！
#   chunk3:     [10,11,12,13,14,15,16,17,18,19]  # 相同范围！
#   question:   [20,21,22,23,24,25,26,27,28]     # 接着chunk范围
```

#### 4.1.3 实现文件

**文件**: `vllm/model_executor/layers/rotary_embedding/chunk_aware_rope.py`

```python
class ChunkAwareRotaryEmbedding(RotaryEmbedding):
    """
    Rotary embedding that allows multiple chunks to share position ranges.

    Position remapping strategy:
    - sys_prompt tokens: keep original positions [0, sys_prompt_len)
    - chunk tokens: remap to [sys_prompt_len, sys_prompt_len + max_chunk_len)
      All chunks share this range!
    - question tokens: continue from [sys_prompt_len + max_chunk_len, ...)

    This allows chunks to have identical position encodings,
    enabling them to attend to sys_prompt with the same relative positions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_chunk_len = kwargs.get('max_chunk_len', 0)

    def forward_with_chunk_info(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        chunk_ids: torch.Tensor | None = None,
        sys_prompt_end: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply rotary embeddings with chunk-aware position remapping.

        Args:
            positions: Original absolute positions [num_tokens]
            query: [num_tokens, num_heads, head_dim]
            key: [num_tokens, num_kv_heads, head_dim]
            chunk_ids: [-1=sys, 0,1,2,...=chunks, -2=question]
            sys_prompt_end: Length of system prompt (exclusive)

        Returns:
            Rotated query and key
        """
        if chunk_ids is None:
            return super().forward(positions, query, key)

        # Remap positions
        remapped_positions = self._remap_positions(
            positions, chunk_ids, sys_prompt_end
        )

        # Apply RoPE with remapped positions
        return super().forward(remapped_positions, query, key)

    def _remap_positions(
        self,
        positions: torch.Tensor,
        chunk_ids: torch.Tensor,
        sys_prompt_end: int,
    ) -> torch.Tensor:
        """Remap positions so chunks share the same range."""
        remapped = positions.clone()

        for i, chunk_id in enumerate(chunk_ids):
            if chunk_id >= 0:  # Chunk token
                # Compute position within this chunk
                chunk_start_pos = self._get_chunk_start(chunk_id)
                pos_in_chunk = positions[i].item() - chunk_start_pos
                remapped[i] = sys_prompt_end + pos_in_chunk
            elif chunk_id == -2:  # Question token
                # Compute position after all chunks
                num_tokens_before = positions[i].item()
                # Keep original (already after chunks)
                pass

        return remapped
```

### 4.2 Chunk-Aware Attention Backend

#### 4.2.1 设计原理

**关键挑战**: 如何让chunk token关注sys_prompt和同chunk内的前序token，但不关注其他chunk？

**解决方案**: 自定义attention mask实现chunk-isolated causal attention

```python
# Attention mask design
# Rows: query tokens, Cols: key/value tokens
# T=True (can attend), F=False (cannot attend)
# causal=causal attention within the group

#          sys_prompt  chunk1_causal  chunk2_causal  chunk3_causal  question_causal
# sys_prompt    T              F               F               F              F
# chunk1        T         causal(T/T)          F               F              F
# chunk2        T              F          causal(T/T)           F              F
# chunk3        T              F               F          causal(T/T)          F
# question      T              T               T               T         causal(T/T)
```

#### 4.2.2 Attention Mask构建

**文件**: `vllm/v1/attention/ops/chunk_attention_mask.py`

```python
def build_chunk_attention_mask(
    num_tokens: int,
    max_seq_len: int,
    chunk_ids: torch.Tensor,
    sys_prompt_end: int,
    chunk_boundaries: list[tuple[int, int]],  # NEW: needed for intra-chunk causal
) -> torch.Tensor:
    """
    Build attention mask for chunk-aware attention.

    Rules:
    1. sys_prompt tokens: standard causal attention within sys_prompt
    2. chunk tokens: can attend to sys_prompt + earlier tokens in SAME chunk (causal)
    3. question tokens: can attend to everything (causal)

    Args:
        num_tokens: Total tokens in sequence
        max_seq_len: Maximum sequence length
        chunk_ids: [-1=sys, 0,1,2,...=chunks, -2=question]
        sys_prompt_end: End of sys_prompt (exclusive)
        chunk_boundaries: [(start1, end1), (start2, end2), ...] token positions

    Returns:
        attention_mask: [num_tokens, max_seq_len] bool tensor
    """
    mask = torch.zeros(num_tokens, max_seq_len, dtype=torch.bool)

    for query_pos in range(num_tokens):
        query_chunk_id = chunk_ids[query_pos].item()

        if query_chunk_id >= 0:  # Chunk token
            # Can attend to sys_prompt
            mask[query_pos, :sys_prompt_end] = True

            # Can attend to earlier tokens in SAME chunk (causal)
            chunk_start, chunk_end = chunk_boundaries[query_chunk_id]
            causal_end = min(query_pos, chunk_end - 1) + 1  # +1 for slice
            mask[query_pos, chunk_start:causal_end] = True

        elif query_chunk_id == -2:  # Question token
            # Can attend to everything before it (causal)
            mask[query_pos, :query_pos] = True

        else:  # sys_prompt token
            # Standard causal attention
            mask[query_pos, :query_pos] = True

    return mask
```

#### 4.2.3 Attention Backend实现

**文件**: `vllm/v1/attention/backends/chunk_attn.py`

```python
@dataclass
class ChunkAwareAttentionMetadata(FlashAttentionMetadata):
    """Extended metadata for chunk-aware attention."""
    # Chunk boundaries
    sys_prompt_end: int = 0
    question_start: int = 0

    # Chunk information
    chunk_ids: torch.Tensor | None = None  # [num_tokens]
    remapped_positions: torch.Tensor | None = None  # [num_tokens]
    chunk_boundaries: list[tuple[int, int]] | None = None  # [(start1, end1), ...]

    # Pre-computed attention mask
    chunk_attn_mask: torch.Tensor | None = None  # [num_tokens, max_seq_len]


class ChunkAwareAttentionImpl(FlashAttentionImpl):
    """Implementation with custom attention mask."""

    def forward(
        self,
        layer,
        query, key, value,
        kv_cache,
        attn_metadata: ChunkAwareAttentionMetadata,
        output,
        **kwargs
    ):
        # Build chunk mask if needed
        if attn_metadata.chunk_attn_mask is None:
            attn_metadata.chunk_attn_mask = build_chunk_attention_mask(
                num_tokens=query.shape[0],
                max_seq_len=kv_cache.shape[1],
                chunk_ids=attn_metadata.chunk_ids,
                sys_prompt_end=attn_metadata.sys_prompt_end,
                chunk_boundaries=attn_metadata.chunk_boundaries,  # NEW parameter
            )

        # Apply mask to attention scores
        # This integrates with FlashAttention
        return super().forward(
            layer, query, key, value, kv_cache,
            attn_metadata, output, **kwargs
        )
```

### 4.3 Chunk边界识别

#### 4.3.1 设计思路

**问题**: 如何从prompt中自动识别chunk边界？

**方案**: 在InputProcessor中解析prompt，找到分隔符" # # "的token位置

#### 4.3.2 实现文件

**文件**: `vllm/v1/engine/input_processor.py` (扩展)

```python
class InputProcessor:
    def _parse_chunks(
        self,
        prompt: str,
        tokenizer,
        separator: str = " # # ",
    ) -> ChunkInfo:
        """
        Parse prompt to identify chunk boundaries.

        Args:
            prompt: Input prompt with separator
            tokenizer: Tokenizer to encode text
            separator: Chunk separator string

        Returns:
            ChunkInfo with boundaries and metadata
        """
        # Tokenize the full prompt
        full_tokens = tokenizer.encode(prompt)

        # Tokenize the separator
        sep_tokens = tokenizer.encode(separator, add_special_tokens=False)

        # Find all occurrences of separator
        sep_positions = find_sublist_positions(full_tokens, sep_tokens)

        # Determine boundaries
        if len(sep_positions) < 2:
            # No chunks, return simple info
            return ChunkInfo(
                num_chunks=0,
                sys_prompt_end=len(full_tokens),
                question_start=len(full_tokens),
                chunk_boundaries=[],
            )

        # Structure: [sys] [sep] [chunk1] [sep] [chunk2] ... [sep] [question]
        # sep_positions: [p1, p2, p3, p4] where:
        #   p1 = end of sys_prompt
        #   p2 = end of chunk1
        #   p3 = end of chunk2
        #   p4 = end of chunk3

        sys_prompt_end = sep_positions[0]
        question_start = sep_positions[-1] + len(sep_tokens)

        # Extract chunk boundaries
        chunk_boundaries = []
        for i in range(len(sep_positions) - 1):
            chunk_start = sep_positions[i] + len(sep_tokens)
            chunk_end = sep_positions[i + 1]
            chunk_boundaries.append((chunk_start, chunk_end))

        # Build chunk_ids tensor
        chunk_ids = self._build_chunk_ids(
            num_tokens=len(full_tokens),
            sys_prompt_end=sys_prompt_end,
            chunk_boundaries=chunk_boundaries,
            question_start=question_start,
        )

        return ChunkInfo(
            num_chunks=len(chunk_boundaries),
            sys_prompt_end=sys_prompt_end,
            question_start=question_start,
            chunk_boundaries=chunk_boundaries,
            chunk_ids=chunk_ids,
        )

    def _build_chunk_ids(
        self,
        num_tokens: int,
        sys_prompt_end: int,
        chunk_boundaries: list[tuple[int, int]],
        question_start: int,
    ) -> list[int]:
        """Build chunk_ids array."""
        chunk_ids = []

        # sys_prompt: -1
        chunk_ids.extend([-1] * sys_prompt_end)

        # chunks: 0, 1, 2, ...
        for chunk_idx, (start, end) in enumerate(chunk_boundaries):
            chunk_ids.extend([chunk_idx] * (end - start))

        # question: -2
        question_len = num_tokens - question_start
        chunk_ids.extend([-2] * question_len)

        return chunk_ids
```

### 4.4 KV Cache策略

#### 4.4.1 问题分析

**核心冲突**:
- KV cache存储的是**已应用位置编码**的K/V
- 多个chunk共享位置范围 → 它们的K/V在编码空间相同
- 但它们代表不同的内容 → 不能简单地复用cache

#### 4.4.2 解决方案

**策略1: 分离存储**
- 为每个chunk分配独立的KV cache slot
- 虽然位置编码相同，但物理存储分离
- 在attention时，根据chunk_id选择正确的K/V

**策略2: Prefix Cache复用**
- sys_prompt部分可以被所有chunk复用
- chunk部分独立存储
- 减少内存占用

**实现**:

```python
class ChunkAwareKVCacheManager:
    """Manage KV cache for chunk-aware attention."""

    def get_kv_cache_layout(
        self,
        chunk_ids: torch.Tensor,
        sys_prompt_end: int,
    ) -> dict:
        """
        Compute KV cache layout for chunks.

        Strategy:
        - sys_prompt: stored in shared prefix cache
        - each chunk: independent slots, cannot overwrite each other
        - question: follows the last chunk

        Returns:
            layout: {
                'sys_prompt_slots': [slot_ids...],
                'chunk_slots': {chunk_idx: [slot_ids...]},
                'question_slots': [slot_ids...],
            }
        """
        layout = {
            'sys_prompt_slots': list(range(sys_prompt_end)),
            'chunk_slots': {},
            'question_slots': [],
        }

        current_slot = sys_prompt_end
        chunk_starts = {}

        # Find slot positions for each chunk
        for i, chunk_id in enumerate(chunk_ids):
            if chunk_id >= 0:  # Chunk token
                if chunk_id not in chunk_starts:
                    chunk_starts[chunk_id] = current_slot
                    layout['chunk_slots'][chunk_id] = []
                layout['chunk_slots'][chunk_id].append(current_slot)
                current_slot += 1
            elif chunk_id == -2:  # Question token
                layout['question_slots'].append(current_slot)
                current_slot += 1

        return layout
```

### 4.5 配置与API

#### 4.5.1 配置参数

**文件**: `vllm/config/attention.py`

```python
@dataclass
class AttentionConfig:
    # Existing params...
    enable_chunk_aware: bool = False
    chunk_separator: str = " # # "
    chunk_position_strategy: Literal["shared", "offset"] = "shared"
    max_chunk_len: int = 8192
```

#### 4.5.2 用户API

```python
# Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_chunk_aware=True,
    chunk_separator=" # # ",
)

# Automatically handled
prompt = (
    "You are a helpful assistant. # # "
    "Document 1: This is the first document... # # "
    "Document 2: This is the second document... # # "
    "Document 3: This is the third document... # # "
    "Question: What are the key points from these documents?"
)

outputs = llm.generate(prompt, SamplingParams(temperature=0.8))
```

---

## 5. 实现路径

### 阶段1: 核心组件实现

**Step 1.1**: Chunk-Aware RoPE
- 文件: `vllm/model_executor/layers/rotary_embedding/chunk_aware_rope.py`
- 实现: ChunkAwareRotaryEmbedding类
- 关键方法: `forward_with_chunk_info()`, `_remap_positions()`

**Step 1.2**: Attention Mask工具
- 文件: `vllm/v1/attention/ops/chunk_attention_mask.py`
- 实现: `build_chunk_attention_mask()`, `apply_chunk_mask_to_attention()`

**Step 1.3**: Chunk-Aware Attention Backend
- 文件: `vllm/v1/attention/backends/chunk_attn.py`
- 实现: ChunkAwareAttentionBackend, ChunkAwareAttentionImpl
- 实现: ChunkAwareAttentionMetadata, ChunkAwareAttentionMetadataBuilder

**Step 1.4**: Backend注册
- 文件: `vllm/v1/attention/backends/registry.py`
- 添加: CHUNK_AWARE到AttentionBackendEnum

### 阶段2: 集成到vLLM引擎

**Step 2.1**: InputProcessor扩展
- 文件: `vllm/v1/engine/input_processor.py`
- 添加: `_parse_chunks()` 方法
- 添加: `_build_chunk_ids()` 方法

**Step 2.2**: Metadata传递
- 文件: `vllm/v1/worker/gpu_model_runner.py`
- 修改: 传递chunk_ids和remapped_positions到model

**Step 2.3**: 模型层集成
- 文件: `vllm/model_executor/models/llama.py`
- 修改: LlamaAttention使用ChunkAwareRotaryEmbedding
- 修改: LlamaAttention使用ChunkAwareAttentionBackend

### 阶段3: 优化与测试

**Step 3.1**: Prefix Cache优化
- 文件: `vllm/v1/engine/core_client.py`
- 实现: sys_prompt部分的KV cache复用

**Step 3.2**: 性能测试
- 文件: `vllm/tests/v1/attention/backends/test_chunk_attn.py`
- 测试: correctness, performance, memory

### 阶段4: 文档与示例

**Step 4.1**: API文档
- 使用示例和最佳实践

**Step 4.2**: 性能调优指南
- 如何设置chunk_len
- 如何优化内存使用

---

## 6. 关键文件清单

### 新增文件

| 文件路径 | 用途 |
|---------|------|
| `vllm/model_executor/layers/rotary_embedding/chunk_aware_rope.py` | Chunk-aware RoPE实现 |
| `vllm/v1/attention/ops/chunk_attention_mask.py` | Attention mask构建工具 |
| `vllm/v1/attention/backends/chunk_attn.py` | Chunk-aware attention backend |
| `vllm/tests/v1/attention/backends/test_chunk_attn.py` | 单元测试 |

### 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| `vllm/v1/attention/backends/registry.py` | 注册CHUNK_AWARE后端 |
| `vllm/v1/engine/input_processor.py` | 添加chunk解析逻辑 |
| `vllm/config/attention.py` | 添加enable_chunk_aware等配置 |
| `vllm/model_executor/models/llama.py` | 使用chunk-aware组件 |
| `vllm/v1/worker/gpu_model_runner.py` | 传递chunk metadata |

---

## 7. 性能考虑

### 7.1 内存开销

**额外内存**:
- `chunk_ids`: [num_tokens] × 4 bytes (int32)
- `remapped_positions`: [num_tokens] × 4 bytes
- `chunk_attn_mask`: [num_tokens, max_seq_len] × 1 byte (bool)

**优化**:
- 不存储完整的mask，按需计算
- 复用sys_prompt的KV cache

### 7.2 计算开销

**额外计算**:
- 位置重映射: O(num_tokens)
- Mask构建: O(num_tokens × max_seq_len) - 可预计算缓存

**优化**:
- 预计算mask模板
- 使用JIT编译优化关键路径

### 7.3 吞吐量影响

**预期影响**:
- 单个请求: +5-10% 延迟（mask计算）
- 批量处理: 如果所有请求使用相同chunk结构，影响最小

---

## 8. 潜在风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| RoPE精度问题 | 位置编码不准确 | 充分测试不同chunk_len |
| FlashAttention兼容性 | 性能下降 | 提供fallback实现 |
| KV cache冲突 | 内存错误 | 严格的slot mapping |
| 复杂性增加 | 维护困难 | 充分的文档和测试 |

---

## 9. 未来扩展

1. **支持可变chunk长度**: 当前假设所有chunk长度相同
2. **支持chunk间attention**: 可选地允许chunk相互关注
3. **动态chunk调整**: 运行时合并/拆分chunk
4. **其他位置编码**: 支持ALiBi等其他编码方式

---

## 10. 总结

本设计通过以下核心技术实现多文档chunk的受限注意力：

1. **位置编码重映射**: 多个chunk共享位置范围
2. **自定义Attention Mask**: 实现chunk-isolated causal attention（chunk内causal，chunk间隔离）
3. **Chunk-Aware Backend**: 无缝集成到vLLM架构
4. **自动Chunk检测**: 用户友好的API

这个设计在保持vLLM高性能的同时，实现了复杂的多文档处理场景。
