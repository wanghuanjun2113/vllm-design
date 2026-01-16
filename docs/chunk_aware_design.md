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
- user question的位置编码从sys_prompt + 单个最大chunk长度之后开始
- 采用相对位置编码

**用户澄清**:
- chunk之间**无重叠**（独立文档）
- chunk内的token**可以关注sys_prompt和同chunk内的前序token**（causal attention，但不能关注其他chunk）
- 位置编码采用**相对位置编码**策略

### 1.4 硬件平台

**目标硬件**: Huawei Ascend NPUs
- 支持芯片: Atlas 800I A2/A3系列、Atlas A2/A3训练系列、Atlas 300I Duo
- 芯片型号: 910B、910C、310P

**软件栈**:
- CANN 8.3.rc2 (Ascend HDK，相当于CUDA之于GPU)
- torch-npu 2.8.0 (PyTorch扩展，支持Ascend NPUs)
- HCCL (Huawei Collective Communication Library，分布式通信)

**实现架构**:
- 本设计基于**vLLM-Ascend插件**实现
- 使用**Sparse Flash Attention (SFA)** v1 backend
- 通过**ACL图编译**进行NPU kernel优化
- 集成点: `vllm-ascend/vllm_ascend/` 目录

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

### 2.5 挑战5: Ascend特定实现 (Ascend-Specific Challenges)

**问题**: 在Ascend NPU上实现需要考虑以下差异：

| 方面 | CUDA vLLM | vLLM-Ascend | 影响 |
|------|-----------|-------------|------|
| 图编译 | CUDA graphs | ACL图编译 | 需要适配ACL图构建流程 |
| 内存分配 | PyTorch default | camem_allocator.cpp | NPU特定的内存管理 |
| 通信库 | NCCL | HCCL | 分布式场景下的通信差异 |
| Backend | FlashAttention | SFA (Sparse Flash Attention) | 利用SFA的chunked prefill支持 |
| 位置编码 | 标准RoPE | NPU-optimized RoPE | `vllm_ascend.ops.rotary_embedding` |

**影响**:
- 需要集成到`vllm-ascend`插件的attention架构
- 扩展`AttentionMaskBuilder`来管理chunk-aware mask
- 在`AscendSFABackend`中集成自定义mask逻辑
- 考虑ACL图编译对mask模式的影响

---

## 3. 整体设计架构

### 3.1 设计思路

采用**分层设计**，基于vLLM-Ascend插件架构实现：

```
┌─────────────────────────────────────────────────────┐
│                   User API Layer                    │
│  LLM(enable_chunk_aware=True, chunk_separator=" # # ")│
│  (使用vLLM-Ascend插件，backend="ascend_sfa")        │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│         vLLM-Ascend Plugin Layer                   │
│  - NPUPlatform registration                         │
│  - Ascend-specific config (VllmConfig)             │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│          Input Processing Layer                     │
│  vllm-ascend worker:                                │
│  - Parse chunks from prompt                         │
│  - Identify chunk boundaries                        │
│  - Build chunk_ids tensor                           │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│          Attention Metadata Layer                   │
│  AscendCommonAttentionMetadata (扩展):              │
│    - chunk_ids: [-1, -1, ..., 0,0,0, ..., 1,1,1, ...]│
│    - chunk_boundaries: [(start1, end1), ...]       │
│    - positions: remapped for RoPE                   │
│    - chunk_attn_mask: computed by AttentionMaskBuilder│
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼─────────┐    ┌────────▼────────────┐
│ Rotary Embedding │    │ Attention Backend   │
│ vllm_ascend.ops  │    │ AscendSFABackend    │
│ .rotary_embedding│    │ (SFA v1)            │
│                  │    │                     │
│ - NPU-optimized  │    │ - SFA kernels       │
│   position remap │    │ - Chunked prefill   │
│ - MLA/GQA support│    │ - ACL graphs        │
└──────────────────┘    └──────────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼─────────────────────┐
         │   AttentionMaskBuilder           │
         │   vllm_ascend.attention.         │
         │   attention_mask                 │
         │                                  │
         │ - get_chunk_aware_mask()         │
         │ - Singleton pattern              │
         │ - NPU-optimized mask cache       │
         └──────────────────────────────────┘
                     │
         ┌───────────▼─────────────────────┐
         │   ACL Graph Compilation         │
         │   vllm_ascend.compilation       │
         │                                  │
         │ - NPU kernel optimization       │
         │ - Mask pattern encoding          │
         │ - Graph modes: FULL/PIECEWISE   │
         └──────────────────────────────────┘
```

### 3.2 核心设计原则

1. **插件化架构**: 基于vLLM-Ascend插件，不修改主vLLM代码
2. **Ascend优化**: 利用SFA (Sparse Flash Attention)和ACL图编译
3. **向后兼容**: 通过配置开关控制，默认禁用chunk-aware模式
4. **性能优先**:
   - 复用SFA的chunked prefill优化
   - 利用NPU内存分配器(camem_allocator)
   - Prefix cache复用sys_prompt的KV cache
5. **用户友好**: 自动检测chunk边界，无需手动指定

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
#   question:   [20,21,22,23,24,25,26,27,28]     # 从sys_prompt_end + max_chunk_len之后开始
```

#### 4.1.3 实现文件

**文件**: `vllm-ascend/vllm_ascend/ops/rotary_embedding.py` (扩展)

在vllm-ascend的现有rotary embedding实现中添加chunk-aware位置重映射功能：

```python
# 在 vllm_ascend/ops/rotary_embedding.py 中添加

def apply_chunk_aware_rope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    chunk_ids: torch.Tensor,
    sys_prompt_end: int,
    max_chunk_len: int,
    rotary_emb,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Apply rotary embeddings with chunk-aware position remapping for Ascend.

    集成到vllm_ascend.ops.rotary_embedding的现有实现中，
    利用NPU-optimized RoPE操作。

    Position remapping strategy:
    - sys_prompt tokens: keep original positions [0, sys_prompt_len)
    - chunk tokens: remap to [sys_prompt_len, sys_prompt_len + max_chunk_len)
      All chunks share this range!
    - question tokens: continue from [sys_prompt_len + max_chunk_len, ...)

    Args:
        positions: Original absolute positions [num_tokens]
        query: [num_tokens, num_heads, head_dim]
        key: [num_tokens, num_kv_heads, head_dim]
        chunk_ids: [-1=sys, 0,1,2,...=chunks, -2=question]
        sys_prompt_end: Length of system prompt (exclusive)
        max_chunk_len: Maximum chunk length
        rotary_emb: Existing rotary embedding instance from vllm_ascend

    Returns:
        Rotated query and key
    """
    if chunk_ids is None:
        # 使用vllm-ascend的标准RoPE流程
        return rotary_emb(positions, query, key)

    # Remap positions for chunk-aware mode
    remapped_positions = positions.clone()
    for i, chunk_id in enumerate(chunk_ids):
        if chunk_id >= 0:  # Chunk token
            # Compute position within this chunk
            pos_in_chunk = positions[i].item() - _get_chunk_start_pos(i, positions, chunk_ids)
            remapped_positions[i] = sys_prompt_end + pos_in_chunk
        elif chunk_id == -2:  # Question token
            # Position after all chunks (already correct if using absolute positions)
            pass

    # 使用vllm-ascend的NPU-optimized RoPE
    return rotary_emb(remapped_positions, query, key)
```

**集成点**:
- 扩展现有的`vllm_ascend.ops.rotary_embedding`模块
- 复用NPU-optimized的cos/sin缓存
- 支持MLA和GQA模型的RoPE变体

### 4.2 Chunk-Aware Attention Backend

#### 4.2.1 设计原理

**关键挑战**: 如何让chunk token关注sys_prompt和同chunk内的前序token，但不关注其他chunk？

**解决方案**: 扩展`AttentionMaskBuilder`实现chunk-isolated causal attention，集成到AscendSFABackend

**关键组件**:
- `AttentionMaskBuilder`: Singleton模式管理mask缓存
- `AscendSFABackend`: Sparse Flash Attention v1 backend
- `AscendCommonAttentionMetadata`: 扩展以包含chunk信息

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

#### 4.2.2 Attention Mask构建

**文件**: `vllm-ascend/vllm_ascend/attention/attention_mask.py` (扩展)

在现有的`AttentionMaskBuilder`类中添加`get_chunk_aware_mask()`方法：

```python
# File: vllm-ascend/vllm_ascend/attention/attention_mask.py

@singleton
class AttentionMaskBuilder:

    def __init__(self, device: torch.device):
        self.attn_mask_cache = None
        self._seq_len_cached = 0
        self.device = device
        self.mla_mask = None
        self.chunked_prefill_attn_mask = None
        self.pcp_mla_mask = None
        self.swa_mask = None
        self.chunk_aware_mask_cache = None  # NEW: cache for chunk-aware masks
        self._chunk_aware_cached = None

    # ... existing methods ...

    def get_chunk_aware_mask(
        self,
        num_tokens: int,
        max_seq_len: int,
        chunk_ids: torch.Tensor,
        sys_prompt_end: int,
        chunk_boundaries: list[tuple[int, int]],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build chunk-isolated causal attention mask for Ascend.

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
            dtype: torch.float16 or torch.bfloat16

        Returns:
            attention_mask: [num_tokens, max_seq_len] tensor
                             Mask values: 0 for can-attend, -inf for cannot
        """
        # Ascend-specific: mask values
        # For fp16: use -inf (float32.min)
        # For bf16/other: use 1 (binary mask)
        mask_value = torch.finfo(torch.float32).min if dtype == torch.float16 else 1.0

        mask = torch.zeros(num_tokens, max_seq_len, dtype=dtype, device=self.device)

        for query_pos in range(num_tokens):
            query_chunk_id = chunk_ids[query_pos].item()

            if query_chunk_id >= 0:  # Chunk token
                # Can attend to sys_prompt
                mask[query_pos, :sys_prompt_end] = 0

                # Can attend to earlier tokens in SAME chunk (causal)
                chunk_start, chunk_end = chunk_boundaries[query_chunk_id]
                causal_end = min(query_pos, chunk_end - 1) + 1
                mask[query_pos, chunk_start:causal_end] = 0

                # Cannot attend to other chunks or future positions
                # (already initialized to 0, need to mask out disallowed positions)
                disallow_start = causal_end
                if disallow_start < max_seq_len:
                    mask[query_pos, disallow_start:] = mask_value

            elif query_chunk_id == -2:  # Question token
                # Standard causal: can attend to everything before current position
                mask[query_pos, :query_pos] = 0
                mask[query_pos, query_pos:] = mask_value

            else:  # sys_prompt token (query_chunk_id == -1)
                # Standard causal attention within sys_prompt
                mask[query_pos, :query_pos] = 0
                mask[query_pos, query_pos:] = mask_value

        return mask
```

**关键特性**:
- **Singleton模式**: 复用mask实例，避免重复创建
- **NPU优化**: 使用NPU-compatible的mask值（fp16用-inf，bf16用1.0）
- **设备管理**: mask创建在正确的NPU device上
- **缓存机制**: 可扩展添加chunk-aware mask缓存

#### 4.2.3 Metadata扩展

**文件**: `vllm-ascend/vllm_ascend/attention/utils.py`

扩展现有的`AscendCommonAttentionMetadata`类：

```python
# File: vllm-ascend/vllm_ascend/attention/utils.py

@dataclass
class AscendCommonAttentionMetadata(CommonAttentionMetadata):
    """
    Per-batch attention metadata, shared across layers and backends.
    Extended with chunk-aware fields.
    """
    # ... existing fields ...
    seq_lens_cpu: torch.Tensor = None
    num_computed_tokens_cpu: torch.Tensor = None
    decode_token_per_req: int = 1
    actual_seq_lengths_q: list[int] = field(default_factory=list)
    positions: torch.Tensor = None
    attn_state: Any = None
    graph_pad_size: int = -1
    num_input_tokens: int = 0

    # NEW: Chunk-aware fields
    chunk_ids: torch.Tensor = None  # [num_tokens], -1=sys, 0,1,2..=chunks, -2=question
    chunk_boundaries: list[tuple[int, int]] = None  # [(start1, end1), ...]
    sys_prompt_end: int = 0  # Length of sys_prompt (exclusive)
    chunk_attn_mask: torch.Tensor = None  # [num_tokens, max_seq_len], computed by AttentionMaskBuilder
```

#### 4.2.4 Backend集成

**文件**: `vllm-ascend/vllm_ascend/attention/sfa_v1.py` (扩展)

在`AscendSFABackend`的forward方法中集成chunk-aware mask：

```python
# File: vllm-ascend/vllm_ascend/attention/sfa_v1.py

class AscendSFABackend:
    # ... existing implementation ...

    def forward(
        self,
        layer,
        query, key, value,
        kv_cache,
        attn_metadata: AscendCommonAttentionMetadata,
        output,
        **kwargs
    ):
        """
        Forward pass with chunk-aware attention support.
        """
        # Check if chunk-aware mask is needed
        if attn_metadata.chunk_ids is not None:
            from vllm_ascend.attention.attention_mask import AttentionMaskBuilder

            # Get or build chunk-aware mask
            if attn_metadata.chunk_attn_mask is None:
                mask_builder = AttentionMaskBuilder(query.device)
                attn_metadata.chunk_attn_mask = mask_builder.get_chunk_aware_mask(
                    num_tokens=query.shape[0],
                    max_seq_len=kv_cache.shape[1],
                    chunk_ids=attn_metadata.chunk_ids,
                    sys_prompt_end=attn_metadata.sys_prompt_end,
                    chunk_boundaries=attn_metadata.chunk_boundaries,
                    dtype=query.dtype,
                )

        # Apply chunk-aware mask in SFA computation
        # Integrate with existing SFA logic
        # ... (具体实现依赖于SFA的内部结构) ...

        return output
```

**集成要点**:
1. **懒加载**: 只在需要时构建chunk-aware mask
2. **复用metadata**: mask存储在`attn_metadata`中，避免重复构建
3. **SFA兼容**: mask应用需要适配SFA kernel的接口
4. **性能考虑**: 利用SFA的chunked prefill优化

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

### 阶段1: 核心组件实现 (vllm-ascend)

**Step 1.1**: 扩展AttentionMaskBuilder
- 文件: `vllm-ascend/vllm_ascend/attention/attention_mask.py`
- 实现: `get_chunk_aware_mask()` 方法
- 集成: 使用现有的singleton模式和缓存机制

**Step 1.2**: 扩展Attention Metadata
- 文件: `vllm-ascend/vllm_ascend/attention/utils.py`
- 实现: 扩展`AscendCommonAttentionMetadata`
- 添加字段: `chunk_ids`, `chunk_boundaries`, `sys_prompt_end`, `chunk_attn_mask`

**Step 1.3**: 集成到AscendSFABackend
- 文件: `vllm-ascend/vllm_ascend/attention/sfa_v1.py`
- 实现: 在`forward()`方法中添加chunk-aware mask逻辑
- 集成: 与现有的SFA kernel和ACL图编译流程配合

**Step 1.4**: Rotary Embedding扩展
- 文件: `vllm-ascend/vllm_ascend/ops/rotary_embedding.py`
- 实现: 添加`apply_chunk_aware_rope()`函数
- 集成: 复用现有的NPU-optimized RoPE实现

### 阶段2: 集成到vLLM-Ascend Worker

**Step 2.1**: Input处理扩展
- 文件: `vllm-ascend/vllm_ascend/worker/model_runner.py` (或对应文件)
- 添加: `_parse_chunks()` 方法解析chunk分隔符
- 添加: `_build_chunk_metadata()` 构建chunk_ids和boundaries

**Step 2.2**: Metadata传递
- 文件: `vllm-ascend/vllm_ascend/worker/model_runner.py`
- 修改: 将chunk信息传递到attention metadata
- 确保在prefill和decode阶段都正确传递

**Step 2.3**: 配置集成
- 文件: `vllm-ascend/vllm_ascend/ascend_config.py`
- 添加: `enable_chunk_aware`配置选项
- 添加: `chunk_separator`配置选项

### 阶段3: 优化与测试

**Step 3.1**: ACL图编译优化
- 文件: `vllm-ascend/vllm_ascend/compilation/acl_graph.py`
- 实现: 为chunk-aware模式优化图编译
- 考虑: Mask模式是否影响图复用

**Step 3.2**: 性能测试
- 文件: `vllm-ascend/tests/test_chunk_aware_attention.py` (新建)
- 测试: correctness (正确性), performance (性能), memory (内存)
- 对比: 与标准attention的对比基准

**Step 3.3**: 硬件测试
- 在Atlas 800I A2/A3上测试
- 验证ACL图编译正确性
- 性能profiling

### 阶段4: 文档与示例

**Step 4.1**: API文档
- 使用示例: 如何启用chunk-aware模式
- 配置说明: chunk_separator, max_chunk_len等参数

**Step 4.2**: 性能调优指南
- 如何设置chunk_len以获得最佳性能
- NPU内存优化建议
- ACL图编译模式选择 (FULL vs PIECEWISE)

---

## 6. 关键文件清单

### 新增文件 (vllm-ascend)

| 文件路径 | 用途 |
|---------|------|
| `vllm-ascend/tests/test_chunk_aware_attention.py` | 单元测试和集成测试 |

### 修改文件 (vllm-ascend)

| 文件路径 | 修改内容 |
|---------|---------|
| `vllm-ascend/vllm_ascend/attention/attention_mask.py` | 添加`get_chunk_aware_mask()`方法 |
| `vllm-ascend/vllm_ascend/attention/utils.py` | 扩展`AscendCommonAttentionMetadata`字段 |
| `vllm-ascend/vllm_ascend/attention/sfa_v1.py` | 集成chunk-aware mask到SFA backend |
| `vllm-ascend/vllm_ascend/ops/rotary_embedding.py` | 添加`apply_chunk_aware_rope()`函数 |
| `vllm-ascend/vllm_ascend/worker/model_runner.py` | 添加chunk解析和metadata构建 |
| `vllm-ascend/vllm_ascend/ascend_config.py` | 添加`enable_chunk_aware`等配置选项 |

### 相关文件 (主vLLM仓库 - 仅参考)

| 文件路径 | 说明 |
|---------|------|
| `vllm/v1/attention/backends/utils.py` | `CommonAttentionMetadata`基类 |
| `vllm/config.py` | `VllmConfig`配置系统 |

---

## 7. 性能考虑

### 7.1 内存开销

**额外内存**:
- `chunk_ids`: [num_tokens] × 4 bytes (int32)
- `chunk_boundaries`: Python list, 可忽略
- `chunk_attn_mask`: [num_tokens, max_seq_len] × 2 bytes (fp16) 或 × 4 bytes (bf16)

**优化**:
- **NPU内存管理**: 使用`camem_allocator.cpp`高效分配
- **Mask缓存**: 在`AttentionMaskBuilder`中缓存常用chunk模式的mask
- **Prefix cache复用**: sys_prompt的KV cache可被所有chunk复用

### 7.2 计算开销

**额外计算**:
- 位置重映射: O(num_tokens) - 轻量级CPU操作
- Mask构建: O(num_tokens × max_seq_len) - 可优化
- ACL图编译: 初次编译有开销，后续可复用

**优化**:
- **SFA优化**: 利用SFA的chunked prefill特性，减少mask计算次数
- **图编译缓存**: ACL图编译后缓存，避免重复编译
- **Lazy evaluation**: 只在需要时构建mask，避免不必要的计算

### 7.3 吞吐量影响

**预期影响**:
- **Prefill阶段**: +3-8% 延迟（取决于chunk数量和mask复杂度）
- **Decode阶段**: 几乎无影响（每个token只关注前面的chunk，mask简单）
- **批量处理**: 如果多个请求使用相同chunk结构，影响更小（图复用）

**NPU特定优化**:
- 利用SFA的sparse attention特性，跳过无效计算
- ACL图编译模式选择: FULL模式适合固定chunk结构，PIECEWISE模式适合动态结构
- 批量处理: 如果所有请求使用相同chunk结构，影响最小

---

## 8. 潜在风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| RoPE精度问题 | 位置编码不准确 | 充分测试不同chunk_len，使用NPU-optimized RoPE |
| SFA兼容性 | 性能下降 | 验证mask模式与SFA kernel的兼容性 |
| ACL图编译失败 | 无法运行 | 提供PIECEWISE或NONE模式作为fallback |
| NPU内存不足 | OOM | 优化mask大小，使用camem_allocator |
| HCCL通信问题 | 分布式场景故障 | 测试多NPU场景，验证chunk metadata传递 |

---

## 9. 未来扩展

### 9.1 Ascend特定扩展

1. **Context Parallel Chunk-Aware Attention**: 利用vllm-ascend的context parallel功能，支持超大文档chunk
2. **MLA优化**: 为Multi-Head Latent Attention优化chunk-aware模式，减少KV cache内存
3. **HCCL分布式处理**: 多NPU间chunk并行处理，提升吞吐量

### 9.2 通用扩展

1. **支持可变chunk长度**: 当前假设所有chunk使用共享位置范围，可扩展为不等长chunk
2. **支持chunk间attention**: 可选地允许某些chunk相互关注（如相关文档）
3. **动态chunk调整**: 运行时根据注意力模式合并/拆分chunk
4. **其他位置编码**: 支持ALiBi等其他编码方式

---

## 10. 总结

本设计通过以下核心技术实现**基于Ascend NPU**的多文档chunk受限注意力：

1. **vLLM-Ascend插件架构**: 无缝集成到Ascend平台，不修改主vLLM代码
2. **Sparse Flash Attention (SFA)**: 利用Ascend-优化的sparse attention backend
3. **扩展AttentionMaskBuilder**: NPU-optimized的chunk-aware mask，支持singleton和缓存
4. **AscendCommonAttentionMetadata扩展**: 添加chunk_ids、chunk_boundaries等字段
5. **ACL图编译**: NPU kernel级优化，支持FULL/PIECEWISE模式
6. **NPU内存管理**: 利用camem_allocator高效管理KV cache
7. **位置编码重映射**: 多个chunk共享位置范围，配合mask实现chunk-isolated causal attention

**关键优势**:
- **性能**: 利用SFA的chunked prefill优化
- **内存**: NPU-optimized的内存分配和mask缓存
- **兼容性**: 基于vLLM-Ascend插件，易于部署
- **可扩展**: 清晰的架构设计，便于未来扩展

这个设计在保持vLLM高性能和Ascend硬件优化的同时，实现了复杂的多文档处理场景，为RAG、长文档QA等应用提供了强大的技术支持。
