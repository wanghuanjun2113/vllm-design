# Attention Mask 生成与传递机制设计文档

## 1. 概述

本文档详细分析了在昇腾 (Ascend) 硬件下，Attention Mask 是如何生成并传递给算子的，以及如果要定制化生成 Attention Mask 需要修改哪些文件。

### 1.1 核心发现

**vLLM 和 vLLM-Ascend 在 Attention Mask 处理上的关键差异**：

| 特性 | vLLM (CUDA) | vLLM-Ascend (NPU) |
|------|-------------|-------------------|
| **Mask 方式** | 隐式 (通过 seq_lens) | 显式 (mask tensor) |
| **Mask 传递** | causal flag + seq_lens | attn_mask tensor |
| **Mask 生成** | 不需要显式生成 | AttentionMaskBuilder |
| **Caching** | 无缓存 | Mask 缓存机制 |

---

## 2. vLLM 主干的 Attention Mask 处理

### 2.1 核心设计理念

vLLM 使用**隐式 Mask** 方式，**不显式创建 Attention Mask 张量**。

```
vLLM 方式: 隐式 Mask
┌─────────────────────────────────────────────────────────────────┐
│  不创建显式 mask 张量，而是通过序列长度信息隐式表示             │
│                                                                 │
│  seq_lens: [10, 20, 15]      → 每个序列的有效长度               │
│  query_start_loc: [0, 10, 30, 45]  → 每个序列在 batch 中的位置   │
│  causal: True                  → 启用因果 attention              │
│                                                                 │
│  Attention Kernel 使用这些信息隐式应用 mask                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键文件

| 文件 | 功能 |
|------|------|
| `vllm/v1/attention/backends/utils.py` | CommonAttentionMetadata 定义 |
| `vllm/v1/attention/backends/flash_attn.py` | FlashAttention 实现 |
| `vllm/v1/worker/gpu_model_runner.py` | 构建注意力元数据 |

### 2.3 CommonAttentionMetadata 结构

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor        # (batch_size + 1,)
    query_start_loc_cpu: torch.Tensor     # CPU 版本
    seq_lens: torch.Tensor               # (batch_size,) 序列长度
    num_reqs: int                        # 请求数量
    num_actual_tokens: int               # 实际 token 数
    max_query_len: int                   # 最大查询长度
    max_seq_len: int                     # 最大序列长度
    block_table_tensor: torch.Tensor     # PagedAttention 块表
    slot_mapping: torch.Tensor           # KV cache slot 映射
    causal: bool = True                  # 是否因果 masking
```

**没有 attn_mask 字段！**

### 2.4 FlashAttention 如何使用隐式 Mask

```python
# vllm/v1/attention/backends/flash_attn.py
flash_attn_varlen_func(
    q=query,
    k=key_cache,
    v=value_cache,
    cu_seqlens_q=attn_metadata.query_start_loc,  # 累积序列长度
    seqused_k=attn_metadata.seq_lens,           # 有效 key 长度
    causal=attn_metadata.causal,                 # 因果 flag
    ...
)
```

**关键点**：
- `cu_seqlens_q`: 告诉 kernel 每个 query 序列从哪里开始
- `seqused_k`: 告诉 kernel 每个 key 序列有多长
- `causal=True`: kernel 内部自动应用因果 mask

---

## 3. vLLM-Ascend 的 Attention Mask 处理

### 3.1 核心设计理念

vLLM-Ascend 使用**显式 Mask** 方式，**创建并传递 Mask 张量**给 NPU 算子。

```
vLLM-Ascend 方式: 显式 Mask
┌─────────────────────────────────────────────────────────────────┐
│  创建显式 mask 张量传递给 NPU 算子                               │
│                                                                 │
│  AttentionMaskBuilder.get_attn_mask() → 生成 mask 张量          │
│                                                                 │
│  attn_mask: (max_seq_len, max_seq_len)                         │
│    [[0, -inf, -inf, ...],                                      │
│     [0, 0, -inf, ...],                                         │
│     [0, 0, 0, ...],                                            │
│     ...]                                                        │
│                                                                 │
│  torch_npu.npu_fused_infer_attention_score(                     │
│      ..., atten_mask=attn_mask, ...                            │
│  )                                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 关键文件

| 文件 | 功能 |
|------|------|
| `vllm-ascend/vllm_ascend/attention/attention_mask.py` | Mask 生成核心类 |
| `vllm-ascend/vllm_ascend/attention/attention_v1.py` | Ascend Attention Backend |
| `vllm-ascend/vllm_ascend/attention/utils.py` | Ascend 通用元数据 |
| `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` | NPU Model Runner |

### 3.3 AttentionMaskBuilder 类

**位置**: `vllm-ascend/vllm_ascend/attention/attention_mask.py`

```python
@singleton
class AttentionMaskBuilder:
    """单例模式，全局共享，缓存 mask 以避免重复生成"""

    def __init__(self, device: torch.device):
        self.attn_mask_cache = None          # 因果 mask 缓存
        self._seq_len_cached = 0             # 缓存的序列长度
        self.device = device
        self.mla_mask = None                 # MLA mask 缓存
        self.chunked_prefill_attn_mask = None
        self.pcp_mla_mask = None
        self.swa_mask = None                 # Sliding Window mask 缓存

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype):
        """获取因果 attention mask"""
        if self.attn_mask_cache is None or max_seq_len > self._seq_len_cached:
            self.attn_mask_cache = _generate_attn_mask(max_seq_len, dtype)
            self._seq_len_cached = max_seq_len
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous()

    def get_splitfuse_attn_mask(self) -> torch.Tensor:
        """获取 chunked prefill mask (2048x2048 上三角)"""
        if self.chunked_prefill_attn_mask is None:
            self.chunked_prefill_attn_mask = torch.triu(
                torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(self.device)
        return self.chunked_prefill_attn_mask

    def get_mla_mask(self, dtype: torch.dtype) -> torch.Tensor:
        """获取 MLA (Multi-head Latent Attention) mask (512x512)"""
        if self.mla_mask is None or self.mla_mask.dtype != dtype:
            mask_value = torch.finfo(torch.float32).min if dtype == torch.float16 else 1
            prefill_mask = torch.triu(torch.ones(512, 512, device=self.device, dtype=dtype), 1)
            self.mla_mask = torch.where(prefill_mask == 1, mask_value, 0).to(dtype)
        return self.mla_mask

    def get_swa_mask(self, dtype: torch.dtype, sliding_window):
        """获取 Sliding Window Attention mask"""
        if self.swa_mask is None or self.swa_mask.dtype != dtype:
            if sliding_window is not None:
                mask = torch.ones(2048, 2048, dtype=torch.bool)
                triu_mask = torch.triu(mask, diagonal=1).to(self.device)
                tril_mask = torch.tril(mask, -sliding_window).to(self.device)
                self.swa_mask = triu_mask + tril_mask
        return self.swa_mask
```

### 3.4 Mask 生成函数

```python
def _generate_attn_mask(max_seq_len, dtype):
    """生成因果 attention mask (下三角矩阵)"""
    # 构建下三角矩阵
    mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
    # 创建上三角矩阵用于标记 mask 位置
    mask_flag = ~mask_flag
    # fp16 dtype 使用 -inf，其他使用 1
    mask_value = float('-inf') if dtype == torch.float16 else 1
    attn_mask = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype) \
        .masked_fill_(mask_flag, mask_value)
    return attn_mask
```

**生成的 Mask 格式**:
```
max_seq_len = 4:
[[0,    -inf, -inf, -inf],
 [0,     0,    -inf, -inf],
 [0,     0,     0,    -inf],
 [0,     0,     0,     0   ]]
```

### 3.5 AscendMetadata 结构

**位置**: `vllm-ascend/vllm_ascend/attention/attention_v1.py`

```python
@dataclass
class AscendMetadata:
    # Mask 相关字段
    attn_mask: Optional[torch.Tensor] = None      # 主 attention mask
    swa_mask: Optional[torch.Tensor] = None        # Sliding Window mask

    # 序列信息
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0
    seq_lens: torch.Tensor = None
    seq_lens_list: List[int] = None
    actual_seq_lengths_q: List[int] = None
    query_start_loc: torch.Tensor = None
    max_query_len: Optional[int] = None

    # KV Cache
    block_tables: torch.Tensor = None
    slot_mapping: torch.Tensor = None

    # 状态
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    causal: bool = True
    model_runner_type: str = ""
```

### 3.6 Mask 传递流程

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. AscendAttentionMetadataBuilder.build()                        │
│    ┌─────────────────────────────────────────────────────────┐   │
│    │ attn_mask = self.attn_mask_builder.get_attention_mask(  │   │
│    │     self.model_config)                                  │   │
│    │                                                          │   │
│    │ if sliding_window:                                      │   │
│    │     swa_mask = self.attn_mask_builder.get_swa_mask(...) │   │
│    └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│ 2. 创建 AscendMetadata(attn_mask=attn_mask, swa_mask=swa_mask)    │
│                              ↓                                   │
│ 3. AscendAttentionBackendImpl.forward()                          │
│    ┌─────────────────────────────────────────────────────────┐   │
│    │ torch_npu.npu_fused_infer_attention_score(             │   │
│    │     query=query,                                        │   │
│    │     key=key,                                            │   │
│    │     value=value,                                        │   │
│    │     atten_mask=attn_metadata.attn_mask,  ← 显式传递    │   │
│    │     ...                                                 │   │
│    │ )                                                       │   │
│    └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Mask 生成与传递的完整流程

### 4.1 流程图

```
用户请求
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ InputProcessor (vllm/v1/engine/input_processor.py)               │
│  - Tokenize 提示词                                               │
│  - 验证采样参数                                                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Scheduler (vllm/v1/core/sched/scheduler.py)                      │
│  - 调度请求                                                      │
│  - 分配 KV cache 块                                             │
│  - 形成 batch                                                    │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ GPUModelRunner / NPUModelRunner                                 │
│  - 构建注意力元数据                                              │
│  - 准备输入批次                                                 │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ AscendAttentionMetadataBuilder.build()                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ AttentionMaskBuilder.get_attention_mask()                  │ │
│  │   - 检查缓存                                                 │ │
│  │   - 生成或复用 mask 张量                                     │ │
│  │   - 返回 (max_seq_len, max_seq_len) 的 mask                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  AscendMetadata(                                                 │
│      attn_mask=attn_mask,                                       │
│      swa_mask=swa_mask,  # 如果启用 sliding window              │
│      seq_lens=seq_lens,                                         │
│      block_tables=block_tables,                                 │
│      ...                                                         │
│  )                                                               │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ AscendAttentionBackendImpl.forward()                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ torch_npu.npu_fused_infer_attention_score(                 │ │
│  │     query=query,                                           │ │
│  │     key=key,                                               │ │
│  │     value=value,                                           │ │
│  │     atten_mask=attn_metadata.attn_mask,  ← 显式传递 mask  │ │
│  │     block_table=attn_metadata.block_tables,                │ │
│  │     actual_seq_lengths=attn_metadata.actual_seq_lengths_q, │ │
│  │     sparse_mode=3,  # 因果 attention                         │ │
│  │     ...                                                    │ │
│  │ )                                                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
    ↓
Attention Output
```

### 4.2 关键调用链

```python
# 1. Mask 生成入口 (attention_v1.py:264-273)
attn_mask = self.attn_mask_builder.get_attention_mask(self.model_config)

if is_swa:
    swa_mask = self.attn_mask_builder.get_swa_mask(
        self.model_config.dtype,
        self.model_config.hf_text_config.sliding_window
    )

# 2. Metadata 构建 (attention_v1.py:279-296)
attn_metadata = AscendMetadata(
    attn_mask=attn_mask,
    swa_mask=swa_mask,
    ...
)

# 3. Forward 传递 (attention_v1.py:596-610)
attn_output, _ = torch_npu.npu_fused_infer_attention_score(
    query=query,
    key=key,
    value=value,
    atten_mask=attn_metadata.attn_mask,  # ← 关键：显式传递 mask
    block_table=block_table,
    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
    sparse_mode=3,  # 因果 attention
    ...
)
```

---

## 5. NPU 算子接口

### 5.1 torch_npu.npu_fused_infer_attention_score

**函数签名**:
```python
torch_npu.npu_fused_infer_attention_score(
    query: torch.Tensor,           # [num_tokens, num_heads, head_size]
    key: torch.Tensor,              # [num_tokens, num_kv_heads, head_size]
    value: torch.Tensor,            # [num_tokens, num_kv_heads, head_size]
    atten_mask: torch.Tensor,       # [max_seq_len, max_seq_len] 或 [bs, seq, seq]
    block_table: torch.Tensor,      # PagedAttention 块表
    input_layout: str,              # "TND", "BSH", etc.
    block_size: int,                # 块大小 (128)
    actual_seq_lengths: List[int],  # 实际序列长度
    actual_seq_lengths_kv: List[int],
    num_key_value_heads: int,
    num_heads: int,
    scale: float,
    sparse_mode: int,               # 0=None, 1=Left, 2=Right, 3=Causal
    pre_tokens: int = None,         # Sliding window: pre tokens
    next_tokens: int = None,        # Sliding window: next tokens
    ...
)
```

### 5.2 sparse_mode 参数

| sparse_mode | 含义 | 用途 |
|-------------|------|------|
| 0 | 无 mask | Encoder / Bidirectional |
| 1 | 左侧 mask | Left Causal |
| 2 | 右侧 mask | Right Causal |
| 3 | 下三角 mask | 标准因果 attention (LLM) |

### 5.3 Sliding Window 专用调用

```python
# attention_v1.py:553-565
output, _ = torch_npu.npu_fused_infer_attention_score(
    query,
    key,
    value,
    num_heads=self.num_heads,
    num_key_value_heads=self.num_kv_heads,
    input_layout="BSH",
    block_size=block_size,
    pre_tokens=self.sliding_window,  # 滑动窗口：向前看的 token 数
    scale=self.scale,
    block_table=attn_metadata.block_tables,
    actual_seq_lengths=[1] * len(attn_metadata.seq_lens),
    actual_seq_lengths_kv=attn_metadata.seq_lens
)
```

---

## 6. 定制化 Attention Mask 需要修改的文件

### 6.1 核心修改文件

如果要添加**自定义 Mask 类型**（如特定模式的 mask），需要修改以下文件：

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `vllm-ascend/vllm_ascend/attention/attention_mask.py` | 添加新的 mask 生成方法 | **高** |
| `vllm-ascend/vllm_ascend/attention/attention_v1.py` | 在 MetadataBuilder 中调用新 mask | **高** |
| `vllm-ascend/vllm_ascend/attention/attention_v1.py` | 在 AscendMetadata 中添加字段 | **高** |
| `vllm-ascend/vllm_ascend/attention/attention_v1.py` | 在 forward 中传递新 mask | **高** |

### 6.2 详细修改指南

#### 修改 1: attention_mask.py - 添加新的 mask 生成方法

```python
# vllm-ascend/vllm_ascend/attention/attention_mask.py

@singleton
class AttentionMaskBuilder:
    # ... 现有代码 ...

    def get_custom_mask(self, dtype: torch.dtype, custom_param: int):
        """添加自定义 mask 生成方法"""
        if self.custom_mask is None or self.custom_mask.dtype != dtype:
            # 实现自定义 mask 生成逻辑
            # 例如：特定稀疏模式、自定义窗口等
            self.custom_mask = self._generate_custom_mask(
                dtype, custom_param
            )
        return self.custom_mask

    def _generate_custom_mask(self, dtype, custom_param):
        """实现自定义 mask 生成"""
        # 示例：创建带特定稀疏模式的 mask
        max_seq_len = 2048
        mask = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)
        # 实现自定义逻辑
        # ...
        mask_value = float('-inf') if dtype == torch.float16 else 1
        attn_mask = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype) \
            .masked_fill_(mask, mask_value)
        return attn_mask
```

#### 修改 2: attention_v1.py - 扩展 AscendMetadata

```python
# vllm-ascend/vllm_ascend/attention/attention_v1.py

@dataclass
class AscendMetadata:
    # ... 现有字段 ...

    # 添加自定义 mask 字段
    custom_mask: Optional[torch.Tensor] = None
```

#### 修改 3: attention_v1.py - 在 MetadataBuilder 中生成自定义 mask

```python
# vllm-ascend/vllm_ascend/attention/attention_v1.py
# AscendAttentionMetadataBuilder.build() 方法

def build(
    self,
    common_prefix_len: int,
    common_attn_metadata: AscendCommonAttentionMetadata,
    fast_build: bool = False,
) -> AscendMetadata:
    # ... 现有代码 ...

    # 获取标准 mask
    attn_mask = self.attn_mask_builder.get_attention_mask(
        self.model_config
    )

    # 获取自定义 mask（如果启用）
    custom_mask = None
    if hasattr(self.model_config, 'custom_mask_enabled') and \
       self.model_config.custom_mask_enabled:
        custom_mask = self.attn_mask_builder.get_custom_mask(
            self.model_config.dtype,
            self.model_config.custom_param
        )

    # 创建 metadata
    attn_metadata = AscendMetadata(
        attn_mask=attn_mask,
        custom_mask=custom_mask,  # 添加自定义 mask
        # ... 其他字段 ...
    )
    return attn_metadata
```

#### 修改 4: attention_v1.py - 在 forward 中使用自定义 mask

```python
# vllm-ascend/vllm_ascend/attention/attention_v1.py
# AscendAttentionBackendImpl.forward_fused_infer_attention() 方法

def forward_fused_infer_attention(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AscendMetadata,
    output: torch.Tensor,
):
    # ... 现有代码 ...

    # 选择使用哪个 mask
    mask_to_use = attn_metadata.attn_mask
    if attn_metadata.custom_mask is not None:
        mask_to_use = attn_metadata.custom_mask

    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        atten_mask=mask_to_use,  # 使用选择的 mask
        # ... 其他参数 ...
    )

    return attn_output
```

### 6.3 配置扩展

如果需要通过配置控制自定义 mask，需要修改：

```python
# vllm/vllm/config.py 或 vllm-ascend 中的配置文件

@dataclass
class ModelConfig:
    # ... 现有字段 ...

    # 自定义 mask 配置
    custom_mask_enabled: bool = False
    custom_mask_type: str = "default"  # "sparse", "window", "custom"
    custom_mask_param: int = 128
```

### 6.4 测试文件

添加测试以验证自定义 mask 的正确性：

```python
# vllm-ascend/tests/test_custom_attention_mask.py

import torch
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder

def test_custom_mask_generation():
    device = torch.device("npu:0")
    builder = AttentionMaskBuilder(device)

    # 测试自定义 mask 生成
    custom_mask = builder.get_custom_mask(torch.float16, 64)

    # 验证 mask 形状
    assert custom_mask.shape[0] == 2048
    assert custom_mask.shape[1] == 2048

    # 验证 mask 值
    assert custom_mask.dtype == torch.float16

    print("Custom mask test passed!")
```

---

## 7. 不同场景下的 Mask 处理

### 7.1 标准因果 Attention

```python
# attention_mask.py:48-56
def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype):
    """下三角矩阵 mask"""
    if self.attn_mask_cache is None or max_seq_len > self._seq_len_cached:
        self.attn_mask_cache = _generate_attn_mask(max_seq_len, dtype)
    return self.attn_mask_cache[:max_seq_len, :max_seq_len]
```

**用途**: LLM 自回归生成

### 7.2 Sliding Window Attention

```python
# attention_mask.py:83-90
def get_swa_mask(self, dtype: torch.dtype, sliding_window):
    """滑动窗口 mask"""
    if self.swa_mask is None:
        mask = torch.ones(2048, 2048, dtype=torch.bool)
        triu_mask = torch.triu(mask, diagonal=1)  # 上三角（因果）
        tril_mask = torch.tril(mask, -sliding_window)  # 下三角（窗口）
        self.swa_mask = triu_mask + tril_mask
    return self.swa_mask
```

**用途**: 长序列模型（如 Mistral）

### 7.3 Multi-head Latent Attention (MLA)

```python
# attention_mask.py:65-75
def get_mla_mask(self, dtype: torch.dtype) -> torch.Tensor:
    """MLA 专用 mask (512x512）"""
    if self.mla_mask is None or self.mla_mask.dtype != dtype:
        mask_value = torch.finfo(torch.float32).min if dtype == torch.float16 else 1
        prefill_mask = torch.triu(torch.ones(512, 512, device=self.device, dtype=dtype), 1)
        self.mla_mask = torch.where(prefill_mask == 1, mask_value, 0).to(dtype)
    return self.mla_mask
```

**用途**: DeepSeek-V3 等 MLA 架构模型

### 7.4 Chunked Prefill

```python
# attention_mask.py:58-63
def get_splitfuse_attn_mask(self) -> torch.Tensor:
    """Chunked prefill mask (2048x2048 上三角)"""
    if self.chunked_prefill_attn_mask is None:
        self.chunked_prefill_attn_mask = torch.triu(
            torch.ones(2048, 2048), diagonal=1
        ).to(torch.int8).to(self.device)
    return self.chunked_prefill_attn_mask
```

**用途**: 分块预填充优化

---

## 8. 性能优化考虑

### 8.1 Mask 缓存机制

```python
@singleton
class AttentionMaskBuilder:
    """单例模式确保全局共享缓存"""
```

**优势**:
- 避免重复生成相同大小的 mask
- 减少 H2D 数据传输
- 降低内存分配开销

### 8.2 不同数据类型的 Mask 值

| dtype | mask 值 | 原因 |
|-------|---------|------|
| float16 | `-inf` | 与 softmax 配合，被 mask 的位置概率为 0 |
| 其他 (bool/int8) | `1` | 标记 mask 位置 |

```python
# attention_mask.py:29-31
mask_value = float('-inf') if dtype == torch.float16 else 1
```

### 8.3 ACL Graph 编译模式下的 Mask 处理

在 ACL Graph 模式下，mask 作为图的输入参数：

```python
# attention_v1.py:378-414 (full_graph_fia 方法)
workspace = graph_params.workspaces.get(num_tokens)

# Mask 作为参数保存
graph_params.attn_params[num_tokens].append((
    weak_ref_tensors(query),
    weak_ref_tensors(key),
    weak_ref_tensors(value),
    weak_ref_tensors(attn_metadata.attn_mask),  # ← Mask 作为图参数
    ...
))
```

---

## 9. 关键差异总结

### 9.1 vLLM vs vLLM-Ascend Mask 处理对比

| 方面 | vLLM (CUDA) | vLLM-Ascend (NPU) |
|------|-------------|-------------------|
| **Mask 类型** | 隐式 | 显式 |
| **Mask 张量** | 不创建 | 创建并缓存 |
| **传递方式** | `causal=True` flag | `atten_mask` tensor |
| **Kernel 接口** | `cu_seqlens`, `seqused_k` | `atten_mask`, `actual_seq_lengths` |
| **内存占用** | 低（无 mask tensor） | 较高（需要存储 mask） |
| **灵活性** | 低（依赖 kernel 实现） | 高（可自定义任何 mask） |

### 9.2 为什么 vLLM-Ascend 使用显式 Mask？

**原因 1: NPU Kernel 接口要求**
- `torch_npu.npu_fused_infer_attention_score` 需要 `atten_mask` 参数
- 昇腾 NPU 算子设计期望显式提供 mask

**原因 2: 更灵活的 Mask 支持**
- 支持 Sliding Window Attention
- 支持 MLA（Multi-head Latent Attention）
- 支持自定义稀疏模式
- 支持非标准 attention pattern

**原因 3: 性能优化**
- Mask 可以预先计算并缓存
- 减少运行时计算开销
- 支持图编译优化

---

## 10. 总结

### 10.1 核心要点

1. **vLLM 主干**：使用隐式 mask，通过 `seq_lens` 和 `causal flag` 实现
2. **vLLM-Ascend**：使用显式 mask，创建并传递 mask tensor
3. **Mask 生成**：`AttentionMaskBuilder` 单例类负责生成和缓存 mask
4. **Mask 传递**：通过 `AscendMetadata.attn_mask` 传递给 NPU 算子
5. **NPU 算子**：`torch_npu.npu_fused_infer_attention_score` 接收 mask 参数

### 10.2 定制化 Mask 的关键文件

| 文件 | 作用 |
|------|------|
| `vllm-ascend/vllm_ascend/attention/attention_mask.py` | 实现 `AttentionMaskBuilder`，添加新 mask 生成方法 |
| `vllm-ascend/vllm_ascend/attention/attention_v1.py` | 扩展 `AscendMetadata`，在 `build()` 和 `forward()` 中使用新 mask |
| `vllm/config.py` | 添加配置项控制自定义 mask |
| `vllm-ascend/tests/` | 添加测试验证自定义 mask |

### 10.3 实现步骤

如果要添加自定义 Attention Mask：

1. **扩展 AttentionMaskBuilder** (`attention_mask.py`)
   - 添加 `get_custom_mask()` 方法
   - 实现自定义 mask 生成逻辑
   - 添加缓存机制

2. **扩展 AscendMetadata** (`attention_v1.py`)
   - 添加 `custom_mask` 字段

3. **修改 MetadataBuilder** (`attention_v1.py`)
   - 在 `build()` 方法中调用 `get_custom_mask()`
   - 根据配置决定是否使用自定义 mask

4. **修改 AttentionImpl** (`attention_v1.py`)
   - 在 `forward()` 中使用 `custom_mask`

5. **添加配置支持**
   - 在 ModelConfig 中添加相关配置项

6. **添加测试**
   - 编写测试验证自定义 mask 的正确性
