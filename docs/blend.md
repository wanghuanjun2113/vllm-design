# LMCache Cache Blend 特性分析

## 任务目标

分析 LMCache 中 Cache Blend 特性的主要流程和实现机制。

## Cache Blend 概述

CacheBlend 是 LMCache 的一个核心特性，允许在**非前缀位置**复用 KV 缓存。通过重新计算部分 token，它可以智能地组合多个预先计算的 KV cache。

### 核心价值

- **适用场景**: RAG（检索增强生成）、多文档问答、长上下文处理
- **性能提升**: 允许在不同顺序组合相同文档时复用 KV cache
- **效率优化**: 只重新计算少量关键 token（如 15%），而非全部重新计算

## 核心流程

### 1. 初始化流程

**入口点**: `LMCBlenderBuilder.get_or_create()`

**关键步骤**:
```
1. 创建 LMCBlender 实例
   ├─ 从 vLLM 推断 layerwise 模型
   ├─ 设置 check_layers（需要检查的层列表）
   ├─ 配置 recompute_ratios（重新计算比例）
   └─ 初始化元数据对象
```

**相关文件**:
- `LMCache/lmcache/v1/compute/blend/utils.py` - LMCBlenderBuilder
- `LMCache/lmcache/v1/compute/blend/blender.py` - LMCBlender.__init__()

### 2. Token 分割与存储流程

**特殊分隔符机制**:
```python
blend_special_str = " # # "  # 默认值

# 文档示例
sys_prompt + " # # " + chunk1 + " # # " + chunk2 + " # # " + chunk3
```

**Token 数据库处理**:
- 使用 `SegmentTokenDatabase`（而非 `ChunkedTokenDatabase`）
- 根据特殊分隔符将 token 序列分割成多个 segment
- 每个 segment 作为独立的缓存单元存储

**相关文件**:
- `LMCache/lmcache/v1/token_database/segment_token_database.py`
- `LMCache/lmcache/config.py` - blend_special_str 配置

### 3. 缓存检索与混合流程

**主要接口**: `LMCBlender.blend()` 和 `blend_layer()`

**详细流程**:

```
blend_layer(tokens, mask)
  │
  ├─ 启动 layerwise_model_executor（计算模型）
  │   └─ 逐层执行模型前向传播
  │
  ├─ 启动 cache_engine.retrieve_layer（检索缓存）
  │   └─ 逐层从存储中检索 KV cache
  │
  └─ 协调执行
      ├─ Layer 0: 检索 + 计算
      ├─ Layer 1-N: 检索 + 计算 + process_qkv 混合
      └─ 清理元数据
```

### 4. QKV 混合核心流程

**关键方法**: `LMCBlender.process_qkv()`

**详细步骤**:

```
process_qkv(q, k, v, residual, layer_id, ...)
  │
  ├─ 1. 获取旧的 KV cache
  │   old_k, old_v = gpu_connector.get_kv(layer_id)
  │
  ├─ 2. 位置编码
  │   rotary_emb(positions, q, k)
  │
  ├─ 3. 检查是否需要混合（仅在 check_layers）
  │   if layer_id in check_layers:
  │       │
  │       ├─ 3.1 计算新旧 K 的差异
  │       │   diff_k = sum((new_k - old_k)^2, dim=1)
  │       │
  │       ├─ 3.2 选择 top-k 重要 token
  │       │   topk_num = int(total_len * recompute_ratio)
  │       │   top_indices = torch.topk(diff_k, k=topk_num).indices
  │       │
  │       ├─ 3.3 仅保留这些 token 进行计算
  │       │   q, k, v = q[top_indices], k[top_indices], v[top_indices]
  │       │   residual = residual[top_indices]
  │       │
  │       └─ 3.4 更新元数据
  │           metadata.imp_indices = top_indices
  │           metadata.positions = positions[top_indices]
  │
  └─ 4. 替换旧 KV cache 中的对应位置
      old_k[imp_indices] = k
      old_v[imp_indices] = v
```

**关键代码位置**:
- `LMCache/lmcache/v1/compute/blend/blender.py:88-117`

### 5. 层间协同流程

```
for layer_id in range(num_layers):
  │
  ├─ retrieve_layer: 从存储获取 KV cache
  │   └─ 返回: (keys, memory_objs, starts, ends)
  │
  ├─ layerwise_model.compute_layer
  │   └─ 调用 process_qkv 进行混合
  │
  └─ storage_manager.batched_put
      └─ 将混合后的 KV 存回缓存
```

## 数据结构

### 元数据类

**LMCBlendCommonMetadata**（静态配置）:
```python
@dataclass
class LMCBlendCommonMetadata:
    check_layers: List[int]           # 检查层列表，如 [1]
    recomp_ratios: List[float]        # 重算比例，如 [0.15]
    thresholds: List[float]           # 阈值列表（未实现）
```

**LMCBlendMetadata**（运行时状态）:
```python
@dataclass
class LMCBlendMetadata:
    imp_indices: torch.Tensor         # 重要 token 的索引
    attn_mask: torch.Tensor           # 注意力掩码
    positions: torch.Tensor           # 位置编码
```

### 关键配置参数

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| `enable_blending` | `LMCACHE_ENABLE_BLENDING` | False | 启用混合功能 |
| `blend_special_str` | `LMCACHE_BLEND_SPECIAL_STR` | " # # " | 分隔符 |
| `blend_check_layers` | `LMCACHE_BLEND_CHECK_LAYERS` | None | 检查层列表 |
| `blend_recompute_ratios` | `LMCACHE_BLEND_RECOMPUTE_RATIOS` | None | 重算比例 |
| `blend_thresholds` | `LMCACHE_BLEND_THRESHOLDS` | None | 阈值（未实现） |
| `use_layerwise` | `LMCACHE_USE_LAYERWISE` | False | 必须启用 |

## 典型使用场景示例

### RAG 场景

```python
# 场景：3个文档 chunk，不同顺序组合

# Prompt 1: sys + chunk1 + chunk2 + chunk3 + question
prompt1 = sys_tokens + sep_tokens + chunk1_tokens + \
          sep_tokens + chunk2_tokens + \
          sep_tokens + chunk3_tokens + \
          sep_tokens + question_tokens

# Prompt 2: sys + chunk2 + chunk1 + chunk3 + question（不同顺序）
prompt2 = sys_tokens + sep_tokens + chunk2_tokens + \
          sep_tokens + chunk1_tokens + \
          sep_tokens + chunk3_tokens + \
          sep_tokens + question_tokens

# 两次请求可以复用 chunk1, chunk2, chunk3 的 KV cache
# 只在连接处重新计算约 15% 的 token
```

## 性能优化原理

### 为什么有效？

1. **局部性原理**: 大部分 token 在不同上下文中保持相似
2. **选择性重计算**: 只重新计算受位置编码影响最大的 token
3. **差异度量**: 使用新旧 K 的平方差识别重要 token

### 优化效果

- **计算开销**: 仅需重新计算 10-20% 的 token
- **缓存复用**: 非前缀位置仍可复用 80-90% 的 KV cache
- **适用性**: 特别适合 RAG、多文档合并等场景

## 关键文件路径总结

| 功能 | 文件路径 |
|------|----------|
| **核心混合逻辑** | `lmcache/v1/compute/blend/blender.py` |
| **元数据定义** | `lmcache/v1/compute/blend/metadata.py` |
| **构建器** | `lmcache/v1/compute/blend/utils.py` |
| **配置** | `lmcache/v1/config.py` |
| **Segment Token DB** | `lmcache/v1/token_database/segment_token_database.py` |
| **vLLM 集成** | `lmcache/integration/vllm/vllm_v1_adapter.py` |
| **示例代码** | `examples/blend_kv_v1/blend.py` |
| **文档** | `docs/source/kv_cache_optimizations/blending.rst` |

## 总结

Cache Blend 通过以下机制实现高效 KV cache 复用：

1. **Segment-based 存储**: 使用特殊分隔符分割独立可复用的缓存单元
2. **Layer-wise 处理**: 逐层检索和混合，支持细粒度控制
3. **智能重计算**: 在指定层使用差异度量选择需要重新计算的 token
4. **位置感知**: 通过 rotary encoding 处理位置信息变化

这使得 LMCache 能够在 RAG 等场景中显著提升性能，即使文档顺序发生变化也能复用大部分 KV cache。
