# vLLM Blend 插件架构设计文档

## 文档信息

- **版本**: v1.0
- **日期**: 2025-01-13
- **状态**: 架构设计完成，Phase 1 实现完成
- **作者**: vLLM Blend Team

## 目录

1. [系统概述](#1-系统概述)
2. [架构设计](#2-架构设计)
3. [核心组件](#3-核心组件)
4. [接口规范](#4-接口规范)
5. [数据流](#5-数据流)
6. [集成方式](#6-集成方式)
7. [扩展机制](#7-扩展机制)
8. [设计决策](#8-设计决策)
9. [性能考虑](#9-性能考虑)
10. [安全考虑](#10-安全考虑)

---

## 1. 系统概述

### 1.1 设计目标

vLLM Blend 插件的设计目标是实现一个**完全解耦、硬件无关的 KV 缓存混合系统**，主要目标：

1. **解耦性**: 完全独立于 LMCache，无任何依赖
2. **硬件无关**: 支持 CUDA、Ascend NPU、ROCm 等多种平台
3. **可扩展性**: 易于添加新的模型支持和硬件后端
4. **性能优先**: 最小化开销，最大化 KV 复用
5. **向后兼容**: 不修改 vLLM 核心代码

### 1.2 核心价值

Blend 解决的核心问题是：**如何在非前缀位置复用 KV cache**

**场景示例**：
```
请求 1: [系统提示] + [文档A] + [文档B] + [问题1]
请求 2: [系统提示] + [文档B] + [文档A] + [问题2]

传统方式: 两次请求完全独立，0% KV 复用
Blend方式: 复用系统提示、文档A、文档B的 KV，只重算连接处 ~15%
```

### 1.3 系统边界

```
┌─────────────────────────────────────────────────────────────┐
│                     vLLM Engine                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Blend Plugin (本系统)                     │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐      │ │
│  │  │ Core Logic │  │ Providers  │  │ Adapters   │      │ │
│  │  └────────────┘  └────────────┘  └────────────┘      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  外部系统（由 vLLM 提供）：                                    │
│  - vLLM Model Executor                                       │
│  - KV Cache Manager                                         │
│  - Attention Backend                                        │
│  - Worker/Scheduler                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 架构设计

### 2.1 分层架构

Blend 采用**四层架构**，每层有明确的职责：

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 4: Adapters                        │
│  (模型适配层 - Llama, Qwen, Mistral 等)                      │
├─────────────────────────────────────────────────────────────┤
│                    Layer 3: Providers                        │
│  (硬件/缓存抽象层 - Cache, GPU, Model)                        │
├─────────────────────────────────────────────────────────────┤
│                    Layer 2: Core Logic                       │
│  (核心混合逻辑 - Blender, Selector, Metadata)                │
├─────────────────────────────────────────────────────────────┤
│                    Layer 1: Config                           │
│  (配置层 - BlendConfig, 参数验证)                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 组件关系图

```
                         ┌──────────────────┐
                         │   BlendConfig    │
                         │  (配置管理)       │
                         └────────┬─────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                         BlendWorker                         │
│  (vLLM Worker 扩展，拦截模型执行)                             │
└────────────────────────────────────┬────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────┐
│                        BlendBlender                          │
│  (核心混合协调器)                                               │
│  ┌──────────────┬──────────────┬──────────────────────┐     │
│  │ TokenSelector│  Metadata    │   Provider Factory   │     │
│  │  (Token选择) │  (元数据)     │   (Provider工厂)      │     │
│  └──────────────┴──────────────┴──────────────────────┘     │
└──────────┬───────────────────────────────────────────────────┘
           │
           ├─────────────────┬─────────────────┬──────────────┐
           ▼                 ▼                 ▼              ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐  ┌─────────┐
    │   Cache   │    │    GPU    │    │   Model   │  │Adapter  │
    │ Provider  │    │ Provider  │    │ Provider  │  │Registry │
    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘  └─────────┘
          │                │                │
          ▼                ▼                ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │  LMCache  │    │  CUDA/    │    │  Llama/   │
    │  CPU/     │    │  NPU/     │    │  Qwen/    │
    │  Remote   │    │  ROCm     │    │  Mistral  │
    └───────────┘    └───────────┘    └───────────┘
```

### 2.3 目录结构

```
vllm-blend/
│
├── vllm_blend/                    # 主包
│   ├── __init__.py                # 插件注册入口
│   ├── config.py                  # Layer 1: 配置系统
│   │
│   ├── core/                      # Layer 2: 核心逻辑
│   │   ├── blender.py             # 核心混合协调器
│   │   ├── metadata.py            # 元数据定义
│   │   └── selector.py            # Token 选择算法
│   │
│   ├── providers/                 # Layer 3: Provider 抽象
│   │   ├── cache_provider.py      # 缓存接口
│   │   ├── gpu_provider.py        # GPU 接口
│   │   └── model_provider.py      # 模型接口
│   │
│   ├── backends/                  # Provider 实现
│   │   ├── cuda/                  # CUDA 后端
│   │   │   ├── gpu_provider.py
│   │   │   ├── cache_provider.py
│   │   │   └── model_provider.py
│   │   ├── npu/                   # NPU 后端
│   │   └── rocm/                  # ROCm 后端
│   │
│   ├── adapters/                  # Layer 4: 模型适配
│   │   ├── base.py                # 适配器基类
│   │   ├── llama.py               # Llama 适配器
│   │   ├── qwen2.py               # Qwen2 适配器
│   │   ├── qwen3.py               # Qwen3 适配器
│   │   └── registry.py            # 适配器注册表
│   │
│   ├── worker/                    # Worker 集成
│   │   ├── blend_worker.py        # Blend Worker
│   │   └── model_runner.py        # Model Runner Mixin
│   │
│   └── utils/                     # 工具函数
│       ├── rope.py                # RoPE 工具
│       └── diagnostics.py         # 诊断工具
│
├── tests/                         # 测试
├── examples/                      # 示例
└── setup.py                       # 安装配置
```

---

## 3. 核心组件

### 3.1 BlendBlender

**职责**: 协调所有 Provider，执行 KV 混合的核心逻辑

**关键方法**:

```python
class BlendBlender:
    def process_qkv(
        self,
        q: torch.Tensor,      # [seq_len, num_heads, head_dim]
        k: torch.Tensor,      # [seq_len, num_kv_heads, head_dim]
        v: torch.Tensor,      # [seq_len, num_kv_heads, head_dim]
        residual: torch.Tensor,  # [seq_len, hidden_dim]
        layer_id: int,
    ) -> Tuple[torch.Tensor, ...]:
        """
        核心混合逻辑：

        1. 从 CacheProvider 获取缓存的 KV
        2. 从 GPUProvider 获取 GPU 中的 KV
        3. 应用 Rotary Embedding
        4. 如果是检查层：
           a. 计算新旧 K 的差异
           b. 选择 top-k 重要 tokens
           c. 更新 GPU cache
           d. 返回选中的 tokens
        5. 否则直接返回

        返回: (q, k, v, residual) - 可能是 subset
        """
```

**执行流程**:

```
process_qkv() 调用
    │
    ├─► cache_provider.retrieve_layer()
    │   └─► 返回 (cached_k, cached_v) 或 None
    │
    ├─► gpu_provider.get_kv(layer_id)
    │   └─► 返回 (gpu_k, gpu_v)
    │
    ├─► model_provider.apply_rotary_emb(q, k, positions, layer_id)
    │   └─► 返回 (q_rot, k_rot)
    │
    ├─► if layer_id in check_layers:
    │   │
    │   ├─► selector.select_important_tokens(new_k, old_k, ratio)
    │   │   └─► 返回 imp_indices
    │   │
    │   ├─► gpu_provider.update_kv(layer_id, k[imp], v[imp], imp)
    │   │
    │   └─► 返回 (q[imp], gpu_k, gpu_v, residual[imp])
    │
    └─► 返回 (q, k, v, residual)
```

**状态管理**:

```python
class BlendBlender:
    def __init__(self, ...):
        # 静态配置
        self.common_metadata = common_metadata

        # 运行时状态（每次混合操作重置）
        self.metadata = BlendMetadata(
            imp_indices=None,    # 当前选中的 token 索引
            positions=None,      # 位置信息
            attn_mask=None,      # 注意力掩码
        )

    def reset_metadata(self):
        """为新的混合操作重置状态"""
        self.metadata.clean()
```

### 3.2 TokenSelector

**职责**: 选择需要重新计算的重要 tokens

**默认策略**: L2 距离 top-k 选择

```python
def select_important_tokens(
    self,
    new_k: torch.Tensor,  # 新计算的 K
    old_k: torch.Tensor,  # 缓存的 K
    ratio: float,        # 选择比例 (0.0-1.0)
) -> torch.Tensor:
    """
    算法步骤：

    1. 计算差异矩阵
       diff = sum((new_k - old_k)^2, dim=[1, 2])
       # 在 heads 和 head_dim 维度上求和
       # 结果: [seq_len] - 每个 token 的差异分数

    2. 选择 top-k
       topk_num = int(seq_len * ratio)
       top_indices = torch.topk(diff, k=topk_num).indices

    3. 排序并返回
       return torch.sort(top_indices).indices
    """
```

**替代策略**: 阈值选择

```python
def select_by_threshold(
    self,
    new_k: torch.Tensor,
    old_k: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    选择所有差异超过阈值的 tokens

    优势:
    - 自适应选择数量
    - 适合差异分布不均匀的场景

    劣势:
    - 可能选择过多或过少 tokens
    - 需要仔细调优阈值
    """
```

### 3.3 Metadata 系统

**BlendCommonMetadata**（静态配置）:
```python
@dataclass
class BlendCommonMetadata:
    check_layers: List[int]           # 检查层列表
    recomp_ratios: List[float]        # 每层的重算比例
    thresholds: List[float]           # 可选的阈值配置
```

**BlendMetadata**（运行时状态）:
```python
@dataclass
class BlendMetadata:
    imp_indices: Optional[torch.Tensor]   # 选中的 token 索引
    attn_mask: Optional[torch.Tensor]     # 注意力掩码
    positions: Optional[torch.Tensor]     # 位置编码

    def clean(self):
        """重置所有运行时状态"""
        self.imp_indices = None
        self.attn_mask = None
        self.positions = None
```

---

## 4. 接口规范

### 4.1 CacheProviderInterface

**目的**: 抽象 KV cache 存储，支持多种后端

```python
class CacheProviderInterface(ABC):
    """
    KV 缓存提供者接口

    实现者需要提供：
    - LMCache adapter
    - CPU memory cache
    - Remote cache (Redis, S3, etc.)
    """

    @abstractmethod
    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        layer_id: int,
        metadata: Dict[str, Any],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        检索指定层的 KV cache

        Args:
            tokens: Token 序列
            layer_id: 层索引
            metadata: 额外元数据（位置、掩码等）

        Returns:
            (k_cache, v_cache) 或 None（未命中）

        实现要点:
        - Token 匹配策略（精确匹配 / 前缀匹配 / segment 匹配）
        - 返回的 KV 形状需要与输入兼容
        - 处理缓存未命中情况
        """
        pass

    @abstractmethod
    def store_layer(
        self,
        tokens: torch.Tensor,
        layer_id: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> None:
        """
        存储 KV cache

        实现要点:
        - 异步存储优化
        - 缓存淘汰策略（LRU, LFU 等）
        - 内存/磁盘管理
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计

        Returns:
            {
                "hit_rate": float,        # 命中率
                "total_entries": int,     # 总条目数
                "size_bytes": int,        # 缓存大小
                "evictions": int,         # 淘汰次数
            }
        """
        pass
```

**实现示例**（CPU Cache Provider）:

```python
class CPUCacheProvider(CacheProviderInterface):
    def __init__(self, max_size_gb: float = 5.0):
        import torch
        self.cache = {}  # {layer_id: {token_tuple: (k, v)}}
        self.max_size = max_size_gb * 1024**3
        self.current_size = 0
        self.lock = threading.Lock()

        # 统计
        self.hits = 0
        self.misses = 0

    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        layer_id: int,
        metadata: Dict[str, Any],
    ):
        # Token 转为 tuple 作为 key
        key = tuple(tokens.tolist())

        with self.lock:
            layer_cache = self.cache.get(layer_id, {})
            if key in layer_cache:
                self.hits += 1
                return layer_cache[key]

            self.misses += 1
            return None

    def store_layer(
        self,
        tokens: torch.Tensor,
        layer_id: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Dict[str, Any],
    ):
        # 实现 LRU 淘汰
        # ...
        pass
```

### 4.2 GPUProviderInterface

**目的**: 抽象 GPU KV cache 访问，支持不同硬件

```python
class GPUProviderInterface(ABC):
    """
    GPU KV 访问接口

    实现者需要处理不同硬件的 KV 管理：
    - CUDA: 通过 vLLM KV connector
    - NPU: 通过 vLLM-Ascend KV pool
    - ROCm: 通过 ROCm 特定接口
    """

    @abstractmethod
    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 GPU 内存获取 KV

        Returns:
            (k_gpu, v_gpu)

        实现要点:
        - 处理分页 KV cache
        - 处理稀疏 attention
        - 避免不必要的 CPU-GPU 同步
        """
        pass

    @abstractmethod
    def update_kv(
        self,
        layer_id: int,
        k_update: torch.Tensor,
        v_update: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """
        更新 GPU KV cache

        Args:
            k_update: 新的 K 值 [n_selected, num_kv_heads, head_dim]
            v_update: 新的 V 值 [n_selected, num_kv_heads, head_dim]
            indices: 要更新的 token 位置 [n_selected]

        实现要点:
        - 高效的 scatter 更新
        - 避免内存碎片
        - 考虑使用 kernel 融合
        """
        pass

    @abstractmethod
    def get_kv_shape(self) -> Tuple[int, ...]:
        """
        获取 KV 形状

        Returns:
            (max_blocks, block_size, num_heads, head_dim)
            或其他分页形状
        """
        pass
```

**实现示例**（CUDA Provider）:

```python
class CUDAGPUProvider(GPUProviderInterface):
    def __init__(self, model_runner):
        """
        Args:
            model_runner: vLLM ModelRunner 实例
                         必须实现 KVConnectorModelRunnerMixin
        """
        self.model_runner = model_runner

    def get_kv(self, layer_id: int):
        """
        通过 KV connector 获取 KV

        vLLM KV connector 提供:
        - get_kv_from_connector(layer_id)
        - 返回分页的 KV cache
        """
        return self.model_runner.get_kv_from_connector(layer_id)

    def update_kv(self, layer_id, k_update, v_update, indices):
        """
        直接更新 GPU cache

        注意:
        - indices 需要是相对于分页的
        - 更新后可能需要同步
        """
        k_gpu, v_gpu = self.get_kv(layer_id)

        # 直接索引更新（高效）
        k_gpu[indices] = k_update
        v_gpu[indices] = v_update

        # 可选: 记录更新位置用于调试
        # logger.debug(f"Updated {len(indices)} tokens at layer {layer_id}")
```

### 4.3 ModelProviderInterface

**目的**: 抽象模型计算，支持不同架构

```python
class ModelProviderInterface(ABC):
    """
    模型计算接口

    实现者需要封装模型特定的操作：
    - QKV projection
    - Rotary embedding
    - Layer normalization
    """

    @abstractmethod
    def get_num_layers(self) -> int:
        """获取模型层数"""
        pass

    @abstractmethod
    def compute_layer_qkv(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算指定层的 QKV projection

        这是模型特定的核心方法：
        - Llama: qkv_proj(hidden_states)
        - Qwen2: 类似 Llama
        - Qwen3: 可能有不同的 attention 实现

        Returns:
            (q, k, v, residual)
        """
        pass

    @abstractmethod
    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用 rotary position encoding

        实现要点:
        - 处理不同位置的 RoPE
        - 支持扩展位置（如 ALiBi）
        - 保持与模型实现一致
        """
        pass
```

**实现示例**（Llama Adapter）:

```python
class LlamaAdapter(ModelProviderInterface):
    def __init__(self, vllm_model):
        """
        Args:
            vllm_model: vLLM LlamaForCausalLM 模型
        """
        self.vllm_model = vllm_model
        self.num_layers = len(vllm_model.model.layers)

    def compute_layer_qkv(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ):
        # 获取层
        layer = self.vllm_model.model.layers[layer_id]

        # Input layernorm
        hidden_states = layer.input_layernorm(hidden_states)

        # QKV projection
        qkv, _ = layer.self_attn.qkv_proj(hidden_states)

        # Split into Q, K, V
        num_q_heads = layer.self_attn.num_q_heads
        num_kv_heads = layer.self_attn.num_kv_heads
        head_dim = layer.self_attn.head_dim

        q, k, v = qkv.split([
            num_q_heads * head_dim,
            num_kv_heads * head_dim,
            num_kv_heads * head_dim,
        ], dim=-1)

        # Reshape for attention
        # [seq_len, num_heads, head_dim]
        # ...

        return q, k, v, residual

    def apply_rotary_emb(self, q, k, positions, layer_id):
        rotary_emb = self.vllm_model.model.layers[layer_id].self_attn.rotary_emb
        return rotary_emb(positions, q, k)
```

---

## 5. 数据流

### 5.1 单次混合操作的数据流

```
用户请求
    │
    ▼
┌─────────────────┐
│  vLLM Scheduler │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BlendWorker    │
│  (初始化时)     │
└────────┬────────┘
         │
         ├─► 创建 BlendConfig
         │
         ├─► 初始化 Providers:
         │   ├─► CacheProvider (LMCache/CPU/Remote)
         │   ├─► GPUProvider (CUDA/NPU/ROCm)
         │   └─► ModelProvider (Llama/Qwen/Mistral)
         │
         ├─► 创建 BlendBlender
         │
         └─► 创建 TokenSelector
         │
         ▼
┌─────────────────────────────────────┐
│  模型执行 (逐层)                     │
└─────────────────────────────────────┘
         │
         ▼
   Layer 0 执行
         │
         ├─► ModelRunner.compute_layer()
         │   │
         │   ├─► 获取 hidden_states
         │   │
         │   ├─► 调用 BlendBlender.process_qkv()
         │   │   │
         │   │   ├─► 1. CacheProvider.retrieve_layer()
         │   │   │   └─► 命中: 返回 (k_cache, v_cache)
         │   │   │       未命中: 返回 None
         │   │   │
         │   │   ├─► 2. GPUProvider.get_kv(layer_id)
         │   │   │   └─► 返回 (k_gpu, v_gpu)
         │   │   │
         │   │   ├─► 3. ModelProvider.apply_rotary_emb()
         │   │   │   └─► 返回 (q_rot, k_rot)
         │   │   │
         │   │   ├─► 4. if layer_id in check_layers:
         │   │   │   │
         │   │   │   ├─► 5a. TokenSelector.select_important_tokens()
         │   │   │   │   ├─► 计算 diff = sum((new_k - old_k)^2)
         │   │   │   │   ├─► 选择 top-k tokens
         │   │   │   │   └─► 返回 imp_indices
         │   │   │   │
         │   │   │   ├─► 5b. GPUProvider.update_kv()
         │   │   │   │   └─► k_gpu[indices] = k_new[indices]
         │   │   │   │       v_gpu[indices] = v_new[indices]
         │   │   │   │
         │   │   │   └─► 5c. 返回选中 tokens 的 (q, k, v, residual)
         │   │   │
         │   │   └─► 6. 返回 (q, k, v, residual) 或 subset
         │   │
         │   ├─► Attention 计算
         │   │
         │   └─► Layer 输出
         │
         ▼
   Layer 1 执行 (类似)
         │
         ▼
   Layer 2 执行 (类似)
         │
         ▼
      ...
         │
         ▼
┌─────────────────┐
│  最终输出        │
└─────────────────┘
```

### 5.2 多请求复用场景

```
请求 1: [系统提示] + [文档A] + [文档B] + [问题1]
         │
         ├─► Layer 0-31 计算
         │
         ├─► CacheProvider.store_layer()
         │   └─► 存储: 系统提示, 文档A, 文档B 的 KV
         │
         └─► 输出 1

请求 2: [系统提示] + [文档B] + [文档A] + [问题2]
         │
         ├─► Layer 0 开始
         │
         ├─► CacheProvider.retrieve_layer(系统提示)
         │   └─► 命中！复用 KV
         │
         ├─► CacheProvider.retrieve_layer(文档B)
         │   └─► 命中！复用 KV
         │   但位置不同！
         │
         ├─► BlendBlender.process_qkv()
         │   ├─► 计算新旧 K 的差异
         │   ├─► TokenSelector 选择 ~15% tokens
         │   └─► 只重算连接处的 tokens
         │
         ├─► CacheProvider.retrieve_layer(文档A)
         │   └─► 命中！复用 KV（大部分）
         │
         └─► 输出 2 (比请求 1 快 ~70%)
```

---

## 6. 集成方式

### 6.1 与 vLLM v1 的集成

**插件注册** (`setup.py`):

```python
entry_points={
    "vllm.platform_plugins": [
        "blend = vllm_blend:register",
    ],
    "vllm.general_plugins": [
        "blend_worker = vllm_blend:register_worker",
        "blend_config = vllm_blend:register_config",
    ],
}
```

**BlendPlatform** (`vllm_blend/platform.py`):

```python
class BlendPlatform(Platform):
    """
    平台插件 - 注入 Blend 配置和能力

    注意: BlendPlatform 不是硬件平台，而是功能平台
    """

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig):
        """
        在 vLLM 初始化时调用

        任务:
        1. 检查是否启用 Blend
        2. 创建 BlendConfig
        3. 注入到 additional_config
        4. 设置 worker_cls
        """
        from vllm_blend.config import BlendConfig

        # 创建配置
        blend_config = BlendConfig.from_vllm_config(vllm_config)

        if blend_config.enabled:
            # 存储配置供后续使用
            vllm_config.additional_config["blend_config"] = blend_config

            # 覆盖 worker 类
            if vllm_config.parallel_config.worker_cls == "auto":
                vllm_config.parallel_config.worker_cls = (
                    "vllm_blend.worker.blend_worker.BlendWorker"
                )

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config):
        """
        可能包装注意力后端

        如果启用 Blend，需要确保 attention backend 兼容
        """
        base_backend = get_base_attn_backend(selected_backend, attn_selector_config)

        if is_blend_enabled():
            # 包装以支持 Blend
            return wrap_attention_for_blend(base_backend)

        return base_backend
```

**BlendWorker** (`vllm_blend/worker/blend_worker.py`):

```python
class BlendWorker(GPUWorker):
    """
    vLLM Worker 扩展

    继承 GPUWorker 并添加 Blend 功能
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 初始化 Blender
        blend_config = self.vllm_config.additional_config.get("blend_config")
        if blend_config and blend_config.enabled:
            self.blender = self._init_blender(blend_config)
        else:
            self.blender = None

    def _init_blender(self, blend_config):
        """
        创建 BlendBlender 及其 providers

        根据硬件类型选择适当的 providers
        """
        device = self.device_config.device.type

        # 获取 providers
        if device == "cuda":
            from vllm_blend.backends.cuda import (
                CUDACacheProvider,
                CUDAGPUProvider,
                CUDAModelProvider,
            )

            cache_provider = CUDACacheProvider(blend_config)
            gpu_provider = CUDAGPUProvider(self.model_runner)
            model_provider = CUDAModelProvider(
                self.model_runner.model,
                self.model_runner,
            )

        elif device == "npu":
            # NPU providers
            ...

        # 创建 blender
        from vllm_blend.core import BlendBlender

        return BlendBlender(
            cache_provider=cache_provider,
            gpu_provider=gpu_provider,
            model_provider=model_provider,
            common_metadata=blend_config.common_metadata,
        )

    def execute_model(self, scheduler_output):
        """
        执行模型

        如果启用 Blend，注入 blender
        """
        if self.blender is None:
            return super().execute_model(scheduler_output)

        # 注入 blender 到 model_runner
        return self.model_runner.execute_with_blend(
            scheduler_output,
            blender=self.blender,
        )
```

**BlendModelRunnerMixin** (`vllm_blend/worker/model_runner.py`):

```python
class BlendModelRunnerMixin:
    """
    Model Runner 混合类

    添加 Blend 支持到任何 ModelRunner
    """

    def execute_with_blend(self, scheduler_output, blender):
        """
        使用 Blend 执行模型

        这是主要的集成点：
        1. 存储 blender 引用
        2. 执行正常的前向传播
        3. 在每层调用 blender.process_qkv()
        4. 清理 blender 引用
        """
        self.blender = blender

        try:
            result = super().execute_model(scheduler_output)
        finally:
            self.blender = None

        return result

    def _blend_process_layer(self, layer_id, q, k, v, residual):
        """
        处理层的 QKV

        在 attention 之前/之后调用
        """
        if hasattr(self, 'blender') and self.blender is not None:
            return self.blender.process_qkv(q, k, v, residual, layer_id)
        return q, k, v, residual
```

### 6.2 配置传递链

```
用户输入
    │
    ├─► 命令行: --enable-blend --blend-check-layers 0 16 32
    │
    ├─► Python API: LLM(enable_blend=True, blend_check_layers=[0, 16, 32])
    │
    ▼
VllmConfig
    │
    ├─► additional_config = {
    │      "blend_config": {
    │          "enabled": True,
    │          "check_layers": [0, 16, 32],
    │          "recompute_ratios": [0.1],
    │      }
    │  }
    │
    ▼
BlendPlatform.check_and_update_config()
    │
    ├─► 解析 additional_config
    │
    ├─► 创建 BlendConfig
    │
    ├─► 验证参数
    │
    ├─► 设置 worker_cls = "BlendWorker"
    │
    ▼
BlendWorker.__init__()
    │
    ├─► 获取 blend_config
    │
    ├─► 创建 Providers
    │
    ├─► 创建 BlendBlender
    │
    └─► 存储为 self.blender
    │
    ▼
模型执行
    │
    └─► 使用 self.blender.process_qkv()
```

---

## 7. 扩展机制

### 7.1 添加新的硬件支持

**步骤**:

1. **创建 backend 目录**
   ```
   vllm_blend/backends/new_hw/
   ```

2. **实现 Provider 接口**
   ```python
   # gpu_provider.py
   class NewHWGPUProvider(GPUProviderInterface):
       def get_kv(self, layer_id):
           # 新硬件特定的 KV 访问
           pass

       def update_kv(self, layer_id, k_update, v_update, indices):
           # 新硬件特定的更新操作
           pass
   ```

3. **注册到 factory**
   ```python
   # backends/__init__.py
   def get_providers_for_device(device, ...):
       if device == "new_hw":
           return NewHWGPUProvider(...)
   ```

### 7.2 添加新的模型支持

**步骤**:

1. **创建适配器**
   ```python
   # adapters/new_model.py
   class NewModelAdapter(BaseModelAdapter):
       def compute_layer_qkv(self, layer_id, hidden_states, residual):
           # 新模型特定的 QKV projection
           pass

       def apply_rotary_emb(self, q, k, positions, layer_id):
           # 新模型特定的 RoPE
           pass
   ```

2. **注册到适配器表**
   ```python
   # adapters/registry.py
   ModelAdapterRegistry.register("new_model", NewModelAdapter)
   ```

3. **添加类型映射**
   ```python
   # adapters/registry.py
   MODEL_TYPE_MAPPING = {
       "NewModelForCausalLM": "new_model",
   }
   ```

### 7.3 添加新的缓存后端

**步骤**:

1. **实现 CacheProviderInterface**
   ```python
   class NewCacheProvider(CacheProviderInterface):
       def retrieve_layer(self, tokens, layer_id, metadata):
           # 从新后端检索
           pass

       def store_layer(self, tokens, layer_id, k_cache, v_cache, metadata):
           # 存储到新后端
           pass
   ```

2. **注册到配置**
   ```python
   # config.py
   valid_providers = ["cpu", "lmcache", "remote", "new_backend"]
   ```

3. **添加到 factory**
   ```python
   def get_cache_provider(config):
       if config.cache_provider == "new_backend":
           return NewCacheProvider(config.cache_config)
   ```

---

## 8. 设计决策

### 8.1 为什么使用 Provider 抽象？

**决策**: 使用三个独立的 Provider 接口

**理由**:

1. **关注点分离**
   - CacheProvider: 存储策略
   - GPUProvider: 硬件访问
   - ModelProvider: 模型计算

2. **易于测试**
   - 可以使用 mock providers 测试核心逻辑
   - 不需要真实的硬件/模型

3. **灵活组合**
   - 任意 Cache + 任意 GPU + 任意 Model
   - 例如: LMCache + NPU + Qwen

**权衡**:
- 增加了一层抽象
- 需要维护多个接口

**结论**: 收益远大于成本

### 8.2 为什么不是平台替换而是平台包装？

**决策**: BlendPlatform 包装底层平台而非替换

**理由**:

1. **保留原有功能**
   - 用户仍然可以使用底层平台的所有特性
   - 例如 CUDA 的 all reduce, NPU 的优化等

2. **可选功能**
   - Blend 是可选的，不强制启用
   - 未启用时零开销

3. **易于维护**
   - 不需要复制整个平台代码
   - 只添加 Blend 相关功能

**实现**:
```python
class BlendPlatform(Platform):
    @classmethod
    def get_attn_backend(cls, selected_backend, attn_selector_config):
        # 获取底层平台的后端
        base_backend = get_underlying_platform().get_attn_backend(...)

        # 如果启用 Blend，包装它
        if blend_enabled:
            return BlendAttentionWrapper(base_backend)

        return base_backend
```

### 8.3 为什么使用分层执行而非整体执行？

**决策**: Layer-wise blending（逐层混合）

**理由**:

1. **内存效率**
   - 不需要同时存储所有层的 KV
   - 逐层处理，峰值内存更低

2. **灵活性**
   - 可以在不同层使用不同策略
   - 例如: Layer 0 用 15% ratio, Layer 16 用 10%

3. **与 vLLM 架构一致**
   - vLLM 本身就是 layer-wise 执行
   - 更容易集成

**实现**:
```python
def blend_layer(tokens, mask):
    for layer_id in range(num_layers):
        retrieve_layer(layer_id)     # 检索
        compute_layer(layer_id)      # 计算
        process_qkv(layer_id)        # 混合
        yield                        # 生成器模式
```

### 8.4 为什么使用适配器模式？

**决策**: 为每个模型架构实现适配器

**理由**:

1. **模型差异**
   - QKV projection 实现不同
   - RoPE 位置编码可能不同
   - Attention 结构可能不同

2. **易于扩展**
   - 添加新模型只需实现适配器
   - 不修改核心代码

3. **类型安全**
   - 明确的接口定义
   - 编译时检查

**实现**:
```python
class BaseModelAdapter:
    @abstractmethod
    def compute_layer_qkv(self, layer_id, hidden_states, residual):
        pass

class LlamaAdapter(BaseModelAdapter):
    def compute_layer_qkv(self, layer_id, hidden_states, residual):
        # Llama 特定实现
        pass
```

---

## 9. 性能考虑

### 9.1 计算开销

**Blend 增加的计算**:

1. **Token 选择**
   - 计算 L2 距离: O(seq_len * num_heads * head_dim)
   - Top-k 选择: O(seq_len * log(seq_len))
   - 通常可忽略（相比 attention）

2. **KV 更新**
   - Scatter 更新: O(n_selected * head_dim)
   - 如果选择 15% tokens，仅增加 15% 的更新开销

3. **RoPE 应用**
   - 已经在原模型中存在
   - 无额外开销

**总开销**: 通常 <5% 的总计算时间

### 9.2 内存开销

**Blend 增加的内存**:

1. **元数据**
   - BlendMetadata: 几 KB
   - BlendConfig: 几 KB
   - 可忽略

2. **Provider 状态**
   - CacheProvider: 取决于缓存大小
   - GPUProvider: 0（直接访问 GPU）
   - ModelProvider: 0（仅引用）

3. **临时张量**
   - Token 选择的中间结果: O(seq_len)
   - 很小

**总内存**: 几 KB 到几 MB（取决于 cache 配置）

### 9.3 通信开销

**分布式场景**:

1. **Worker 间通信**
   - 如果使用共享 cache (如 Redis)
   - 需要序列化/反序列化 KV
   - 可以优化（增量更新、压缩）

2. **优化策略**
   - 本地 cache 优先
   - 异步更新
   - 批量操作

### 9.4 性能优化建议

1. **异步操作**
   ```python
   async def store_layer_async(self, ...):
       # 异步存储，不阻塞计算
       await self.cache_store_async(...)
   ```

2. **Kernel 融合**
   ```python
   # 融合的 scatter update kernel
   fused_update_kv(gpu_kv, indices, new_kv)
   ```

3. **稀疏更新**
   ```python
   # 只更新真正改变的 tokens
   changed_indices = find_changed_tokens(old_k, new_k, threshold)
   update_kv(changed_indices)
   ```

4. **缓存预热**
   ```python
   # 预先加载常用文档的 KV
   preheat_cache(common_documents)
   ```

---

## 10. 安全考虑

### 10.1 输入验证

**配置验证**:
```python
def __post_init__(self):
    # 检查层的有效性
    for layer_id in self.check_layers:
        if layer_id < 0 or layer_id >= max_layers:
            raise ValueError(f"Invalid layer_id: {layer_id}")

    # 检查比例的有效性
    for ratio in self.recompute_ratios:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"Invalid ratio: {ratio}")
```

**Token 验证**:
```python
def validate_tokens(tokens):
    if len(tokens) == 0:
        raise ValueError("Empty token sequence")

    if tokens.max() >= vocab_size:
        raise ValueError("Token ID exceeds vocab size")
```

### 10.2 资源管理

**内存限制**:
```python
class CPUCacheProvider:
    def __init__(self, max_size_gb: float):
        self.max_size = max_size_gb * 1024**3
        self.current_size = 0

    def store_layer(self, ...):
        # 检查是否超过限制
        if self.current_size + new_size > self.max_size:
            self.evict_lru()  # 淘汰最少使用的项
```

**线程安全**:
```python
class ThreadSafeCacheProvider:
    def __init__(self):
        self.lock = threading.Lock()
        self.cache = {}

    def retrieve(self, ...):
        with self.lock:
            return self.cache.get(key)
```

### 10.3 错误处理

**优雅降级**:
```python
def process_qkv(self, ...):
    try:
        # 尝试混合
        return self._blend_qkv(...)
    except Exception as e:
        logger.warning(f"Blend failed: {e}, falling back to normal execution")
        # 降级到正常执行
        return q, k, v, residual
```

**超时处理**:
```python
def retrieve_with_timeout(self, tokens, layer_id, timeout_ms=100):
    try:
        return self.cache.retrieve(tokens, layer_id, timeout=timeout_ms)
    except TimeoutError:
        logger.warning(f"Cache retrieval timed out for layer {layer_id}")
        return None
```

---

## 11. 附录

### 11.1 术语表

| 术语 | 定义 |
|------|------|
| KV Cache | Key-Value 缓存，transformer 模型中间结果 |
| TTFT | Time To First Token，首字延迟 |
| Blend | 混合，指复用和重新计算 KV 的过程 |
| Provider | 提供者，抽象接口的实现 |
| Adapter | 适配器，模型特定的封装 |
| Segment | 文本段，可独立缓存的最小单位 |
| RoPE | Rotary Position Embedding，旋转位置编码 |

### 11.2 参考资料

1. **LMCache**: https://github.com/LMCache/LMCache
2. **vLLM**: https://github.com/vllm-project/vllm
3. **vLLM-Ascend**: https://github.com/ascend/vllm-ascend
4. **CacheGen 论文**: Cachegen: KV Cache Compression and Streaming
5. **CacheBlend 论文**: CacheBlend: Fast Large Language Model Serving for RAG

### 11.3 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2025-01-13 | 初始设计，Phase 1 实现 |
| v1.0 | TBD | 完整实现，生产就绪 |

### 11.4 贡献指南

贡献者请阅读：

1. [DESIGN.md](DESIGN.md) - 设计文档
2. [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南
3. [README.md](README.md) - 快速开始

---

## 总结

本文档详细描述了 vLLM Blend 插件的架构设计，包括：

- **四层架构**: Config → Core → Providers → Adapters
- **核心组件**: BlendBlender, TokenSelector, Metadata
- **抽象接口**: Cache, GPU, Model Providers
- **集成方式**: 通过 vLLM 插件系统无缝集成
- **扩展机制**: 易于添加新硬件、模型、缓存后端

设计原则：
1. **解耦优先**: 完全独立于 LMCache 和特定硬件
2. **性能至上**: 最小化开销，最大化复用
3. **易于扩展**: 清晰的接口和扩展点
4. **生产就绪**: 完整的错误处理和资源管理

下一步实施：Phase 2 - Provider 具体实现
