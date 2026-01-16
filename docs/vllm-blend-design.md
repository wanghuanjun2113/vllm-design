# vLLM Blend 插件独立化设计方案

## 文档版本

- **版本**: v1.0
- **日期**: 2025-01-13
- **状态**: 设计阶段

## 1. 项目概述

### 1.1 背景

LMCache 提供了 CacheBlend 功能，允许在非前缀位置复用 KV cache，特别适合 RAG、多文档问答等场景。目前 Blend 功能深度耦合在 LMCache 中，限制了其使用范围。

### 1.2 目标

将 Blend 功能从 LMCache 中抽离出来，设计一个独立的 vLLM 插件，实现：

1. **完全解耦**: 移除对 LMCache 的所有依赖
2. **硬件无关**: 支持 CUDA、Ascend NPU、ROCm 等多种硬件平台
3. **插件化集成**: 通过 vLLM 的插件系统注册
4. **向后兼容**: 与 vLLM v1 保持完全兼容
5. **易于扩展**: 方便添加新的模型和硬件支持

### 1.3 适用场景

- **RAG (检索增强生成)**: 复用检索到的文档 KV cache
- **多文档问答**: 不同顺序组合相同文档时复用 KV cache
- **长上下文处理**: 分段处理长文档，减少重复计算
- **对话系统**: 复用对话历史，仅重算新问题部分

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                         vLLM v1 Engine                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Blend Plugin (vllm-blend)               │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │                                                     │   │
│  │  ┌─────────────┐    ┌──────────────────────────┐   │   │
│  │  │ BlendWorker │◄──┤  BlendModelRunnerMixin   │   │   │
│  │  └──────┬──────┘    └──────────────┬───────────┘   │   │
│  │         │                          │                │   │
│  │         ▼                          ▼                │   │
│  │  ┌──────────────────────────────────────────────┐ │   │
│  │  │            BlendBlender (Core)               │ │   │
│  │  │  ┌────────────┐  ┌──────────┐  ┌─────────┐  │ │   │
│  │  │  │  Selector │  │ Metadata │  │  Utils  │  │ │   │
│  │  │  └────────────┘  └──────────┘  └─────────┘  │ │   │
│  │  └──────────────────────────────────────────────┘ │   │
│  │         ▲                ▲                ▲        │   │
│  │         │                │                │        │   │
│  │  ┌──────┴──────┐  ┌─────┴─────┐  ┌─────┴─────┐   │   │
│  │  │   Cache     │  │    GPU    │  │   Model    │   │   │
│  │  │  Provider   │  │ Provider  │  │  Provider  │   │   │
│  │  └──────────────┘  └───────────┘  └────────────┘   │   │
│  │         │                │                │        │   │
│  │         └────────────────┼────────────────┘        │   │
│  │                          ▼                         │   │
│  │  ┌──────────────────────────────────────────────┐ │   │
│  │  │          Backend Implementations             │ │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────────┐    │ │   │
│  │  │  │  CUDA   │ │   NPU   │ │    ROCm     │    │ │   │
│  │  │  └─────────┘ └─────────┘ └─────────────┘    │ │   │
│  │  └──────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Model Adapters (Pluggable)                  │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐               │   │
│  │  │ Llama   │ │  Qwen   │ │ Mistral │  ...         │   │
│  │  └─────────┘ └─────────┘ └─────────┘               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

#### 2.2.1 Provider 抽象层

**设计理念**: 通过抽象接口实现与具体实现的解耦，支持多种硬件和缓存后端。

**三个核心接口**:

```python
# CacheProviderInterface - KV 缓存提供者
class CacheProviderInterface(ABC):
    """抽象 KV 缓存访问接口"""

    @abstractmethod
    def retrieve_layer(
        self,
        tokens: torch.Tensor,
        layer_id: int,
        metadata: dict,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """从缓存中检索指定层的 KV tensors

        Returns:
            (k_cache, v_cache) 或 None（缓存未命中）
        """
        pass

    @abstractmethod
    def store_layer(
        self,
        tokens: torch.Tensor,
        layer_id: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: dict,
    ) -> None:
        """存储 KV tensors 到缓存"""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """获取缓存统计信息（命中率等）"""
        pass

# GPUProviderInterface - GPU KV 访问
class GPUProviderInterface(ABC):
    """抽象 GPU KV cache 访问接口"""

    @abstractmethod
    def get_kv(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """从 GPU 内存获取当前 KV tensors"""
        pass

    @abstractmethod
    def update_kv(
        self,
        layer_id: int,
        k_update: torch.Tensor,
        v_update: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """更新 GPU KV cache 中指定位置的值"""
        pass

    @abstractmethod
    def get_kv_shape(self) -> tuple:
        """获取 KV cache tensor 的形状"""
        pass

# ModelProviderInterface - 模型计算
class ModelProviderInterface(ABC):
    """抽象模型访问接口"""

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算指定层的 QKV projection

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """应用 rotary position encoding"""
        pass
```

#### 2.2.2 BlendBlender 核心逻辑

```python
class BlendBlender:
    """核心混合逻辑 - 完全解耦的实现"""

    def __init__(
        self,
        cache_provider: CacheProviderInterface,
        gpu_provider: GPUProviderInterface,
        model_provider: ModelProviderInterface,
        common_metadata: BlendCommonMetadata,
    ):
        self.cache_provider = cache_provider
        self.gpu_provider = gpu_provider
        self.model_provider = model_provider
        self.common_metadata = common_metadata

        self.metadata = BlendMetadata()
        self.selector = TokenSelector(common_metadata)

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理 QKV 并执行混合

        这是每个层执行时调用的核心方法
        """
        # 1. 尝试从缓存获取 KV
        cached_k, cached_v = self.cache_provider.retrieve_layer(...)

        if cached_k is None:
            return q, k, v, residual  # 缓存未命中

        # 2. 获取 GPU 中的 KV（用于比较）
        gpu_k, gpu_v = self.gpu_provider.get_kv(layer_id)

        # 3. 应用 rotary embedding
        q, k = self.model_provider.apply_rotary_emb(...)

        # 4. 如果是检查层，执行混合
        if layer_id in self.common_metadata.check_layers:
            imp_indices = self.selector.select_important_tokens(
                new_k=k, old_k=gpu_k,
                ratio=self.common_metadata.recomp_ratios[0]
            )

            # 更新 GPU cache
            self.gpu_provider.update_kv(
                layer_id=layer_id,
                k_update=k[imp_indices],
                v_update=v[imp_indices],
                indices=imp_indices,
            )

            # 只返回选中的 tokens
            return q[imp_indices], gpu_k, gpu_v, residual[imp_indices]

        # 5. 非检查层直接返回
        return q, k, v, residual
```

#### 2.2.3 TokenSelector 选择算法

```python
class TokenSelector:
    """选择需要重新计算的重要 tokens"""

    def select_important_tokens(
        self,
        new_k: torch.Tensor,  # 新计算的 K
        old_k: torch.Tensor,  # 缓存的 K
        ratio: float,        # 重算比例
    ) -> torch.Tensor:
        """基于 L2 距离选择 top-K tokens

        算法：
        1. 计算新旧 K 的 L2 距离（在 heads 和 head_dim 维度上平均）
        2. 选择距离最大的 top-K tokens
        3. 返回排序后的索引
        """
        # 计算差异
        diff_k = torch.sum(
            (new_k.to(torch.float32) - old_k.to(torch.float32)) ** 2,
            dim=[1, 2],  # 在 heads 和 head_dim 上求和
        )

        total_len = diff_k.shape[0]
        topk_num = max(int(total_len * ratio), 1)

        # 获取 top-k 索引
        top_indices = torch.topk(diff_k, k=topk_num).indices
        top_indices, _ = torch.sort(top_indices)

        return top_indices
```

#### 2.2.4 模型适配器系统

```python
class BaseModelAdapter(ModelProviderInterface):
    """模型适配器基类"""

    def __init__(self, vllm_model):
        self.vllm_model = vllm_model
        self.num_layers = len(vllm_model.model.layers)

    @abstractmethod
    def extract_qkv_from_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从层中提取 QKV projection - 模型特定实现"""
        pass

# Llama 适配器示例
class LlamaAdapter(BaseModelAdapter):
    """Llama 模型特定适配器"""

    def extract_qkv_from_layer(self, layer_id, hidden_states):
        layer = self.vllm_model.model.layers[layer_id]

        # QKV projection
        qkv, _ = layer.self_attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([
            layer.self_attn.num_q_heads * layer.self_attn.head_dim,
            layer.self_attn.num_kv_heads * layer.self_attn.head_dim,
            layer.self_attn.num_kv_heads * layer.self_attn.head_dim,
        ], dim=-1)

        return q, k, v

    def get_rotary_emb(self, layer_id):
        return self.vllm_model.model.layers[layer_id].self_attn.rotary_emb

    def apply_rotary_emb(self, q, k, positions, layer_id):
        rotary_emb = self.get_rotary_emb(layer_id)
        return rotary_emb(positions, q, k)
```

### 2.3 插件集成

#### 2.3.1 平台注册

```python
class BlendPlatform(Platform):
    """Blend 平台插件"""

    _enum = PlatformEnum.OOT
    device_name = "blend"
    device_type = "blend"  # 不实际使用，仅用于标识

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        """集成 Blend 配置到 vLLM"""
        from vllm_blend.config import BlendConfig

        blend_config = BlendConfig.from_vllm_config(vllm_config)

        if blend_config.enabled:
            # 存储配置
            vllm_config.additional_config["blend_config"] = blend_config

            # 设置 Worker 类
            if vllm_config.parallel_config.worker_cls == "auto":
                vllm_config.parallel_config.worker_cls = (
                    "vllm_blend.worker.blend_worker.BlendWorker"
                )

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config):
        """获取注意力后端，可能包装以支持 Blend"""
        # 获取底层平台的后端
        base_backend = cls._get_base_platform().get_attn_backend_cls(
            selected_backend, attn_selector_config
        )

        # 如果启用 Blend，包装后端
        if cls._is_blend_enabled():
            from vllm_blend.backends import wrap_attention_for_blend
            return wrap_attention_for_blend(base_backend)

        return base_backend
```

#### 2.3.2 Worker 集成

```python
class BlendWorker(GPUWorker):
    """支持 Blend 的 Worker"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 初始化 Blender
        blend_config = self.vllm_config.additional_config.get("blend_config")
        if blend_config and blend_config.enabled:
            self.blender = self._init_blender(blend_config)
        else:
            self.blender = None

    def _init_blender(self, blend_config):
        """初始化 BlendBlender 及其 providers"""
        from vllm_blend.backends import get_providers_for_device

        cache_provider = get_providers_for_device(
            provider_type="cache",
            device=self.device,
            config=blend_config,
        )
        gpu_provider = get_providers_for_device(
            provider_type="gpu",
            device=self.device,
            model_runner=self.model_runner,
        )
        model_provider = get_providers_for_device(
            provider_type="model",
            device=self.device,
            vllm_model=self.model_runner.model,
        )

        from vllm_blend.core.blender import BlendBlender
        return BlendBlender(
            cache_provider=cache_provider,
            gpu_provider=gpu_provider,
            model_provider=model_provider,
            common_metadata=blend_config.common_metadata,
        )

    def execute_model(self, scheduler_output):
        """执行模型（带 Blend 支持）"""
        if self.blender is None:
            return super().execute_model(scheduler_output)

        # 将 blender 注入到 model_runner
        return self.model_runner.execute_with_blend(
            scheduler_output,
            blender=self.blender,
        )
```

### 2.4 Backend 实现

#### 2.4.1 CUDA Backend

```python
class CUDAGPUProvider(GPUProviderInterface):
    """CUDA GPU KV cache provider"""

    def __init__(self, model_runner):
        self.model_runner = model_runner
        assert isinstance(model_runner, KVConnectorModelRunnerMixin)

    def get_kv(self, layer_id: int):
        """通过 KV connector 获取 KV"""
        return self.model_runner.get_kv_from_connector(layer_id)

    def update_kv(self, layer_id, k_update, v_update, indices):
        """更新 GPU KV cache"""
        k_gpu, v_gpu = self.get_kv(layer_id)
        k_gpu[indices] = k_update
        v_gpu[indices] = v_update

class CUDACacheProvider(CacheProviderInterface):
    """CUDA 缓存 provider（可以包装 LMCache）"""

    def __init__(self, config):
        self.config = config
        # 可以选择使用 LMCache、CPU memory 等

        if config.cache_provider == "lmcache":
            from lmcache.v1 import LMCacheEngine
            self.cache_engine = LMCacheEngine(...)
        else:
            self.cache_storage = {}  # 简单内存缓存

    def retrieve_layer(self, tokens, layer_id, metadata):
        if hasattr(self, 'cache_engine'):
            return self.cache_engine.retrieve_layer(tokens, layer_id, **metadata)
        else:
            key = (tuple(tokens.tolist()), layer_id)
            return self.cache_storage.get(key)
```

#### 2.4.2 NPU Backend (Ascend)

```python
class NPUGPUProvider(GPUProviderInterface):
    """Ascend NPU GPU KV cache provider"""

    def __init__(self, model_runner):
        self.model_runner = model_runner
        # NPU 特定初始化

    def get_kv(self, layer_id: int):
        """从 NPU 内存获取 KV

        实现取决于 vLLM-Ascend 的 KV 管理方式
        可能通过 HCCL 或 NPU 内存访问
        """
        # vLLM-Ascend 特定实现
        pass

    def update_kv(self, layer_id, k_update, v_update, indices):
        """使用 NPU 优化操作更新 KV cache"""
        import torch_npu

        k_gpu, v_gpu = self.get_kv(layer_id)

        # 使用 torch_npu 高效更新
        torch_npu.copy_(k_gpu[indices], k_update)
        torch_npu.copy_(v_gpu[indices], v_update)

class NPUCacheProvider(CacheProviderInterface):
    """NPU 缓存 provider"""

    def __init__(self, config):
        self.config = config
        # 可以与 vLLM-Ascend 的 KV pool 集成

    def retrieve_layer(self, tokens, layer_id, metadata):
        # 从 Ascend KV pool 或远程存储检索
        pass
```

## 3. 目录结构

```
vllm-blend/
├── setup.py                      # 插件入口定义
├── README.md                     # 项目说明
├── requirements.txt              # 依赖
├── pyproject.toml               # 项目配置
│
├── tests/                        # 测试
│   ├── __init__.py
│   ├── test_blender.py          # 核心逻辑测试
│   ├── test_providers.py        # Provider 测试
│   ├── test_adapters.py         # 适配器测试
│   └── integration/             # 集成测试
│       ├── test_cuda.py
│       └── test_npu.py
│
├── examples/                     # 示例
│   ├── basic_usage.py
│   ├── rag_example.py
│   └── multi_doc_qa.py
│
└── vllm_blend/                   # 主代码
    ├── __init__.py              # 插件注册
    ├── config.py                # BlendConfig
    ├── platform.py              # BlendPlatform
    │
    ├── core/                    # 核心逻辑
    │   ├── __init__.py
    │   ├── blender.py           # BlendBlender
    │   ├── metadata.py          # 元数据
    │   └── selector.py          # TokenSelector
    │
    ├── providers/               # 抽象接口
    │   ├── __init__.py
    │   ├── cache_provider.py    # CacheProviderInterface
    │   ├── gpu_provider.py      # GPUProviderInterface
    │   └── model_provider.py    # ModelProviderInterface
    │
    ├── adapters/                # 模型适配器
    │   ├── __init__.py
    │   ├── base.py              # BaseModelAdapter
    │   ├── llama.py             # LlamaAdapter
    │   ├── qwen2.py             # Qwen2Adapter
    │   ├── qwen3.py             # Qwen3Adapter
    │   ├── mistral.py           # MistralAdapter
    │   └── registry.py          # 适配器注册表
    │
    ├── backends/                # 硬件实现
    │   ├── __init__.py
    │   ├── factory.py           # Backend factory
    │   │
    │   ├── cuda/                # CUDA 实现
    │   │   ├── __init__.py
    │   │   ├── gpu_provider.py
    │   │   ├── cache_provider.py
    │   │   └── model_provider.py
    │   │
    │   ├── npu/                 # Ascend NPU 实现
    │   │   ├── __init__.py
    │   │   ├── gpu_provider.py
    │   │   ├── cache_provider.py
    │   │   ├── model_provider.py
    │   │   └── attention.py      # NPU 特定注意力优化
    │   │
    │   └── rocm/                # ROCm 实现
    │       ├── __init__.py
    │       ├── gpu_provider.py
    │       └── cache_provider.py
    │
    ├── worker/                  # Worker 集成
    │   ├── __init__.py
    │   ├── blend_worker.py      # BlendWorker
    │   └── model_runner.py      # BlendModelRunnerMixin
    │
    └── utils/                   # 工具
        ├── __init__.py
        ├── rope.py              # RoPE 工具
        └── diagnostics.py       # 诊断和监控
```

## 4. 配置系统

### 4.1 BlendConfig

```python
@dataclass
class BlendConfig:
    """Blend 配置"""

    # 启用/禁用
    enabled: bool = False

    # 检查层列表
    check_layers: List[int] = field(default_factory=lambda: [0, 16, 32])

    # 重算比例
    recompute_ratios: List[float] = field(default_factory=lambda: [0.1])

    # 决策阈值（可选）
    thresholds: Optional[List[float]] = None

    # 缓存后端选择
    cache_provider: str = "cpu"  # lmcache, cpu, remote

    # 缓存配置
    cache_config: dict = field(default_factory=dict)

    def __post_init__(self):
        """验证配置"""
        if not self.check_layers:
            raise ValueError("check_layers 不能为空")

        for ratio in self.recompute_ratios:
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"recompute_ratio 必须在 [0, 1] 之间，得到 {ratio}")

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "BlendConfig":
        """从 VllmConfig 创建"""
        additional_config = vllm_config.additional_config or {}
        blend_config_dict = additional_config.get("blend_config", {})
        return cls(**blend_config_dict)

    @property
    def common_metadata(self) -> BlendCommonMetadata:
        """转换为 BlendCommonMetadata"""
        return BlendCommonMetadata(
            check_layers=self.check_layers,
            recomp_ratios=self.recompute_ratios,
            thresholds=self.thresholds,
        )
```

### 4.2 命令行参数

```python
def register_blend_config():
    """注册 Blend 参数到 vLLM argument parser"""
    def add_blend_args(parser):
        parser.add_argument(
            "--enable-blend",
            action="store_true",
            help="启用 Blend 功能",
        )
        parser.add_argument(
            "--blend-check-layers",
            type=int,
            nargs="+",
            default=[0, 16, 32],
            help="执行混合检查的层索引",
        )
        parser.add_argument(
            "--blend-recompute-ratios",
            type=float,
            nargs="+",
            default=[0.1],
            help="每层重新计算的 token 比例",
        )
        parser.add_argument(
            "--blend-cache-provider",
            type=str,
            default="cpu",
            choices=["cpu", "lmcache", "remote"],
            help="缓存后端选择",
        )

    # 注册到 vLLM
    from vllm.cli.args import LooseArgumentParser
    LooseArgumentParser.register_argument_adder(add_blend_args)
```

## 5. 使用方式

### 5.1 命令行

```bash
# 基础用法
vllm serve meta-llama/Llama-2-7b-chat-hf \
    --enable-blend

# 自定义配置
vllm serve meta-llama/Llama-2-7b-chat-hf \
    --enable-blend \
    --blend-check-layers 0 8 16 24 32 \
    --blend-recompute-ratios 0.15

# 使用 LMCache 后端
vllm serve meta-llama/Llama-2-7b-chat-hf \
    --enable-blend \
    --blend-cache-provider lmcache
```

### 5.2 Python API

```python
from vllm import LLM, SamplingParams

# 基础用法
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_blend=True,
)

# 自定义配置
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_blend=True,
    blend_check_layers=[0, 16, 32],
    blend_recompute_ratios=[0.1],
)

# 推理
outputs = llm.generate("Hello, world!", SamplingParams(max_tokens=10))
```

### 5.3 RAG 示例

```python
from vllm import LLM, SamplingParams

# 初始化
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_blend=True,
    blend_check_layers=[0],
    blend_recompute_ratios=[0.15],
)

# 文档 chunks
documents = {
    "doc1": "Content of document 1...",
    "doc2": "Content of document 2...",
    "doc3": "Content of document 3...",
}

# 多个查询，不同顺序
queries = [
    (["doc1", "doc2", "doc3"], "What is the summary?"),
    (["doc3", "doc1", "doc2"], "Compare doc3 and doc1"),
    (["doc2", "doc3", "doc1"], "What are the key points?"),
]

for doc_order, question in queries:
    # 构建提示（使用特殊分隔符）
    prompt = build_prompt_with_separator(documents, doc_order, question)

    # 第一次请求会计算并缓存
    # 后续请求会复用文档的 KV cache，只重算连接处
    output = llm.generate(prompt, SamplingParams(max_tokens=100))
    print(output[0].outputs[0].text)
```

## 6. 实施计划

### Phase 1: 核心基础设施 (Week 1-2)

**目标**: 实现核心 Blend 逻辑和配置系统

**关键文件**:
- `vllm_blend/core/blender.py`
- `vllm_blend/core/metadata.py`
- `vllm_blend/core/selector.py`
- `vllm_blend/config.py`

**任务清单**:
- [ ] 实现元数据类（BlendCommonMetadata, BlendMetadata）
- [ ] 实现 TokenSelector 算法
- [ ] 实现 BlendBlender（使用抽象 providers）
- [ ] 实现 BlendConfig
- [ ] 实现参数注册系统
- [ ] 编写单元测试

### Phase 2: Provider 抽象层 (Week 3)

**目标**: 实现抽象接口和 CUDA backend

**关键文件**:
- `vllm_blend/providers/*.py`
- `vllm_blend/backends/cuda/gpu_provider.py`
- `vllm_blend/backends/cuda/cache_provider.py`
- `vllm_blend/backends/cuda/model_provider.py`

**任务清单**:
- [ ] 定义三个抽象 provider 接口
- [ ] 实现 CUDA GPU provider
- [ ] 实现 LMCache adapter 作为 cache provider
- [ ] 实现 CPU cache provider（用于测试）
- [ ] 实现 CUDA model provider
- [ ] 编写 provider 测试

### Phase 3: 模型适配器 (Week 4)

**目标**: 实现模型适配器系统

**关键文件**:
- `vllm_blend/adapters/base.py`
- `vllm_blend/adapters/llama.py`
- `vllm_blend/adapters/qwen2.py`
- `vllm_blend/adapters/qwen3.py`
- `vllm_blend/adapters/registry.py`

**任务清单**:
- [ ] 实现 BaseModelAdapter
- [ ] 实现 LlamaAdapter
- [ ] 实现 Qwen2Adapter
- [ ] 实现 Qwen3Adapter
- [ ] 实现适配器注册表
- [ ] 编写适配器测试

### Phase 4: Worker 集成 (Week 5)

**目标**: 集成到 vLLM Worker

**关键文件**:
- `vllm_blend/worker/blend_worker.py`
- `vllm_blend/worker/model_runner.py`
- `vllm_blend/platform.py`
- `vllm_blend/__init__.py`
- `setup.py`

**任务清单**:
- [ ] 实现 BlendPlatform
- [ ] 实现 BlendWorker
- [ ] 实现 BlendModelRunnerMixin
- [ ] 实现插件注册函数
- [ ] 创建 setup.py
- [ ] 端到端集成测试

### Phase 5: Ascend NPU 支持 (Week 6)

**目标**: 实现 NPU backend

**关键文件**:
- `vllm_blend/backends/npu/gpu_provider.py`
- `vllm_blend/backends/npu/cache_provider.py`
- `vllm_blend/backends/npu/model_provider.py`

**任务清单**:
- [ ] 实现 NPUGPUProvider
- [ ] 实现 NPUCacheProvider
- [ ] 实现 NPUModelProvider
- [ ] 与 vLLM-Ascend KV 管理集成
- [ ] 在 Ascend 硬件上测试
- [ ] NPU 特定优化

### Phase 6: 测试与文档 (Week 7-8)

**目标**: 完善测试和文档

**任务清单**:
- [ ] 完整的单元测试覆盖
- [ ] 集成测试（多模型）
- [ ] 性能基准测试
- [ ] 用户文档
- [ ] API 参考
- [ ] 示例代码
- [ ] README

## 7. 测试策略

### 7.1 单元测试

```python
# tests/test_blender.py

def test_token_selector():
    """测试 Token 选择算法"""
    selector = TokenSelector(
        BlendCommonMetadata(check_layers=[0], recomp_ratios=[0.5])
    )

    new_k = torch.randn(100, 32, 128)
    old_k = torch.randn(100, 32, 128)

    indices = selector.select_important_tokens(new_k, old_k, ratio=0.5)

    assert len(indices) == 50
    assert torch.all(indices < 100)

def test_blender_with_mock_providers():
    """使用 mock providers 测试 Blender"""
    blender = BlendBlender(
        cache_provider=MockCacheProvider(),
        gpu_provider=MockGPUProvider(),
        model_provider=MockModelProvider(),
        common_metadata=BlendCommonMetadata(
            check_layers=[0],
            recomp_ratios=[0.1]
        ),
    )

    q, k, v, residual = blender.process_qkv(
        q=torch.randn(10, 32, 128),
        k=torch.randn(10, 32, 128),
        v=torch.randn(10, 32, 128),
        residual=torch.randn(10, 4096),
        layer_id=0,
    )

    # 应该只返回 10% 的 tokens
    assert q.shape[0] == 1
```

### 7.2 集成测试

```python
# tests/integration/test_cuda.py

def test_blend_worker_cuda():
    """测试 CUDA 上的 Blend worker"""
    vllm_config = VllmConfig(
        model="meta-llama/Llama-2-7b-chat-hf",
        additional_config={
            "blend_config": {
                "enabled": True,
                "check_layers": [0, 16],
                "recompute_ratios": [0.1],
            }
        }
    )

    llm = LLM(config=vllm_config)
    outputs = llm.generate("Hello, world!", SamplingParams(max_tokens=10))

    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0
```

### 7.3 性能测试

```python
# benchmarks/benchmark_blend.py

def benchmark_blend_vs_baseline():
    """对比 Blend 和 baseline 的性能"""

    prompts = ["Hello, world!"] * 100
    sampling_params = SamplingParams(max_tokens=100)

    # Baseline
    llm_baseline = LLM(model="meta-llama/Llama-2-7b-chat-hf")
    start = time.time()
    outputs_baseline = llm_baseline.generate(prompts, sampling_params)
    time_baseline = time.time() - start

    # With Blend
    llm_blend = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        enable_blend=True,
        blend_recompute_ratios=[0.1],
    )
    start = time.time()
    outputs_blend = llm_blend.generate(prompts, sampling_params)
    time_blend = time.time() - start

    speedup = time_baseline / time_blend
    print(f"Blend 加速比: {speedup:.2f}x")

    # 验证输出质量
    assert len(outputs_baseline) == len(outputs_blend)
```

## 8. 关键设计决策

### 8.1 Provider 抽象模式

**决策**: 使用三个抽象接口（CacheProvider, GPUProvider, ModelProvider）

**原因**:
- 完全解耦具体实现
- 支持多种硬件和缓存后端
- 易于测试和扩展

**权衡**:
- 增加了一层抽象
- 需要为每个平台实现 adapter

**结论**: 收益大于成本，是实现硬件无关性的最佳方式

### 8.2 平台包装 vs 平台替换

**决策**: BlendPlatform 包装底层平台而非替换

**原因**:
- 保留底层平台的所有功能
- Blend 作为可选功能添加
- 用户可以同时使用其他平台特性

**实现**:
```python
class BlendPlatform(Platform):
    @classmethod
    def get_attn_backend(cls, ...):
        # 获取底层平台的后端
        base = get_underlying_platform()
        base_backend = base.get_attn_backend(...)

        # 如果启用 Blend，包装它
        if blend_enabled:
            return BlendBackendWrapper(base_backend)
        return base_backend
```

### 8.3 适配器模式

**决策**: 使用适配器模式支持不同模型

**原因**:
- 不同模型的 QKV projection 实现不同
- 适配器封装模型特定逻辑
- 易于添加新模型支持

**实现**:
```python
class LlamaAdapter(BaseModelAdapter):
    def extract_qkv(self, layer_id, hidden_states):
        # Llama 特定实现
        pass

class Qwen3Adapter(BaseModelAdapter):
    def extract_qkv(self, layer_id, hidden_states):
        # Qwen3 特定实现
        pass
```

### 8.4 分层执行

**决策**: 保持与 LMCache 相同的 layer-wise 执行方式

**原因**:
- 逐层执行更适合混合逻辑
- 可以在特定层（如 layer 0）执行 token 选择
- 减少内存峰值使用

**实现**:
```python
def blend_layer(tokens, mask):
    for layer_id in range(num_layers):
        retrieve_layer(layer_id)
        compute_layer(layer_id)
        process_qkv(layer_id)  # 混合
        yield
```

## 9. 兼容性保证

### 9.1 API 兼容性

- **向后兼容**: 不修改 vLLM 现有 API
- **可选功能**: 未启用时不影响性能
- **渐进采用**: 用户可以选择性启用

### 9.2 模型兼容性

支持的模型架构：
- ✅ Llama (Llama 1, 2, 3, Mistral, Mixtral)
- ✅ Qwen (Qwen 1.5, 2, 2.5, 3)
- ✅ 其他基于 Llama 架构的模型

扩展新模型需要：
1. 实现适配器（~100 行代码）
2. 注册到适配器表
3. 测试验证

### 9.3 硬件兼容性

| 硬件 | 状态 | 备注 |
|------|------|------|
| CUDA (NVIDIA) | ✅ 计划支持 | Phase 2 |
| Ascend NPU | ✅ 计划支持 | Phase 5 |
| ROCm (AMD) | 🔄 未来支持 | Phase 7+ |
| Intel CPU/GPU | 🔄 未来支持 | 待定 |

### 9.4 与 LMCache 共存

- 可以同时使用 LMCache 和 Blend
- Blend 通过 LMCache cache provider 访问 LMCache
- LMCache 处理存储，Blend 处理混合逻辑

## 10. 性能预期

### 10.1 理论分析

假设场景：RAG 应用，3个文档 chunk，不同顺序组合

| 指标 | 无 Blend | 有 Blend (15%) | 改进 |
|------|---------|----------------|------|
| TTFT (首次) | 100ms | 100ms | 0% |
| TTFT (后续) | 100ms | ~30ms | 70% ↓ |
| GPU 内存 | 100% | 100% | 0% |
| Cache 复用 | 0% | 85% | +85% |

### 10.2 实际测量

测试环境：
- 模型: Llama-2-7b
- 硬件: NVIDIA A100
- 场景: 多文档 RAG

预期结果：
- TTFT 减少: 30-60%
- 吞吐量提升: 1.5-2x
- 质量损失: <1%（困惑度对比）

## 11. 风险与挑战

### 11.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Provider 接口设计不当 | 高 | 充分的原型验证，迭代设计 |
| 性能不达预期 | 中 | 性能测试和优化 |
| 硬件特定 bug | 中 | 早期测试，单元测试覆盖 |
| 与 vLLM 版本兼容性 | 中 | 版本锁定测试，CI 集成 |

### 11.2 实施挑战

1. **解耦复杂度**: LMCache Blend 深度耦合，需要仔细梳理依赖
   - **解决方案**: 使用 mock providers 测试核心逻辑，逐步迁移

2. **硬件差异**: 不同硬件的 KV 管理方式不同
   - **解决方案**: Provider 抽象层隔离差异

3. **测试覆盖**: 需要在多种硬件上测试
   - **解决方案**: CI/CD 集成，定期测试

## 12. 后续优化

### 12.1 Phase 7+ (未来功能)

1. **更多硬件支持**
   - ROCm (AMD GPU)
   - Intel CPU/GPU
   - TPU

2. **高级混合策略**
   - 自适应重算比例
   - 基于阈值的混合
   - 多层混合策略

3. **性能优化**
   - Kernel 融合
   - 异步执行
   - 分布式混合

4. **功能扩展**
   - 支持流式输入
   - 支持多模态
   - 支持批处理优化

## 13. 总结

本设计方案提供了一个完全解耦、硬件无关的 Blend 插件，具有以下优势：

1. **清晰架构**: Provider 抽象 + 适配器模式
2. **易于扩展**: 新增硬件/模型只需实现接口
3. **性能优先**: 最小化开销，最大化复用
4. **生产就绪**: 完整的测试和文档计划

通过分阶段实施，可以逐步验证和优化，确保项目成功。
