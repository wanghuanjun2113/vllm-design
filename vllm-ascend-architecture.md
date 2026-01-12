# vLLM 与 vLLM-Ascend 架构设计文档

## 1. 概述

### 1.1 vLLM 简介

vLLM 是一个高吞吐量、内存高效的 LLM 推理引擎，主要创新包括：
- **PagedAttention**: 高效的 KV cache 管理机制
- **Continuous Batching**: 动态批处理以最大化 GPU 利用率

### 1.2 vLLM-Ascend 简介

vLLM-Ascend 是**硬件插件**（非 fork），通过 vLLM 的平台插件接口（RFC #11162）将 vLLM 扩展到华为昇腾 NPU。保持与主 vLLM 项目完全的 API 兼容性。

---

## 2. vLLM 核心架构

### 2.1 主要模块

```
┌─────────────────────────────────────────────────────────────────┐
│                      Entry Points (入口层)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Python API   │  │   CLI        │  │ OpenAI-Compatible    │  │
│  │ (LLM class)  │  │  (vllm cmd)  │  │ API Server (FastAPI) │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Engine Layer (引擎层)                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    LLMEngine                                │ │
│  │  ┌─────────────────┐  ┌─────────────────┐                 │ │
│  │  │ InputProcessor  │  │ OutputProcessor │                 │ │
│  │  └─────────────────┘  └─────────────────┘                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   EngineCore (核心调度层)                         │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │   Scheduler     │  │ KVCacheManager  │                      │
│  │ (请求调度/批处理) │  │ (PagedAttention)│                      │
│  └─────────────────┘  └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Executor (执行器层)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ UniProc      │  │ MultiProc    │  │ RayDistributed       │  │
│  │ (单进程)      │  │ (多进程)      │  │ (多节点)              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Worker (工作进程层)                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    GPUModelRunner                           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │ │
│  │  │Attention     │  │ Model        │  │ Sampler        │  │ │
│  │  │Backend       │  │Forward Pass  │  │(Token Gen)     │  │ │
│  │  └──────────────┘  └──────────────┘  └────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Model Executor (模型执行层)                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  300+ Models (LLaMA, Mixtral, Qwen, GPT, etc.)            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键文件位置

| 组件 | 文件路径 | 功能描述 |
|------|----------|----------|
| **入口层** | | |
| Python API | `vllm/entrypoints/llm.py` | LLM 类，离线推理 API |
| OpenAI API | `vllm/entrypoints/openai/api_server.py` | RESTful API 服务器 |
| **引擎层** | | |
| LLMEngine | `vllm/v1/engine/llm_engine.py` | 主引擎实现 |
| EngineCore | `vllm/v1/engine/core.py` | 核心调度循环 |
| InputProcessor | `vllm/v1/engine/input_processor.py` | 输入处理（分词、验证） |
| OutputProcessor | `vllm/v1/engine/output_processor.py` | 输出处理（去分词、格式化） |
| **调度层** | | |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | 请求调度、连续批处理 |
| KVCacheManager | `vllm/v1/core/kv_cache_manager.py` | PagedAttention 内存管理 |
| **执行器层** | | |
| UniProcExecutor | `vllm/v1/executor/uniproc_executor.py` | 单进程执行器 |
| MultiprocExecutor | `vllm/v1/executor/multiproc_executor.py` | 多进程执行器 |
| RayExecutor | `vllm/v1/executor/ray_executor.py` | Ray 分布式执行器 |
| **工作层** | | |
| WorkerBase | `vllm/v1/worker/worker_base.py` | Worker 抽象基类 |
| GPUModelRunner | `vllm/v1/worker/gpu_model_runner.py` | GPU 模型执行 |
| **注意力层** | | |
| Backend | `vllm/v1/attention/backend.py` | 注意力后端抽象 |
| FlashAttention | `vllm/v1/attention/backends/flash_attn.py` | FlashAttention 实现 |
| **配置系统** | | |
| VllmConfig | `vllm/config/vllm.py` | 中央配置对象 |

### 2.3 主要执行流程

```
用户请求 (generate / API 调用)
         ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. InputProcessor 处理                                       │
│    - Tokenize 提示词                                         │
│    - 验证采样参数                                            │
│    - 处理多模态输入                                          │
│    - 设置 LoRA                                               │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Scheduler 调度                                            │
│    - 从等待队列选择请求                                       │
│    - 分配 KV cache 块 (PagedAttention)                        │
│    - 形成执行批次                                             │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Worker 模型执行                                           │
│    - 准备输入批次                                             │
│    - 执行模型前向传播                                         │
│    - 计算 PagedAttention                                      │
│    - 采样生成 token                                           │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. OutputProcessor 处理输出                                   │
│    - Detokenize tokens                                       │
│    - 格式化响应                                               │
│    - 处理流式输出                                             │
└─────────────────────────────────────────────────────────────┘
         ↓
返回响应给用户
```

**连续批处理循环**:
```
迭代 N:
1. Schedule: 添加新请求，移除完成的请求
2. Execute: 在当前批次上运行模型
3. Output: 返回生成的 token
4. Repeat: 用更新后的批次重复
```

### 2.4 核心技术创新

#### PagedAttention

```
传统 KV Cache (固定分配):
┌─────────────────────────────────┐
│ Request 1: ████████████████████  │  浪费空间
└─────────────────────────────────┘

PagedAttention (按页分配):
┌─────────────────────────────────┐
│ Request 1: [Block1][Block2][Block3][Block4/16]  │
│ Request 2:         [Block5][Block6/16]           │
│ Request 3:                 [Block7/16]           │
└─────────────────────────────────┘

- 块大小: 通常 16 个 token
- 页表: 每个请求维护逻辑位置到物理块的映射
- 动态管理: 按需分配/释放块
```

**优势**:
- 消除内存碎片
- 支持灵活的批次大小
- 通过块交换支持超长序列
- 前缀缓存减少重复计算

#### Continuous Batching

```
静态批处理 (传统):
批次 1: [Req1(100), Req2(100), Req3(100)] → 等待全部完成
批次 2: [Req4(100), Req5(100), Req6(100)] → 等待全部完成

连续批处理 (vLLM):
步骤 1: [Req1(100), Req2(100), Req3(100)] → 各生成 1 token
步骤 2: [Req2(101), Req3(101), Req4(NEW-100)] → Req1 完成，添加 Req4
步骤 3: [Req2(102), Req5(NEW-100), Req6(NEW-100)] → Req3 完成
```

**优势**:
- 最大化 GPU 利用率
- 减少短请求延迟
- 高效处理变长请求

---

## 3. vLLM 硬件抽象层设计

### 3.1 平台抽象架构

vLLM 通过插件系统实现硬件抽象，核心是 `Platform` 接口：

```python
# vllm/platforms/interface.py
class Platform:
    _enum: PlatformEnum           # 平台枚举
    device_name: str              # 设备名称
    device_type: str              # 设备类型 (cuda, npu, etc.)
    dispatch_key: str             # PyTorch 分发 key
    ray_device_key: str           # Ray 资源 key
    dist_backend: str             # 分布式后端 (nccl, hccl, etc.)

    # 关键抽象方法
    @classmethod
    def get_device_capability(cls, device: int) -> Tuple[int, int]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, config)

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig)
```

### 3.2 硬件抽象层次

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│                  (User Code / APIs)                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Configuration Layer                           │
│              (VllmConfig - Platform Agnostic)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Platform Abstraction                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  CUDA    │  │  ROCm    │  │   XPU    │  │  OOT (Ascend)│   │
│  │Platform  │  │Platform  │  │Platform  │  │    Platform   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Backend Implementations                       │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ Attention Backend│  │ Device           │                   │
│  │ (FlashAttn,      │  │ Communicator     │                   │
│  │  FlashInfer,     │  │ (NCCL, HCCL,     │                   │
│  │  Custom)         │  │  Gloo, etc.)     │                   │
│  └──────────────────┘  └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Hardware Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ NVIDIA   │  │   AMD    │  │  Intel   │  │  Huawei      │   │
│  │   GPU    │  │   GPU    │  │   XPU    │  │   NPU        │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 平台发现机制

```python
# vllm/platforms/__init__.py

# 内置平台
_BUILTIN_PLATFORMS = [
    ("cuda", CudaPlatform),
    ("rocm", RocmPlatform),
    ("tpu", TpuPlatform),
    ("xpu", XpuPlatform),
    ("cpu", CpuPlatform),
]

# 插件平台 (通过 entry_points 发现)
_EXTERNAL_PLATFORMS = discover_platform_plugins()

# 平台选择流程:
# 1. 检查环境变量 VLLM_TARGET_PLATFORM
# 2. 自动检测可用硬件
# 3. 只能有一个活动平台
```

---

## 4. vLLM-Ascend 架构设计

### 4.1 整体架构

vLLM-Ascend 作为**插件**集成到 vLLM：

```
┌─────────────────────────────────────────────────────────────────┐
│                        vLLM Core                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           Platform Plugin Interface (RFC #11162)           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                          ↕ Plugin API
┌─────────────────────────────────────────────────────────────────┐
│                    vLLM-Ascend Plugin                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   NPU        │  │   NPU        │  │  Custom Ops          │  │
│  │  Platform    │  │   Worker     │  │  (C++ Kernels)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Attention   │  │ Distributed  │  │  Quantization        │  │
│  │  Backends    │  │  (HCCL)      │  │  Schemes             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────────────┐
│                    Ascend Hardware                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Atlas 800I  │  │  Atlas A2    │  │  Atlas 300I Duo     │  │
│  │    A2 (910B) │  │  (910B)      │  │   (310P)             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 vLLM-Ascend 主要模块

| 模块 | 文件路径 | 功能描述 |
|------|----------|----------|
| **平台层** | | |
| NPUPlatform | `vllm-ascend/vllm_ascend/platform.py` | 昇腾 NPU 平台实现 |
| **Worker 层** | | |
| NPUWorker | `vllm-ascend/vllm_ascend/worker/worker.py` | NPU Worker 实现 |
| NPUModelRunner | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` | v1 引擎模型运行器 |
| **注意力层** | | |
| AscendAttentionBackend | `vllm-ascend/vllm_ascend/attention/attention_v1.py` | 主注意力实现 |
| AscendMLABackend | `vllm-ascend/vllm_ascend/attention/mla_v1.py` | 多头潜在注意力 |
| AscendSFABackend | `vllm-ascend/vllm_ascend/attention/sfa_v1.py` | 稀疏 Flash Attention |
| **C++ 内核** | | |
| Sparse Flash Attention | `vllm-ascend/csrc/sparse_flash_attention/` | 稀疏注意力内核 |
| MoE Kernels | `vllm-ascend/csrc/moe_*/` | MoE 路由/分发/合并内核 |
| MLA Kernels | `vllm-ascend/csrc/mla_preprocess/` | MLA 预处理内核 |
| **分布式层** | | |
| NPUCommunicator | `vllm-ascend/vllm_ascend/distributed/communicator.py` | NPU 通信器 |
| PyHCCL | `vllm-ascend/vllm_ascend/distributed/pyhccl.py` | 华为集合通信库封装 |
| **量化层** | | |
| W8A8/W4A8 | `vllm-ascend/vllm_ascend/quantization/` | 8-bit/4-bit 量化方案 |
| **编译层** | | |
| AscendCompiler | `vllm-ascend/vllm_ascend/compilation/compiler_interface.py` | ACL Graph 编译器 |
| Graph Fusion | `vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py` | 图融合优化 |

### 4.3 插件注册机制

```python
# vllm-ascend/setup.py
entry_points = {
    # 平台插件
    "vllm.platform_plugins": [
        "ascend = vllm_ascend:register"
    ],
    # 通用插件
    "vllm.general_plugins": [
        "ascend_kv_connector = vllm_ascend:register_connector",
        "ascend_model_loader = vllm_ascend:register_model_loader",
        "ascend_service_profiling = vllm_ascend:register_service_profiling"
    ],
}

# vllm_ascend/__init__.py
def register() -> str | None:
    """平台插件入口函数"""
    if is_ascend_available():
        return "vllm_ascend.platform.NPUPlatform"
    return None
```

### 4.4 vLLM-Ascend 相比 vLLM 的主要变化

```
┌─────────────────────────────────────────────────────────────────┐
│                    vLLM (CUDA 原生)                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ CudaPlatform                                               │ │
│  │  - device_type = "cuda"                                    │ │
│  │  - dispatch_key = "CUDA"                                   │ │
│  │  - dist_backend = "nccl"                                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ GPUModelRunner                                             │ │
│  │  - torch.cuda                                              │ │
│  │  - FlashAttention / FlashInfer                             │ │
│  │  - CUDA Graphs                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ CudaCommunicator (NCCL)                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

                            ↓ 替换为

┌─────────────────────────────────────────────────────────────────┐
│                    vLLM-Ascend (NPU 适配)                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ NPUPlatform (继承 Platform)                                │ │
│  │  - device_type = "npu"                                     │ │
│  │  - dispatch_key = "NPU"                                    │ │
│  │  - dist_backend = "hccl"                                   │ │
│  │  + 支持 ACL Graph 编译模式                                 │ │
│  │  + 自定义算子注册                                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ NPUModelRunner (继承 WorkerBase)                           │ │
│  │  - torch.npu (torch-npu)                                   │ │
│  │  - AscendAttentionBackend (Sparse Flash Attention)         │ │
│  │  - ACL Graph 编译优化                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ NPUCommunicator (HCCL)                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ C++ Custom Ops (昇腾 NPU 优化内核)                          │
│  │  - Sparse Flash Attention                                  │
│  │  - MoE (gating, dispatch, combine)                         │
│  │  - MLA preprocessing                                        │
│  │  - Communication ops                                       │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 昇腾硬件适配的关键设计决策

### 5.1 为什么采用插件架构？

**原因 1: 代码维护性**
- vLLM 主仓库专注 CUDA 生态
- 昇腾适配独立开发，不阻塞主分支
- 减少合并冲突

**原因 2: 发布周期独立**
- vLLM-Ascend 可独立发版
- 快速响应昇腾硬件/软件更新
- 不受 vLLM 发布周期约束

**原因 3: 硬件特性隔离**
- 昇腾特有功能（ACL Graph 编译）不影响 CUDA 代码
- 避免硬件特定代码污染主仓库
- 清晰的抽象边界

### 5.2 为什么需要 C++ 自定义算子？

**原因 1: 昇腾 NPU 架构差异**
```
NVIDIA GPU:
- CUDA Core + Tensor Core
- 统一内存架构
- NVENC/NVDEC 专用单元

Huawei NPU:
- AI Core (矩阵计算)
- Vector Core (向量计算)
- Cube Unit (立方体计算)
- 独立内存层次 (HBM->L3->L2->L1)
```

**原因 2: 算子优化需求**
- FlashAttention 不直接支持 NPU
- 需要实现 Sparse Flash Attention (SFA)
- MoE 模型的路由/分发需要定制内核
- MLA (Multi-head Latent Attention) 需要特殊预处理

**原因 3: 性能优化**
- 昇腾 CANN 算子库可能有性能限制
- 自定义内核可针对特定模型优化
- 支持 ACL Graph 编译模式进行图融合

### 5.3 为什么使用 ACL Graph 编译？

**ACL Graph** (Ascend Computing Language Graph) 是昇腾的计算图编译优化技术：

```
Eager 模式:
┌─────────────────────────────────────────────────────────────────┐
│  Op1 → Op2 → Op3 → Op4 → Op5 → Op6 → Op7 → Op8                 │
│  每次调用都通过 Python → CANN → NPU                            │
│  开销大，调度延迟高                                             │
└─────────────────────────────────────────────────────────────────┘

ACL Graph 模式:
┌─────────────────────────────────────────────────────────────────┐
│  [Op1+Op2+Op3] → [Op4+Op5] → [Op6+Op7+Op8]                    │
│  编译时融合，运行时执行融合后的算子                              │
│  减少调度开销，提升性能                                          │
└─────────────────────────────────────────────────────────────────┘
```

**优势**:
- 图融合减少 kernel launch 次数
- 内存访问优化
- 算子间并行化
- 减少内存拷贝

**vLLM-Ascend 编译模式**:
- `NONE`: 纯 Eager 模式
- `VLLM_COMPILE`: 选择性编译关键路径
- `FULL_DECODE_ONLY`: 完全编译解码阶段

### 5.4 为什么实现 Sparse Flash Attention？

**FlashAttention 的问题**:
- 为 NVIDIA GPU 优化，利用 CUDA 特性
- NPU 内存层次不同，直接移植性能差

**Sparse Flash Attention (SFA) 设计**:
```
标准 Attention:
Q @ K^T → Softmax → @ V
O(N²) 内存访问

Sparse Flash Attention:
- 分块计算 (Tiling)
- 注意力矩阵稀疏化
- 在线 Softmax (Online Softmax)
- 减少 HBM 访问

针对 NPU 优化:
- 利用 Vector Core 并行
- 适配 L1/L2 缓存大小
- 融合算子减少内存往返
```

### 5.5 为什么需要 MoE 自定义内核？

**MoE (Mixture of Experts) 关键操作**:
```
1. Gating: 计算每个 token 应该路由到哪些专家
2. Dispatch: 将 token 分发到对应专家
3. Compute: 各专家独立计算
4. Combine: 合并专家输出
```

**需要定制原因**:
- MoE 通信模式特殊 (All-to-All)
- 需要充分利用昇腾的集群通信能力 (HCCL)
- 动态路由需要高效实现
- 支持专家并行 (Expert Parallelism)

**vLLM-Ascend MoE 内核**:
- `moe_gating_top_k/`: Top-K 路由选择
- `moe_init_routing_custom/`: 路由表初始化
- `moe_dispatch_normal/`: Token 分发
- `moe_combine_normal/`: 输出合并

### 5.6 支持的昇腾硬件

| 系列型号 | 芯片 | 设备类型 | 状态 |
|---------|------|----------|------|
| Atlas 800I A2 | 910B | AscendDeviceType.A2 | 稳定 |
| Atlas A2 训练系列 | 910B | AscendDeviceType.A2 | 稳定 |
| Atlas 800I A3 | 910C | AscendDeviceType.A3 | 稳定 |
| Atlas A3 训练系列 | 910C | AscendDeviceType.A3 | 稳定 |
| Atlas 300I Duo | 310P | AscendDeviceType._310P | 实验性 |

---

## 6. 主要流程对比

### 6.1 初始化流程对比

```
vLLM (CUDA):
┌─────────────────────────────────────────────────────────────────┐
│ 1. VllmConfig 创建                                             │
│    ↓                                                           │
│ 2. current_platform = CudaPlatform                             │
│    ↓                                                           │
│ 3. CudaPlatform.check_and_update_config()                      │
│    - 选择 FlashAttention/FlashInfer 后端                        │
│    - 设置 NCCL 通信后端                                         │
│    ↓                                                           │
│ 4. LLMEngine 初始化                                            │
│    ↓                                                           │
│ 5. GPUModelRunner 初始化                                       │
│    - 加载模型到 CUDA                                            │
│    - 初始化 CUDA Graphs                                        │
│    ↓                                                           │
│ 6. CudaCommunicator 初始化 (NCCL)                              │
└─────────────────────────────────────────────────────────────────┘

vLLM-Ascend (NPU):
┌─────────────────────────────────────────────────────────────────┐
│ 1. VllmConfig 创建                                             │
│    ↓                                                           │
│ 2. current_platform = NPUPlatform (通过插件发现)                │
│    ↓                                                           │
│ 3. NPUPlatform.check_and_update_config()                       │
│    - 选择 AscendAttentionBackend (SFA)                          │
│    - 设置 HCCL 通信后端                                         │
│    - 配置 ACL Graph 编译模式                                    │
│    - 注册自定义算子                                             │
│    ↓                                                           │
│ 4. LLMEngine 初始化                                            │
│    ↓                                                           │
│ 5. NPUModelRunner 初始化                                       │
│    - 加载模型到 NPU (torch-npu)                                │
│    - 初始化 ACL Graph                                          │
│    - 编译优化关键算子                                           │
│    ↓                                                           │
│ 6. NPUCommunicator 初始化 (HCCL)                               │
│    - PyHCCL 初始化                                              │
│    - 建立设备间通信                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 推理流程对比

```
vLLM (CUDA):
┌─────────────────────────────────────────────────────────────────┐
│ 1. Scheduler.schedule()                                        │
│    - 分配 KV cache 块                                           │
│    - 形成输入批次                                               │
│    ↓                                                           │
│ 2. GPUModelRunner.execute_model()                              │
│    - 准备输入张量 (CUDA)                                        │
│    - FlashAttention 前向传播                                    │
│    - 采样生成 token                                             │
│    - 更新 KV cache                                              │
│    ↓                                                           │
│ 3. OutputProcessor.process_outputs()                           │
│    - Detokenize                                                │
│    - 返回结果                                                   │
└─────────────────────────────────────────────────────────────────┘

vLLM-Ascend (NPU):
┌─────────────────────────────────────────────────────────────────┐
│ 1. Scheduler.schedule()                                        │
│    - 分配 KV cache 块                                           │
│    - 形成输入批次                                               │
│    ↓                                                           │
│ 2. NPUModelRunner.execute_model()                              │
│    - 准备输入张量 (NPU)                                         │
│    - AscendAttentionBackend 前向传播                             │
│    │  - Sparse Flash Attention (C++ 内核)                       │
│    │  - MLA 预处理 (如果使用 MLA)                               │
│    - 采样生成 token                                             │
│    - 更新 KV cache                                              │
│    ↓                                                           │
│ 3. OutputProcessor.process_outputs()                           │
│    - Detokenize                                                │
│    - 返回结果                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 分布式训练/推理对比

```
vLLM (CUDA):
┌─────────────────────────────────────────────────────────────────┐
│ CudaCommunicator                                               │
│  - all_reduce: NCCL all_reduce                                 │
│  - all_gather: NCCL all_gather                                 │
│  - broadcast: NCCL broadcast                                   │
│                                                                 │
│ 支持:                                                          │
│  - Tensor Parallelism (TP)                                     │
│  - Pipeline Parallelism (PP)                                   │
│  - Data Parallelism (DP)                                       │
└─────────────────────────────────────────────────────────────────┘

vLLM-Ascend (NPU):
┌─────────────────────────────────────────────────────────────────┐
│ NPUCommunicator (PyHCCL)                                       │
│  - all_reduce: HCCL all_reduce                                 │
│  - all_gather: HCCL all_gather                                 │
│  - broadcast: HCCL broadcast                                   │
│                                                                 │
│ 支持相同并行策略:                                               │
│  - Tensor Parallelism (TP)                                     │
│  - Pipeline Parallelism (PP)                                   │
│  - Data Parallelism (DP)                                       │
│                                                                 │
│ 额外支持:                                                      │
│  - Expert Parallelism (EPLB) - MoE 专家并行                     │
│  - Context Parallelism - 长上下文并行                           │
│  - KV Pool - 分布式 KV cache 管理                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键技术特性对比

| 特性 | vLLM (CUDA) | vLLM-Ascend |
|------|-------------|-------------|
| **硬件抽象** | CudaPlatform | NPUPlatform |
| **设备类型** | `cuda` | `npu` (torch-npu) |
| **框架** | PyTorch + CUDA | PyTorch + torch-npu |
| **注意力实现** | FlashAttention, FlashInfer | Sparse Flash Attention (SFA) |
| **通信后端** | NCCL | HCCL (PyHCCL) |
| **图编译** | torch.compile, CUDA Graphs | ACL Graph 编译 |
| **C++ 内核** | CUDA kernels | 昇腾 C++ 自定义算子 |
| **MoE 支持** | 基础支持 | 自定义 MoE 内核 + EPLB |
| **量化方案** | 标准量化 | W8A8, W4A8, W4A4, MXFP8 等 |
| **KV cache** | PagedAttention | PagedAttention + CPU offload |
| **LoRA 支持** | Punica | Punica NPU 适配 |

---

## 8. 总结

### 8.1 vLLM 核心价值

1. **PagedAttention**: 高效 KV cache 管理
2. **Continuous Batching**: 动态批处理，最大化吞吐
3. **模块化设计**: 清晰的抽象层次
4. **插件架构**: 硬件无关的核心设计

### 8.2 vLLM-Ascend 适配策略

1. **插件集成**: 非侵入式硬件适配
2. **平台抽象**: 实现 NPUPlatform 接口
3. **算子定制**: C++ 自定义内核优化性能
4. **编译优化**: ACL Graph 图融合
5. **通信适配**: HCCL 替代 NCCL
6. **完整功能**: 支持 TP/PP/DP/MoE/量化等

### 8.3 架构优势

- **解耦设计**: 硬件特定代码与核心逻辑分离
- **可维护性**: 独立开发周期，减少冲突
- **可扩展性**: 易于添加新硬件支持
- **性能优化**: 硬件特定优化不受限制

---

## 9. 后续深入研究方向

基于此架构文档，可以深入研究的方向：

1. **PagedAttention 实现细节**: `vllm/v1/core/kv_cache_manager.py`
2. **Scheduler 调度策略**: `vllm/v1/core/sched/scheduler.py`
3. **Sparse Flash Attention 实现**: `vllm-ascend/csrc/sparse_flash_attention/`
4. **ACL Graph 编译流程**: `vllm-ascend/vllm_ascend/compilation/`
5. **HCCL 通信优化**: `vllm-ascend/vllm_ascend/distributed/pyhccl.py`
6. **MoE EPLB 机制**: `vllm-ascend/vllm_ascend/eplb/`
7. **量化方案实现**: `vllm-ascend/vllm_ascend/quantization/`
