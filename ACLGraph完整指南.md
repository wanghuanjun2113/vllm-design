# ACLGraph 完整指南

> **面向用户和开发者的ACLGraph技术文档**
>
> ACLGraph（Ascend Computing Language Graph）是vLLM-Ascend插件的核心优化技术，通过图编译机制大幅提升LLM推理性能。本文档面向用户和开发者，全面介绍ACLGraph的使用方法、工作原理和实现细节。

---

# 第一部分：用户指南

## 1. 简介

### 1.1 什么是ACLGraph？

ACLGraph是针对华为昇腾（Ascend）NPU硬件的图编译实现，相当于CUDA Graph在GPU上的作用。它通过将多个操作符（operator）捕获到一个计算图中，然后在推理时重放该图，从而大幅减少主机（CPU）与设备（NPU）之间的通信开销。

### 1.2 为什么需要ACLGraph？

在LLM推理中，每个token的生成都需要执行近千个算子。当主机启动算子的速度慢于设备执行速度时，会出现"主机受限"（host bound）问题。严重情况下，设备有一半以上的时间处于空闲状态。

**Eager模式 vs Graph模式对比：**

```
Eager模式（传统方式）：

主机:   | 启动op1  |  启动op2  |  启动op3  |  启动op4  |  启动op5  |

设备:                | 运行op1 |空闲| 运行op2 |空闲| 运行op3 |空闲| 运行op4 |空闲| 运行op5 |

        | <-----                           总时间                                 -----> |

Graph模式（优化方式）：

主机:   |  启动图  |

设备:                  | 运行op1 | 运行op2 | 运行op3 | 运行op4 | 运行op5 |

        | <-----                    总时间                      -----> |
```

从图中可以看出，Graph模式通过一次性启动整个计算图，消除了主机端重复启动算子的开销，显著减少了总执行时间。

### 1.3 ACLGraph的优势

- **性能提升**：首token延迟（TTFT）降低3-10倍
- **资源利用率**：显著提高NPU利用率，减少设备空闲时间
- **内存优化**：通过图池和弱引用机制优化内存使用
- **灵活模式**：支持多种图模式以适应不同场景

### 1.4 与CUDA Graph的对比

| 特性 | CUDA Graph (NVIDIA) | ACLGraph (Ascend) |
|------|---------------------|-------------------|
| 硬件平台 | NVIDIA GPU | 华为昇腾NPU |
| 软件栈 | CUDA | CANN (Compute Architecture for Neural Networks) |
| 通信库 | NCCL | HCCL (Huawei Collective Communication Library) |
| 图编译 | CUDA Graphs | ACL (Ascend Computing Language) |
| 内存分配器 | PyTorch默认 | 自定义camem_allocator |

---

## 2. 核心概念

### 2.1 图捕获（Capture）和图重放（Replay）

ACLGraph的工作原理分为两个阶段：

**捕获阶段（Capture）：**
- 引擎初始化时，捕获模型前向传播中的所有算子
- 为不同的批次大小创建多个图对象
- 将捕获的图存储在图池中供后续使用

**重放阶段（Replay）：**
- 推理请求到达时，直接重放预先捕获的图
- 避免重复计算相同操作，大幅降低开销

### 2.2 Padding和Bucketing策略

**问题**：计算图只能重放之前捕获的操作，且要求输入形状一致。但模型输入的形状取决于调度器的决策，无法保证一致性。

**解决方案**：不捕获最大形状（会导致大量冗余计算），而是捕获多个不同形状的图（bucket），将模型输入填充到最近的图大小。

**Bucketing策略示意图：**

```
|    graph1    |
|           graph2           |
|                    graph3                    |
|                              graph4                              |    # 阈值

| input1 | 填充 |    # 使用graph1
|           input2           |  # 不需要填充
|                      input3                      |  填充 |    # 使用graph4
|                                    input4                                    |    # 使用eager模式
```

**优化策略：**
1. 设置一个阈值
2. 当`num_scheduled_tokens`超过阈值时，使用eager模式
3. 在阈值范围内捕获多个不同大小的图

### 2.3 图模式类型

vLLM-Ascend支持以下图模式：

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| **FULL** | 整个模型捕获为单个图 | 性能最优，但兼容性有限 |
| **PIECEWISE** | 模型分割为子图（每个注意力层一个） | 兼容性和性能的平衡 |
| **FULL_DECODE_ONLY** | 仅解码操作使用图模式 | 同时处理prefill和decode的批次 |
| **NONE** | Eager模式（回退） | 图模式失败时的回退方案 |

**模式选择逻辑：**
- 当注意力算子可以在图中运行时，倾向于选择FULL模式以获得最佳性能
- 当FULL模式不可行时，使用PIECEWISE模式作为替代
- 当PIECEWISE模式性能不佳且FULL模式被阻塞时，分离prefill和decode，在**仅解码**情况下使用FULL图模式

> **注意**：由于流资源限制，目前PIECEWISE模式仅支持少量bucket，可能导致冗余计算，在某些情况下性能可能不如eager模式。

### 2.4 流资源管理

每个ACL图至少需要一个专用的流（stream）。当前限制：
- 最大图数量：**1800个**
- 总流数：2048（保留248个缓冲）

影响bucket数量的变量：
- **PIECEWISE图**：将模型分为`num_hidden_layers + 1`个子模块，每个子模块都是一个独立的图，消耗流资源
- **通信域**：每个通信域会增加一个图消耗的流
- **多流调用**：子模块中显式调用多流会消耗额外的流

---

## 3. 快速开始

### 3.1 默认行为

从v0.9.1rc1版本开始，在V1引擎中，vLLM-Ascend默认以图模式运行，以保持与vLLM相同的行为。只需确保`enforce_eager`未设置为`True`即可。

### 3.2 基本使用示例

#### 离线推理

```python
from vllm import LLM

# ACLGraph默认启用
model = LLM(model="path/to/Qwen2-7B-Instruct")
outputs = model.generate("Hello, how are you?")
```

#### 在线服务

```bash
vllm serve Qwen/Qwen2-7B-Instruct
```

就这么简单！ACLGraph会在后台自动处理图捕获和重放。

### 3.3 支持的图模式类型

vLLM-Ascend支持两种图模式：
- **ACLGraph**：默认模式，从v0.9.1rc1开始，Qwen和DeepSeek系列模型经过充分测试
- **XliteGraph**：openeuler xlite图模式，从v0.11.0开始，支持Llama、Qwen稠密系列模型和Qwen3-vl

### 3.4 回退到Eager模式

如果遇到问题，可以临时回退到eager模式：

```python
from vllm import LLM

model = LLM(model="some_model_weight", enforce_eager=True)
outputs = model.generate("Hello, how are you?")
```

或通过命令行：

```bash
vllm serve some_model_weight --enforce-eager
```

---

## 4. 配置选项

### 4.1 cudagraph_capture_sizes

控制捕获的图大小（bucket sizes）。默认配置针对昇腾硬件优化。

**配置示例：**

```python
from vllm import LLM

model = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    # 自定义捕获大小
    cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256]
)
```

**注意事项：**
- 较小的值提供更细粒度的匹配，但会增加图数量
- 较大的值减少图数量，但可能增加填充开销
- 系统会自动调整以确保不超过流资源限制

### 4.2 图模式选择

通过`CUDAGraphMode`配置图模式：

```python
from vllm import LLM
from vllm.config import CUDAGraphMode

model = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    # 显式设置图模式
    compilation_config={"cudagraph_mode": CUDAGraphMode.PIECEWISE}
)
```

### 4.3 序列并行调整

启用序列并行时，图捕获大小会自动调整以适应并行策略。

### 4.4 MTP（Multi-Token Prediction）配置

使用ACLGraph和MTP且`num_speculative_tokens > 1`时，需要显式设置`cudagraph_capture_sizes`。

---

## 5. 支持的模型和硬件

### 5.1 支持的模型

从v0.9.0rc1开始，以下模型经过充分测试：

| 模型系列 | 测试状态 | 备注 |
|---------|---------|------|
| **Qwen系列** | ✅ 充分测试 | 推荐使用 |
| **DeepSeek系列** | ✅ 充分测试 | 推荐使用 |
| **Llama系列** | ⚠️ 有限支持 | 可能需要调整 |
| **Qwen3-VL** | ⚠️ 有限支持 | 实验性支持 |

### 5.2 硬件要求

**支持的硬件变体：**
- Atlas 800I A2/A3系列
- Atlas A2/A3训练系列
- Atlas 300I Duo
- 芯片变体：910B、910C、310P

### 5.3 软件栈要求

| 组件 | 版本要求 |
|------|----------|
| **CANN** | 8.3.rc2或更高版本 |
| **torch-npu** | 2.8.0或更高版本 |
| **HCCL** | 随CANN安装 |
| **ACL** | 随CANN安装 |

---

## 6. 性能优化

### 6.1 TTFT提升

ACLGraph可将首token延迟（TTFT）降低**3-10倍**，具体取决于：
- 模型大小和复杂度
- 批次大小
- 硬件配置

### 6.2 内存管理优化

- **弱引用**：使用弱引用管理图输出和工作空间，减少内存占用
- **图池**：通过全局图池重用计算图，避免重复分配
- **工作空间预分配**：预先计算并重用工作空间张量

### 6.3 异步操作

在异步调度或多线程场景中，使用`torch.npu.synchronize()`确保正确的执行顺序：

```python
# 在重放图之前同步，确保参数更新完成
torch.npu.synchronize()
entry.aclgraph.replay()
```

### 6.4 注意力参数更新

对于Full图模式，注意力参数需要在执行前更新：
- **普通注意力**：`update_attn_params()`
- **MLA注意力**：`update_mla_attn_params()`
- **DCP/PCP注意力**：`update_attn_dcp_pcp_params()`

使用`torch.npu.ExternalEvent`确保参数更新和算子执行之间的顺序。

---

## 7. 故障排除

### 7.1 常见问题

**Q1: ACLGraph捕获失败**

**症状**：启动时报错或性能下降

**解决方案**：
1. 检查硬件和软件栈版本
2. 尝试回退到eager模式：`enforce_eager=True`
3. 检查日志中的详细错误信息

**Q2: 内存不足**

**症状**：OOM错误

**解决方案**：
1. 减少`cudagraph_capture_sizes`中的bucket数量
2. 使用较小的批次大小
3. 启用弱引用模式（默认启用）

**Q3: 性能不如预期**

**症状**：性能提升不明显或性能下降

**解决方案**：
1. 检查是否使用了正确的图模式
2. 对于PIECEWISE模式，可能bucket数量不足导致过多填充
3. 对于大批次，可能触发了eager模式阈值
4. 确认模型在支持列表中

### 7.2 限制和约束

当前版本的限制：

1. **FULL和FULL_AND_PIECEWISE模式**：暂不完全支持
2. **MTP + num_speculative_tokens > 1**：需要显式设置`cudagraph_capture_sizes`
3. **use_inductor**：暂不支持
4. **流资源限制**：最多1800个图

### 7.3 调试技巧

启用调试日志：

```bash
export VLLM_LOGGING_LEVEL=DEBUG
```

这会启用：
- 图捕获日志
- 输入地址验证（捕获和重放时）
- 详细的性能指标

### 7.4 回退机制

如果ACLGraph失败，系统会自动回退到eager模式。也可以手动启用：

```python
from vllm import LLM

model = LLM(
    model="your_model",
    enforce_eager=True  # 强制使用eager模式
)
```

---

# 第二部分：开发者指南

## 1. 架构设计

### 1.1 整体架构

ACLGraph是vLLM图编译模式在昇腾NPU上的实现，整体架构如下：

```
┌─────────────────────────────────────────────────────────┐
│                      vLLM V1 Engine                     │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Model (with @support_torch_compile)      │  │
│  │                                                  │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │        ACLGraphWrapper                     │  │  │
│  │  │                                            │  │  │
│  │  │  ┌──────────────┐      ┌──────────────┐   │  │  │
│  │  │  │   Capture    │      │    Replay    │   │  │  │
│  │  │  │              │      │              │   │  │  │
│  │  │  │ torch.npu.   │      │ aclgraph.    │   │  │  │
│  │  │  │   graph()    │      │   replay()   │   │  │  │
│  │  │  └──────────────┘      └──────────────┘   │  │  │
│  │  │                                            │  │  │
│  │  │  ┌────────────────────────────────────┐   │  │  │
│  │  │  │     Graph Pool (global)            │   │  │  │
│  │  │  │  - NPUGraph instances              │   │  │  │
│  │  │  │  - Output tensors                  │   │  │  │
│  │  │  │  - Workspace tensors               │   │  │  │
│  │  │  └────────────────────────────────────┘   │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │         NPUPlatform                             │  │
│  │  - get_static_graph_wrapper_cls()               │  │
│  │  - get_global_graph_pool()                      │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Ascend NPU (CANN Stack)                    │
│                                                         │
│  - ACL (Ascend Computing Language)                     │
│  - HCCL (Huawei Collective Communication Library)      │
│  - torch-npu                                           │
└─────────────────────────────────────────────────────────┘
```

### 1.2 核心组件关系

```
ACLGraphEntry (数据结构)
        │
        │ 存储
        ▼
ACLGraphWrapper (核心包装器)
        │
        │ 使用
        ▼
Graph Pool (全局图池)
        │
        │ 管理
        ▼
NPUGraph (PyTorch NPU图对象)
```

### 1.3 与vLLM主干的集成点

**集成点1：平台注册**
```python
# vllm_ascend/platform.py
@classmethod
def get_static_graph_wrapper_cls(cls) -> str:
    return "vllm_ascend.compilation.acl_graph.ACLGraphWrapper"
```

**集成点2：模型包装**
```python
# vLLM使用support_torch_compile装饰器自动包装模型
@support_torch_compile
class Model:
    def forward(self, ...):
        # forward方法会被ACLGraphWrapper拦截
```

**集成点3：前向上下文**
```python
# 通过ForwardContext传递运行时信息
forward_context.batch_descriptor
forward_context.cudagraph_runtime_mode
```

---

## 2. 核心组件详解

### 2.1 ACLGraphEntry

**位置**：`vllm_ascend/compilation/acl_graph.py:28-36`

**数据结构**：
```python
@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor  # 批次描述符（图的关键）
    aclgraph: Optional[torch.npu.NPUGraph] = None  # 捕获的ACL图
    output: Optional[Any] = None  # 图的输出
    input_addresses: Optional[list[int]] = None  # 调试：输入地址
```

**作用**：
- 存储单个批次描述符对应的图信息
- 跟踪输入地址用于调试（DEBUG模式）
- 管理图输出引用

### 2.2 ACLGraphWrapper

**位置**：`vllm_ascend/compilation/acl_graph.py:38-203`

**核心职责**：
1. 包装可运行对象（runnable），添加图捕获/重放能力
2. 管理图池和具体图条目
3. 根据运行时模式分发（FULL、PIECEWISE、NONE）
4. 实现图捕获和重放逻辑

**初始化流程**：
```python
def __init__(self, runnable, vllm_config, runtime_mode, cudagraph_options):
    self.runnable = runnable  # 被包装的模型/层
    self.runtime_mode = runtime_mode  # FULL或PIECEWISE
    self.graph_pool = current_platform.get_global_graph_pool()
    self.concrete_aclgraph_entries = {}  # BatchDescriptor -> ACLGraphEntry
```

**调用流程**：
```python
def __call__(self, *args, **kwargs):
    # 1. 获取前向上下文
    forward_context = get_forward_context()
    batch_descriptor = forward_context.batch_descriptor
    aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

    # 2. 模式检查
    if aclgraph_runtime_mode != self.runtime_mode:
        return self.runnable(*args, **kwargs)  # 直接调用

    # 3. 获取或创建图条目
    if batch_descriptor not in self.concrete_aclgraph_entries:
        self.concrete_aclgraph_entries[batch_descriptor] = \
            ACLGraphEntry(batch_descriptor=batch_descriptor)

    entry = self.concrete_aclgraph_entries[batch_descriptor]

    # 4. 捕获或重放
    if entry.aclgraph is None:
        return self._capture(entry, *args, **kwargs)  # 捕获
    else:
        return self._replay(entry)  # 重放
```

**捕获实现**（`_capture`方法的简化）：
```python
def _capture(self, entry, *args, **kwargs):
    # 1. 验证捕获合法性
    validate_cudagraph_capturing_enabled()

    # 2. 记录输入地址（调试）
    entry.input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]

    # 3. 创建图对象
    aclgraph = torch.npu.NPUGraph()

    # 4. 捕获图
    with torch.npu.graph(aclgraph, pool=self.graph_pool):
        output = self.runnable(*args, **kwargs)
        output = weak_ref_tensors(output)  # 弱引用优化

    # 5. 存储图
    entry.output = weak_ref_tensors(output)
    entry.aclgraph = aclgraph

    return output  # 返回实际输出（非弱引用）
```

**重放实现**（`_replay`方法的简化）：
```python
def _replay(self, entry):
    # 1. 调试模式：验证输入地址
    if self.is_debugging_mode:
        new_input_addresses = [...]
        assert new_input_addresses == entry.input_addresses

    # 2. 同步（确保参数更新完成）
    torch.npu.synchronize()

    # 3. 重放图
    entry.aclgraph.replay()

    # 4. 返回缓存的输出
    return entry.output
```

### 2.3 AscendCompiler

**位置**：`vllm_ascend/compilation/compiler_interface.py:116-140`

**职责**：提供昇腾特定的编译接口

**编译模式**：
```python
class AscendCompiler:
    def npugraph_ex_compile(self, graph: torch.fx.Graph, args):
        # 使用torch-air后端的增强ACL图编译
        pass

    def fusion_pass_compile(self, graph: torch.fx.Graph, args):
        # 基本融合pass编译
        pass
```

### 2.4 AclGraphManager

**位置**：`vllm_ascend/worker/v2/aclgraph_utils.py:36-61`

**职责**：
- 继承自`CudaGraphManager`
- 处理NPU特定的图捕获
- 准备NPU输入数据

**关键方法**：
```python
class AclGraphManager(CudaGraphManager):
    def capture(self, *args, **kwargs):
        # NPU特定的输入准备
        # 调用父类捕获方法
        pass
```

---

## 3. 工作原理

### 3.1 捕获阶段详解

**触发时机**：
- 引擎初始化后的第一次前向传播
- 遇到新的`BatchDescriptor`（新的批次大小）

**捕获流程**：
```
1. 前向传播开始
   │
   ▼
2. ForwardContext创建，设置batch_descriptor和cudagraph_runtime_mode
   │
   ▼
3. ACLGraphWrapper.__call__被调用
   │
   ▼
4. 检查batch_descriptor是否存在于concrete_aclgraph_entries
   │
   ▼
5. 不存在 → 进入捕获流程
   │
   ├─► 创建ACLGraphEntry
   │
   ├─► 创建torch.npu.NPUGraph对象
   │
   ├─► 进入torch.npu.graph()上下文
   │   │
   │   └─► 执行self.runnable(*args, **kwargs)
   │       │
   │       └─► 记录所有算子到图中
   │
   ├─► 退出上下文，图捕获完成
   │
   ├─► 使用弱引用保存输出和工作空间
   │
   └─► 存储到concrete_aclgraph_entries
```

**内存优化**：
```python
# 使用弱引用减少内存占用
output = weak_ref_tensors(output)
weak_ref_workspaces(_graph_params)
```

### 3.2 重放阶段详解

**触发时机**：
- 推理请求到达
- `BatchDescriptor`已存在于缓存中

**重放流程**：
```
1. 前向传播开始
   │
   ▼
2. ForwardContext创建
   │
   ▼
3. ACLGraphWrapper.__call__被调用
   │
   ▼
4. 检查batch_descriptor是否存在于缓存
   │
   ▼
5. 存在 → 进入重放流程
   │
   ├─► 调试模式：验证输入地址一致性
   │
   ├─► torch.npu.synchronize()（确保参数更新完成）
   │
   ├─► entry.aclgraph.replay()
   │
   └─► 返回entry.output（弱引用）
```

**同步机制**：
在异步调度或多线程场景中，可能出现迭代i的CPU记录事件在迭代i-1的图重放之前完成的情况。为确保正确顺序，需要在重放前调用`torch.npu.synchronize()`。

### 3.3 Padding和Bucketing实现

**Bucket大小配置**：
```python
cudagraph_capture_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
```

**Padding策略**：
```python
def get_nearest_bucket_size(num_tokens, capture_sizes):
    # 找到最接近且不小于num_tokens的bucket
    for size in sorted(capture_sizes):
        if size >= num_tokens:
            return size
    return None  # 超过最大bucket，使用eager模式
```

**填充实现**（在模型输入准备阶段）：
```python
# 假设需要处理的token数量是10
nearest_bucket = 16

# 将输入填充到16
padded_input = torch.nn.functional.pad(input, (0, nearest_bucket - num_tokens))
```

### 3.4 流资源分配策略

**流计算**：
```
每个图所需流数 = 1（基础）+ comm_domains数量 + 显式多流调用数
```

**Bucket限制计算**（`update_aclgraph_sizes`函数）：
```python
def update_aclgraph_sizes(capture_sizes, num_layers, num_comm_domains):
    # 每层一个图（PIECEWISE模式）
    graphs_per_bucket = num_layers + 1

    # 每个图消耗的流
    streams_per_graph = 1 + num_comm_domains

    # 最大可用流
    max_streams = 2048 - 248  # 保留缓冲

    # 最大bucket数
    max_buckets = max_streams // (graphs_per_bucket * streams_per_graph)

    # 调整capture_sizes
    return capture_sizes[:max_buckets]
```

---

## 4. 关键数据结构

### 4.1 GraphParams

**位置**：`vllm_ascend/compilation/acl_graph.py:514-518`

**数据结构**：
```python
@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]  # 同步事件
    workspaces: dict[int, torch.Tensor]  # 工作空间张量
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]  # 任务句柄
    attn_params: dict[int, list[tuple]]  # 注意力参数
```

**作用**：
- 存储Full图模式所需的注意力参数
- 管理同步事件和工作空间
- 支持主模型和草稿模型（投机解码）

**初始化**：
```python
def set_graph_params(aclgraph_capture_sizes: list[int]):
    global _graph_params
    _graph_params = GraphParams(
        events={size: [] for size in aclgraph_capture_sizes},
        workspaces={size: None for size in aclgraph_capture_sizes},
        handles={size: [] for size in aclgraph_capture_sizes},
        attn_params={size: [] for size in aclgraph_capture_sizes},
    )
```

### 4.2 图池管理

**全局图池**：
```python
# 通过平台获取全局图池
graph_pool = current_platform.get_global_graph_pool()

# 捕获时指定图池
with torch.npu.graph(aclgraph, pool=self.graph_pool):
    output = self.runnable(*args, **kwargs)
```

**图池的作用**：
- 共享内存资源
- 优化图对象生命周期
- 支持图对象重用

### 4.3 弱引用内存管理

**弱引用工具函数**：
```python
def weak_ref_tensors(tensor):
    # 将张量转换为弱引用，减少内存占用
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(weak_ref_tensors(t) for t in tensor)
    # ... 其他类型的弱引用转换
```

**使用场景**：
- 图输出：`entry.output = weak_ref_tensors(output)`
- 工作空间：`weak_ref_workspaces(_graph_params)`

**优势**：
- 减少内存占用
- 允许Python垃圾回收器及时释放内存
- 避免循环引用

### 4.4 工作空间预分配

**工作空间计算**（针对不同注意力类型）：

```python
# PagedAttention工作空间
workspace = torch_npu._npu_paged_attention_get_workspace(
    query=query,
    key_cache=key_cache,
    value_cache=value_cache,
    # ... 其他参数
)

# FusedInferAttention工作空间
workspace = graph_params.workspaces.get(runtime_shape)
```

**预分配时机**：
- 图捕获期间计算并存储
- 重放时直接重用，避免重复分配

---

## 5. 注意力参数更新

### 5.1 为什么需要更新注意力参数？

在Full图模式下，注意力算子的参数需要在每次执行前更新，因为：
- 序列长度在每次请求中不同
- Block table（块表）动态变化
- 注意力mask需要根据实际序列调整

### 5.2 update_attn_params()

**位置**：`vllm_ascend/compilation/acl_graph.py:319-324`

**作用**：根据注意力类型分发到不同的更新函数

```python
def update_attn_params(update_stream, forward_context, runtime_shape, vllm_config):
    if using_paged_attention(runtime_shape, vllm_config):
        _update_attn_pa_params(update_stream, forward_context, runtime_shape)
    else:
        _update_attn_fia_params(update_stream, forward_context, runtime_shape)
```

### 5.3 _update_attn_pa_params()

**位置**：`vllm_ascend/compilation/acl_graph.py:215-270`

**用途**：更新PagedAttention参数

**流程**：
```python
def _update_attn_pa_params(update_stream, forward_context, runtime_shape):
    graph_params = get_graph_params()

    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
            forward_context.attn_metadata,
            graph_params.attn_params[runtime_shape],
            graph_params.handles[runtime_shape],
            graph_params.events[runtime_shape],
        ):
            # 1. 解包参数
            (query, key_cache, value_cache, num_kv_heads, num_heads,
             scale, block_table, seq_lens, output) = param

            # 2. 更新序列长度
            seq_lens = forward_context.attn_metadata[key].seq_lens

            # 3. 获取工作空间（处理FULL_DECODE_ONLY模式的特殊情况）
            workspace = torch_npu._npu_paged_attention_get_workspace(...)

            # 4. 更新图任务
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu._npu_paged_attention(...)
            torch.npu.graph_task_update_end(update_stream)

            # 5. 记录事件
            event.record(update_stream)
```

### 5.4 _update_attn_fia_params()

**位置**：`vllm_ascend/compilation/acl_graph.py:273-316`

**用途**：更新FusedInferAttention参数

**关键点**：
- 处理Qwen3-next的linear_attn和self_attn分离
- 使用`npu_fused_infer_attention_score.out`
- 工作空间从`graph_params.workspaces`获取

### 5.5 update_mla_attn_params()

**位置**：`vllm_ascend/compilation/acl_graph.py:327-396`

**用途**：更新Multi-Head Latent Attention (MLA)参数

**特殊处理**：
```python
def update_mla_attn_params(update_stream, forward_context, runtime_shape,
                           speculative_config):
    # 1. 获取图参数
    if forward_context.is_draft_model:
        graph_params = get_draft_graph_params()
    else:
        graph_params = get_graph_params()

    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(...):
            # 2. 解包MLA参数
            (q_nope, k_nope, q_pe, k_pe, num_heads, num_kv_heads,
             input_layout, attn_mask, sparse_mode, scale, ...)

            # 3. 获取序列长度
            seq_lens_list = forward_context.attn_metadata[key].decode.seq_lens_list

            # 4. 处理MTP（Multi-Token Prediction）
            if speculative_config and speculative_config.method == "mtp":
                spec_multiple = speculative_config.num_speculative_tokens + 1
                seq_lens_list = [...]
                actual_seq_lengths = [...]

            # 5. 更新MLA注意力
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(...)
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)
```

### 5.6 update_attn_dcp_pcp_params()

**位置**：`vllm_ascend/compilation/acl_graph.py:399-458`

**用途**：更新DCP（Data Context Parallel）和PCP（Pipeline Context Parallel）注意力参数

**关键点**：
- 处理`num_computed_tokens_of_pcp_dcp`
- 根据dcp_size调整num_heads
- 填充序列长度以匹配runtime_shape

### 5.7 ExternalEvent同步机制

**用途**：确保参数更新在算子执行之前完成

**流程**：
```
1. 在参数更新流上记录事件
   event.record(update_stream)

2. 在图执行流上等待事件
   torch.npu.stream(wait_stream)
   event.wait(wait_stream)

3. 执行图
   aclgraph.replay()
```

**为什么需要同步？**
- 参数更新和图执行在不同的流上
- 需要确保参数更新完成后才能执行图
- ExternalEvent提供跨流同步机制

---

## 6. 集成点

### 6.1 平台注册

**位置**：`vllm_ascend/platform.py:425-429`

```python
class NPUPlatform(Platform):
    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        """获取piecewise后端类"""
        return "vllm_ascend.compilation.acl_graph.ACLGraphWrapper"
```

**vLLM调用点**：
```python
# vLLM代码中
wrapper_cls = current_platform.get_static_graph_wrapper_cls()
wrapper = import_obj(wrapper_cls)(...)
```

### 6.2 模型自动包装

**装饰器**：`@support_torch_compile`

**作用**：
- 替换模型的`__init__`方法
- 包装`forward`方法，使其通过ACLGraphWrapper执行

**示例**：
```python
@support_torch_compile
class Qwen2Model(nn.Module):
    def forward(self, ...):
        # 实际执行时，这个方法会被ACLGraphWrapper拦截
```

**包装后的执行流程**：
```
用户调用：model.forward(input)
    │
    ▼
ACLGraphWrapper.__call__(input)
    │
    ▼
检查batch_descriptor和runtime_mode
    │
    ▼
捕获或重放图
    │
    ▼
调用原始的model.forward(input)（捕获时）或返回缓存输出（重放时）
```

### 6.3 V1引擎集成

**集成点**：
1. **配置加载**：`VllmConfig.compilation_config`
2. **图池初始化**：`current_platform.init_graph_pool()`
3. **模型包装**：通过`@support_torch_compile`
4. **前向上下文**：`ForwardContext`传递运行时信息

**执行流程**：
```
V1 Engine初始化
    │
    ├─► 加载VllmConfig
    │
    ├─► 初始化图池
    │
    ├─► 加载模型（自动应用@support_torch_compile）
    │
    └─► 模型被ACLGraphWrapper包装

推理请求到达
    │
    ├─► 创建ForwardContext
    │   │
    │   ├─► batch_descriptor
    │   └─► cudagraph_runtime_mode
    │
    └─► 调用model.forward()
            │
            ▼
        ACLGraphWrapper.__call__()
            │
            ├─► 检查runtime_mode
            ├─► 捕获或重放图
            └─► 返回结果
```

### 6.4 与KV Cache Connector的交互

**LMCache集成**：
- KV缓存通过KV connector接口与vLLM V1集成
- ACLGraph不直接影响KV缓存操作
- 但需要注意图捕获时KV缓存的一致性

---

## 7. 扩展和开发

### 7.1 如何添加新的图模式

**步骤**：

1. **定义新的模式枚举**：
```python
# vllm/config.py
class CUDAGraphMode(Enum):
    FULL = "full"
    PIECEWISE = "piecewise"
    # YOUR_NEW_MODE = "your_new_mode"
```

2. **扩展ACLGraphWrapper**：
```python
# vllm_ascend/compilation/acl_graph.py
class ACLGraphWrapper:
    def __call__(self, *args, **kwargs):
        if aclgraph_runtime_mode == CUDAGraphMode.YOUR_NEW_MODE:
            # 实现新模式的逻辑
            pass
```

3. **更新模式选择逻辑**：
```python
# vllm_ascend/platform.py
def select_graph_mode(model_config, ...):
    # 添加新模式的选择逻辑
    pass
```

### 7.2 如何支持新的模型

**步骤**：

1. **确保模型使用`@support_torch_compile`装饰器**：
```python
@support_torch_compile
class YourModel(nn.Module):
    ...
```

2. **检查注意力兼容性**：
   - 确认注意力类型是否支持图模式
   - 可能需要实现新的参数更新函数

3. **测试图捕获**：
```python
from vllm import LLM

model = LLM(model="your_model", enforce_eager=False)
outputs = model.generate("test")
```

4. **如果失败，添加回退逻辑**：
```python
# 在模型配置中
SUPPORTED_MODELS = {
    "your_model": {
        "graph_mode": CUDAGraphMode.PIECEWISE,
        "capture_sizes": [1, 2, 4, 8, 16],
    }
}
```

### 7.3 性能调优指南

**调优维度**：

1. **Bucket大小配置**：
```python
# 较小的bucket → 更精确匹配，但更多图
cudagraph_capture_sizes = [1, 2, 4, 8, 16, 32]

# 较大的bucket → 更少图，但更多填充
cudagraph_capture_sizes = [16, 32, 64, 128, 256]
```

2. **图模式选择**：
```python
# 最佳性能 → FULL（如果兼容）
compilation_config = {"cudagraph_mode": CUDAGraphMode.FULL}

# 兼容性 → PIECEWISE
compilation_config = {"cudagraph_mode": CUDAGraphMode.PIECEWISE}
```

3. **内存优化**：
```python
# 启用弱引用（默认）
CUDAGraphOptions(weak_ref_output=True)

# 禁用GC（捕获期间）
CUDAGraphOptions(gc_disable=True)
```

4. **异步优化**：
```python
# 确保同步点正确
torch.npu.synchronize()  # 在重放前
```

**性能分析**：
```bash
# 启用性能日志
export VLLM_LOGGING_LEVEL=DEBUG

# 查看捕获的图数量
# 日志中搜索 "num_cudagraph_captured"
```

### 7.4 测试策略

**单元测试**：
```python
def test_aclgraph_capture():
    model = LLM(model="Qwen/Qwen2-7B-Instruct")
    outputs = model.generate("test")
    assert outputs is not None
```

**集成测试**：
```python
def test_aclgraph_with_batch_sizes():
    model = LLM(model="Qwen/Qwen2-7B-Instruct",
                cudagraph_capture_sizes=[1, 2, 4, 8])

    # 测试不同的批次大小
    for batch_size in [1, 2, 4, 8]:
        outputs = model.generate(["test"] * batch_size)
        assert len(outputs) == batch_size
```

**性能测试**：
```python
def test_aclgraph_performance():
    import time

    model = LLM(model="Qwen/Qwen2-7B-Instruct")

    # 预热
    model.generate("test")

    # 测试
    start = time.time()
    for _ in range(100):
        model.generate("test")
    elapsed = time.time() - start

    print(f"Average time: {elapsed/100:.4f}s")
```

---

## 8. 实现细节

### 8.1 文件路径和关键代码位置

| 文件 | 关键组件 | 行号 |
|------|---------|------|
| `vllm_ascend/compilation/acl_graph.py` | ACLGraphEntry | 28-36 |
| | ACLGraphWrapper | 38-203 |
| | GraphParams | 514-518 |
| | update_attn_params | 319-324 |
| | update_mla_attn_params | 327-396 |
| | update_attn_dcp_pcp_params | 399-458 |
| `vllm_ascend/compilation/compiler_interface.py` | AscendCompiler | 116-140 |
| `vllm_ascend/worker/v2/aclgraph_utils.py` | AclGraphManager | 36-61 |
| `vllm_ascend/platform.py` | get_static_graph_wrapper_cls | 425-429 |

### 8.2 线程安全考虑

**全局状态**：
```python
_graph_params: Optional[GraphParams] = None
_draft_graph_params: Optional[GraphParams] = None
```

**线程安全策略**：
- 图参数在初始化时设置，之后只读
- 图池使用PyTorch的内部同步机制
- ExternalEvent确保跨线程同步

**注意事项**：
- 不要在多个线程中同时修改`GraphParams`
- 使用`torch.npu.synchronize()`确保跨线程同步

### 8.3 内存管理策略

**弱引用使用**：
```python
# 图输出
entry.output = weak_ref_tensors(output)

# 工作空间
weak_ref_workspaces(_graph_params)
```

**内存生命周期**：
```
捕获阶段：
1. 创建图对象
2. 捕获算子到图中
3. 保存输出为弱引用
4. 释放强引用，允许GC

重放阶段：
1. 获取弱引用
2. 重放图
3. 返回弱引用（图池管理实际内存）
```

### 8.4 同步原语

**使用的同步机制**：

1. **torch.npu.synchronize()**：
   - 等待NPU上所有操作完成
   - 在图重放前调用

2. **torch.npu.ExternalEvent**：
   - 跨流同步
   - 用于参数更新和图执行之间的同步

3. **torch.npu.stream()**：
   - 指定执行流
   - 参数更新在专用流上进行

**同步流程示例**：
```python
# 参数更新流
with torch.npu.stream(update_stream):
    torch.npu.graph_task_update_begin(update_stream, handle)
    # 更新参数
    torch.npu.graph_task_update_end(update_stream)
    event.record(update_stream)

# 图执行流（默认）
event.wait()  # 等待参数更新完成
aclgraph.replay()
```

---

# 第三部分：参考

## API参考

### 主要类

#### ACLGraphEntry

```python
@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: Optional[torch.npu.NPUGraph] = None
    output: Optional[Any] = None
    input_addresses: Optional[list[int]] = None
```

#### ACLGraphWrapper

```python
class ACLGraphWrapper:
    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: Optional[CUDAGraphOptions] = None
    )

    def __call__(self, *args, **kwargs) -> Any

    def unwrap(self) -> Callable
```

#### GraphParams

```python
@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]
```

### 主要函数

#### 图参数管理

```python
def set_graph_params(aclgraph_capture_sizes: list[int]) -> None
def get_graph_params() -> Optional[GraphParams]
def set_draft_graph_params(aclgraph_capture_sizes: list[int]) -> None
def get_draft_graph_params() -> Optional[GraphParams]
def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor) -> None
```

#### 注意力参数更新

```python
def update_attn_params(
    update_stream: torch.npu.Stream,
    forward_context: ForwardContext,
    runtime_shape: int,
    vllm_config: VllmConfig
) -> None

def update_mla_attn_params(
    update_stream: torch.npu.Stream,
    forward_context: ForwardContext,
    runtime_shape: int,
    speculative_config: Optional[SpeculativeConfig]
) -> None

def update_attn_dcp_pcp_params(
    update_stream: torch.npu.Stream,
    forward_context: ForwardContext,
    runtime_shape: int
) -> None
```

---

## 配置参数说明

### VllmConfig相关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `compilation_config.cudagraph_mode` | CUDAGraphMode | PIECEWISE | 图模式 |
| `compilation_config.cudagraph_capture_sizes` | list[int] | 自动计算 | 捕获的图大小 |
| `enforce_eager` | bool | False | 强制使用eager模式 |

### CUDAGraphOptions

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `debug_log_enable` | bool | False | 启用调试日志 |
| `weak_ref_output` | bool | True | 使用弱引用 |
| `gc_disable` | bool | True | 捕获期间禁用GC |

---

## 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| **ACL** | Ascend Computing Language | 华为昇腾计算语言，类似CUDA |
| **CANN** | Compute Architecture for Neural Networks | 昇腾AI计算架构 |
| **HCCL** | Huawei Collective Communication Library | 华为集合通信库，类似NCCL |
| **NPUGraph** | NPU Graph | PyTorch NPU的图对象 |
| **Bucket** | Bucket | 预先捕获的不同大小的图 |
| **Padding** | Padding | 填充输入以匹配图大小 |
| **Capture** | Capture | 图捕获阶段 |
| **Replay** | Replay | 图重放阶段 |
| **TTFT** | Time To First Token | 首token延迟 |
| **MLA** | Multi-Head Latent Attention | 多头潜在注意力 |
| **DCP** | Data Context Parallel | 数据上下文并行 |
| **PCP** | Pipeline Context Parallel | 流水线上下文并行 |
| **MTP** | Multi-Token Prediction | 多token预测 |
| **GQA** | Grouped Query Attention | 分组查询注意力 |
| **ExternalEvent** | External Event | 跨流同步事件 |

---

## 相关资源

### 文档链接

- [vLLM官方文档](https://docs.vllm.ai/)
- [vLLM CUDA Graphs设计](https://docs.vllm.ai/en/latest/design/cuda_graphs.html)
- [vLLM-Ascend图模式用户指南](./source/user_guide/feature_guide/graph_mode.md)
- [vLLM-Ascend ACL Graph开发者指南](./source/developer_guide/feature_guide/ACL_Graph.md)

### 代码路径

- **核心实现**：`vllm_ascend/compilation/acl_graph.py`
- **编译器接口**：`vllm_ascend/compilation/compiler_interface.py`
- **工具函数**：`vllm_ascend/worker/v2/aclgraph_utils.py`
- **平台集成**：`vllm_ascend/platform.py`
- **配置定义**：`vllm/config.py`

### 相关RFC和提案

- RFC #11162: Platform Plugin Interface
- RFC #...: (相关设计文档)

---

## 附录

### A. 完整的使用示例

#### 示例1：基本使用

```python
from vllm import LLM, SamplingParams

# 初始化模型（ACLGraph默认启用）
llm = LLM(model="Qwen/Qwen2-7B-Instruct")

# 创建采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 生成文本
prompts = ["Hello, my name is", "The future of AI is"]
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
```

#### 示例2：自定义捕获大小

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    # 自定义捕获大小
    cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128]
)
```

#### 示例3：回退到eager模式

```python
from vllm import LLM

# 遇到问题时回退到eager模式
llm = LLM(
    model="some_model",
    enforce_eager=True
)
```

#### 示例4：调试模式

```python
import os
from vllm import LLM

# 启用调试日志
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

llm = LLM(model="Qwen/Qwen2-7B-Instruct")
outputs = llm.generate("test")
# 查看日志中的图捕获和地址验证信息
```

### B. 常见错误和解决方案

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `RuntimeError: out of memory` | 图太多导致内存不足 | 减少`cudagraph_capture_sizes` |
| `RuntimeError: graph capture failed` | 不兼容的操作 | 检查模型是否支持，回退到eager |
| `ValueError: invalid batch size` | 批次大小不在捕获范围内 | 调整`cudagraph_capture_sizes` |
| `AssertionError: Input addresses...` | 调试模式检测到地址不一致 | 检查输入准备逻辑 |

### C. 性能基准

（具体数据取决于硬件和模型配置，仅供参考）

| 模型 | 模式 | TTFT提升 | 吞吐量提升 |
|------|------|---------|-----------|
| Qwen2-7B | Eager | - | - |
| Qwen2-7B | PIECEWISE | 3-5x | 1.5-2x |
| Qwen2-7B | FULL | 5-10x | 2-3x |

---

**文档版本**: 1.0
**最后更新**: 2025年1月
**维护者**: vLLM-Ascend团队

---

如有问题或建议，请：
1. 查阅[vLLM-Ascend GitHub仓库](https://github.com/your-repo)
2. 提交Issue或Pull Request
3. 参与社区讨论
