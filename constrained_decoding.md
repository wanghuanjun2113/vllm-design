# vLLM 约束解码设计文档

## 1. 概述 (Overview)

### 1.1 什么是约束解码

约束解码（Constrained Decoding），在vLLM中称为**Structured Outputs（结构化输出）**，是一种在文本生成过程中强制输出符合特定格式或约束的技术。通过在采样阶段限制模型的token选择空间，确保生成的输出满足预定义的结构要求，如JSON schema、正则表达式、上下文无关文法等。

### 1.2 核心价值

- **格式保证**：确保输出符合JSON、XML等结构化格式
- **类型安全**：强制生成特定类型的数据（如数字、日期）
- **模式验证**：基于JSON Schema或正则表达式进行输出验证
- **降低成本**：避免无效token生成，减少重试和后处理

### 1.3 vLLM实现方式

vLLM通过**有限状态机（FSM）+ Bitmask**的方式实现约束解码：
1. 将约束（JSON schema、regex等）编译为有限状态机
2. 在每个生成步骤，FSM计算当前状态允许的下一token集合
3. 生成bitmask掩码，将不允许的token的logits置为-inf
4. 应用采样策略，从允许的token中选择

### 1.4 支持的约束类型

| 约束类型 | 说明 | 后端支持 |
|---------|------|---------|
| `json` | JSON Schema验证 | xgrammar, guidance, outlines |
| `json_object` | 严格JSON对象 | xgrammar |
| `regex` | 正则表达式匹配 | xgrammar, outlines |
| `grammar` | EBNF上下文无关文法 | xgrammar |
| `choice` | 从预定义列表中选择 | xgrammar, guidance, outlines |
| `structural_tag` | 标签内的JSON内容 | xgrammar |

---

## 2. 核心架构设计 (Core Architecture)

### 2.1 组件层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                      vLLM Engine                            │
├─────────────────────────────────────────────────────────────┤
│  StructuredOutputManager (vllm/v1/structured_output/)      │
│  - 引擎级别的约束输出管理器                                  │
│  - 负责后端初始化、grammar编译、bitmask生成                 │
│  - 管理异步编译线程池                                       │
├─────────────────────────────────────────────────────────────┤
│  StructuredOutputBackend (抽象基类)                         │
│  ┌──────────┬──────────┬──────────┬──────────┐            │
│  │ XGrammar │ Guidance │ Outlines │ LMFormat │            │
│  │ Backend  │ Backend  │ Backend  │ Enforcer │            │
│  └──────────┴──────────┴──────────┴──────────┘            │
│  - 可插拔的后端系统                                         │
│  - 每个后端实现特定库的集成                                 │
├─────────────────────────────────────────────────────────────┤
│  StructuredOutputGrammar (抽象基类)                         │
│  - 表示编译后的约束（FSM实例）                              │
│  - 状态推进：accept_tokens()                                │
│  - 状态回滚：rollback()                                     │
│  - Bitmask填充：fill_bitmask()                              │
├─────────────────────────────────────────────────────────────┤
│  GPU Integration (vllm/v1/structured_output/utils.py)      │
│  - apply_grammar_bitmask()                                  │
│  - 使用xgrammar的CUDA kernel应用bitmask                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块交互流程

```
用户请求 (Request with structured_output params)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ InputProcessor                                            │
│ - 解析StructuredOutputsParams                            │
│ - 创建StructuredOutputRequest                            │
│ - 设置初始状态：WAITING_FOR_FSM                          │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ StructuredOutputManager.grammar_init()                   │
│ - 选择后端（xgrammar/guidance/outlines）                  │
│ - 异步编译grammar（Thread Pool Executor）                 │
│ - 编译完成后，状态变更为READY                            │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Scheduler (async_scheduler.py)                           │
│ - 调度带约束的请求                                        │
│ - 调用grammar_bitmask()生成bitmask                       │
│ - 将bitmask序列化后发送给GPU workers                     │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ GPU Model Runner (model_runner_v1.py)                   │
│ - 接收grammar_bitmask（numpy array）                     │
│ - 模型前向传播得到logits                                 │
│ - apply_grammar_bitmask(logits, grammar_bitmask)         │
│ - 将不允许的token logits置为-inf                         │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Sampler                                                   │
│ - 从约束后的logits中采样                                 │
│ - 调用grammar.accept_tokens()推进FSM状态                 │
│ - 如果使用speculative decoding，失败时rollback()         │
└──────────────────────────────────────────────────────────┘
    │
    ▼
输出token（符合约束）
```

### 2.3 关键类和职责

#### StructuredOutputManager
**文件**: `vllm/v1/structured_output/__init__.py`

**核心职责**:
1. **后端管理**: 根据请求的backend参数初始化相应的后端
2. **异步编译**: 使用线程池异步编译grammar，避免阻塞请求处理
3. **Bitmask生成**: 为每个batch生成token bitmask
4. **推理模式支持**: 集成ReasoningParser，支持在推理结束后应用约束

**关键方法**:
```python
def grammar_init(self, request: Request) -> None:
    """初始化并编译grammar（异步）"""
    if self.backend is None:
        self.backend = XgrammarBackend(...)  # 或其他后端
    grammar = self.executor.submit(self._create_grammar, request)

def grammar_bitmask(self, requests, structured_output_request_ids,
                    scheduled_spec_decode_tokens) -> np.ndarray:
    """生成batch的token bitmask"""
    # 并行填充bitmask（大batch）
    # 串行填充bitmask（小batch）
    # 返回numpy array用于序列化传输
```

#### StructuredOutputBackend
**文件**: `vllm/v1/structured_output/backend_types.py`

所有后端实现的抽象基类，定义了统一的接口：
- `compile_grammar(request_type, grammar_spec)`: 编译约束为FSM
- `allocate_token_bitmask(size)`: 分配bitmask tensor
- `destroy()`: 清理后端资源

#### XgrammarBackend
**文件**: `vllm/v1/structured_output/backend_xgrammar.py`

主要使用的后端，基于[xgrammar](https://github.com/mlc-ai/xgrammar)库：
- 支持JSON schema、regex、EBNF grammar
- 高效的bitmask填充实现
- 内置CUDA kernel优化

---

## 3. 实现机制详解 (Implementation Details)

### 3.1 Grammar编译流程

#### 3.1.1 FSM（有限状态机）构建

约束首先被编译为有限状态机（FSM），FSM定义了：
- **状态（States）**: 表示约束解析过程中的不同阶段
- **转换（Transitions）**: 从一个状态到另一个状态需要消费的token
- **接受状态（Accepting States）**: 表示满足完整约束的结束状态

**示例：JSON Schema**

```python
# JSON Schema示例
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

# 编译为FSM
grammar = backend.compile_grammar("json", schema)
# FSM初始状态: 期望 '{'
# 状态转换:
#   '{' -> 期望 '"name"' 或 '"age"'
#   '"name"' -> 期望 ':' -> 期望字符串值
#   '"age"' -> 期望 ':' -> 期望整数值
#   最终接受状态: 完整的JSON对象
```

#### 3.1.2 Bitmask生成机制

在每个生成步骤，FSM根据当前状态计算允许的下一token集合：

```
┌──────────────────────────────────────────────────────────┐
│ 当前FSM状态 + 已生成token历史                             │
└──────────────────────────────────────────────────────────┘
            │
            ▼ grammar.fill_bitmask()
┌──────────────────────────────────────────────────────────┐
│ 生成bitmask (形状: [vocab_size])                          │
│ - 允许的token位置: 保持原值                               │
│ - 禁止的token位置: -1 (特殊标记)                         │
└──────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────┐
│ bitmask存储在共享tensor中                                 │
│ - 形状: [batch_size * (1 + num_spec_tokens), vocab_size] │
│ - 每个请求一行，speculative decoding时多行               │
└──────────────────────────────────────────────────────────┘
```

**代码实现**: `vllm/v1/structured_output/__init__.py:170-181`

```python
def _fill_bitmasks(self, batch: Iterable[tuple[Grammar, int, bool]]) -> None:
    """为batch中的每个请求填充bitmask"""
    assert self._grammar_bitmask is not None
    for grammar, index, apply_bitmask in batch:
        if apply_bitmask and not grammar.is_terminated():
            grammar.fill_bitmask(self._grammar_bitmask, index)
        else:
            # 对于推理中的请求或已终止的请求，填充全允许mask
            self._grammar_bitmask[index].fill_(self._full_mask)
```

#### 3.1.3 异步编译优化

为了避免grammar编译阻塞请求处理，vLLM使用线程池进行异步编译：

**配置**: `vllm/v1/structured_output/__init__.py:48-73`

```python
# 异步编译开关（external_launcher模式下禁用）
self._use_async_grammar_compilation = (
    vllm_config.parallel_config.distributed_executor_backend
    != "external_launcher"
)

# 线程池配置
# - 用于grammar编译: CPU密集型，max_workers = cpu_count // 2
# - 用于bitmask填充: 并行优化，max_workers = min(cpu_count // 2, 8)
```

**工作流程**:
```
Request 1: grammar_init() ──┐
                             ├──> ThreadPoolExecutor ──> 异步编译
Request 2: grammar_init() ──┘                              │
                                                            │
Request调度等待编译完成 <────────────────────────────────────┘
```

### 3.2 Logits约束应用

#### 3.2.1 apply_grammar_bitmask函数

**文件**: `vllm/v1/structured_output/utils.py:44-119`

这是约束解码的核心函数，在GPU端应用bitmask到logits：

```python
def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    logits: torch.Tensor,
) -> None:
    """
    应用grammar bitmask到模型输出的logits

    参数:
        scheduler_output: 调度器输出（包含请求映射信息）
        grammar_output: 包含grammar_bitmask的输出
        input_batch: GPU runner的输入batch
        logits: 模型前向传播的输出 [batch_size, vocab_size]
    """
    # 1. 获取bitmask（numpy array格式，序列化高效）
    grammar_bitmask = grammar_output.grammar_bitmask

    # 2. 重新排序bitmask以匹配GPU runner的batch顺序
    #    （调度器顺序可能与GPU runner不同）
    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens

    for batch_index, req_id in enumerate(input_batch.req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    # 3. 创建排序后的bitmask tensor
    sorted_bitmask = np.full(
        shape=(logits.shape[0], grammar_bitmask.shape[1]),
        fill_value=-1,
        dtype=grammar_bitmask.dtype,
    )

    # 4. 填充排序后的bitmask
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
        cumulative_index += 1 + num_spec_tokens

    # 5. 异步传输到GPU
    grammar_bitmask = torch.from_numpy(sorted_bitmask).to(
        logits.device, non_blocking=True
    )

    # 6. 调用xgrammar的CUDA kernel应用bitmask
    xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
```

**Bitmask应用逻辑**:
```
输入logits: [batch_size, vocab_size]
例如: [2.3, -1.5, 0.8, 3.2, ...]

输入bitmask: [batch_size, vocab_size]
例如: [0, -1, 0, 0, ...]  (第2个token被禁止)

输出logits: [batch_size, vocab_size]
应用后: [2.3, -inf, 0.8, 3.2, ...]  (第2个token logit被置为-inf)
```

#### 3.2.2 Triton Kernel优化

xgrammar使用自定义的CUDA kernel（通过Triton实现）来高效应用bitmask：

**优势**:
- **向量化操作**: 批量处理整个vocab的mask
- **内存高效**: In-place操作，避免额外内存分配
- **GPU优化**: 充分利用GPU并行能力

### 3.3 状态管理

#### 3.3.1 FSM状态推进

在采样生成token后，需要推进FSM状态：

**接口**: `vllm/v1/structured_output/backend_types.py`

```python
def accept_tokens(self, req_id: str, tokens: list[int]) -> bool:
    """
    接受token并推进FSM状态

    参数:
        req_id: 请求ID
        tokens: 生成的token列表（通常为1个）

    返回:
        bool: token是否被FSM接受
               True: token有效，FSM状态已推进
               False: token无效（不应发生，因已应用bitmask）
    """
```

**工作流程**:
```
当前FSM状态: 期望 '"name"' 或 '"age"'
    │
    ▼ 采样得到token: 对应 '"name"'
    │
    ▼ grammar.accept_tokens('"name"')
    │
    ▼ FSM新状态: 期望 ':'
```

#### 3.3.2 状态回滚（Speculative Decoding支持）

Speculative decoding会预先生成多个draft tokens，验证时可能需要回滚：

```python
def rollback(self, num_tokens: int) -> None:
    """
    回滚FSM状态

    用于speculative decoding：
    - 先推进状态验证draft tokens
    - 如果draft tokens被拒绝，回滚状态
    - 然后使用target token重新推进
    """
```

**场景示例**:
```
FSM当前状态: S0
Draft tokens: [t1, t2, t3]
    │
    ▼ accept_tokens([t1, t2, t3]) -> 状态变为 S3
    │
    ▼ 验证发现 t2 与target不符
    │
    ▼ rollback(3) -> 状态回滚到 S0
    │
    ▼ accept_tokens([target_t]) -> 正确推进到 S1
```

#### 3.3.3 终止检测

```python
def is_terminated(self) -> bool:
    """
    检查FSM是否已到达接受状态

    返回:
        True: 已满足完整约束，可停止生成
        False: 约束未满足，需继续生成
    """
```

**使用场景**:
- JSON对象完整生成后停止
- 正则表达式匹配完成后停止
- 避免生成无效的多余token

---

## 4. Ascend NPU适配 (Ascend Adaptations)

### 4.1 核心适配点

#### 4.1.1 缺少torch.compile支持的处理

**问题**: NPU不支持`torch.compile`，而xgrammar的`apply_grammar_bitmask`使用了torch.compile优化。

**解决方案**: `vllm_ascend/worker/model_runner_v1.py`

```python
# Apply structured output bitmasks if present
if grammar_output is not None:
    # 由于NPU不支持torch.compile，需要手动处理
    logits_dtype = logits.dtype
    # 将logits移到CPU（float格式）
    logits = logits.to("cpu").float()
    # 在CPU上应用bitmask
    apply_grammar_bitmask(scheduler_output, grammar_output,
                         self.input_batch, logits)
    # 将结果移回NPU并恢复原始dtype
    logits = logits.to(self.device).to(logits_dtype)
```

**性能影响**:
- **CPU-NPU传输开销**: 每次生成需要往返传输
- **类型转换开销**: CPU float32与NPU dtype之间转换
- **权衡**: 牺牲性能换取功能完整性

#### 4.1.2 CPU-NPU数据传输方案

**流程图**:
```
NPU logits: [batch, vocab_size] (float16/bfloat16)
    │
    ▼ to("cpu").float()
CPU logits: [batch, vocab_size] (float32)
    │
    ▼ apply_grammar_bitmask()
CPU masked logits: [batch, vocab_size] (float32)
    │
    ▼ to("npu").to(dtype)
NPU masked logits: [batch, vocab_size] (原始dtype)
```

### 4.2 性能优化

#### 4.2.1 自定义Triton Kernels

**文件**: `vllm_ascend/ops/triton/reject_sample.py`

针对Ascend向量核心优化的拒绝采样kernels：

**1. Greedy Sampling Kernel**

```python
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_triton(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    vec_len,
    max_spec_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Speculative decoding的greedy采样kernel

    优化点:
    - 向量化加载draft和target tokens
    - 批量验证draft tokens
    - 仅在全部接受时写入bonus token
    """
```

**2. Random Sampling Kernel**

```python
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,  # [num_tokens, vocab_size]
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,
    vec_len,
    max_spec_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Speculative decoding的概率采样kernel

    优化点:
    - 块验证减少 rejection采样次数
    - 并行计算accept/reject概率
    - Token recovery机制
    """
```

**3. 向量核心优化**

```python
def cal_grid_and_block_size(batch_size: int):
    """
    计算适合Ascend向量核心的grid和block size

    Ascend NPU架构特点:
    - 多个向量核心并行执行
    - 需要根据batch size动态调整
    """
    vectorcore_num = get_vectorcore_num()
    if batch_size <= vectorcore_num:
        grid = batch_size
        block_size = 1
    else:
        grid = vectorcore_num
        block_size = triton.next_power_of_2(triton.cdiv(batch_size, grid))
    return grid, block_size
```

#### 4.2.2 异步指数采样

**文件**: `vllm_ascend/sample/sampler.py`

为了避免`torch.multinomial`导致的CPU-NPU同步，使用指数分布技巧：

```python
def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """
    使用指数分布技巧进行随机采样

    数学原理:
    - X ~ Exp(1), Y ~ Exp(1) 独立
    - P(X / (X+Y) < p) = p
    - 因此 argmin(X / prob) 等价于按概率采样

    优势:
    - 避免torch.multinomial的CPU-NPU同步
    - 可在不同stream中异步执行
    """
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        if len(generators) != probs.shape[0]:
            q.exponential_()  # 生成指数分布随机数
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())
    return probs.div_(q).argmax(dim=-1).view(-1)
```

**异步优化**:

```python
def do_async_exponential(self, b_s, head_dim, generators):
    """
    在独立stream中预计算指数随机数
    与模型执行重叠，隐藏延迟
    """
    with torch.npu.stream(global_stream()):
        global_stream().wait_stream(torch.npu.current_stream())
        q = torch.empty((b_s, head_dim), device="npu", dtype=torch.float32)
        q.exponential_()
        self.async_exponential_event.record()
    self.set_q_event(q, self.async_exponential_event)
```

**执行流程**:
```
时间线:
t0: 模型前向传播 (current_stream)
    │
    ├─> global_stream: 预计算指数随机数 (异步)
    │
t1: 模型完成，采样需要随机数
    │
    ▼ wait_stream(global_stream)
t2: 使用预计算的随机数进行采样
```

### 4.3 硬件限制和权衡

#### 4.3.1 支持的后端

| 后端 | 支持状态 | 说明 |
|------|---------|------|
| **xgrammar** | ✅ 完全支持 | 主要推荐后端，性能最佳 |
| **guidance** | ⚠️ 部分支持 | 可能有错误，不推荐 |
| **outlines** | ❌ 不支持 | 正则表达式约束不可用 |
| **lm-format-enforcer** | ✅ 支持 | 轻量级备选方案 |

**推荐配置**:

```python
# OpenAI API
extra_body={
    "structured_outputs": {
        "backend": "xgrammar",  # 强制使用xgrammar
        "disable_any_whitespace": False  # JSON格式化选项
    }
}

# 离线推理
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    structured_outputs={"backend": "xgrammar"}
)
```

#### 4.3.2 已知问题和局限性

**1. CPU-NPU传输开销**

- **问题**: 每次生成需要CPU-NPU往返传输
- **影响**: 高吞吐场景下性能下降明显
- **缓解方案**:
  - 批处理多个请求以摊销传输成本
  - 使用更大的batch size

**2. Chunked Prefill兼容性**

```python
# 对于chunked prefill，使用-1作为mask而非0
# 因为约束解码可能回滚speculative tokens
num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
```

**3. External Launcher模式限制**

```python
# External launcher模式下禁用异步编译
# 因为不同TP rank的FSM状态可能不一致
self._use_async_grammar_compilation = (
    vllm_config.parallel_config.distributed_executor_backend
    != "external_launcher"
)
```

**4. 推理模式（Reasoning）支持**

约束解码支持在推理（thinking）模式下应用：

```python
# enable_in_reasoning配置
self.enable_in_reasoning = (
    self.vllm_config.structured_outputs_config.enable_in_reasoning
)

# 如果enable_in_reasoning=True，约束在整个生成过程中生效
# 否则，约束仅在推理结束后生效
```

---

## 5. 使用指南 (Usage Guide)

### 5.1 API使用示例

#### 5.1.1 OpenAI兼容API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token"
)

# JSON Schema约束
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["name", "age"]
}

completion = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "Generate a person profile"}
    ],
    extra_body={
        "structured_outputs": {
            "json": json_schema
        }
    }
)

print(completion.choices[0].message.content)
# 输出: {"name": "Alice", "age": 30, "skills": ["Python", "ML"]}
```

#### 5.1.2 正则表达式约束

```python
completion = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "Generate an email address"}
    ],
    extra_body={
        "structured_outputs": {
            "regex": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\n"
        }
    }
)

print(completion.choices[0].message.content)
# 输出: user@example.com
```

#### 5.1.3 Choice约束

```python
completion = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "Classify this sentiment"}
    ],
    extra_body={
        "structured_outputs": {
            "choice": ["positive", "negative", "neutral"]
        }
    }
)

print(completion.choices[0].message.content)
# 输出: positive
```

#### 5.1.4 离线推理API

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# 初始化LLM
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    structured_outputs={"backend": "xgrammar"}
)

# 定义JSON Schema
json_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "count": {"type": "integer"}
    },
    "required": ["title", "count"]
}

# 创建采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=100,
    structured_outputs=StructuredOutputsParams(json=json_schema)
)

# 生成
prompts = ["Generate a book recommendation"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### 5.2 配置选项

#### 5.2.1 全局配置

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    structured_outputs={
        "backend": "xgrammar",           # 后端选择
        "disable_fallback": False,       # 禁用回退
        "disable_any_whitespace": False, # JSON紧凑模式
        "reasoning_parser": "deepseek_r1",  # 推理模式解析器
        "enable_in_reasoning": False     # 在推理中启用约束
    }
)
```

**配置说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | str | "auto" | 后端选择：auto/xgrammar/guidance/outlines/lm-format-enforcer |
| `disable_fallback` | bool | False | 禁用错误时回退到其他后端 |
| `disable_any_whitespace` | bool | False | 禁用JSON中的空白字符（紧凑输出） |
| `reasoning_parser` | str | None | 推理模式解析器（如deepseek_r1） |
| `enable_in_reasoning` | bool | False | 是否在推理过程中应用约束 |

#### 5.2.2 后端选择建议

**选择流程图**:
```
需要约束解码
    │
    ▼
是否需要JSON Schema?
    │
    ├─ 是 ──> xgrammar (推荐)
    │         │
    │         └─ 备选: guidance (如果有特殊需求)
    │
    └─ 否 ──> 需要正则表达式?
              │
              ├─ 是 ──> xgrammar (Ascend推荐)
              │         │
              │         └─ GPU: outlines
              │
              └─ 否 ──> 需要choice?
                        │
                        └─ xgrammar / lm-format-enforcer
```

### 5.3 性能考虑

#### 5.3.1 批处理优化

vLLM自动对约束解码请求进行批处理优化：

**并行Bitmask填充**:

```python
# 大batch（>128请求）: 使用线程池并行填充
if len(structured_output_request_ids) > 128:
    # 分批并行填充
    batch_size = 16
    max_workers = min(cpu_count // 2, 8)
    # 使用ThreadPoolExecutor并行执行
```

**小batch**: 串行填充，避免线程开销

#### 5.3.2 内存管理

**Bitmask内存分配**:

```python
# 预分配最大batch的bitmask
max_batch_size = scheduler_config.max_num_seqs
max_spec_tokens = speculative_config.num_speculative_tokens

# 分配: [max_batch * (1 + max_spec_tokens), vocab_size]
# 内存占用估算:
# - batch_size=256, spec_tokens=5, vocab=128K
# - 256 * 6 * 128K * 4 bytes (int32) ≈ 750 MB
```

**优化建议**:
- 根据实际batch调整`max_num_seqs`
- Speculative decoding时控制`num_speculative_tokens`
- 优先使用xgrammar（内存效率更高）

#### 5.3.3 Ascend特定优化

**1. 使用xgrammar后端**

```python
# 强制使用xgrammar以获得最佳性能
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    structured_outputs={"backend": "xgrammar"}
)
```

**2. 启用异步指数采样**

```python
# vLLM-Ascend配置
from vllm_ascend import AscendConfig

ascend_config = AscendConfig(
    enable_async_exponential=True  # 隐藏采样延迟
)
```

**3. 批处理策略**

```python
# 使用较大的batch size摊销CPU-NPU传输成本
sampling_params = SamplingParams(
    max_tokens=100,
    structured_outputs=StructuredOutputsParams(json=schema)
)

# 批量处理多个请求
llm.generate(prompts, sampling_params=sampling_params)
```

---

## 6. 关键代码路径索引 (Key Code References)

### 6.1 vLLM核心实现

#### 6.1.1 StructuredOutputManager

**文件**: `vllm/v1/structured_output/__init__.py`

**关键代码位置**:
- **类定义**: 第35行 - `class StructuredOutputManager`
- **后端初始化**: 第99-149行 - `grammar_init()`
- **Bitmask生成**: 第188-284行 - `grammar_bitmask()`
- **异步编译**: 第151-155行
- **推理模式支持**: 第286-333行

**阅读建议**:
1. 先阅读`__init__`了解初始化流程
2. 理解`grammar_init()`的后端选择机制
3. 重点研究`grammar_bitmask()`的bitmask生成和排序逻辑

#### 6.1.2 XGrammar后端

**文件**: `vllm/v1/structured_output/backend_xgrammar.py`

**关键类和方法**:
- **XgrammarBackend类**: 主后端实现
- **compile_grammar()**: 将约束编译为xgrammar的Grammar对象
- **allocate_token_bitmask()**: 分配bitmask tensor
- **XgrammarGrammar类**: 封装xgrammar.Grammar，实现vLLM接口

**核心流程**:
```
JSON Schema / Regex
    │
    ▼ xgrammar.Grammar.from_json_schema() / from_regex()
Grammar对象（FSM）
    │
    ▼ grammar.fill_bitmask()
Bitmask tensor
```

#### 6.1.3 GPU集成

**文件**: `vllm/v1/structured_output/utils.py`

**关键函数**:
- **apply_grammar_bitmask()**: 第44-119行
  - Bitmask排序和重新对齐
  - 调用xgrammar CUDA kernel
  - Speculative decoding支持

**重要逻辑**:
```python
# 第100-102行: 异步传输到GPU
grammar_bitmask = torch.from_numpy(sorted_bitmask).to(
    logits.device, non_blocking=True
)

# 第119行: xgrammar kernel调用
xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
```

#### 6.1.4 调度器集成

**文件**: `vllm/v1/core/sched/async_scheduler.py`

**集成点**:
- **StructuredOutputRequest管理**: 跟踪带约束的请求
- **WAITING_FOR_FSM状态**: 等待grammar编译完成
- **Bitmask传递**: 将grammar_bitmask发送给GPU workers

**代码位置**:
- 请求过滤: 跳过仍在等待FSM的请求
- Speculative token处理: 传递给grammar_bitmask()

### 6.2 vLLM-Ascend适配

#### 6.2.1 模型运行器集成

**文件**: `vllm_ascend/worker/model_runner_v1.py`

**关键适配**:
- **导入**: 第74行 - `from vllm.v1.structured_output.utils import apply_grammar_bitmask`
- **CPU-NPU转换**: 约束解码特殊处理
  ```python
  if grammar_output is not None:
      logits_dtype = logits.dtype
      logits = logits.to("cpu").float()
      apply_grammar_bitmask(...)
      logits = logits.to(self.device).to(logits_dtype)
  ```

**查找技巧**:
- 搜索`grammar_output`变量
- 查看logits处理流程

#### 6.2.2 Triton拒绝采样Kernels

**文件**: `vllm_ascend/ops/triton/reject_sample.py`

**关键函数**:
- **rejection_greedy_sample_triton()**: 第88-141行
  - Greedy采样带约束的speculative decoding
  - 批量验证draft tokens
  - Bonus token处理

- **rejection_random_sample_kernel()**: 第144行起
  - 概率采样带约束
  - 块验证优化
  - Token recovery机制

- **cal_grid_and_block_size()**: 第23-31行
  - Ascend向量核心优化
  - 动态grid/block计算

**理解要点**:
1. Triton JIT装饰器: `@triton.jit`
2. 向量化加载: `tl.load()`
3. 批量处理: `for pos in tl.range(0, BLOCK_SIZE)`

#### 6.2.3 NPU优化采样

**文件**: `vllm_ascend/sample/sampler.py`

**关键函数**:
- **random_sample()**: 第11-34行
  - 指数分布技巧避免torch.multinomial
  - Stream切换避免CPU-NPU同步

- **do_async_exponential()**: 第48-61行
  - 异步预计算随机数
  - 与模型执行重叠

- **apply_top_k_top_p()**: 第93-126行
  - Ascend优化的top-k/top-p实现

**关键概念**:
```python
# 第24-33行: Stream切换避免同步
with npu_stream_switch(global_stream()):
    q = torch.empty_like(probs)
    q.exponential_()  # 在global_stream中执行
torch.npu.current_stream().wait_stream(global_stream())
```

#### 6.2.4 测试覆盖

**文件**: `vllm-ascend/tests/test_guided_decoding.py`

**测试用例**:
- JSON Schema约束测试
- 正则表达式约束测试
- 多后端测试（xgrammar, guidance）
- 批处理测试

**运行测试**:
```bash
pytest vllm-ascend/tests/test_guided_decoding.py -v
```

### 6.3 调试和监控

#### 6.3.1 日志配置

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看约束解码相关日志
logger = logging.getLogger("vllm.v1.structured_output")
logger.setLevel(logging.DEBUG)
```

**关键日志点**:
- Grammar编译开始/完成
- Bitmask生成时间
- FSM状态推进
- Speculative decoding rejection

#### 6.3.2 性能分析

**关键指标**:
1. **Grammar编译时间**: 首次请求的延迟
2. **Bitmask填充时间**: 每步的CPU开销
3. **Bitmask应用时间**: GPU kernel执行时间
4. **CPU-NPU传输时间**: Ascend特定开销

**分析工具**:
```python
import time

# 在apply_grammar_bitmask周围添加计时
start = time.time()
apply_grammar_bitmask(...)
elapsed = time.time() - start
logger.info(f"Bitmask application took {elapsed*1000:.2f}ms")
```

---

## 7. 总结

### 7.1 设计优势

1. **模块化架构**: 可插拔的后端系统，易于扩展
2. **性能优化**: 异步编译、批处理、GPU kernel优化
3. **灵活集成**: 支持多种约束类型，统一API
4. **Speculative Decoding兼容**: 完整支持状态回滚
5. **推理模式支持**: 智能检测推理阶段并延迟约束应用

### 7.2 Ascend适配挑战

1. **缺少torch.compile**: CPU-NPU传输是主要瓶颈
2. **优化方向**: Triton kernels、异步采样、向量核心优化
3. **权衡取舍**: 功能完整性 vs 性能开销

### 7.3 未来改进方向

1. **NPU原生kernel**: 直接在NPU上实现bitmask应用
2. **零拷贝传输**: 减少CPU-NPU数据拷贝
3. **更智能的批处理**: 自适应的并行填充策略
4. **编译缓存**: Grammar编译结果持久化

---

## 附录

### A. 术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| 约束解码 | Constrained Decoding | 限制模型生成输出的技术 |
| 结构化输出 | Structured Outputs | vLLM中的约束解码实现 |
| 有限状态机 | Finite State Machine (FSM) | 用于表示约束的自动机 |
| 位掩码 | Bitmask | 标记允许/禁止token的数组 |
| 推测解码 | Speculative Decoding | 使用draft tokens加速生成的技术 |
| 拒绝采样 | Rejection Sampling | 验证draft tokens的采样方法 |

### B. 参考资料

1. **vLLM文档**: https://docs.vllm.ai/
2. **xgrammar库**: https://github.com/mlc-ai/xgrammar
3. **vLLM-Ascend项目**: https://github.com/huawei-noah/vllm-ascend
4. **RFC #11162**: vLLM Platform Plugin Interface
5. **Speculative Decoding论文**: "Speculative Sampling: Accelerating Large Language Model Inference"

### C. 版本信息

- **vLLM版本**: v0.7.x (v1 engine)
- **vLLM-Ascend版本**: 基于8.3.rc2 CANN
- **torch-npu版本**: 2.8.0
- **文档更新日期**: 2025年1月
