# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This workspace contains three related projects:

1. **vLLM** (`vllm/`) - A high-throughput and memory-efficient inference engine for Large Language Models (LLMs). Written primarily in Python with C++/CUDA kernels for GPU acceleration. Key innovations include **PagedAttention** (efficient KV cache management) and **continuous batching** for optimal throughput.

2. **vLLM-Ascend** (`vllm-ascend/`) - A hardware plugin that extends vLLM to run on Huawei Ascend NPUs. This is **NOT a fork** but a plugin that integrates with vLLM through the platform plugin interface (RFC #11162), maintaining full API compatibility with the main vLLM project.

3. **LMCache** (`LMCache/`) - A high-performance KV cache management system that extends vLLM with multi-tier KV cache storage (GPU/CPU/Disk/Remote). Reduces TTFT by 3-10x through intelligent cache reuse, cross-instance sharing, and disaggregated prefill-decode. Integrates with vLLM v1 via the KV connector interface.


## 如何设计一个新的特性
我会按照如下步骤进行分析
1. 通过分析vllm/vllm-Ascend代码来了解如何实现新特性 
2. 从整体架构上分析需要改的模块和新增的模块
3. 明确受影响的模块和新增的模块都主要作用和对外接口
3. 分析实现该特性的主要流程
4. 分析该特性涉及的主要代码
5. Then I'll generate documentation xxx.md and save to current folder:
6. 提交github

# Documentation: xxx.md , xxx 是特性名称
## 概述
[通过2,3个段落来描述该特性的主要功能]

## 架构设计
[画出vllm/vllm-ascend的完整的架构图,并标出该特性影响到的所有模块和新增模块,架构图要全面]
[每个模块完成的主要功能]
[每个模块对外的接口的简要描述]

## 主要流程
[该特性的流程,画出各个流程的时序图]

## 相关代码
[该特性所涉及的代码的文件和相关信息]

## 使用说明
[如何使用该特性的简要说明]

This documentation will follow our guidelines for clarity, completeness, and actionability.