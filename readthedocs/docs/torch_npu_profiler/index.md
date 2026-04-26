---
title: torch_npu.profiler
---

# Ascend Pytorch Profiler

Ascend Pytorch Profiler 是面向 PyTorch 训练与在线推理场景的性能数据采集工具，支持采集框架侧算子信息、CANN 软件栈数据、底层 NPU 算子信息以及显存占用信息，并可衔接自动解析、离线解析、MindStudio Insight 和 `msprof-analyze` 进行后续分析。

## 文档导读

- 初次使用，建议先阅读 [快速开始](./source/getting_started/quick_start.md)。
- 需要完成常规性能采集，优先阅读 [使用 `torch_npu.profiler.profile` 采集](./source/user_guide/profile_api.md)。
- 需要在训练或推理过程中按需触发采集，阅读 [dynamic_profile 动态采集](./source/user_guide/dynamic_profile.md)。
- 采集完成后需要手动解析数据，阅读 [离线解析](./source/user_guide/offline_analyse.md)。
- 需要查参数、配置文件和扩展能力，阅读 [参考说明](./source/reference/index.md)。

## 典型流程

1. 完成环境准备并选择采集方式。
2. 在脚本中接入 `torch_npu.profiler.profile`、或使用`dynamic_profile`动态采集。
3. 执行训练或在线推理任务，生成性能数据。
4. 选择自动解析或离线解析。
5. 使用 MindStudio Insight 或 `msprof-analyze` 查看和分析结果。
