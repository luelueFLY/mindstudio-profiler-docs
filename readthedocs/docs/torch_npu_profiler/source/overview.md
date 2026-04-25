# 概览

Ascend PyTorch Profiler 是专为 PyTorch 框架设计的性能分析工具，对标框架原生能力。用户仅需在训练脚本中插入 Profiler 接口，即可在训练过程中自动采集性能数据，并在训练完成后直接输出可视化结果，显著提升分析效率。该工具支持全面采集 PyTorch 层算子、CANN 层算子、底层 NPU 算子及算子内存占用等多维度性能信息，实现对训练过程的全方位性能分析。

## 能力概览

- 采集 PyTorch 层算子信息、CANN 层信息和底层 NPU 算子信息。
- 支持 `torch_npu.profiler.profile`接口采集、`dynamic_profile` 动态采集。
- 支持 mstx 打点、显存时间线、环境变量信息、自定义元数据、子线程采集等扩展能力。
- 支持生成文本结果、数据库结果，并可对接 MindStudio Insight 与 `msprof-analyze`。

## 阅读建议

- 新用户先看 [快速开始](./getting_started/quick_start.md)。
- 日常采集和调优优先看 [profile 接口采集](./user_guide/profile_api.md)。
- 长任务运行中触发采集时看 [dynamic_profile 动态采集](./user_guide/dynamic_profile.md)。
- 需要查参数字典时看 [参考说明](./reference/index.md)。
