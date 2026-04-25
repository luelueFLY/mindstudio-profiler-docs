# 性能数据交付件


## 结果文件

单卡Ascend Pytorch Profiler 性能结果文件如下：
   ```ColdFusion 
    XXX_ascend_pt
    ├── ASCEND_PROFILER_OUTPUT   // 解析后的性能数据
    │    ├── analysis.db   
    │    ├── api_statistic.csv 
    │    ├── ascend_pytorch_profiler.db
    │    ├── kernel_details.csv
    │    ├── operator_details.csv
    │    ├── op_statistic.csv
    │    ├── step_trace_time.csv
    │    ├── ...
    │    └── trace_view.json
    ├── FRAMEWORK   // 框架侧性能原始数据，用户无需关注
    ├── PROF_000001_20260424092602791_02445978DJECPLIB
    │    └── device_0  // Device侧性能原始数据，用户无需关注
    │    └── host // Host侧性能原始数据，用户无需关注
    └── profiler_info.json   // 性能数据采集配置信息
    └── profiler_metadata.json   // 性能数据相关的元数据
   ```

- 文本结果：包括 `.json`、`.csv` 等 timeline 和 summary 文件。
- 数据库结果：`ascend_pytorch_profiler_{Rank_ID}.db`、`analysis.db`。
- 元数据文件：如 `profiler_metadata.json`。
- 原始性能数据：位于 `PROF_*`、`ASCEND_PROFILER_OUTPUT`、`FRAMEWORK` 等目录下。


## 相关说明

- 推荐使用 MindStudio Insight 可视化查看 trace、内核、通信、显存等数据。
- 结果目录和字段详情可参考 [MindSpore&PyTorch 框架性能数据文件参考](https://www.hiascend.com/document/detail/zh/mindstudio/830/T&ITools/Profiling/atlasprofiling_16_0203.html#ZH-CN_TOPIC_0000002536038401)。

