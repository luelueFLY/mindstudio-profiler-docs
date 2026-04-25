# dynamic_profile 动态采集

`dynamic_profile` 允许在训练或在线推理任务运行过程中随时启动采集，适合长时间运行任务和需要按需触发采集的场景。

以下三种方式只能选择一种使用，不能同时启用两种及以上方式。

## 动态采集方式对比

| 方式 | 适用场景 | 特点 |
| --- | --- | --- |
| 环境变量方式 | 仅训练场景 | 不需要修改用户代码，通过配置文件触发采集。 |
| `dp.init()` 方式 | 训练和在线推理 | 需要在脚本中接入 `dynamic_profile`，通过修改配置文件触发采集。 |
| `dp.start()` 方式 | 训练和在线推理 | 在代码中预置精确触发点，适合缩小采集范围。 |

## 环境变量方式

### 步骤

1. 配置环境变量：

```bash
export PROF_CONFIG_PATH="/path/to/profiler_config_path"
```

2. 启动训练任务。`dynamic_profile` 会在 `profiler_config_path` 下自动生成模板文件 `profiler_config.json`。
3. 重新开启一个终端，修改 `profiler_config.json` 使能 Profiling。
4. 采集结束后，选择自动解析或手动解析结果。

### 说明

- 仅支持训练场景。
- 不支持采集第一个迭代 `step0`。
- 依赖 PyTorch 原生 `Optimizer.step()` 划分 step，不支持自定义 Optimizer。
- `PROF_CONFIG_PATH` 路径要求有读写权限，且路径仅支持字母、数字、下划线和连字符，不支持软链接。

## `dp.init()` 方式

```python
from torch_npu.profiler import dynamic_profile as dp

dp.init("profiler_config_path")

for step in steps:
    train_one_step()
    dp.step()
```

### 说明

- `dp.init()` 执行时会在 `profiler_config_path` 下自动创建模板文件 `profiler_config.json`。
- 运行任务后，可在新的终端中修改 `profiler_config.json`，动态触发采集。
- `dynamic_profile` 会轮询配置文件变化；若采集执行期间再次修改配置，则会在当前采集结束后启动最后一次修改对应的采集任务。

## `dp.start()` 方式

```python
from torch_npu.profiler import dynamic_profile as dp

dp.init("profiler_config_path")

for step in steps:
    if step == 5:
        dp.start("/home/xx/start_config_path/profiler_config.json")
    train_one_step()
    dp.step()
```

### 说明

- `dp.start()` 适合把采集范围缩小到指定代码位置附近。
- `dp.start()` 不感知后续配置文件修改，在一次任务执行过程中只触发一次采集。
- 若 `dp.init()` 触发的动态采集已经在执行中，`dp.start()` 不会生效；若 `dp.init()` 对应采集已结束，则 `dp.start()` 可继续触发新的采集任务。


# `profiler_config.json` 配置文件

`profiler_config.json` 推荐使用共享存储保存。以下是默认配置示例：

```json
{
  "activities": ["CPU", "NPU"],
  "prof_dir": "./",
  "analyse": false,
  "record_shapes": false,
  "profile_memory": false,
  "with_stack": false,
  "with_flops": false,
  "with_modules": false,
  "active": 1,
  "warmup": 0,
  "start_step": 0,
  "is_rank": false,
  "rank_list": [],
  "experimental_config": {
    "profiler_level": "Level0",
    "aic_metrics": "AiCoreNone",
    "l2_cache": false,
    "op_attr": false,
    "gc_detect_threshold": null,
    "data_simplification": true,
    "record_op_args": false,
    "export_type": ["text"],
    "mstx": false,
    "mstx_domain_include": [],
    "mstx_domain_exclude": [],
    "host_sys": [],
    "sys_io": false,
    "sys_interconnection": false
  }
}
```

### 主要字段

| 参数 | 说明 |
| --- | --- |
| `start_step` | 开始采集的 step。`0` 表示不采集，`-1` 表示保存配置后的下个 step 开始采集。 |
| `activities` | 采集 CPU、NPU 事件列表。 |
| `prof_dir` | 采集结果落盘目录。 |
| `analyse` | 是否自动解析。 |
| `record_shapes` | 是否采集输入 shape 和 type。 |
| `profile_memory` | 是否采集显存占用。 |
| `with_stack` | 是否采集调用栈。 |
| `with_flops` | 是否采集浮点操作信息。 |
| `with_modules` | 是否采集 modules 层级 Python 调用栈。 |
| `active` | 实际采集的 step 数。 |
| `warmup` | 预热 step 数。 |
| `is_rank` | 是否开启指定 Rank 采集。 |
| `rank_list` | 需要采集的 Rank ID 列表。 |
| `async_mode` | 是否开启异步解析。 |
| `experimental_config` | 扩展采集配置。 |
| `metadata` | 自定义元数据。 |

## dynamic_profile 维测日志

启用 `dynamic_profile` 后，会在 `profiler_config_path` 下自动生成日志目录，典型结构如下：

```text
profiler_config_path/
├── log
│   ├── dp_ubuntu_xxxxxx_rank_*.log
│   ├── dp_ubuntu_xxxxxx_rank_*.log.1
│   ├── monitor_dp_ubuntu_xxxxxx_rank_*.log
│   ├── monitor_dp_ubuntu_xxxxxx_rank_*.log.1
├── profiler_config.json
└── shm
```

- `dp_*.log`：记录动态采集执行过程中的 INFO、WARNING、ERROR。
- `monitor_dp_*.log`：记录 `profiler_config.json` 每次修改是否生效，以及 dynamic_profile 进程的结束。
- `shm`：Python 3.7 环境下用于共享内存映射的目录。程序异常终止时，需要手动清理残留文件。
