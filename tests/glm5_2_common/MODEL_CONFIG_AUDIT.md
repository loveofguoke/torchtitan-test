# 实验模型配置来源审计

模型结构由 TorchTitan 配置工厂定义；实验只选择配置并设置训练/采集参数。
`steps`、batch、sequence length、采集窗口的默认值不等于重复定义模型。

| 实验 | 配置来源及调用点 | 检查结果 |
|---|---|---|
| parity | `glm5_2_parity/model_config.py::load_model_config` → `glm5_configs[flavor]()` | 已移除复制的 debug 尺寸；默认原生 Config，HF 从其提取结构；CLI `--model-config` 选 GLM flavor |
| smoke | `glm5_2_smoke/train_smoke.py` 的 `--module/--config` | 默认选择 `glm5/glm5_debugmodel`，交给训练入口解析，不复制模型尺寸 |
| precision | `FormalTrainingConfig.module/config` → `glm5_2_precision/workflow.py` 的 fixture/capture 命令 | token-plan 和 checkpoint/capture 都传同一个配置名 |
| checkpoint、stability | 复用 `FormalTrainingConfig`，各训练命令传 `--module/--config` | 无额外模型结构副本 |
| performance、MindStudio 系统性能 | `PerformanceConfig.module/model_config` → `glm5_2_performance/workflow.py` | 传递配置名给 capture_metrics，不复制模型结构 |
| MindStudio accuracy | `config.training` → `glm5_2_mindstudio/workflow.py` 的 capture_training | 复用选定训练配置；不另外定义模型 |
| graph、combination | 通过 precision/performance 训练配置构造对应工作流 | 编译设置叠加在选定训练配置上，不单独复制模型尺寸 |
| nsys | `glm5_2_nsys/workflow.py` 的 `--module/--config` | 通过训练启动路径使用所选配置 |

注意两种配置名不是同一层：parity 的 `debugmodel` 是模型注册表 flavor；
其他训练入口的 `glm5_debugmodel` 是 TorchTitan 训练配置名，内部选择模型。
目前没有把任意模型家族接入 GLM/HF parity；更换 GLM flavor 不等于支持其他家族。

parity 的 `router_unit_profile()` 是独立 CPU 单元测试夹具，刻意使用微型尺寸，
不是运行实验时的 debug 默认配置。此类测试夹具无需跟随训练模型变大。

修改 TorchTitan 的模型配置会影响**新构建的模型**，但不能让已有 checkpoint
或采集结果自动变成新配置。其他工作流已完成结果的复用规则本次未变更；
不要在修改模型后把旧 fixture 当作新初始化。选择新实验或显式重新生成 fixture。
本审计确认配置来源，不宣称所有旧工作流均能自动追踪配置工厂源码变更。

parity 回归检查：工厂内容改变会更新有效尺寸及实验标识；选择其他 flavor
确实调用对应工厂；显式 CLI 覆盖优先；默认构建保留原生层对象及非尺寸设置。
