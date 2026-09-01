# MindStudio 参考约束

- 在处理不熟悉或不确定的昇腾 NPU 问题时，尤其是精度对齐、性能数据采集与分析、算子调试、内存问题和训练/推理故障定位，应优先参考华为昇腾 MindStudio 官方文档：https://www.hiascend.com/document/detail/zh/mindstudio/ 。
- 使用文档中的命令、参数、指标或结论前，应先确认当前项目的硬件型号以及 CANN、MindStudio、PyTorch 和 `torch_npu` 版本，并选用相匹配的文档版本；不得未经核验直接照搬其他版本的操作。
- 从 MindStudio 文档提炼到本项目的经验，应记录适用场景、环境与版本、操作步骤、观察指标、结论和官方文档来源，便于复现与复核。
