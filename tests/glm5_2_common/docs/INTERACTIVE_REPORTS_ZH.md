# 交互式报告栈

项目自己生成的 HTML 报告统一使用三层结构：

- **Panel 1.9.4** 负责页面布局、组件组合和单文件导出；
- **pyecharts 2.1.0** 负责在 Python 中声明图表；
- **Apache ECharts** 在浏览器中完成交互绘制。

安装：

```bash
python -m pip install -r requirements-reporting.txt
```

版本固定在 `requirements-reporting.txt`，测试环境通过
`requirements-test.txt` 自动包含同一组依赖。报告生成过程不需要启动
Panel server；`Panel.save(resources=INLINE, embed=True)` 将 Panel、Bokeh 和
ECharts 运行时代码以及实验数据写进一个 HTML。产物可直接下载、离线打开，
不引用相邻 JavaScript、CSS、CSV 或 JSON 文件。

公共实现位于 `tests/glm5_2_common/reporting.py`。业务报告只负责提供指标、
序列、阈值线和诊断区间，不自行复制 JavaScript。精度训练观察、NPU/GPU
性能单端报告和 profiler-off 重复实验对比均使用这套实现，提供：

- 鼠标悬停显示 step 与精确值；
- 滚轮/框选缩放、平移、滑块和恢复；
- 图例开关、数据视图和单图导出；
- 首 steps 区间、首次超过 Loss 指导线后的区间；
- Loss 与 Grad Norm 的原始曲线、相对误差、有符号误差；
- NaN/Inf 计数和首次异常证据。

性能报告另外提供逐 Step 耗时、吞吐、TFLOPS、MFU 和活跃显存，明确标出暖机区间；
重复实验报告叠加每个 repeat 的曲线，展示 median、p90、p95、CV 和候选相对基准
变化。报告中的自动诊断只负责选择下一步分析分支，不产生统一 PASS/FAIL。

静态 SVG 暂时保留为兼容产物，供文本审阅、旧报告和无 JavaScript 环境使用；
交互 HTML 是面向人的主要入口。聚合报告把交互 HTML 内嵌为 `srcdoc`，因此
下载聚合 HTML 后也不依赖原始目录。

外部工具的原生页面和数据库不属于项目自有报告。`tlparse`、Profiler、
TensorBoard/`.vis.db`、Nsight 等产物只在项目报告中建立索引，不重新生成、
改名或伪装成 Panel 报告。

官方依据：

- [Panel 导出与嵌入](https://panel.holoviz.org/how_to/export/index.html)
- [Panel ECharts pane](https://panel.holoviz.org/reference/panes/ECharts.html)
- [pyecharts](https://github.com/pyecharts/pyecharts)
- [Apache ECharts](https://github.com/apache/echarts)
