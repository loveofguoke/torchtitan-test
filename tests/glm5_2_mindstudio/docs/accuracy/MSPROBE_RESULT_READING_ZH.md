# msProbe 判定与分级图阅读：状态、统计量和 Norm

## 1. 审计范围

2026-09-07 核对本地 `D:/yyb/repos/msprobe`，commit
`86a64ee303cf27adedcefc0b661fd9d5ab9af615`。该 checkout 属于 26.2 开发历史，
不能直接代表服务器 `26.1.0.post1` wheel。下文是此源码普通单卡 statistics/tensor
比较的实现；服务器精确版本须用同名源码确认。我们的 CLI 不覆盖这些阈值。

源码前缀为 `python/msprobe/`。主要位置：

- `core/compare/indicator_analysis/algorithm.py`：所有 checker 与阈值。
- `core/compare/indicator_analysis/api_data.py::set_result`：状态只升级不降级。
- `core/compare/indicator_analysis/calculator.py::get_api_indicator_and_msg`：节点取其输入输出行最严重状态。
- `core/compare/indicator_analysis/utils.py::str2float`：百分数除以 100。
- `visualization/compare/mode_adapter.py::parse_result`：给调试侧节点添加比较字段。
- `visualization/graph/node_colors.py`、`visualization/utils.py`：图状态和颜色。

## 2. pass、warning、error、unmatched 不是四档误差

```text
两端采集数据
  ├─节点/参数未找到对应项 → unmatched：没有完成该项比较
  └─建立对应关系 → 按模式运行 checker
                     ├─命中 error → error
                     ├─无 error，命中 warning → warning
                     └─均未命中 → pass
```

`unmatched` 是匹配状态，不是“误差超过阈值”。缺少 API、模块类名不同、
融合与非融合实现、采集范围不同、调用顺序改变，都可能影响匹配；原因需要逐项确认。
同名 API 编号也不是语义等价证明。已匹配但 shape 不一致则属于结构 error。

行状态遵循 `error > warning > pass`；后运行的 warning 不覆盖已有 error。
节点取自身输入和输出行的最严重状态，错误原因可以同时保留多条。
普通模块的判定不是把所有子模块/API 递归取最差；因此模块边界 pass 可以与内部
API error/unmatched 同时出现。`api_collection` 集合节点有显式取子 API 最差指标的逻辑，
见 `graph_comparator.py::_handle_api_collection_index`，不要把两类节点混同。

图内部映射为 pass=0、warning=0.5、error=1；颜色分段的 0.3/0.6 是显示边界，
不是张量 30%/60% 的误差阈值。未匹配节点为灰色。

## 3. 全部普通统计/tensor checker

### 3.1 数值规则：变量、分支、例子和标记位置

以下输入/输出均指当前一个已匹配的模块或 API 的输入/输出，而非整个模型的
最初 token 和最终 logits。对每一对张量定义相对范数误差：

\[
e(X)=\frac{|\|X_N\|_2-\|X_B\|_2|}{\|X_B\|_2}.
\]

令 `e_in` 为有效输入比较项中最大的 e，`e_out` 为有效输出比较项中最大的 e。
“最大”是在同一操作的多个输入或输出之间选取，绝不是张量元素最大值，
也不是原始 Norm。比如输入有两项误差 0.1%、0.2%，则 e_in=0.2%。

**统计 warning 的两个分支：**

1. `e_in != 0`：只有 `e_out / e_in > 10` 才触发。例如 0.2% → 3%，
   放大 15 倍，warning；0.2% → 2%，恰好 10 倍，不触发这一条。
2. `e_in == 0`：不能除以零，改为判断 `e_out > 0.1`，即超过 10%。
   0% → 12% 触发；0% → 2% 不触发。输入范数误差为零不保证输入元素一致。

两种情况都把提示写在第一条输出记录。这是记录位置，不证明该输出是触发来源：
第二个输出的误差最大，也可能让第一条输出显示 warning。

**统计 error 独立执行：**当 e_in <10%，逐项检查每个输出，某输出 e >50%
就标记该项 error。例如输入 0%、输出 60%，同时触发 warning 和 error，
最终 error 优先。输入不需要达到 10%，条件恰好是小于 10%。

**tensor Cosine warning：**设 c_in 为所有有效输入 Cosine 的最小值，c_out
为所有有效输出 Cosine 的最小值，要求 `c_in >0.9` 且 `c_in-c_out >0.1`。
例如输入为 0.999/0.995，输出为 0.98/0.85，取 0.995 和 0.85，下降
0.145，触发 warning，并写在第一输出。这里用差值衡量相似度下降，
不是用张量值相减，也不是要求输出 Cosine <0.1。

**tensor 比例 error：**每个张量先计算元素相对误差 <0.001 的比例，再分别
取输入、输出中最小的比例。该 checkout 判断输入最小比例 >90% 且输出最小比例
<10%，然后标记第一输出。不要把“千分之一误差带”和“达标元素的百分比”混淆。

这些都是定位启发式。正确实现也可能因输入扰动经过敏感操作而放大误差；
没有触发则可能只是规则较宽或统计量丢失信息，不是严格验收通过。

| 模式 | error | warning |
| --- | --- | --- |
| statistics | 最大输入 NormRelativeErr <10%，某输出 >50%，标记该输出 | 输入误差非零时，最大输出/最大输入 >10；输入为零时，最大输出 >10%；标记第一输出 |
| tensor | 最小输入千分之一达标比例 >90%，最小输出比例 <10%；标记第一输出 | 最小输入 Cosine >0.9 且最小输入减最小输出 >0.1；标记第一输出 |

全部是严格不等号。tensor 的输出 <10% 是当前源码 `output_threshold=0.1`；
其注释、错误消息仍写 <0.6，不能根据提示文字反推实际阈值。

统计 `Max diff / Min diff / Mean diff / L2norm diff / MaxRelativeErr /
MinRelativeErr / MeanRelativeErr` **均没有独立数值通过线**。
tensor 的 Cosine >0.99、MaxAbsErr <0.001 是文档建议，不是这些 checker 中
独立注册的硬判定。EucDist、MaxRelativeErr、千分之五比例也没有统一硬通过线。

### 3.2 结构、属性和异常值规则

- `InfNanErrChecker`：检查 NPU Max/Min 的 NaN/Inf。该源码中只要 Bench Max 或
  Min 任一也是 NaN/Inf，就跳过此行，并非严格逐字段证明两端异常相同。
- `RequiresGradErrChecker`：已匹配且字段有效时，requires_grad 不同标记 error；
  源码对缺失/falsy 字段有跳过分支，不能把缺失当成核验成功。
- `DTypeErrChecker`：已匹配且类型字段有效时，不同标记 error；有 qkv/gate_up
  聚合项豁免，不应当作所有节点无条件规则。
- `ShapeErrChecker`：已匹配、有效 shape 不同标记 error；并行合并/特定后端
  使用不同 checker，当前单卡不应套用这些例外。
- `ParametersErrChecker`：有效、已匹配的非 tensor 输入标量值不同标记 error。
- MD5 模式单独使用 `CRC32ErrChecker`：有效校验值不同为 error；值缺失造成
  不能匹配为 warning。不要将该 warning 规则套到普通 statistics 的 unmatched。

`calculator.py` 还定义忽略列表：`empty/empty_like/numpy/to/__setitem__/
empty_with_format/new_empty_strided/new_empty/empty_strided` 等跳过相应检查；
`_reduce_scatter_base/_all_gather_base/all_to_all_single/batch_isend_irecv`
跳过依赖输入的规则。因此 pass 是“所适用规则未触发”，不是逐元素完全一致证明。

## 4. Max、Min、Mean、Norm 是哪一侧的

每一行是一个输入或输出张量，不是整个节点只有一份统计。
每个张量在每一侧各有一套 Max/Min/Mean/Norm：

```text
input.0
  调试侧（本实验 NPU）：Max_N、Min_N、Mean_N、Norm_N
  标杆侧（本实验 GPU）：Max_B、Min_B、Mean_B、Norm_B
  比较字段：两侧差值、相对误差、Result、Err_message
```

调试侧页签显示 NPU 原值；标杆侧页签显示 GPU 原值。配对表也可能以两行
展示两端，比较字段绑定在调试侧行，标杆行显示 `-`。不是把两边取平均后只显示
一个值。复制成纯文本会丢掉页签、行标识和配对关系，应结合侧别阅读。
若贴出的表没有侧别，不能仅凭重复节点名判断那一份是谁。

用户实例：NPU Norm=0.0028076171875，GPU Norm=0.00054168701171875；
差值=0.00226593017578125，相对差=418.309859%。这组数确实来自两个原值，
不是“一份 Norm 和自己比较”。输入误差为 0 时，它触发上述 error 和 warning。
但两端 shape 不同，应先处理配对问题，不能将该百分比认作同义算子的精度退化。

## 5. Norm 是张量的整体幅度，不是误差本身

把张量所有元素展开成 x，L2 范数为：

\[
\|x\|_2=\sqrt{\sum_{i=1}^n |x_i|^2}.
\]

例如张量 `[3,4]` 的 Norm 为 5。它不是均值，也不是模型中的 LayerNorm/RMSNorm
归一化操作。相同元素尺度下，元素数量增大会使 Norm 大致随平方根增长，因此不同
shape 的 Norm 不适合直接当作同义输出比较。

statistics 的 NormRelativeErr 为：

\[
\frac{|\|N\|_2-\|B\|_2|}{\|B\|_2}.
\]

完整 tensor 的相对 L2 误差则是：

\[
\frac{\|N-B\|_2}{\|B\|_2}.
\]

前者只比较两个长度，后者比较逐元素差，绝不能混用。

## 6. 为什么用 Norm，以及为什么绝对不能只靠 Norm

Norm 易于压缩为一个标量，避免传输完整张量；均值可能正负抵消而 Norm 不会。
输入幅度接近、输出幅度突然严重偏离，适合提示“从此处开始误差被放大”，
这是一条排查启发式，不是严密的等价性证明。

反例：N=`[1,2]`、B=`[2,1]`。两端 Max=2、Min=1、Mean=1.5、Norm=√5
全部相同，但元素差为 `[-1,1]`，差的 L2 范数为 √2，Cosine=4/5=0.8。
因此连四个统计量全相同，也不能证明 tensor 相同；专家索引顺序错误尤其可能漏过摘要。

正确使用顺序：统计筛查 → 核对模块/API 语义及 shape → 定点 tensor 采集 →
逐元素指标和任务输出/训练行为验证。先解决 unmatched，不通过放宽阈值掩盖结构错配。

## 7. 当前端到端比较与相同输入模块独立比较的区别

本项目此处约定“端到端”为一批输入完整经过模型前向和反向，不要求数千训练步。
“模块独立比较”为两端以相同模块输入、权重、状态执行；反向还要固定同一上游梯度。

当前 migration capture 的实际流程：

```text
同一 checkpoint + 同一批 token
  GPU：Embedding → Layer0 → Layer1 → … → loss → backward
                    沿途记录模块/API 输入输出
  NPU：Embedding → Layer0 → Layer1 → … → loss → backward
                    沿途记录模块/API 输入输出
```

中间没有用 GPU 张量覆盖 NPU 模块输入。因此后续模块收到的值可能已不同，
观测到的是上游误差传播、当前操作新增误差及其放大的共同结果。L0 是模块边界观测，
L1 是 API 边界观测，mix 同时记录二者；均不自动隔离上游误差。

真正独立比较需要：

```text
同一输入 X、权重 W、其他状态/参数
          ├─GPU 模块 → Y_GPU
          └─NPU 模块 → Y_NPU

同一 X、W，以及同一上游梯度 dY
          ├─GPU backward → dX_GPU、dW_GPU
          └─NPU backward → dX_NPU、dW_NPU
```

还需控制 dropout 掩码等随机行为和可变状态。只固定前向输入而不固定 dY，
反向仍混入后续层传播来的差异。当前 migration 工作流**没有实现这种逐模块固定
输入重放**；仅增加 `--tensor-log`、选 scope 或改为 tensor dump 都不会自动实现它。
API 预检和编译精度 checker 是其他工具路径，不能冒充本次 capture 已经执行的验证。

| 维度 | 选项 | 作用 |
| --- | --- | --- |
| 保存内容 | statistics / tensor | 摘要 / 完整张量，不决定是否端到端 |
| 观测粒度 | L0 / L1 / mix | 模块边界 / API 边界 / 两者 |
| 采集范围 | step、rank、scope/list | 选择时刻、进程和观测位置 |
| 执行方式 | 当前完整训练 / 独立重放 | 决定是否保留上游误差传播；独立重放需另外实现 |

因此 `statistics+mix` 和 `tensor+mix` 都可以是端到端连续运行。
定点采集完整 tensor 只是提供重放所需的证据，不等于重放已经发生。

## 8. 当前使用的指标词典：定义、公式、单位和用途

### 8.1 当前配置与符号

当前命令为 `--dump-task statistics --level mix --dump-steps 0,1`。
因此实际数值证据是模块/API 的输入输出摘要与摘要差异，不包含完整 tensor
逐元素误差。mix 不会把 statistics 自动升级为 tensor。

以下 N 为调试侧 NPU 张量，B 为标杆侧 GPU 张量。两者必须先确认属于同一
数学对象、shape 和元素对应关系一致。展开后元素数为 n。原始数值单位随
张量而定；不能把激活、梯度、概率等不同对象的绝对误差直接横向排名。

### 8.2 单侧张量属性与统计量（当前已采集）

| 字段 | 明确定义 | 怎样使用 |
| --- | --- | --- |
| name | 当前输入/输出参数位置，例如 input.0、output.1 | 是参数编号，不是数值指标；两端必须配对正确 |
| type | Python 对象类型，如 torch.Tensor、None 或标量类型 | 首先判断两侧是不是同类对象 |
| dtype | tensor 元素的数据类型 | BF16、FP32、int64 语义不同；不是允许误差大小的直接证明 |
| shape | 张量各维长度，元素总数为各维长度乘积 | 不同 shape 应先处理映射，不能直接解释误差百分比 |
| requires_grad | 此张量是否要求 autograd 跟踪梯度 | 不是“这个张量有没有非零梯度”；普通反向产生的梯度可为 False |
| Max | max_i X_i | 最大元素，不是最大绝对值；全负数时仍为负数 |
| Min | min_i X_i | 最小元素；结合 Max 判断范围和异常值 |
| Mean | sum_i X_i / n | 有符号均值；正负抵消后可能接近零 |
| Norm | sqrt(sum_i abs(X_i)^2) | 整体 L2 幅度；不除以 n，不是 RMS，不是归一化层 |

每侧各自计算这些统计。数值归约的计算 dtype、返回 dtype和舍入由采集实现
决定，不能把显示小数位数当成真实有效精度。null、空白、N/A 通常表示未提供
或无法计算，必须与数值 NaN 区分；索引 tensor 缺少 Mean/Norm 不等于索引损坏。

### 8.3 两侧摘要差异（当前已比较）

对统计函数 s∈{max,min,mean,L2norm}，定义：

\[
\Delta_s=s(N)-s(B),\qquad
r_s=\frac{|s(N)-s(B)|}{|s(B)|}.
\]

| 字段 | 具体公式 | 单位与判读 |
| --- | --- | --- |
| Max diff | max(N)−max(B) | 有符号差，负数表示 NPU 最大值更小 |
| Min diff | min(N)−min(B) | 有符号差，不是逐元素最小误差 |
| Mean diff | mean(N)−mean(B) | 有符号差，接近零仍可能掩盖元素错误 |
| L2norm diff | norm(N)−norm(B) | 有符号差；不是 norm(N−B) |
| MaxRelativeErr | abs(Max diff)/abs(max(B)) | 无量纲，界面按百分数显示；不是 tensor 最大逐元素相对误差 |
| MinRelativeErr | abs(Min diff)/abs(min(B)) | 无量纲，界面按百分数显示 |
| MeanRelativeErr | abs(Mean diff)/abs(mean(B)) | 无量纲；均值近零时比例非常敏感 |
| NormRelativeErr | abs(L2norm diff)/norm(B) | 无量纲；自动数值 checker 使用这一项，见 §3.1 |

例子：B=[1,2]，N=[1.1,2.2]。Max diff=0.2、Min diff=0.1、Mean diff=0.15，
L2norm diff=0.1√5；四种相对摘要误差均为 10%。自动 error 并不是“四项任一
超过 10%”：它使用输入到输出的联合条件，参见 §3.1。

分母为零时以上数学比值未定义；近零时很小的绝对差也会变成很大的百分比。
应查看两侧原值、绝对差和工具错误信息，不能将 inf/NaN 或缺失字段自行当作 0。
只有核实安装版本的特殊处理后才能解释其具体显示。

### 8.4 完整 tensor 指标（当前 statistics 没有提供）

先定义逐元素误差：

\[
AE_i=|N_i-B_i|,\qquad RE_i=\frac{|N_i-B_i|}{|B_i|}.
\]

| 指标 | 公式 | 意义与限制 |
| --- | --- | --- |
| Cosine | (N·B)/(norm(N)norm(B)) | 方向相似度，越接近 1 越好；整体乘正数可仍为 1，因此不能单独判断幅度正确 |
| EucDist | sqrt(sum_i (N_i−B_i)^2) | 完整逐元素差的 L2 范数，越接近 0 越好；受元素数量和幅度影响 |
| MaxAbsErr | max_i AE_i | 最坏元素的绝对误差，保留原数值单位 |
| MaxRelativeErr | max_i RE_i | 最坏元素的相对误差；golden 小值/零值会放大或使其未定义 |
| One Thousandth Err Ratio | count(RE_i<0.001)/n | 相对误差小于 0.1% 的元素占比；越接近 100% 越好 |
| Five Thousandth Err Ratio | count(RE_i<0.005)/n | 相对误差小于 0.5% 的元素占比；不是“误差值为0.5%” |

两个模式都叫 MaxRelativeErr，但含义不同：statistics 比较两个最大值，tensor
取所有元素相对误差的最大值。阅读 CSV 时必须先看采集 task。

例子：B=[1,2]，N=[1,2.002]，忽略表示舍入：AE=[0,0.002]，
RE=[0,0.001]，EucDist=0.002、MaxAbsErr=0.002、MaxRelativeErr=0.1%。
由于误差带用严格小于，千分之一比例为 1/2=50%，千分之五比例为100%。
实际浮点运算可能使边界略偏，因此测试阈值边界要考虑实际 dtype 表示。

Cosine 的反例：N=2B 时 Cosine=1，但 NormRelativeErr=100%，说明方向相同
不代表幅度正确。相反，§6 的元素置换例子四个摘要完全相同，Cosine却只有0.8。
两种例子说明需要组合证据，不存在一个指标万能通过。

本地源码的自动 Result 并不为每项设置独立阈值。Cosine>0.99 与
MaxAbsErr<0.001 是官方参考建议；实际 checker、版本与文字不一致见 §3。

### 8.5 判定与完整性字段不是数值精度指标

| 字段/状态 | 含义 |
| --- | --- |
| Requires_grad Consistent | 两端属性是否一致，不是梯度张量数值一致 |
| Result | 当前适用 checker 的 pass/warning/error；未命中不代表元素完全一致 |
| Err_message | 命中的规则及结构问题，可同时包含多条 |
| unmatched | 尚未建立对应关系，不是第四档数值误差 |
| Stack/Data_Name | 定位调用和原始数据的证据，不参与数值优劣排名 |
| MD5/CRC-32 | 另一个校验模式的内容指纹；当前 statistics 没有启用，不能当作误差大小 |
| 工具成功退出/complete.json | 工具执行和产物完整，不等于精度通过 |

当前不能从这些表中推出长程收敛、loss 轨迹通过率或独立模块因果定位结论。
完整前向/反向连续误差定位与相同输入独立模块重放的区别见 §7。

## 9. 多 step、多 rank 与图合并是三种不同的可视化语义

本次命令把 `candidate-r1/official` 和 `reference-r1/official` 交给
`msprobe graph_visualize`。目录结构是：

```text
official/
├── step0/
│   ├── rank0/{construct.json,dump.json,stack.json}
│   ├── rank1/{...}
│   └── ... rank7/{...}
└── step1/
    ├── rank0/{...}
    └── ... rank7/{...}
```

官方文档规定，传入 rank 目录是单 rank，传入 step 目录是多 rank，传入
`official` 这种 step 的父目录是多 step。源码
[`check_directory_content`](https://github.com/Ascend/msprobe/blob/master/python/msprobe/visualization/utils.py)
先识别目录内容；随后
[`_compare_graph_steps`](https://github.com/Ascend/msprobe/blob/master/python/msprobe/visualization/graph_service.py)
遍历相同 step，默认再调用 `_compare_graph_ranks`。后者取两侧 rank 名称交集并按
编号配对。因此本次真实执行逻辑是：

```text
step0: candidate rank0 ↔ reference rank0
       candidate rank1 ↔ reference rank1
       ...
       candidate rank7 ↔ reference rank7
step1: candidate rank0 ↔ reference rank0
       ...
       candidate rank7 ↔ reference rank7
```

每个 rank 对各自从 `construct.json` 重建层级图，用 `dump.json` 比较统计值，
并读取 `stack.json` 提供调用定位。源码 `_export_compare_graph_result` 把这些
step/rank 图写入同一个时间戳命名的 `compare_*.vis.db`，同时记录 `step_list`、
`rank_list`、当前 step 和 rank。TensorBoard 打开的不是“一张由八卡合成的图”，
而是一个包含多张 step/rank 图的数据库；应先在界面的 step/rank 选择器确认当前
查看的是哪一对。

### 9.1 为什么 FSDP8 看起来与单卡接近

FSDP2 分片的是参数、梯度和优化器状态的物理存储与通信生命周期，不会把八个
Transformer block 分别交给八个 rank。每个 rank 仍执行同一套完整模块调用结构：

```text
rank0: embedding → layer0 → ... → layer7 → output
rank1: embedding → layer0 → ... → layer7 → output
...
rank7: embedding → layer0 → ... → layer7 → output
```

所以 `construct.json` 重建出的逻辑 Module/API 层级天然接近单卡；差别主要存在于
张量局部形态、FSDP hook 触发的 all-gather/reduce-scatter、数值和调用栈，而不是
模型主干多出七份层。msProbe 的图也不是 DeviceMesh/DTensor placement 可视化，
不能期待它自动画出 `Shard(0)`、`Replicate()` 或每段参数属于哪张卡。

本次 GPU 与 NPU 两侧又使用相同的 FSDP8 拓扑，因此 rank0 对 rank0 的模型结构
理应高度相似。这正是同拓扑迁移对比希望得到的结果，不表示工具只读取了 rank0。
判断是否真的处理了八卡，应查看构图 `runtime.log` 中的 rank 处理记录，以及
`.vis.db` 界面的 rank 列表，而不能只凭主干图外观判断。

### 9.2 “开启不同切分策略下的图合并”才会进入 GraphMerger

普通多 rank 批量比对与图合并不是一回事。只有显式传入
`--rank_size`、`--tp`、`--pp`（可选 `--vpp`、`--order`）时，源码才令
`parallel_merge=True`，走 `_compare_graph_ranks_parallel`：先逐 rank 构图，
再用 `GraphMerger` 合并两侧，最后比较合并图。

官方目前明确支持的切分是 TP、PP、VPP；不支持 CP、EP。图合并主要面向
Megatron、MindSpeed-LLM，其他套件的效果需要验证。DP 数必须一致，因为 DP
副本不会合成一个更大的模型图：例如 `rank_size=8,tp=1,pp=1` 意味着 dp=8，
结果仍是八个副本，而不是一张“FSDP 全局图”。官方依据见
[PyTorch 分级可视化构图比对](https://github.com/Ascend/msprobe/blob/master/docs/zh/user_guide/accuracy_compare/pytorch_visualization_instruct.md)。

因此对当前 TorchTitan 拓扑应采用以下边界：

| 拓扑 | 当前可靠用法 | 是否把所有 rank 合成一张图 |
| --- | --- | --- |
| single | 单 rank 双图比对 | 本来只有一张 |
| ddp/fsdp/hsdp | 同 rank 批量比对 | 否；每个 DP/FSDP 副本独立查看 |
| TP/PP | 默认先做同 rank 批量比对 | 否；官方合并算法不是 TorchTitan 已验证能力 |
| CP/EP及其组合 | 同 rank 批量比对，结合普通 compare 解释 | 否；官方明确不支持对应图合并 |
| Megatron/MindSpeed 的 TP/PP/VPP | 配齐并行参数后可用官方 GraphMerger | 是，但 DP 副本数必须一致 |

当前 GLM wrapper 虽然底层 `graph_visualize_command()` 已能组装图合并参数，公开
`migration_benchmark.py` CLI 并未把这些参数传给 `run_graph_visualization()`；所以
本次 `--graph-visualize --topology all` 明确属于“多 step、多 rank 批量比对”，
不是图合并。MindStudio 页面里的“开启不同切分策略下的图合并”是官方工具的另一
模式；对 FSDP8 不应开启，也不能借用 Megatron 的 TP/PP 参数假装还原 TorchTitan
FSDP 分片。

### 9.3 页面上其他开关改变什么

| 页面选项 | CLI | 实际作用 |
| --- | --- | --- |
| 单个算子日志打屏 | `--is_print_compare_log` / `-tensor_log` | 只支持 tensor dump；本次 statistics 不适用 |
| 不同切分策略图合并 | `--rank_size --tp --pp ...` | 进入 GraphMerger，不是普通多 rank 批量构图 |
| 跨框架/跨套件比对 | `--layer_mapping` / `-lm` | 用 Layer 映射辅助结构不同的两端匹配；同名同类模块通常无需开启 |
| 溢出检测 | `--overflow_check` / `-oc` | 给图节点标记输入/输出 Inf、NaN 传播等级 |
| 模糊匹配 | `--fuzzy_match` / `-fm` | 节点无法严格一一对应时尝试结构匹配；可能扩大匹配范围，不能无条件开启 |

本节源码链路与官方说明核对日期为 2026-09-09。源码 checkout 与服务器安装包
版本不同时，应以服务器 `runtime.log` 中保存的最终命令和实际生成的 rank/step
索引为准。
