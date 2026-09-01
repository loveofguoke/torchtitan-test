# MindStudio 官方工具源码安装、版本锁定与环境检查

> 状态：安装与复现方案。
>
> 本文优先给出“源码可阅读、版本可锁定”的方案，但不主张把 driver、firmware、CANN、torch_npu 和所有 GUI 工具都强行从源码重编译。硬件运行栈必须先满足官方兼容矩阵；源码 checkout 用于学习、构建支持源码安装的工具，以及把实验绑定到明确 commit。

## 1. 安装原则

### 1.1 源码优先不等于全部源码编译

把环境分成三层：

| 层 | 示例 | 建议 |
| --- | --- | --- |
| 硬件运行栈 | NPU driver、firmware、CANN Toolkit/OPS、HCCL | 使用服务器/容器已验证的官方兼容组合，不在测试脚本中安装或升级 |
| Python 训练栈 | Python、torch、torch_npu、torchtitan、TorchTitanTurbo | 三仓源码 editable 安装；torch/torch_npu 必须和 CANN 匹配 |
| MindStudio 工具 | msProbe、msprof-analyze、msInsight、msOT 等 | 能源码构建的使用固定 commit 源码；GUI 或随 CANN 发布的组件按官方包安装，同时保留源码用于阅读 |

错误做法包括：

- 为了得到最新 msProbe，顺手升级 torch_npu 或 CANN；
- 在同一环境同时装两个不同来源/版本的同名工具；
- 永远跟随 `master`，但 manifest 不记录 commit；
- 把整个官方源码复制进 torchtitan-test；
- 只同步 Python 虚拟环境，不记录 driver/firmware/CANN；
- doctor 发现缺包后自动 `pip install`，导致实验环境静默变化。

### 1.2 为什么使用仓外源码目录加 lock

推荐布局：

```text
/data/yyb_hys/
├── torchtitan/
├── TorchTitanTurbo/
├── torchtitan-test/
└── mindstudio-tools/
    ├── msprobe/
    ├── msprof-analyze/
    ├── mstt/                 # 若选用聚合仓库
    ├── msinsight/            # 阅读/构建所选版本
    ├── msot/                 # 算子工具，按需
    └── wheels/               # 本地构建包，不提交 Git
```

torchtitan-test 只保存一个小型 lock/manifest，记录外部目录、remote、commit、版本和构建方式。这样：

- 不把大仓库和构建产物塞进实验仓库；
- 不让 Git subtree 把上游历史复制进当前仓库；
- 可以在 GPU、NPU、CPU 分析机分别 clone 相同 commit；
- 升级工具时 lock 的 diff 清楚；
- release 只携带 lock 和结果，不携带整个工具源码。

如果团队强制要求 Git 追踪外部源码关系，**submodule 比 subtree 更合适**，因为 submodule 只记录 commit 指针。但本项目优先采用“仓外 checkout + lock”，避免现有三仓再次嵌套第四、第五个 Git 仓库。

## 2. 版本锁定文件

仓库已经提供 `tests/glm5_2_mindstudio/toolchain.lock.json`。它记录：

- runtime compatibility 选择原则；
- msProbe、msprof、msprof-analyze、msinsight 的官方 URL、外部目录和 ref；
- Python distribution/import/CLI 名；
- 每个工具的 bootstrap 策略；
- 哪些工具是精度流程必需、性能流程必需或只读可选源码。

当前 lock 的探索默认可能指向官方 `master`。正式实验必须把 `ref` 改成经过
验证的 tag/commit，并填完整 `commit`/版本字段；否则只能复现“安装当天 master”，
不能严格复现实验。

bootstrap 实际执行后会在仓库外工具根写入 `toolchain.resolved.json`。两类文件的
职责不同：

```text
tests/glm5_2_mindstudio/toolchain.lock.json
  = 期望来源、ref/commit、构建策略和运行入口（纳入代码评审）

/data/yyb_hys/mindstudio-tools/toolchain.resolved.json
  = 本次实际 checkout HEAD、构建产物、Python、命令记录（随实验归档）
```

正式实验不能只写 `ref=master` 后依赖 resolved manifest 补救；必须先把 lock 固定到
经过验收的 tag 或完整 commit，再确认 resolved 中实际 HEAD 与之相等。探索阶段可
使用 master 快速验证新能力，但产物只能标为 exploratory，不能与已锁定正式实验
混在同一 generation。

锁定要求：

1. 探索完成后把 branch ref 替换为实际 tag 或完整 commit；
2. 同一实验的 reference/candidate manifest 保存各自工具元信息；
3. compare 使用的工具版本也要记录，不能只记采集端；
4. 若两端工具版本不同，报告必须显式提示，不能静默继续；
5. Git dirty 状态可记录但不应被误作“不同实验”的唯一判据；真正配对依赖 fixture/config/tool hash。

## 3. NPU 服务器运行栈准备

### 3.1 先确认已有 CANN 环境

不在 benchmark 内修改系统环境。进入官方容器或已配置 shell 后做只读检查：

```bash
npu-smi info

python - <<'PY'
import sys
import torch
import torch_npu

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("npu_available:", torch.npu.is_available())
print("compile_backends:", torch.compiler.list_backends())
PY

command -v msprof || true
msprof --help >/dev/null 2>&1 && echo "msprof: available" || echo "msprof: unavailable"
```

还要记录而不是仅打印：

- driver/firmware 版本；
- CANN Toolkit 和 OPS 版本、安装路径；
- HCCL 版本；
- SoC 型号；
- torch/torch_npu build string；
- `ASCEND_HOME_PATH`、库搜索路径的有效摘要；
- 编译器和 Triton-Ascend 版本（图模式/自定义 kernel 时）。

当前项目资料中的图模式实验基于 CANN 9.1 系列，并使用与该数据格式兼容的 MindStudio Insight 26.1 系列桌面端。新环境仍需按所选官方兼容矩阵复核，不能把这一组合推广为所有服务器的固定答案。

### 3.2 三仓源码安装

在已经激活的目标 Python 环境中分别执行：

```bash
cd /data/yyb_hys/torchtitan
python -m pip install -e . --no-deps

cd /data/yyb_hys/TorchTitanTurbo
python -m pip install -e . --no-deps

cd /data/yyb_hys/torchtitan-test
python -m pip install -r requirements-test.txt
```

`--no-deps` 只适用于已由兼容环境准备好 torch/torch_npu 等核心依赖的源码重装。首次建环境必须依据三个仓库各自依赖文件安装缺失包，并用 `python -m pip check` 审计；不能把 `--no-deps` 当成“依赖不重要”。

```bash
python -m pip check
python - <<'PY'
import torchtitan
import torchtitanturbo
print("torchtitan:", torchtitan.__file__)
print("torchtitanturbo:", torchtitanturbo.__file__)
PY
```

## 4. 克隆并锁定官方源码

仓库提供一个安全 bootstrap：目标目录必须在 torchtitan-test 仓库外；已有 checkout
必须 origin 匹配、工作区 clean；更新只允许 fast-forward。先 dry-run：

```bash
cd /data/yyb_hys/torchtitan-test
python tools/bootstrap_mindstudio_toolchain.py \
  --root /data/yyb_hys/mindstudio-tools \
  --dry-run
```

确认命令、remote、ref 和安装环境后执行：

```bash
python tools/bootstrap_mindstudio_toolchain.py \
  --root /data/yyb_hys/mindstudio-tools

export TORCHTITAN_MINDSTUDIO_TOOLCHAIN_ROOT=/data/yyb_hys/mindstudio-tools

# 记录本次实际解析结果；正式实验归档该文件及 lock
python -m json.tool \
  /data/yyb_hys/mindstudio-tools/toolchain.resolved.json | less
```

默认选择 lock 中 `bootstrap.default=true` 的工具。需要同时 clone 可选的 Insight
源码辅助脚本：

```bash
python tools/bootstrap_mindstudio_toolchain.py \
  --root /data/yyb_hys/mindstudio-tools \
  --include-optional
```

`mstt` 与独立工具仓库可能存在聚合/迁移关系。一个环境只选定一种权威来源，并在 lock 中记录。例如选独立 `msprof-analyze` 后，不应又从 `mstt/profiler/msprof_analyze` 安装第二份同名包。

官方源码入口：

- [msProbe](https://gitcode.com/Ascend/msprobe)
- [msprof-analyze](https://gitcode.com/Ascend/msprof-analyze)
- [msTT 聚合仓库](https://gitcode.com/Ascend/mstt)
- [MindStudio Insight](https://gitcode.com/Ascend/msinsight)
- [MindStudio Operator Tools](https://gitcode.com/Ascend/msot)
- [MindStudio Inference Tools](https://gitcode.com/Ascend/msit)

## 5. msProbe 源码安装

当前 lock 按官方安装指南执行 `build.py`，安装生成的 wheel：

```bash
cd /data/yyb_hys/mindstudio-tools/msprobe
git status --short
git rev-parse HEAD

python -m pip install uv
node --version   # 官方推荐 20.19.3
npm --version    # 官方推荐 10.8.2
python build.py -e include-mod=tb_graph_ascend
python -m pip install ./artifacts/mindstudio_probe*.whl
```

等价的自动入口是上一节 bootstrap。该脚本会记录实际执行命令和 wheel 路径；
不会假设 msProbe 支持 editable install。`tb_graph_ascend` 是官方
`graph_visualize` 生成 `.vis.db` 后在 TensorBoard 中分级展示所需的插件；缺少
Node/npm 时可以先只做 dump/compare，但不能声称分级图可视化环境完整。

安装后做能力检查：

```bash
command -v msprobe
msprobe --help
python -m pip show mindstudio-probe
python - <<'PY'
import msprobe
print("msprobe module:", msprobe.__file__)
PY
```

升级 revision 后必须重新阅读该 revision 安装指南；如果官方更改 build 入口，先
更新 lock/bootstrap 测试，再运行正式实验。

## 6. msprof-analyze 源码构建

当前 bootstrap 优先安装仓库声明的 `requirements/build.txt`，不存在时回退
`requirements.txt`，然后 editable 安装锁定的源码 checkout：

```bash
cd /data/yyb_hys/mindstudio-tools/msprof-analyze
git rev-parse HEAD

python -m pip install -r requirements/build.txt
python -m pip install --no-deps --editable .

msprof-analyze --help
python -m pip show msprof-analyze
```

如果所锁定 revision 没有 `requirements/build.txt`，以 bootstrap 输出的实际 fallback
为准。不要把 build 目录、wheel 或 editable 源码提交到 torchtitan-test；artifact
会记录 source path、commit、dirty 状态和 lock hash。

官方入口：[msprof-analyze 源码](https://gitcode.com/Ascend/msprof-analyze)

## 7. MindStudio Insight 安装与源码阅读

Insight 是交互式 GUI。对于无图形界面的 NPU 服务器，推荐分工：

```text
NPU 服务器：采集、解析、生成可导入数据
Windows/桌面：安装兼容版本 Insight，导入产物交互分析
源码 checkout：学习数据格式、视图和兼容逻辑
```

源码 clone 不代表桌面 GUI 已安装成功。正式使用应按照兼容 CANN 的官方安装包和安装指南；Windows 可能需要管理员授权、企业策略放行或 IT 支持，不能通过测试脚本绕过安全策略。

验证点：

- Insight 版本与 CANN/profiler 数据格式兼容；
- 能导入一个最小单卡 profile；
- timeline 中能看到 Host/runtime/NPU kernel；
- 分布式数据能正确识别 rank；
- 大型原始数据先在服务器完成必要解析，桌面只同步需要导入的产物。

入口：

- [MindStudio Insight 源码](https://gitcode.com/Ascend/msinsight)
- [Insight 系统调优数据说明](https://gitcode.com/Ascend/msinsight/blob/master/docs/en/user_guide/system_tuning.md)

## 8. 算子与推理工具按需安装

当前标准精度流程不要求安装整个 msOT/msIT。只有对应实验进入算子或推理阶段才安装：

### 8.1 msOT

```bash
cd /data/yyb_hys/mindstudio-tools
git clone https://gitcode.com/Ascend/msot.git
git -C msot checkout <MSOT_PINNED_COMMIT>
cd msot
python build.py
```

构建和安装必须遵循所选 revision 的官方说明。msOT 常与 CANN/编译器/SoC 强耦合，不能因为 Python build 成功就认为目标 NPU 可用。

### 8.2 msIT

```bash
cd /data/yyb_hys/mindstudio-tools
git clone https://gitcode.com/Ascend/msit.git
git -C msit checkout <MSIT_PINNED_COMMIT>
```

推理工具应在单独环境验证，避免其依赖改写训练环境。官方文档已存在组件迁移/日落，必须按所选版本确认当前权威工具。

## 9. Python 依赖分层

不建议把所有工具依赖塞进根 `requirements-test.txt`。目标分层：

```text
requirements-test.txt                  # 基础单元测试
tests/glm5_2_mindstudio/...            # 官方工作流 adapter 的轻量项目依赖
官方源码 requirements/build            # 工具自己的依赖，由锁定 revision 管理
```

关键原则：

- CANN/torch/torch_npu 不由普通 requirements 自动升级；
- matplotlib、pandas、openpyxl 等分析依赖应和官方工具要求兼容；
- GPU reference 环境不应安装 torch_npu 才能运行基础 capture adapter；
- CPU compare 环境只安装 compare 所需依赖；
- 训练与桌面 GUI 环境分离；
- 每次安装后执行 `python -m pip check`。

## 10. doctor

doctor 已实现为只读 JSON 检查，不自动修复。按工作端选择 scope：

```bash
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope accuracy-gpu

python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope accuracy-npu

# 默认 msProf 采集机
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope performance-capture

# torch_npu.profiler 深度采集机
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope performance-torch-npu-capture

# CPU/独立离线分析机
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope performance-analysis

python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope full
```

也可以显式指定外部源码根，或只输出紧凑 JSON：

```bash
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope accuracy-npu \
  --toolchain-root /data/yyb_hys/mindstudio-tools \
  --compact

python -m tests.glm5_2_mindstudio.toolchain metadata \
  --scope accuracy-npu --compact
```

`doctor` 用 required/failure 语义判断 readiness；`metadata` 只采集可写入 manifest
的版本、路径、source commit 和能力信息，不把缺少可选 GUI/source checkout 误判为
训练失败。未传 `--scope` 时默认是 `full`。

`accuracy-gpu` 不把 torch_npu/CANN/msprof 当成 GPU capture 必需条件；
`accuracy-npu` 要求 torch_npu/CANN；`performance-capture` 额外要求 msProf，
`performance-torch-npu-capture` 只要求 torch/torch_npu/CANN，
`performance-analysis` 只要求 msprof-analyze；`full` 要求精度与性能全栈。所有
scope 都会报告可选工具，只是 required/failure 语义不同。

doctor 不替代 resolved manifest：doctor 回答“当前 shell 现在能否运行”，resolved
回答“bootstrap 当时实际安装了什么”。正式 capture 前应同时检查 lock、resolved、
doctor；任一工具 checkout 或安装包变化都应建立新 generation。

### 10.1 通用检查

- Python 版本、可执行路径；
- Python、torch、msProbe import 与 distribution 版本；
- external source path、remote、commit 是否匹配 lock；
- lock 文件 hash 和工具 checkout 状态。

### 10.2 NPU 检查

- CANN 关键环境根与版本文件；
- torch、torch_npu；
- `msprof`；
- msProbe、msprof-analyze；
- 可选 `npu-smi` 和 Insight 源码脚本。

### 10.3 GPU 检查

- torch 和 msProbe；
- 不要求 torch_npu、CANN 或 Insight。

### 10.4 CPU compare/Windows 检查

- 当前 doctor 负责工具/环境 readiness；artifact/hash 匹配由 compare workflow 负责；
- Insight GUI 安装和本地磁盘由桌面侧人工检查。

doctor 失败分级：

- `ERROR`：该工作流无法运行，例如 NPU 不可用、工具缺失、lock commit 不匹配；
- `WARN`：可以运行但证据不完整，例如 Git dirty、Insight 未安装、可选 advisor 缺失；
- `INFO`：版本、路径和设备摘要。

## 11. 实验 manifest 必须记录什么

每个 capture 至少保存：

```text
experiment_id / generation_id / config_hash
fixture_hash / token_plan_hash / checkpoint identity
role / device / topology / rank selection
training command / environment allowlist
torchtitan / turbo / test commit and dirty summary
python / torch / torch_npu
driver / firmware / CANN / HCCL
msProbe / msprof / msprof-analyze / Insight version or commit
official config file hash
start/end/exit status
all attachments with size and sha256
```

环境变量只保存白名单摘要，禁止把 proxy 凭据、token、私有 registry 密码或完整环境直接写进 artifact。

## 12. 升级流程

一次只升级一个层次：

1. 新建 toolchain profile/lock，不覆盖旧 lock；
2. 在最小单卡 smoke 上运行 doctor；
3. 运行 msProbe eager/eager 自检；
4. 运行 GPU/NPU 单卡 module/API 最小比较；
5. 运行一个最小 profiler 和 Insight 导入；
6. 运行 NPU eager/compile；
7. 最后再扩到分布式和 all topology；
8. 比较新旧工具输出 schema，更新 parser 测试；
9. 保留旧 profile 以复现历史报告。

禁止在活动 capture 进行中修改 lock 或 editable checkout 的 commit。否则同一目录可能
含有两套工具语义，应当终止该 generation 并用 `--force` 建立新 generation。

## 13. 安全与数据治理

msProbe tensor dump、checkpoint、token plan、source stack、profile op args 都可能包含敏感信息。必须：

- 默认使用合成/授权数据；
- 默认只采统计量或必要 tensor；
- raw 数据保存在受控服务器；
- 对外同步前生成 compact artifact 并审查；
- 不把凭据、完整环境、私有数据路径写入报告；
- 不将 raw dump 直接提交 Git/GitHub Release；
- 记录删除策略和保留期限；
- 运行官方工具时使用普通用户和最小权限。

## 14. 官方参考

- [MindStudio 总入口](https://www.hiascend.com/document/detail/zh/mindstudio/latest/index/index.html)
- [msTT](https://gitcode.com/Ascend/mstt)
- [msProbe](https://gitcode.com/Ascend/msprobe)
- [msprof-analyze](https://gitcode.com/Ascend/msprof-analyze)
- [MindStudio Insight](https://gitcode.com/Ascend/msinsight)
- [msOT](https://gitcode.com/Ascend/msot)
- [msIT](https://gitcode.com/Ascend/msit)
