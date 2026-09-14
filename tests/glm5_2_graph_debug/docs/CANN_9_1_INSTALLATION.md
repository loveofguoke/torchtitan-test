# CANN 9.1.0 独立安装与激活

## 目标

在 `glm5-npu-dev` 中增加一套独立的 CANN 9.1.0 GA，供图模式调试进程使用，同时：

- 不覆盖现有 CANN 9.0.0；
- 不修改宿主机 Driver；
- 不修改 `/usr/local/Ascend/cann`；
- 不修改 `/usr/local/Ascend/ascend-toolkit/latest`；
- 不污染容器的默认 shell 环境。

最终目录：

```text
/usr/local/Ascend/cann-9.1.0
```

## 前置检查

本次来源容器为同一宿主机上的 `yhr_cann9.1.0`，目标容器为
`glm5-npu-dev`。复制前确认：

```bash
docker exec yhr_cann9.1.0 test -d /usr/local/Ascend/cann-9.1.0
docker exec glm5-npu-dev npu-smi info
docker exec yhr_cann9.1.0 npu-smi info
```

两边使用同一宿主 Driver `26.0.rc1`，目标设备为 Ascend 910B2，来源 CANN 为
9.1.0 GA。只有在架构、Driver 和 CANN 来源确认兼容时才应复制已有安装目录；跨主机
或来源不明时应改用华为官方 CANN 9.1 安装包，并按照该安装包的 `--help` 指定独立
安装目录。

## 本次实际安装命令

在宿主机执行流式 `docker cp`，不在宿主机生成 9 GB 临时副本：

```bash
docker cp \
  yhr_cann9.1.0:/usr/local/Ascend/cann-9.1.0 - \
  | docker cp - glm5-npu-dev:/usr/local/Ascend
```

复制完成后：

```bash
docker exec glm5-npu-dev \
  test -f /usr/local/Ascend/cann-9.1.0/set_env.sh

docker exec glm5-npu-dev \
  du -sh /usr/local/Ascend/cann-9.1.0
```

实测安装目录约 9 GB。

## 没有来源容器时：使用官方 run 包

从华为官方渠道获取与服务器架构、NPU 型号匹配的 CANN 9.1.0 GA 安装包。本机为
`aarch64`、Ascend 910B2，需要 toolkit 包和 910B kernels/ops 包。不同发布渠道的
文件名可能不同，先以实际下载文件执行 `--help`，确认支持 `--install-path`：

```bash
chmod +x Ascend-cann-toolkit_*_linux-aarch64.run
chmod +x Ascend-cann-kernels-910b_*_linux-aarch64.run

./Ascend-cann-toolkit_*_linux-aarch64.run --help
./Ascend-cann-kernels-910b_*_linux-aarch64.run --help
```

在目标容器内以 root 安装到独立目录，先 toolkit、后 910B kernels/ops：

```bash
./Ascend-cann-toolkit_*_linux-aarch64.run \
  --install --install-path=/usr/local/Ascend/cann-9.1.0

./Ascend-cann-kernels-910b_*_linux-aarch64.run \
  --install --install-path=/usr/local/Ascend/cann-9.1.0
```

若下载包的 `--help` 给出的参数形式与上面不同，以安装包自身帮助为准，不要省略
独立安装目录。安装完成后检查实际元数据：

```bash
grep -E '^(package_name|version|arch|path)=' \
  /usr/local/Ascend/cann-9.1.0/aarch64-linux/ascend_toolkit_install.info \
  /usr/local/Ascend/cann-9.1.0/aarch64-linux/ascend_ops_install.info
```

本次已安装目录记录为：

```text
package_name=Ascend-cann-toolkit  version=9.1.0  arch=aarch64
package_name=Ascend-cann-910b-ops version=9.1.0  arch=aarch64
path=/usr/local/Ascend/cann-9.1.0
```

随后继续执行下文的 SHA-256、ACL 类型、全局软链和隔离环境检查。不要运行会修改
`/usr/local/Ascend/cann` 或 `ascend-toolkit/latest` 的全局切换命令。

## 完整性校验

分别在来源和目标容器执行：

```bash
sha256sum \
  /usr/local/Ascend/cann-9.1.0/compiler/version.info \
  /usr/local/Ascend/cann-9.1.0/include/acl/acl_rt.h \
  /usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so \
  /usr/local/Ascend/cann-9.1.0/set_env.sh
```

本次确认的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `compiler/version.info` | `f20359296d5a7127f841aa35f4a788137f444dc36d72641e948cb9e39e47e4f4` |
| `include/acl/acl_rt.h` | `22c603cb80ab582006d4f26697ceb14c059d7f3613cf13b3c72aeb0d3d36e3c1` |
| `lib64/libascendcl.so` | `123d67c313f743e5d6f2e856f40c24d5edac376975cf0a7a1019e079b4171480` |
| `set_env.sh` | `de50f1bf5b960f8305792ef407e3bfc3fc1b6d82931d0feae39ac4c965690aaf` |

同时确认 9.1 头文件包含图模式 launcher 需要的新 ACL 类型：

```bash
grep -nE 'aclmdlRICondHandle|aclmdlRICondTaskParams' \
  /usr/local/Ascend/cann-9.1.0/include/acl/acl_rt.h
```

## 不修改全局软链

安装前后都应检查：

```bash
docker exec glm5-npu-dev readlink -f /usr/local/Ascend/cann
docker exec glm5-npu-dev readlink -f /usr/local/Ascend/ascend-toolkit/latest
```

本次两者仍解析到：

```text
/usr/local/Ascend/cann-9.0.0
```

不要为了图模式调试执行 `ln -sfn` 切换全局 CANN，否则会影响容器中其他实验。

## 按进程激活

推荐直接使用调试脚本，它会通过 `env -i` 创建干净子进程，再加载 CANN 9.1：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology single
```

需要手工验证时：

```bash
env -i \
  HOME=/root USER=root LANG=C.UTF-8 \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /bin/bash --noprofile --norc

set +u
source /usr/local/Ascend/cann-9.1.0/set_env.sh
set -u
export PATH=/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin:$PATH
export PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:\
${LD_LIBRARY_PATH:-}
export TORCHTITAN_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=0
unset CUDA_VISIBLE_DEVICES
```

CANN 自带的 `set_env.sh` 会读取若干可选变量，因此在启用 Bash `nounset` 时需要在
source 前后临时执行 `set +u` / `set -u`。

在 `env -i` 干净子进程中，本次 CANN 9.1.0 的 `set_env.sh` 实际导出如下。这里把
完整值列出用于审计；日常运行应 source `set_env.sh`，不要手工复制这组内部路径，
否则安装目录升级后很容易漏项：

```bash
export ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0
export TOOLCHAIN_HOME=/usr/local/Ascend/cann-9.1.0/toolkit

export PATH=/usr/local/Ascend/cann-9.1.0/bin:\
/usr/local/Ascend/cann-9.1.0/tools/ccec_compiler/bin:\
/usr/local/Ascend/cann-9.1.0/tools/profiler/bin:\
/usr/local/Ascend/cann-9.1.0/tools/ascend_system_advisor/asys:\
/usr/local/Ascend/cann-9.1.0/tools/show_kernel_debug_data:\
/usr/local/Ascend/cann-9.1.0/tools/msobjdump:$PATH

export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/lib64:\
/usr/local/Ascend/cann-9.1.0/lib64/plugin/opskernel:\
/usr/local/Ascend/cann-9.1.0/lib64/plugin/nnengine:\
/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:\
/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64/driver

export PYTHONPATH=/usr/local/Ascend/cann-9.1.0/python/site-packages:\
/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe
```

common 随后再把图模式 Conda 和 ATB 放到上述环境前面；HCCL、Inductor、
NPUGraphs、cache 和 launcher-only 兼容项的完整 export 见
`GRAPH_MODE_COMMON.md`。

## 验证实际加载路径

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env
```

应看到：

```text
CANN_ROOT=/usr/local/Ascend/cann-9.1.0
CONDA_ENV=/root/miniconda3/envs/torchtitan-0803-graph-adapt
```

训练日志中的 `LD_LIBRARY_PATH` 不应出现 CANN 9.0 或
`ascend-toolkit/latest`。必要时还可在 Python 进程内检查 `/proc/self/maps`。
Inductor、NPUGraphs、HCCL 和缓存所需的完整 export 以及所有可覆盖项见
`GRAPH_MODE_COMMON.md`；不要只 source CANN 后遗漏这些 profile 变量。

## 容器重建后的恢复

CANN 9.1.0 安装在容器可写层；删除并重建 `glm5-npu-dev` 后需要重新执行复制和校验。
Conda 环境也位于容器内，应一并确认。编译缓存位于宿主机主目录挂载下：

```text
/home/y50064852_yyb/.cache/torchtitan-test/graph_mode/
```

因此重建容器后，只要软件版本和路径保持一致，Inductor/Triton 缓存仍可继续复用。
