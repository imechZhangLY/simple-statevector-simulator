# Benchmarks

本目录包含两类基准：

| 文件 | 用途 |
|---|---|
| [benchmark_backends.py](benchmark_backends.py) | 项目内部后端对比（NumPy / Torch，CPU / CUDA） |
| [framework_comparison.py](framework_comparison.py) | 与 qulacs、qiskit-aer 的跨框架对比 |
| [results/](results/) | 已记录的结果 JSON 与折线图 |

## 快速开始

脚本**不创建也不检查环境**，只设置线程环境变量、运行基准并输出 JSON 与折线图。运行前先激活对应环境，创建方法见 [envs/README.md](../envs/README.md)。

每个场景对应一个环境：

| 场景 | 环境 | 平台 |
|---|---|---|
| `cpu-single`、`cpu-multi` | `.venv-bench-cpu` | Windows, Linux |
| `gpu` | `.venv-bench-cuda` | 仅 Linux |
| `supa` | `.venv-bench-supa` | 仅 Linux |

### Windows

```powershell
.\envs\bench-cpu\create-env.ps1        # 仅首次需要
.\.venv-bench-cpu\Scripts\Activate.ps1

.\benchmarks\scripts\run_benchmark.ps1 -Scenario cpu-single
.\benchmarks\scripts\run_benchmark.ps1 -Scenario cpu-multi
```

### Linux

```bash
bash envs/bench-cpu/create-env.sh        # 仅首次需要
source .venv-bench-cpu/bin/activate

bash benchmarks/scripts/run_benchmark.sh --scenario cpu-single
bash benchmarks/scripts/run_benchmark.sh --scenario cpu-multi

# GPU 场景换用 bench-cuda 环境
source .venv-bench-cuda/bin/activate
bash benchmarks/scripts/run_benchmark.sh --scenario gpu
```

可用选项：`--qubits 4,8,12,16,20`、`--repeats 5`（PowerShell 为 `-Qubits`、`-Repeats`）。

脚本开头会打印实际使用的解释器路径，可据此确认激活的是否为预期环境。若环境缺少某个框架，[framework_comparison.py](framework_comparison.py) 会指名缺失的包并退出，而不是静默跳过该实现：

```text
unavailable implementation(s):
  qulacs:cpu:complex128: qulacs is not installed
  qiskit-aer:cpu:complex128: qiskit-aer is not installed
```

它还会区分“包没装”与“包装了但设备不可用”，例如 `torch is installed but no CUDA device is available`。

本项目的每个后端都有 fusion 开关对照。原标签表示 fusion 关闭，带 `:fusion` 后缀的标签表示 fusion 打开，例如：

```text
ours:numpy:complex128          fusion off
ours:numpy:complex128:fusion   fusion on
```

场景脚本默认同时运行这两种配置。也可以直接运行最小对照：

```powershell
python benchmarks\framework_comparison.py `
  --qubits 4,8,12 `
  --implementations ours:numpy:complex128,ours:numpy:complex128:fusion `
  --reference ours:numpy:complex128
```

选用 bash 而非 POSIX sh，是因为 `set -o pipefail`、数组和 `[[ ]]` 均为 bash 特有；脚本只用 bash 3.2 特性，因此 macOS 自带的 bash 也能直接运行。

## 场景

| 场景 | 线程 | 对比实现 | 误差基准 |
|---|---|---|---|
| `cpu-single` | 1 | numpy-128、torch-cpu-128、qulacs、qiskit-aer | qiskit-aer |
| `cpu-multi` | 系统逻辑核数 | 同上 | qiskit-aer |
| `gpu`（仅 Linux） | — | torch-cuda-64、torch-cuda-128、qiskit-aer-gpu | qiskit-aer-gpu |
| `supa`（仅 Linux） | 系统逻辑核数 | numpy-64、torch-cpu-64、torch-supa-64、qulacs、qiskit-aer | qiskit-aer |

误差定义为消去全局相位后的最大振幅偏差

$$
\varepsilon=\max_i\left|a_i-e^{-i\arg\langle a|b\rangle}\,b_i\right|
$$

**为什么不用保真度。** 保真度 $F$ 紧贴 1，双精度根本存不下差异：振幅误差为 $10^{-12}$ 时 $F$ 已经返回 `1.0000000000000002`，$1-F$ 恒等于 $-2.22\times10^{-16}$ 这个浮点噪声，既分辨不出误差大小，还会因舍入越过 1。$F$ 是振幅误差的**二次**量，而上式是**一次**量，因此能一直用科学计数法读到 $10^{-16}$。

消去全局相位是必要的：整体相差 $e^{i\varphi}$ 的两个态在物理上完全相同，若直接相减会得到 $10^{-1}$ 量级的假误差。

## 精度说明

| 框架 | GPU 精度 |
|---|---|
| **qiskit-aer** | `precision` 可选 `"single"` / `"double"`，**默认 `"double"`**，GPU 同样适用 |

本基准中采用 qiskit-aer **双精度**，作为统一的误差基准。

## 方法学

跨框架基准采用 [qulacs 官方 benchmark](https://github.com/qulacs/qulacs/blob/main/benchmark/circuits/qulacsbench.py) 的电路：

```text
first_rotation   每个 qubit → RX(θ), RZ(θ)                    2n 门
entangler        CNOT ring，pairs = [(i, (i+1) % n)]           n 门
× depth (=9)     mid_rotation 每 qubit → RZ, RX, RZ           3n 门
                 + entangler                                   n 门
last_rotation    每个 qubit → RZ, RX                          2n 门
                                                    合计 41n 门
```

$n=20$ 时为 820 门。所有角度随机。

计时规则：

- 电路构造在计时区**之外**，只测态演化；
- fusion on 使用 `StateVectorSimulator(fusion=True)`，每次 `run()` 的门融合编译成本包含在计时内，体现开关的端到端性能；
- 每次运行都从 $|0\cdots0\rangle$ 开始，三个框架口径一致；
- 一次未计时的预热，之后取多次重复的**最小值**；
- CUDA 路径在读表前调用 `torch.cuda.synchronize()`，否则测到的只是 kernel launch。

三个框架执行**同一条电路**：门序列先生成为与框架无关的指令列表，再由各实现翻译成原生电路。这样避免了各自抽随机数导致电路不同。

### 修正的问题

**qulacs 的旋转门符号相反。** qulacs 用 $e^{+i\theta P/2}$，本项目与 Qiskit 用 $e^{-i\theta P/2}$。基准向 qulacs 传入 $-\theta$，三者才计算同一个态。

## 环境

基准环境与日常开发环境分开，避免 qulacs、qiskit-aer 等对比框架污染 `.venv-cpu`。三个基准环境定义在 [envs/](../envs/)：`bench-cpu`、`bench-cuda`、`bench-supa`，创建与激活方法见 [envs/README.md](../envs/README.md)。

## SUPA 张量布局限制

在 `.venv-bench-supa` 中运行以下实验，对比相同 PyTorch 张量布局操作在 SUPA 和 CPU 上的行为：

```bash
python experiments/test-reshape-perf.py --device supa
python experiments/test-reshape-perf.py --device cpu
```

下表为每次操作的平均耗时，单位为毫秒。`共享 storage` 表示结果张量与输入 statevector 的底层 storage 地址相同，即操作没有复制振幅数据。

| 设备 | 操作 | 16 qubits | 20 qubits | 24 qubits | 共享 storage |
|---|---|---:|---:|---:|:---:|
| SUPA | `reshape((2,) * num_qubits)` | 0.004595 | 0.004144 | 0.004328 | 是 |
| SUPA | `reshape(2, -1)` | 0.011608 | 0.042867 | 0.473035 | 否 |
| SUPA | `reshape(tensor).permute` | 0.006121 | 0.006620 | 0.006860 | 是 |
| SUPA | `reshape(tensor).permute.reshape(2, -1)` | 0.581659 | 20.800177 | 168.789613 | 否 |
| CPU | `reshape((2,) * num_qubits)` | 0.002126 | 0.001803 | 0.002606 | 是 |
| CPU | `reshape(2, -1)` | 0.001062 | 0.001020 | 0.001012 | 是 |
| CPU | `reshape(tensor).permute` | 0.003331 | 0.003456 | 0.004063 | 是 |
| CPU | `reshape(tensor).permute.reshape(2, -1)` | 0.004755 | 0.004964 | 0.005283 | 是 |

CPU 上四种操作都只是修改 shape、stride 等张量元数据，不复制底层数据，耗时基本不随 statevector 大小增长。SUPA 上保持每个 qubit 一个轴的 `reshape((2,) * num_qubits)` 和随后的 `permute` 同样是轻量 view；但展平为 `(2, -1)` 会创建新的 storage。

限制在 `permute` 后尤其明显：24 qubits 时，`permute` 后再 `reshape(2, -1)` 需要约 168.79 ms，而单独 `permute` 仅约 0.0069 ms。这说明耗时来自 SUPA 对不兼容 stride 的展平和数据重排，而不是 `permute` 本身。

因此 SUPA 后端应尽量让 statevector 始终保持 `(2,) * num_qubits` 的多轴形式，并保留 `permute` 返回的 view。热路径中应避免将 statevector 或 permuted view 展平；若矩阵乘法接口要求连续的二维输入，需要使用支持 strided tensor 的算子或融合 kernel，否则每个量子门都可能额外复制全部 $2^n$ 个振幅。

仓库提供两个候选优化实验：

```bash
# 对 qubit 0 的单比特门，用显式二维 stride 绕过 reshape 复制
python experiments/test-as-strided-perf.py --device supa

# 对单、双、三比特门，直接在多轴张量上 contraction，避免二维展平
python experiments/test-tensordot-perf.py --device supa
```

`test-as-strided-perf.py` 会同时验证 `as_strided` 是否与输入共享 storage、门计算结果是否与原实现一致，并比较包含矩阵乘法的完整耗时。该实验只构造能够严格表示为二维 stride `(1, 2)` 的 qubit 0 布局，不能直接推广到任意目标 qubit 集合。

`test-tensordot-perf.py` 会比较当前的 `permute + reshape + matmul` 与不展平的 `tensordot`，并对单比特、非相邻双比特和非相邻三比特 workload 执行数值一致性检查。只有 SUPA 实测显示完整 contraction 更快时，才应替换后端热路径；CPU 结果不能代表 SUPA 算子实现。

## 平台限制

| 包 | 可用性 | 说明 |
|---|---|---|
| `qulacs` | Windows / Linux / macOS | CPU，complex128 |
| `qiskit-aer` | Windows / Linux / macOS | CPU |
| `qiskit-aer-gpu` | **仅 Linux** | 只发布 manylinux x86_64 wheel；当前最新 0.15.1，落后于 CPU 版 0.17.2；它**替代**而非叠加 `qiskit-aer` |

