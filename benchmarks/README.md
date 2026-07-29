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
- 每次运行都从 $|0\cdots0\rangle$ 开始，三个框架口径一致；
- 一次未计时的预热，之后取多次重复的**最小值**；
- CUDA 路径在读表前调用 `torch.cuda.synchronize()`，否则测到的只是 kernel launch。

三个框架执行**同一条电路**：门序列先生成为与框架无关的指令列表，再由各实现翻译成原生电路。这样避免了各自抽随机数导致电路不同。

### 修正的问题

**qulacs 的旋转门符号相反。** qulacs 用 $e^{+i\theta P/2}$，本项目与 Qiskit 用 $e^{-i\theta P/2}$。基准向 qulacs 传入 $-\theta$，三者才计算同一个态。

## 环境

基准环境与日常开发环境分开，避免 qulacs、qiskit-aer 等对比框架污染 `.venv-cpu`。三个基准环境定义在 [envs/](../envs/)：`bench-cpu`、`bench-cuda`、`bench-supa`，创建与激活方法见 [envs/README.md](../envs/README.md)。

## 平台限制

| 包 | 可用性 | 说明 |
|---|---|---|
| `qulacs` | Windows / Linux / macOS | CPU，complex128 |
| `qiskit-aer` | Windows / Linux / macOS | CPU |
| `qiskit-aer-gpu` | **仅 Linux** | 只发布 manylinux x86_64 wheel；当前最新 0.15.1，落后于 CPU 版 0.17.2；它**替代**而非叠加 `qiskit-aer` |

