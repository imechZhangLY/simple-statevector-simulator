# Benchmarks

本目录包含两类基准：

| 文件 | 用途 |
|---|---|
| [benchmark_backends.py](benchmark_backends.py) | 项目内部后端对比（NumPy / Torch，CPU / CUDA） |
| [framework_comparison.py](framework_comparison.py) | 与 qulacs、qiskit-aer 的跨框架对比 |
| [results/](results/) | 已记录的结果 JSON 与折线图 |

## 快速开始

脚本会自动创建 venv、安装依赖、设置线程环境变量、运行基准并输出 JSON 与折线图。

```powershell
# Windows
powershell -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\scripts\run_benchmark.ps1 -Scenario cpu-single
powershell -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\scripts\run_benchmark.ps1 -Scenario cpu-multi
```

```bash
# Linux / macOS
bash benchmarks/scripts/run_benchmark.sh --scenario cpu-single
bash benchmarks/scripts/run_benchmark.sh --scenario cpu-multi
bash benchmarks/scripts/run_benchmark.sh --scenario gpu      # 仅 Linux
```

可用选项：`--qubits 4,8,12,16,20`、`--repeats 5`、`--skip-install`（PowerShell 为 `-Qubits`、`-Repeats`、`-SkipInstall`）。

选用 bash 而非 POSIX sh，是因为 `set -o pipefail`、数组和 `[[ ]]` 均为 bash 特有；脚本只用 bash 3.2 特性，因此 macOS 自带的 bash 也能直接运行。

### 依赖检查

安装前先判断是否真的需要安装，两个条件都满足才跳过：

1. 依赖文件的 SHA-256 与 venv 内 `.requirements-hash` 标记一致；
2. [scripts/check_dependencies.py](scripts/check_dependencies.py) 能定位到全部模块。

第二条不可省略——哈希只能反映依赖文件没变，无法发现有人手动卸载了某个包。

## 场景

| 场景 | 线程 | 对比实现 | 保真度基准 |
|---|---|---|---|
| `cpu-single` | 1 | numpy-128、torch-cpu-128、qulacs、qiskit-aer | qiskit-aer |
| `cpu-multi` | 系统逻辑核数 | 同上 | qiskit-aer |
| `gpu`（仅 Linux） | — | torch-cuda-64、torch-cuda-128、qiskit-aer-gpu | qiskit-aer-gpu |

保真度定义为

$$
F=\frac{|\langle a|b\rangle|^{2}}{\langle a|a\rangle\langle b|b\rangle}
$$

分母的归一化不可省略：低精度后端的末态并非严格归一，直接用 $|\langle a|b\rangle|^2$ 会得到大于 1 的“保真度”。加上归一化后由 Cauchy-Schwarz 保证 $F\le1$。

## 精度说明

| 框架 | GPU 精度 |
|---|---|
| **qiskit-aer** | `precision` 可选 `"single"` / `"double"`，**默认 `"double"`**，GPU 同样适用 |

本基准中采用 qiskit-aer **双精度**，作为统一的保真度基准。

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

benchmark使用独立环境，和正常环境 `.venv`区分：

```powershell
python -m venv .venv-bench
.\.venv-bench\Scripts\python.exe -m pip install -r requirements-bench.txt
```

## 平台限制

| 包 | 可用性 | 说明 |
|---|---|---|
| `qulacs` | Windows / Linux / macOS | CPU，complex128 |
| `qiskit-aer` | Windows / Linux / macOS | CPU |
| `qiskit-aer-gpu` | **仅 Linux** | 只发布 manylinux x86_64 wheel；当前最新 0.15.1，落后于 CPU 版 0.17.2；它**替代**而非叠加 `qiskit-aer` |

