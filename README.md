# Simple StateVector Simulator

一个用 Python 和 NumPy 实现的轻量级量子计算模拟器。使用完整 statevector 表示纯态，通过局部酉矩阵执行量子门，并支持可插拔的计算后端（NumPy / PyTorch，CPU 与 CUDA）。

## 特性

- 单比特、双比特、三比特量子门，支持任意 qubit 子集上的通用门应用
- `Circuit` 电路结构与电路级 dagger
- 可插拔后端：`NumpyBackend`、`TorchBackend`（CPU / CUDA，`complex64` / `complex128`）
- 不破坏量子态的多次采样
- `Observable`：加权 Pauli 串求和，用于计算期望值
- OpenQASM 2 导出与解析

暂不包含：投影测量与态坦缩（[有意排除](docs/architecture.md)）、电路中间测量、OpenQASM 3、密度矩阵与噪声。

## 快速上手

```python
import numpy as np

from circuit import Circuit
from observable import Observable
from simulator import StateVectorSimulator
from single_qubit_gates import H
from two_qubit_gates import CX

circuit = Circuit(2).append(H(0)).append(CX(0, 1))
state = StateVectorSimulator().run(circuit)

state.amplitudes
# array([0.70710678+0.j, 0.+0.j, 0.+0.j, 0.70710678+0.j])

state.sample(1000, np.random.default_rng(7))
# {0: 502, 3: 498}

state.expectation(Observable([(1.0, {0: "Z", 1: "Z"})]))
# 0.9999999999999998
```

采样只依赖注入的 `numpy.random.Generator`，因此相同 seed 在所有后端上结果一致。

指定后端：

```python
from torch_backend import TorchBackend

simulator = StateVectorSimulator(TorchBackend(device="cuda", dtype="complex64"))
```

## 环境准备

项目使用两个虚拟环境，定义已提交到仓库，环境目录本身被 Git 忽略。

| 环境 | 用途 | 依赖文件 |
|---|---|---|
| `.venv` | 主环境，CUDA 版 PyTorch | [requirements-torch-cuda.txt](requirements-torch-cuda.txt) |
| `.venv-cpu` | CPU 版 PyTorch 对照 | [requirements-torch-cpu.txt](requirements-torch-cpu.txt) |

两者均使用 Python 3.10，并从 [requirements.txt](requirements.txt) 安装 NumPy。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\.vscode\bootstrap-env.ps1 -EnvironmentName .venv -Requirements requirements-torch-cuda.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\.vscode\bootstrap-env.ps1 -EnvironmentName .venv-cpu -Requirements requirements-torch-cpu.txt
```

Linux 与 macOS 使用等价的 bash 脚本。它在 Linux 上默认选择 CUDA 版，在 macOS 上默认选择 [requirements-torch-macos.txt](requirements-torch-macos.txt)：`+cu126` 与 `+cpu` 这类 wheel 只为 Linux 和 Windows 发布，macOS 必须用 PyPI 上的普通版本。

```bash
bash .vscode/bootstrap-env.sh --environment-name .venv
bash .vscode/bootstrap-env.sh --environment-name .venv-cpu --requirements requirements-torch-cpu.txt
```

脚本是幂等的：环境不存在时创建，依赖文件未变化时直接跳过。用 VS Code 打开工作区会自动触发这两个任务。

不确定该用哪个脚本时，交给分发器自动判断平台与 CUDA（见下一节）。

torch 是可选依赖，只安装 [requirements.txt](requirements.txt) 时 NumPy 后端可正常工作，相关测试会自动跳过。

## 用 Agent 编写程序

仓库自带一个 skill，放在 [.agents/skills/quantum-simulator/](.agents/skills/quantum-simulator/)。VS Code、opencode 等支持 Agent Skills 的工具打开本目录即可发现它，无需额外配置。`.agents/` 是厂商中立的约定位置，与仓库已有的 [AGENTS.md](AGENTS.md) 一致。

直接用自然语言描述需求即可，例如：

```text
构造一个 3 qubit 的 GHZ 态，采样 2000 次
用 QFT 变换 |5>，把振幅向量导出来
算一下 Bell 态上 Z0Z1 的期望值
```

也可以在聊天框输入 `/quantum-simulator` 显式调用。

skill 把工作拆成三步：

**1. 初始化环境。** 分发器自行探测平台与 CUDA，再调用对应的引导脚本：

```powershell
python .agents/skills/quantum-simulator/scripts/setup_environment.py
```

| 平台 | CUDA | 脚本 | 依赖文件 |
|---|---|---|---|
| Windows | 有 | `bootstrap-env.ps1` | `requirements-torch-cuda.txt` |
| Windows | 无 | `bootstrap-env.ps1` | `requirements-torch-cpu.txt` |
| Linux | 有 | `bootstrap-env.sh` | `requirements-torch-cuda.txt` |
| Linux | 无 | `bootstrap-env.sh` | `requirements-torch-cpu.txt` |
| macOS | — | `bootstrap-env.sh` | `requirements-torch-macos.txt` |

加 `--print-only` 只报告判断结果而不改动任何东西，`--force-cpu` 跳过 CUDA 探测。CUDA 用 `nvidia-smi` 探测，因为此时 torch 还没装上，无法靠 import 判断。

**2. 编写代码。** agent 依据 [docs/api.md](docs/api.md) 和 [docs/gates.md](docs/gates.md) 生成程序，而不是靠记忆猜门名和参数顺序——这两份文档的内容是对照代码实测生成的。

**3. 运行并整理结果。** 用 venv 中的解释器执行，再按诉求分流输出：

| 诉求 | 输出 |
|---|---|
| 采样 | 全量写入 `results/sampling.csv`，前 10 条打印到对话 |
| 期望值 | 直接打印到对话，不落盘 |
| 振幅向量 | 写入 `results/amplitudes.json` |

CSV 的 key 已从整数转成二进制串，按次数从大到小排序，次数为 0 的结果不输出。`results/` 已加入 `.gitignore`。

格式化逻辑集中在 [qsim_report.py](.agents/skills/quantum-simulator/scripts/qsim_report.py)，手写程序也可以直接复用：

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\.agents\skills\quantum-simulator\scripts"
.\.venv\Scripts\python.exe your_program.py
```

```python
from qsim_report import write_sampling_csv, format_rows

path, top = write_sampling_csv(state.sample(2000), state.num_qubits)
print(format_rows(top))
```

打印到对话的表格形如：

```text
bitstring     count   probability
000             979      0.489500
011             945      0.472500
100              40      0.020000
```

两点提醒：`bitstring` 列在 Excel 中可能被当成数字而丢掉前导零，程序化读回时请用 `index` 列；振幅 JSON 是全量 $2^n$ 条记录，20 qubit 时约 100 MB，大寄存器应改用采样或期望值。

## 运行测试

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前 162 个测试全部通过（1 个在缺少 CUDA 时跳过）。其中后端一致性测试是参数化的，会在每个可用后端上重复验证 Bell 态、纠缠态局部门、非相邻 qubit、三比特门以及 dagger 还原。

## 示例

两个示例都自行把 `src` 加入 `sys.path`，无需设置 `PYTHONPATH`；都支持 `--backend` 参数：

```text
numpy:complex128  numpy:complex64
torch:cpu:complex128  torch:cpu:complex64
torch:cuda:complex128  torch:cuda:complex64
```

### 量子傅里叶变换

计算 QFT 末态振幅，并与解析解对比保真度：

$$
\text{QFT}|x\rangle=\frac{1}{\sqrt N}\sum_{k=0}^{N-1} e^{2\pi i xk/N}|k\rangle
$$

```powershell
.\.venv\Scripts\python.exe examples\qft_demo.py --qubits 3 --value 5 --show-amplitudes
```

```text
backend        : numpy:complex128
qubits         : 3
input state    : |5> = |101>
gate count     : 9
norm           : 1.000000000000
fidelity       : 1.000000000000
max abs error  : 9.328e-16

  index                         simulated                          expected
      0                0.353553+0.000000j                0.353553+0.000000j
      1               -0.250000-0.250000j               -0.250000-0.250000j
      2                0.000000+0.353553j                0.000000+0.353553j
      3                0.250000-0.250000j                0.250000-0.250000j
      4               -0.353553+0.000000j               -0.353553+0.000000j
      5                0.250000+0.250000j                0.250000+0.250000j
      6               -0.000000-0.353553j               -0.000000-0.353553j
      7               -0.250000+0.250000j               -0.250000+0.250000j
```

不同后端在 8 qubit、输入 $|173\rangle$ 下的精度对比：

| backend | fidelity | max abs error |
|---|---:|---:|
| `numpy:complex128` | 1.000000000000 | 1.1e-14 |
| `numpy:complex64` | 0.999999521755 | 2.7e-08 |
| `torch:cuda:complex64` | 0.999999554395 | 2.7e-08 |

### 哈密顿量模拟

横场 Ising 模型的一阶 Trotter 演化，与精确矩阵指数对比：

$$
H=-J\sum_j Z_jZ_{j+1}-h\sum_j X_j
$$

```powershell
.\.venv\Scripts\python.exe examples\hamiltonian_simulation_demo.py --qubits 5 --steps 1,3,9,27,81
```

```text
backend    : numpy:complex128
qubits     : 5
H          : -1.0 * sum ZZ - 0.7 * sum X
time       : 1.0

energy reference
  <H> from Observable       : -3.700000000000
  <H> from dense matrix     : -3.700000000000
  <H> after exact evolution : -3.700000000000  (conserved)

  steps    gates      infidelity    2-norm error  error ratio  step ratio     <H> error
      1       18       8.030e-01       1.148e+00                              4.020e+00
      3       52       8.013e-02       3.187e-01         3.60        3.00     3.897e-01
      9      154       9.111e-03       1.023e-01         3.12        3.00     7.169e-02
     27      460       1.027e-03       3.390e-02         3.02        3.00     1.786e-02
     81     1378       1.147e-04       1.129e-02         3.00        3.00     5.289e-03
```

这个示例包含三层验证：

**能量三方交叉验证。** `Observable`（Pauli 串求和）、稠密矩阵 $\langle\psi|H|\psi\rangle$（Kronecker 积构造）与解析值三者一致。初态 $|+\rangle_0|0000\rangle$ 可手算：

$$
\langle H\rangle=-J(0+1+1+1)-h(1+0+0+0)=-3.7
$$

任何一处端序写反都会让三者分道扬镳。

**误差标度律。** 一阶 Trotter 误差为 $O(t^2\|[A,B]\|/r)$，即 $\propto 1/\text{steps}$，因此 error ratio 应趋近 step ratio。表中收敛到 3.00。

这个指标比误差大小更有判别力：它能区分**离散化误差**和**实现错误**。把 `RX` 的符号故意写反后：

| steps | 正确实现 | ratio | 符号写反 | ratio |
|---:|---:|---:|---:|---:|
| 3 | 3.187e-01 | | 1.298e+00 | |
| 9 | 1.023e-01 | 3.12 | 1.311e+00 | 0.99 |
| 27 | 3.390e-02 | 3.02 | 1.316e+00 | 1.00 |
| 81 | 1.129e-02 | 3.00 | 1.317e+00 | 1.00 |
| 243 | 3.761e-03 | 3.00 | 1.318e+00 | 1.00 |

实现错误是系统性的，不随步数减小，误差卡在平台上、ratio 恒为 1.00。仅看误差大小无法发现这一点。

**能量守恒。** $[H,e^{-iHt}]=0$，所以 $\langle H\rangle$ 在精确演化下严格守恒，能量偏差构成一个独立于保真度的判据。

## OpenQASM

导出：

```python
from qasm_exporter import export_qasm

print(export_qasm(circuit, measure_all=True))
```

```text
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0], q[1];

measure q -> c;
```

解析：

```python
from qasm_parser import parse_qasm

program = parse_qasm(text)
program.circuit        # 纯酉电路，可 dagger、可重复执行、可再导出
program.measurements   # ((0, 0), (1, 1))
```

两点设计选择：

**未知门报错并指名**，绝不跳过。跳过不认识的语句会静默产生语义不同的电路，而调用方看不出异常。

**测量解析到 `Circuit` 之外**，保住"电路只含酉演化"的不变量；若测量之后又出现门操作，则按电路中间测量拒绝——那是模拟器确实无法表示的情况。

参数表达式（`pi/2`、`sqrt(4)` 等）用 `ast` 白名单求值，**不使用 `eval()`**，因此解析来路不明的文件不会执行任意代码。

## 性能基准

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_backends.py --qubits 16,20,22
```

22 qubit 下每个门的 `apply()` 耗时（微秒，取最小值，Quadro P620）：

| backend | 1q `H` | 2q `CX` | 3q `CCX` |
|---|---:|---:|---:|
| `numpy:complex128` | 73958 | 123499 | 147981 |
| `numpy:complex64` | 32454 | 58390 | 65671 |
| `torch:cpu:complex64` | 12004 | 39015 | 44049 |
| `torch:cuda:complex128` | 16686 | 77436 | 41784 |
| `torch:cuda:complex64` | **7331** | **7424** | **9948** |

GPU 上 `complex64` 最快，比 `numpy:complex128` 快约 17 倍。但 GPU 上的 `complex128` 是陷阱：双比特门比 `complex64` 慢 10 倍，甚至慢于 torch CPU——这是 Pascal 架构 FP64 只有 1/32 速率的直接后果。

## 约定

- **全局索引使用 little-endian**：qubit 0 是 statevector 索引的最低有效位，索引 $i$ 对应 $|q_{n-1}\cdots q_1q_0\rangle$
- **局部门矩阵**按 `operation.qubits` 的顺序定义，第一个 qubit 为局部基态的最高有效位，例如 `CX(control, target)` 的基序为 $|control,target\rangle$
- **电路只含酉演化**，测量不能作为 operation 写入 `Circuit`

## 文档

- [docs/api.md](docs/api.md)：公开类型与函数的签名、语义与异常
- [docs/gates.md](docs/gates.md)：全部已实现量子门、矩阵与 dagger 元数据
- [docs/architecture.md](docs/architecture.md)：完整架构、数据模型、算法与设计决策
- [AGENTS.md](AGENTS.md)：面向 coding agent 的开发约束
