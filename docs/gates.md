# 支持的量子门

本文档列出所有已实现的量子门。表中的元数据（`qasm_name`、参数、dagger 形式）与 [src/single_qubit_gates.py](../src/single_qubit_gates.py)、[src/two_qubit_gates.py](../src/two_qubit_gates.py)、[src/three_qubit_gates.py](../src/three_qubit_gates.py) 中的定义一一对应。

API 用法见 [api.md](api.md)，设计动机见 [architecture.md](architecture.md)。

## 1. 调用约定

所有门函数直接返回 `Operation`，而不是先构造门对象再绑定 qubit：

```python
from single_qubit_gates import H, RX
from two_qubit_gates import CX

H(0)              # Operation
RX(0.5, 1)        # 连续参数在前，qubit 在后
CX(0, 1)          # 保持 control、target 顺序
```

规则：

- **连续参数一律排在 qubit 参数之前**；
- 角度必须是有限实数，否则抛出 `TypeError` 或 `ValueError`；
- qubit 必须是非负整数且在同一个 operation 内互不相同；
- 常量门共享唯一的模块级 `Gate` 对象；含参门每次调用构造一个绑定了参数的 `Gate`。

### 局部基序

局部门矩阵按 `operation.qubits` 的顺序定义，**第一个 qubit 是局部基态的最高有效位**。因此：

| 门 | 局部基序 |
|---|---|
| `CX(control, target)` | $\lvert control,target\rangle$ |
| `CCX(control1, control2, target)` | $\lvert control1,control2,target\rangle$ |
| `CSWAP(control, target1, target2)` | $\lvert control,target1,target2\rangle$ |

这与全局 statevector 的 little-endian 索引是两套独立约定，由 `StateVector.apply()` 的轴置换负责调和。下面所有矩阵都写在局部基序下。

## 2. 单比特门

导出自 `single_qubit_gates`，共 14 个函数。

| 函数 | 签名 | `qasm_name` | dagger 的 `qasm_name` | dagger 参数 |
|---|---|---|---|---|
| `I` | `I(qubit)` | `id` | `id` | — |
| `X` | `X(qubit)` | `x` | `x` | — |
| `Y` | `Y(qubit)` | `y` | `y` | — |
| `Z` | `Z(qubit)` | `z` | `z` | — |
| `H` | `H(qubit)` | `h` | `h` | — |
| `S` | `S(qubit)` | `s` | **`sdg`** | — |
| `T` | `T(qubit)` | `t` | **`tdg`** | — |
| `RX` | `RX(theta, qubit)` | `rx` | `rx` | $(-\theta)$ |
| `RY` | `RY(theta, qubit)` | `ry` | `ry` | $(-\theta)$ |
| `RZ` | `RZ(theta, qubit)` | `rz` | `rz` | $(-\theta)$ |
| `P` | `P(theta, qubit)` | `p` | `p` | $(-\theta)$ |
| `U1` | `U1(lam, qubit)` | `u1` | `u1` | $(-\lambda)$ |
| `U2` | `U2(phi, lam, qubit)` | `u2` | `u2` | $(\pi-\lambda,\ -\pi-\phi)$ |
| `U3` | `U3(theta, phi, lam, qubit)` | `u3` | `u3` | $(-\theta,\ -\lambda,\ -\phi)$ |

`I`、`X`、`Y`、`Z`、`H` 是自伴的，正向与 dagger 复用同一个矩阵对象。

### 常量门矩阵

$$
I=\begin{pmatrix}1&0\\0&1\end{pmatrix}\quad
X=\begin{pmatrix}0&1\\1&0\end{pmatrix}\quad
Y=\begin{pmatrix}0&-i\\i&0\end{pmatrix}\quad
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}
$$

$$
H=\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}\quad
S=\begin{pmatrix}1&0\\0&i\end{pmatrix}\quad
T=\begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix}
$$

### 含参门矩阵

$$
R_X(\theta)=\begin{pmatrix}\cos\frac{\theta}{2}&-i\sin\frac{\theta}{2}\\[2pt]-i\sin\frac{\theta}{2}&\cos\frac{\theta}{2}\end{pmatrix}\quad
R_Y(\theta)=\begin{pmatrix}\cos\frac{\theta}{2}&-\sin\frac{\theta}{2}\\[2pt]\sin\frac{\theta}{2}&\cos\frac{\theta}{2}\end{pmatrix}\quad
R_Z(\theta)=\begin{pmatrix}e^{-i\theta/2}&0\\0&e^{i\theta/2}\end{pmatrix}
$$

$$
P(\theta)=U_1(\theta)=\begin{pmatrix}1&0\\0&e^{i\theta}\end{pmatrix}\qquad
U_3(\theta,\phi,\lambda)=\begin{pmatrix}\cos\frac{\theta}{2}&-e^{i\lambda}\sin\frac{\theta}{2}\\[2pt]e^{i\phi}\sin\frac{\theta}{2}&e^{i(\phi+\lambda)}\cos\frac{\theta}{2}\end{pmatrix}
$$

$U_2(\phi,\lambda)=U_3\!\left(\frac{\pi}{2},\phi,\lambda\right)$。

注意 `P` 与 `U1` 矩阵完全相同，区别只在名称与 `qasm_name`：`P` 是现代写法，`U1` 保留是为了与 `qelib1.inc` 一一对应。

### 关于 RZ 的相位

`RZ(θ)` 采用 $\mathrm{diag}(e^{-i\theta/2}, e^{i\theta/2})$，而不是 $\mathrm{diag}(1, e^{i\theta})$。两者相差一个全局相位 $e^{-i\theta/2}$，对单独作用的最终概率没有影响，但**作为受控门的目标时全局相位会变成相对相位**。需要 $\mathrm{diag}(1,e^{i\theta})$ 语义时请使用 `P` 或 `U1`。

## 3. 双比特门

导出自 `two_qubit_gates`，共 10 个函数（含 1 个别名）。局部基序为 $\lvert q_0,q_1\rangle$，其中 $q_0$ 是列表中的第一个 qubit。

| 函数 | 签名 | `qasm_name` | dagger 参数 |
|---|---|---|---|
| `CX` | `CX(control, target)` | `cx` | — |
| `CNOT` | `CNOT(control, target)` | `cx` | — |
| `CY` | `CY(control, target)` | `cy` | — |
| `CZ` | `CZ(control, target)` | `cz` | — |
| `CH` | `CH(control, target)` | `ch` | — |
| `SWAP` | `SWAP(first, second)` | `swap` | — |
| `CP` | `CP(theta, control, target)` | `cp` | $(-\theta)$ |
| `CRX` | `CRX(theta, control, target)` | `crx` | $(-\theta)$ |
| `CRY` | `CRY(theta, control, target)` | `cry` | $(-\theta)$ |
| `CRZ` | `CRZ(theta, control, target)` | `crz` | $(-\theta)$ |

`CNOT` 是 `CX` 的别名，二者共享同一个 `Gate` 对象，并且都序列化为 `cx`。

除 `SWAP` 外的双比特门都是受控门，形如分块对角矩阵 $\mathrm{diag}(I, U)$，即控制位为 $\lvert 1\rangle$ 时对目标位作用 $U$：

$$
CX=\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0\end{pmatrix}\quad
CZ=\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&-1\end{pmatrix}\quad
SWAP=\begin{pmatrix}1&0&0&0\\0&0&1&0\\0&1&0&0\\0&0&0&1\end{pmatrix}
$$

$CY=\mathrm{diag}(I,Y)$，$CH=\mathrm{diag}(I,H)$，$CP(\theta)=\mathrm{diag}(1,1,1,e^{i\theta})$，$CR_\sigma(\theta)=\mathrm{diag}(I,R_\sigma(\theta))$。

## 4. 三比特门

导出自 `three_qubit_gates`，共 4 个函数（含 2 个别名）。

| 函数 | 签名 | `qasm_name` | 作用 |
|---|---|---|---|
| `CCX` | `CCX(control1, control2, target)` | `ccx` | 交换 $\lvert 110\rangle \leftrightarrow \lvert 111\rangle$ |
| `TOFFOLI` | `TOFFOLI(control1, control2, target)` | `ccx` | 同上 |
| `CSWAP` | `CSWAP(control, target1, target2)` | `cswap` | 交换 $\lvert 101\rangle \leftrightarrow \lvert 110\rangle$ |
| `FREDKIN` | `FREDKIN(control, target1, target2)` | `cswap` | 同上 |

两者都是 $8\times 8$ 置换矩阵、自伴，dagger 复用同一矩阵。上表中的基态写在局部基序下。

## 5. dagger 行为

`Operation.dagger()` 只翻转方向标志，不做任何矩阵运算——正向与 dagger 的矩阵和序列化元数据在 `Gate` 构造时就已预先算好并存储。

dagger 的规则**不是**统一的「参数取负」或「名字加后缀」，本项目至少存在四种情况：

| 情况 | 例子 |
|---|---|
| 自伴，完全不变 | `X` → `x`，`CCX` → `ccx` |
| 专用的 dagger 名称 | `S` → `sdg`，`T` → `tdg` |
| 参数取负 | `RX(θ)` → `rx(-θ)`，`CP(θ)` → `cp(-θ)` |
| 参数重排并取负 | `U3(θ,φ,λ)` → `u3(-θ,-λ,-φ)`，`U2(φ,λ)` → `u2(π-λ,-π-φ)` |

因此不要从矩阵反推序列化参数，也不要假设某条统一规则。

电路级的 `Circuit.dagger()` 会**反转操作顺序**并对每个操作取 dagger。只取 dagger 而不反转，在门不对易时就是错的。

## 6. OpenQASM 覆盖范围

上述 25 个不同的门加上 `sdg`、`tdg` 两个 dagger 专用名称，恰好构成 [src/qasm_parser.py](../src/qasm_parser.py) 中 `GATE_TABLE` 的 27 个条目。导出器与解析器双向都不做名称翻译——`Gate.qasm_name` 从一开始就按 `qelib1.inc` 命名。

导出后再解析，得到的矩阵与作用 qubit 与原电路完全一致，但 `Operation.matrix_key` 可能不同：daggered 操作会以规范的正向形式导出（`RX†(0.7)` 写成 `rx(-0.7)`，自伴的 `X†` 写成 `x`），解析回来时 `is_dagger=False`。这是表示形式的差异，不是语义差异。`S†`/`T†` 因为有专属名称，往返后 `matrix_key` 保持不变。

## 7. 新增门的步骤

1. 在本项目的局部基序下确认矩阵；
2. 同时定义正向与 dagger 矩阵；
3. 同时定义正向与 dagger 的序列化名称和参数；
4. 校验所有角度为有限实数；
5. 通过模块的 `__all__` 导出；
6. 补充测试：矩阵、酉性、dagger 元数据、qubit 顺序、非法输入。

`qasm_name` 必须与 `qelib1.inc` 保持一致，不要引入名称翻译表。
