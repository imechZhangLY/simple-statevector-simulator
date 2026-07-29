# 环境说明

本仓库用 Python venv 管理依赖。**所有环境的依赖定义与创建脚本都集中在本目录**，每个环境一个子目录，内含 `requirements.txt` 和创建脚本。

命名遵循一条规律：

```text
envs/<name>  ->  <ReposRoot>/.venv-<name>
```

例如 `envs/bench-cpu` 创建的是 `.venv-bench-cpu`。venv 目录始终位于仓库根，已被 `.gitignore` 忽略。

## 工作目录

> **本文档所有命令都以仓库根目录（下称 `ReposRoot`）为工作目录。**

创建脚本本身与当前目录无关——它按脚本自身位置推算仓库根，无论从哪里调用，venv 都会建在 `ReposRoot` 下。但**激活命令和后续的运行命令使用相对路径**，只有在 `ReposRoot` 下执行才能逐字照搬。

```powershell
cd D:\path\to\simple-quantum-simulator
```

```bash
cd /path/to/simple-quantum-simulator
```

## 环境一览

| 环境 | venv | 平台 | 使用场景 |
|---|---|---|---|
| `cpu` | `.venv-cpu` | Windows, Linux | 日常开发与测试，torch CPU |
| `cuda` | `.venv-cuda` | Windows, Linux | GPU 模拟，torch cu126 |
| `supa` | `.venv-supa` | Linux（supa 云镜像） | supa 加速器 |
| `bench-cpu` | `.venv-bench-cpu` | Windows, Linux | 跨框架 CPU 基准 |
| `bench-cuda` | `.venv-bench-cuda` | Linux | 跨框架 GPU 基准 |
| `bench-supa` | `.venv-bench-supa` | Linux（supa 云镜像） | supa 基准 |

前三个是**项目环境**，只装运行模拟器所需的依赖；后三个是**基准环境**，额外包含 qulacs、qiskit-aer、matplotlib 等对比框架。两类分开，避免基准依赖污染日常开发环境。

关键依赖：

| 环境 | torch | 其他 |
|---|---|---|
| `cpu` | `2.13.0+cpu` | numpy |
| `cuda` | `2.13.0+cu126` | numpy |
| `supa` | 由云镜像提供 | numpy |
| `bench-cpu` | `2.13.0+cpu` | qulacs、qiskit-aer、matplotlib |
| `bench-cuda` | `2.13.0+cu126` | qiskit-aer-gpu、matplotlib |
| `bench-supa` | 由云镜像提供 | qulacs、qiskit-aer、matplotlib |

平台限制的原因：

- **`cuda` 与 `bench-cuda`** 依赖 PyTorch 的 `+cu126` wheel，只为 Linux 和 Windows 发布；`bench-cuda` 还需要 qiskit-aer-gpu，后者仅发布 manylinux wheel，因此实际只能在 Linux 上装成。
- **`supa` 与 `bench-supa`** 的 torch 和 torch_br 由 supa 云镜像以系统包形式提供，不经 pip 安装，所以这两个环境创建时会带 `--system-site-packages`。也正因如此它们没有 Windows 脚本。
- **macOS 暂无对应环境**：`+cpu` 与 `+cu126` 这类 wheel 不为 macOS 发布，需要单独钉不带后缀的 PyPI 版本。

## 创建环境

Windows 用 `.ps1`，Linux 用 `.sh`。脚本是幂等的：环境不存在时创建，依赖文件未变化时直接跳过，因此可以反复运行。

### Windows（PowerShell）

```powershell
# 日常开发
.\envs\cpu\create-env.ps1

# GPU 模拟
.\envs\cuda\create-env.ps1

# CPU 基准
.\envs\bench-cpu\create-env.ps1
```

若提示脚本被禁止执行，先在当前会话放开策略：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### Linux（bash）

```bash
# 日常开发
bash envs/cpu/create-env.sh

# GPU 模拟
bash envs/cuda/create-env.sh

# supa
bash envs/supa/create-env.sh

# 基准
bash envs/bench-cpu/create-env.sh
bash envs/bench-cuda/create-env.sh
bash envs/bench-supa/create-env.sh
```

### 自定义 venv 名称

两种脚本都接受覆盖参数，便于并行试验：

```powershell
.\envs\cpu\create-env.ps1 -EnvironmentName .venv-experiment
```

```bash
bash envs/cpu/create-env.sh --environment-name .venv-experiment
```

## 激活环境

### Windows

```powershell
# PowerShell
.\.venv-cpu\Scripts\Activate.ps1

# cmd
.venv-cpu\Scripts\activate.bat
```

### Linux

```bash
source .venv-cpu/bin/activate
```

退出：

```text
deactivate
```

### 不激活直接使用

激活会改写当前会话的 `PATH`，在脚本和 CI 中容易出现用错解释器的问题。**推荐直接按路径调用解释器**：

```powershell
.\.venv-cpu\Scripts\python.exe -m unittest discover -s tests
```

```bash
./.venv-cpu/bin/python -m unittest discover -s tests
```

## 运行项目代码

`src/` 是扁平布局。Python 只把**脚本自身所在的目录**加入 `sys.path`，因此写在仓库根的脚本、`python -c` 以及 `unittest discover` 需要把 `src` 加入 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv-cpu\Scripts\python.exe -m unittest discover -s tests -v
```

```bash
export PYTHONPATH="$PWD/src"
./.venv-cpu/bin/python -m unittest discover -s tests -v
```

`examples/` 与 `benchmarks/` 下的脚本自行插入了 `src`，不设 `PYTHONPATH` 也能直接运行。

## 故障排查

以下两个问题都出现在 Linux 上，而且**第二个通常是第一个的后果**。

### `ensurepip is not available`

```text
Creating virtual environment in .venv-bench-cuda ...
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python3.12-venv
```

Debian 与 Ubuntu 把 `venv` 拆成了独立的包，默认不随 Python 一起安装。按提示装上即可：

```bash
sudo apt install python3.12-venv
```

**把 `python3.12` 换成报错信息里的版本号**，它对应的是 `python3` 实际指向的解释器，不同发行版各不相同。可以先确认：

```bash
python3 --version
```

### `No module named pip`

```text
No module named pip
```

删掉已创建的环境，再重新运行创建脚本：

```bash
rm -rf .venv-bench-cuda
bash envs/bench-cuda/create-env.sh
```

**为什么不能直接重跑。** 上一个问题失败时，`python3 -m venv` 已经建好了目录和 `bin/python`，只是在执行 ensurepip 这一步才中断，于是留下一个有解释器、没有 pip 的半成品。创建脚本判断环境是否存在的依据是 `bin/python` 是否可执行，它无法区分半成品和正常环境，因此会跳过创建、直接去装依赖，从而报出这个错误。删除目录才能让它重新走一遍创建流程。

## 新增环境

1. 建目录 `envs/<name>/`；
2. 写 `requirements.txt`；
3. 复制一份同类环境的 `create-env.ps1` / `create-env.sh`，改掉 `--environment-name` 与 `--requirements`，两者都必须指向 `envs/<name>`；
4. 保持 `envs/<name>` → `.venv-<name>` 的命名规律；
5. 在上面的表格中补一行。

创建逻辑本身不要复制：`envs/scripts/create-env.ps1` 和 `envs/scripts/create-env.sh` 是共享引擎，各环境的脚本只是给它传参的薄封装。
