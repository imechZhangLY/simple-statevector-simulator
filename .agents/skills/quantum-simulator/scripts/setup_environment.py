"""Platform-aware environment bootstrap dispatcher.

Detects the operating system and whether an NVIDIA GPU is present, then invokes
the matching bootstrap script with the matching requirements file:

    Windows + CUDA   -> .vscode/bootstrap-env.ps1  requirements-torch-cuda.txt
    Windows, no CUDA -> .vscode/bootstrap-env.ps1  requirements-torch-cpu.txt
    Linux   + CUDA   -> .vscode/bootstrap-env.sh   requirements-torch-cuda.txt
    Linux, no CUDA   -> .vscode/bootstrap-env.sh   requirements-torch-cpu.txt
    macOS            -> .vscode/bootstrap-env.sh   requirements-torch-macos.txt

macOS has no CUDA build at all, and the "+cpu" wheels are published only for
Linux and Windows, so macOS must use the plain PyPI build.

Run with the system interpreter; it only creates the environment.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def detect_cuda() -> bool:
    """Probe for an NVIDIA driver before torch is installed.

    nvidia-smi ships with the driver, so it is the only reliable pre-install
    signal. Importing torch is not an option here: the environment that would
    provide it is exactly what this script is creating.
    """
    if platform.system() == "Darwin":
        return False
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def select_requirements(system: str, cuda: bool) -> str:
    if system == "Darwin":
        return "requirements-torch-macos.txt"
    return "requirements-torch-cuda.txt" if cuda else "requirements-torch-cpu.txt"


def interpreter_path(environment_name: str) -> Path:
    environment = REPOSITORY_ROOT / environment_name
    if platform.system() == "Windows":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def build_command(environment_name: str, requirements: str) -> list[str]:
    if platform.system() == "Windows":
        script = REPOSITORY_ROOT / ".vscode" / "bootstrap-env.ps1"
        # pwsh is tried first on purpose. Launching Windows PowerShell 5.1 from a
        # non-PowerShell parent such as this script leaks PSModulePath entries
        # that point at PowerShell 7's modules, so 5.1 binds
        # Microsoft.PowerShell.Utility to an incompatible copy and Get-FileHash
        # disappears. pwsh sanitizes that variable for itself; python does not.
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if shell is None:
            raise RuntimeError("neither pwsh nor powershell was found on PATH")
        return [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-EnvironmentName",
            environment_name,
            "-Requirements",
            requirements,
        ]

    script = REPOSITORY_ROOT / ".vscode" / "bootstrap-env.sh"
    shell = shutil.which("bash")
    if shell is None:
        raise RuntimeError("bash was not found on PATH")
    return [
        shell,
        str(script),
        "--environment-name",
        environment_name,
        "--requirements",
        requirements,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-name", default=".venv")
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="skip CUDA detection and install the CPU build",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="report the detected plan without running it",
    )
    arguments = parser.parse_args()

    system = platform.system()
    cuda = False if arguments.force_cpu else detect_cuda()
    requirements = select_requirements(system, cuda)
    python = interpreter_path(arguments.environment_name)

    print(f"platform:     {system} ({platform.machine()})")
    print(f"cuda:         {'yes' if cuda else 'no'}")
    print(f"requirements: {requirements}")
    print(f"environment:  {arguments.environment_name}")
    print(f"interpreter:  {python}")

    if arguments.print_only:
        return 0

    if not (REPOSITORY_ROOT / requirements).is_file():
        print(f"requirements file not found: {requirements}", file=sys.stderr)
        return 1

    command = build_command(arguments.environment_name, requirements)
    print(f"running:      {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT)
    if completed.returncode != 0:
        return completed.returncode

    if not python.is_file():
        print(f"bootstrap finished but {python} is missing", file=sys.stderr)
        return 1

    print(f"\nready. run project code with:\n  {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
