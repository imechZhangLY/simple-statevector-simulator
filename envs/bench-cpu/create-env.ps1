#Requires -Version 5.1
[CmdletBinding()]
param([string]$EnvironmentName = '.venv-bench-cpu')

& (Join-Path $PSScriptRoot '..\scripts\create-env.ps1') `
    -EnvironmentName $EnvironmentName `
    -Requirements 'envs/bench-cpu/requirements.txt'
