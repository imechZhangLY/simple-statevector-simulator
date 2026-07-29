#Requires -Version 5.1
[CmdletBinding()]
param([string]$EnvironmentName = '.venv-cuda')

& (Join-Path $PSScriptRoot '..\scripts\create-env.ps1') `
    -EnvironmentName $EnvironmentName `
    -Requirements 'envs/cuda/requirements.txt'
