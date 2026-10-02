# Start the GPT-SoVITS api_v2.py TTS server for AIVtuber (Windows).
#
# Usage: double-click start_tts_server.bat, or in PowerShell:
#   .\start_tts_server.ps1 [-GsvDir D:\GPT-SoVITS-v2pro-20250604] [-BindHost 127.0.0.1] [-Port 9880] [-Python C:\path\to\python.exe]
# The same settings can come from environment variables:
#   GPT_SOVITS_DIR, TTS_HOST, TTS_PORT, GPT_SOVITS_PYTHON
# Python selection: -Python / GPT_SOVITS_PYTHON, else <GsvDir>\runtime\python.exe (official Windows
# package), else "python" of the active conda env (conda activate GPTSoVits).
# This file is ASCII on purpose: Windows PowerShell 5.1 misreads UTF-8 scripts that have no BOM.
param(
    [string]$GsvDir = $env:GPT_SOVITS_DIR,
    [string]$BindHost = $env:TTS_HOST,
    [string]$Port = $env:TTS_PORT,
    [string]$Python = $env:GPT_SOVITS_PYTHON
)
$ErrorActionPreference = "Stop"

if (-not $GsvDir) { $GsvDir = Join-Path $PSScriptRoot "..\GPT-SoVITS" }
if (-not $BindHost) { $BindHost = "127.0.0.1" }
if (-not $Port) { $Port = "9880" }

if (-not (Test-Path -LiteralPath (Join-Path $GsvDir "api_v2.py"))) {
    Write-Host "[ERROR] api_v2.py not found in $GsvDir. Run setup_gpt_sovits.ps1, or point -GsvDir / GPT_SOVITS_DIR at GPT-SoVITS (for example the extracted Windows package)." -ForegroundColor Red
    exit 1
}
$GsvDir = (Resolve-Path -LiteralPath $GsvDir).Path

# api_v2.py rewrites the file passed with -c whenever it loads weights, so it only gets a copy.
$ExampleCfg = Join-Path $PSScriptRoot "tts_infer.example.yaml"
$UserCfg = Join-Path $PSScriptRoot "tts_infer.yaml"
$RuntimeCfg = Join-Path $PSScriptRoot "runtime\tts_infer.yaml"
if (-not (Test-Path -LiteralPath $UserCfg)) {
    Copy-Item -LiteralPath $ExampleCfg -Destination $UserCfg
    Write-Host "[INFO] Created $UserCfg from the template; edit it to change models or device."
}
New-Item -ItemType Directory -Force -Path (Split-Path $RuntimeCfg) | Out-Null
Copy-Item -LiteralPath $UserCfg -Destination $RuntimeCfg -Force

# Pretrained models used by the default (v2ProPlus zero-shot) configuration.
$Required = @(
    "GPT_SoVITS\pretrained_models\s1v3.ckpt",
    "GPT_SoVITS\pretrained_models\v2Pro\s2Gv2ProPlus.pth",
    "GPT_SoVITS\pretrained_models\sv\pretrained_eres2netv2w24s4ep4.ckpt",
    "GPT_SoVITS\pretrained_models\chinese-hubert-base\pytorch_model.bin",
    "GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large\pytorch_model.bin"
)
foreach ($f in $Required) {
    if (-not (Test-Path -LiteralPath (Join-Path $GsvDir $f))) {
        Write-Host "[WARN] Missing $GsvDir\$f" -ForegroundColor Yellow
    }
}

$PyArgs = @()
$RuntimePython = Join-Path $GsvDir "runtime\python.exe"
if ($Python) {
    # A relative path must stay valid after Set-Location below; a bare command name (python) is left as is.
    if (Test-Path -LiteralPath $Python -PathType Leaf) { $Python = (Resolve-Path -LiteralPath $Python).Path }
    $PyExe = $Python
    # A conda env python used without "conda activate" still needs the env's DLL folders
    # (for example FFmpeg for torchcodec) on PATH.
    $PyHome = Split-Path -Parent $Python
    if ($PyHome -and (Test-Path -LiteralPath (Join-Path $PyHome "Library\bin"))) {
        $env:PATH = $PyHome + ";" + (Join-Path $PyHome "Library\bin") + ";" + (Join-Path $PyHome "Scripts") + ";" + $env:PATH
    }
} elseif (Test-Path -LiteralPath $RuntimePython) {
    # Official Windows package: like upstream go-webui.ps1, put runtime (ffmpeg.exe) on PATH and use isolated mode.
    $PyExe = $RuntimePython
    $PyArgs += "-I"
    $env:PATH = (Join-Path $GsvDir "runtime") + ";" + $GsvDir + ";" + $env:PATH
} else {
    $PyExe = "python"
}

Set-Location -LiteralPath $GsvDir   # api_v2.py finds its modules and the relative model paths from the working directory
# From here on only python runs. Its stderr (tracebacks, uvicorn logs) must not stop this script when
# Windows PowerShell 5.1 turns redirected stderr into error records (e.g. .\start_tts_server.ps1 *> log.txt).
$ErrorActionPreference = "Continue"
$ImportOk = $false
try {
    & $PyExe @PyArgs -c "import torch, fastapi"
    $ImportOk = ($LASTEXITCODE -eq 0)
} catch {
    $ImportOk = $false   # python not found
}
if (-not $ImportOk) {
    Write-Host "[ERROR] '$PyExe' cannot import torch/fastapi. Run 'conda activate GPTSoVits' first, or pass -Python / set GPT_SOVITS_PYTHON." -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] GPT-SoVITS : $GsvDir"
Write-Host "[INFO] Python     : $PyExe $PyArgs"
Write-Host "[INFO] Config     : $RuntimeCfg"
Write-Host "[INFO] Serving http://${BindHost}:${Port} once the models are loaded (GET /docs returns 200 when ready)"
& $PyExe @PyArgs -X utf8 api_v2.py -a $BindHost -p $Port -c $RuntimeCfg
exit $LASTEXITCODE
