# Install GPT-SoVITS for AIVtuber at the pinned upstream commit (Windows, conda route).
# Users of the official Windows package (GPT-SoVITS-v2pro-*.7z) do not need this script:
# extract the package and point start_tts_server.bat at it (-GsvDir or GPT_SOVITS_DIR).
#
# Prerequisites: Git, curl.exe (Windows 10 1803+), Miniforge/Miniconda, and an activated env:
#   conda create -n GPTSoVits python=3.10 -y
#   conda activate GPTSoVits
# Usage (PowerShell 7 is what the upstream README uses):
#   pwsh -ExecutionPolicy Bypass -File tts_server\setup_gpt_sovits.ps1 -Device CU128 [-Source HF] [-FullModels] [-SkipInstall]
#     -Device       CU126 | CU128 | CPU, passed to upstream install.ps1
#     -Source       HF | HF-Mirror | ModelScope (default HF)
#     -FullModels   let install.ps1 download all pretrained models (pretrained_models.zip, about 4.6 GB)
#     -SkipInstall  only clone/check out and download models; do not run install.ps1
# Environment: GPT_SOVITS_DIR (default GPT-SoVITS in the repository root), GPT_SOVITS_COMMIT, GPT_SOVITS_REPO_URL, HF_ENDPOINT
# This file is ASCII on purpose: Windows PowerShell 5.1 misreads UTF-8 scripts that have no BOM.
param(
    [ValidateSet("CU126", "CU128", "CPU")][string]$Device,
    [ValidateSet("HF", "HF-Mirror", "ModelScope")][string]$Source = "HF",
    [switch]$FullModels,
    [switch]$SkipInstall,
    [string]$GsvDir = $env:GPT_SOVITS_DIR
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest in install.ps1 is very slow with a progress bar on PowerShell 5.1

$GsvCommit = "48b1a0169a28582a8984402f82cf438d3bfa6aca"   # RVC-Boss/GPT-SoVITS main, 2026-08-18
if ($env:GPT_SOVITS_COMMIT) { $GsvCommit = $env:GPT_SOVITS_COMMIT }
$HfRev = "336b2ec4e8d4ac74740798dd40af44e74659ecaf"       # huggingface.co/lj1995/GPT-SoVITS, 2025-06-04
$RepoUrl = "https://github.com/RVC-Boss/GPT-SoVITS.git"
if ($env:GPT_SOVITS_REPO_URL) { $RepoUrl = $env:GPT_SOVITS_REPO_URL }
if (-not $GsvDir) { $GsvDir = Join-Path $PSScriptRoot "..\GPT-SoVITS" }
$GsvDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($GsvDir)

function Invoke-Checked([string]$Exe, [string[]]$ArgList) {
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($ArgList -join ' ') failed (exit code $LASTEXITCODE)" }
}

if (-not $SkipInstall) {
    if (-not $Device) { throw "-Device is required (CU126, CU128 or CPU) unless -SkipInstall is given" }
    if (-not $env:CONDA_PREFIX) { throw "Activate a conda env first: conda create -n GPTSoVits python=3.10 -y; conda activate GPTSoVits" }
    if ($PSVersionTable.PSVersion.Major -lt 7) {
        # install.ps1 captures conda/pip output with 2>&1 under ErrorActionPreference Stop; on 5.1 any stderr line aborts it.
        Write-Host "[WARN] Upstream install.ps1 is meant for PowerShell 7 (pwsh). Windows PowerShell 5.1 may abort at the first warning printed by conda or pip." -ForegroundColor Yellow
    }
}

# 1. Clone and pin the upstream source.
if (-not (Test-Path -LiteralPath (Join-Path $GsvDir ".git"))) {
    if ((Test-Path -LiteralPath $GsvDir) -and (Get-ChildItem -Force -LiteralPath $GsvDir | Select-Object -First 1)) {
        throw "$GsvDir exists but is not a git clone (the old vendored copy?). Move it away first."
    }
    Invoke-Checked "git" @("clone", $RepoUrl, $GsvDir)
}
Invoke-Checked "git" @("-C", $GsvDir, "fetch", "--quiet", "origin")
Invoke-Checked "git" @("-C", $GsvDir, "-c", "advice.detachedHead=false", "checkout", "--quiet", $GsvCommit)

# 2. Pretrained models needed for v2ProPlus inference (about 1.3 GB). This creates
#    pretrained_models\sv, which makes install.ps1 skip its 4.6 GB pretrained_models.zip.
if ((-not $FullModels) -and ($Source -ne "ModelScope")) {
    $HfBase = "https://huggingface.co"
    if ($env:HF_ENDPOINT) { $HfBase = $env:HF_ENDPOINT }
    if ($Source -eq "HF-Mirror") { $HfBase = "https://hf-mirror.com" }
    $Files = @(
        "s1v3.ckpt",
        "v2Pro/s2Gv2ProPlus.pth",
        "sv/pretrained_eres2netv2w24s4ep4.ckpt",
        "chinese-hubert-base/config.json",
        "chinese-hubert-base/preprocessor_config.json",
        "chinese-hubert-base/pytorch_model.bin",
        "chinese-roberta-wwm-ext-large/config.json",
        "chinese-roberta-wwm-ext-large/tokenizer.json",
        "chinese-roberta-wwm-ext-large/pytorch_model.bin"
    )
    foreach ($f in $Files) {
        $Dest = Join-Path $GsvDir ("GPT_SoVITS\pretrained_models\" + $f.Replace("/", "\"))
        if ((Test-Path -LiteralPath $Dest) -and ((Get-Item -LiteralPath $Dest).Length -gt 0)) { continue }
        New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null
        Write-Host "[INFO] Downloading $f"
        Invoke-Checked "curl.exe" @("-fL", "--retry", "5", "-C", "-", "-o", "$Dest.part", "$HfBase/lj1995/GPT-SoVITS/resolve/$HfRev/$f")
        Move-Item -LiteralPath "$Dest.part" -Destination $Dest -Force
    }
}

# 3. Upstream installer: FFmpeg, PyTorch, requirements, G2PW, NLTK data, Open JTalk dictionary.
if (-not $SkipInstall) {
    & (Join-Path $GsvDir "install.ps1") -Device $Device -Source $Source
    if ($LASTEXITCODE -ne 0) { throw "install.ps1 failed (exit code $LASTEXITCODE)" }
}

Write-Host "[SUCCESS] GPT-SoVITS $GsvCommit is ready at $GsvDir"
Write-Host "          Start the server with tts_server\start_tts_server.bat"
