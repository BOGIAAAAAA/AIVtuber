#!/usr/bin/env bash
# Install GPT-SoVITS for aiVtuber at the pinned upstream commit (macOS / Linux).
#
# Prerequisites: git, curl, conda (Miniforge recommended) and an activated env:
#   conda create -n GPTSoVits python=3.10 -y && conda activate GPTSoVits
# Usage:
#   bash aiVtuber/tts_server/setup_gpt_sovits.sh --device CU128|CU126|ROCM|MPS|CPU [--source HF|HF-Mirror|ModelScope] [--full-models] [--skip-install]
#     --device        passed to upstream install.sh (MPS installs the CPU wheel; inference runs on CPU)
#     --source        download source (default: HF)
#     --full-models   let install.sh download all pretrained models (pretrained_models.zip, about 4.6 GB)
#                     instead of the about 1.3 GB needed for v2ProPlus inference
#     --skip-install  only clone/check out and download models; do not run install.sh
# Environment: GPT_SOVITS_DIR (default: aiVtuber/GPT-SoVITS), GPT_SOVITS_COMMIT, GPT_SOVITS_REPO_URL, HF_ENDPOINT
set -euo pipefail

GSV_COMMIT="${GPT_SOVITS_COMMIT:-48b1a0169a28582a8984402f82cf438d3bfa6aca}"  # RVC-Boss/GPT-SoVITS main, 2026-08-18
HF_REV="336b2ec4e8d4ac74740798dd40af44e74659ecaf"                          # huggingface.co/lj1995/GPT-SoVITS, 2025-06-04
GSV_REPO_URL="${GPT_SOVITS_REPO_URL:-https://github.com/RVC-Boss/GPT-SoVITS.git}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSV_DIR="${GPT_SOVITS_DIR:-$SCRIPT_DIR/../GPT-SoVITS}"

DEVICE=""
SOURCE="HF"
FULL_MODELS=0
SKIP_INSTALL=0

die() { echo "[ERROR] $*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --device) [ $# -ge 2 ] || usage 1; DEVICE="$2"; shift 2 ;;
        --source) [ $# -ge 2 ] || usage 1; SOURCE="$2"; shift 2 ;;
        --full-models) FULL_MODELS=1; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        -h | --help) usage 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$SOURCE" in
    HF) HF_BASE="${HF_ENDPOINT:-https://huggingface.co}" ;;
    HF-Mirror) HF_BASE="https://hf-mirror.com" ;;
    ModelScope) HF_BASE=""; FULL_MODELS=1 ;;  # single files are only fetched from Hugging Face
    *) die "--source must be HF, HF-Mirror or ModelScope" ;;
esac
if [ "$SKIP_INSTALL" = 0 ]; then
    case "$DEVICE" in
        CU126 | CU128 | ROCM | MPS | CPU) ;;
        *) die "--device must be CU126, CU128, ROCM, MPS or CPU" ;;
    esac
    [ -n "${CONDA_PREFIX:-}" ] ||
        die "Activate a conda env first: conda create -n GPTSoVits python=3.10 -y && conda activate GPTSoVits"
fi

# 1. Clone and pin the upstream source.
if [ ! -d "$GSV_DIR/.git" ]; then
    if [ -d "$GSV_DIR" ] && [ -n "$(ls -A "$GSV_DIR")" ]; then
        die "$GSV_DIR exists but is not a git clone (the old vendored copy?). Move it away first."
    fi
    git clone "$GSV_REPO_URL" "$GSV_DIR"
fi
git -C "$GSV_DIR" fetch --quiet origin
git -C "$GSV_DIR" -c advice.detachedHead=false checkout --quiet "$GSV_COMMIT"
GSV_DIR="$(cd "$GSV_DIR" && pwd)"

# 2. Pretrained models needed for v2ProPlus inference (about 1.3 GB). This creates
#    pretrained_models/sv, which makes install.sh skip its 4.6 GB pretrained_models.zip.
if [ "$FULL_MODELS" = 0 ]; then
    for f in \
        s1v3.ckpt \
        v2Pro/s2Gv2ProPlus.pth \
        sv/pretrained_eres2netv2w24s4ep4.ckpt \
        chinese-hubert-base/config.json \
        chinese-hubert-base/preprocessor_config.json \
        chinese-hubert-base/pytorch_model.bin \
        chinese-roberta-wwm-ext-large/config.json \
        chinese-roberta-wwm-ext-large/tokenizer.json \
        chinese-roberta-wwm-ext-large/pytorch_model.bin; do
        dest="$GSV_DIR/GPT_SoVITS/pretrained_models/$f"
        [ -s "$dest" ] && continue
        mkdir -p "$(dirname "$dest")"
        echo "[INFO] Downloading $f"
        curl -fL --retry 5 -C - -o "$dest.part" "$HF_BASE/lj1995/GPT-SoVITS/resolve/$HF_REV/$f"
        mv "$dest.part" "$dest"
    done
fi

# 3. Upstream installer: FFmpeg, PyTorch, requirements, G2PW, NLTK data, Open JTalk dictionary.
if [ "$SKIP_INSTALL" = 0 ]; then
    if ! command -v wget >/dev/null 2>&1; then  # install.sh downloads with wget, which stock macOS lacks
        conda install -y -q -c conda-forge wget
    fi
    (cd "$GSV_DIR" && bash install.sh --device "$DEVICE" --source "$SOURCE")
fi

echo "[SUCCESS] GPT-SoVITS $GSV_COMMIT is ready at $GSV_DIR"
echo "          Start the server: bash $SCRIPT_DIR/start_tts_server.sh"
