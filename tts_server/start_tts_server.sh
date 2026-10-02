#!/usr/bin/env bash
# Start the GPT-SoVITS api_v2.py TTS server for AIVtuber (macOS / Linux).
#
# Usage:
#   bash tts_server/start_tts_server.sh
#
# Environment variables (all optional):
#   GPT_SOVITS_DIR        GPT-SoVITS root (default: GPT-SoVITS in the repository root, created by setup_gpt_sovits.sh)
#   GPT_SOVITS_PYTHON     Python of the GPT-SoVITS environment (default: python of the active conda env
#                         $GPT_SOVITS_CONDA_ENV, else "conda run -n $GPT_SOVITS_CONDA_ENV python", else python)
#   GPT_SOVITS_CONDA_ENV  conda env name (default: GPTSoVits, as in the upstream README)
#   TTS_HOST / TTS_PORT   bind address / port (default: 127.0.0.1 / 9880; use 0.0.0.0 for LAN access)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSV_DIR="${GPT_SOVITS_DIR:-$SCRIPT_DIR/../GPT-SoVITS}"
HOST="${TTS_HOST:-127.0.0.1}"
PORT="${TTS_PORT:-9880}"
CONDA_ENV="${GPT_SOVITS_CONDA_ENV:-GPTSoVits}"

EXAMPLE_CFG="$SCRIPT_DIR/tts_infer.example.yaml"
USER_CFG="$SCRIPT_DIR/tts_infer.yaml"
RUNTIME_CFG="$SCRIPT_DIR/runtime/tts_infer.yaml"

die() { echo "[ERROR] $*" >&2; exit 1; }

[ -f "$GSV_DIR/api_v2.py" ] || die "api_v2.py not found in $GSV_DIR. Run setup_gpt_sovits.sh first or set GPT_SOVITS_DIR."
GSV_DIR="$(cd "$GSV_DIR" && pwd)"

# api_v2.py rewrites the file passed with -c whenever it loads weights, so it only gets a copy.
if [ ! -f "$USER_CFG" ]; then
    cp "$EXAMPLE_CFG" "$USER_CFG"
    echo "[INFO] Created $USER_CFG from the template; edit it to change models or device."
fi
mkdir -p "$(dirname "$RUNTIME_CFG")"
cp "$USER_CFG" "$RUNTIME_CFG"

# Pretrained models used by the default (v2ProPlus zero-shot) configuration.
for f in \
    GPT_SoVITS/pretrained_models/s1v3.ckpt \
    GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth \
    GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt \
    GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin \
    GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin; do
    [ -s "$GSV_DIR/$f" ] || echo "[WARN] Missing $GSV_DIR/$f (run setup_gpt_sovits.sh)" >&2
done

if [ -n "${GPT_SOVITS_PYTHON:-}" ]; then
    case "$GPT_SOVITS_PYTHON" in
        /*) PY=("$GPT_SOVITS_PYTHON") ;;
        */*) PY=("$PWD/$GPT_SOVITS_PYTHON") ;;  # relative path: keep it valid after the cd below
        *) PY=("$GPT_SOVITS_PYTHON") ;;         # command name, looked up on PATH
    esac
elif [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
    PY=(python)
elif command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
else
    PY=(python)
fi

cd "$GSV_DIR"   # api_v2.py finds its modules and the relative model paths from the working directory
"${PY[@]}" -c "import torch, fastapi" ||
    die "'${PY[*]}' cannot import torch/fastapi. Run 'conda activate $CONDA_ENV' or set GPT_SOVITS_PYTHON."

VERSION="not a git checkout"
[ -d .git ] && VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown commit")"
echo "[INFO] GPT-SoVITS : $GSV_DIR ($VERSION)"
echo "[INFO] Python     : ${PY[*]}"
echo "[INFO] Config     : $RUNTIME_CFG"
echo "[INFO] Serving http://$HOST:$PORT once the models are loaded (GET /docs returns 200 when ready)"
exec "${PY[@]}" -X utf8 api_v2.py -a "$HOST" -p "$PORT" -c "$RUNTIME_CFG"
