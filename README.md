# AIVtuber

用語音和 AI 角色（Zora）練習英文會話的 AI VTuber：
麥克風收音 → Google 語音辨識 → Groq LLM 串流產生回覆 → 按句切分，GPT-SoVITS 逐句合成角色聲音、邊合成邊播放；
另外可以依文字情緒切換 VTube Studio 模型的表情，並附有用自己的對話資料微調 GPT-2 的工具。

| 模式 | 指令 | 說明 |
|---|---|---|
| 語音聊天（預設） | `aivtuber` | 說話 → AI 用角色聲音回答，說 `exit` 結束 |
| 聊天 + VTube Studio | `aivtuber --api` | 同上，並保持與 VTube Studio 的連線 |
| 微調 GPT-2 | `aivtuber --train` | 用 `data/` 的對話資料微調 gpt2-medium |
| 文字生成 | `aivtuber --generate` | 分別輸出微調 GPT-2 與 Groq 的結果 |
| 情緒分類 | `aivtuber --classify --api` | 判斷正面／負面並切換 VTube Studio 表情 |

`aivtuber` 是安裝後的指令，等同 `python -m vtuber`（安裝方式見〈安裝〉）。

## 專案結構

```
AIVtuber/
├── src/vtuber/            # 主程式套件（各模組的職責見〈架構〉）
├── tests/                 # pytest 測試，不需要網路、麥克風、GPU 或 torch
├── data/                  # GPT-2 微調用的對話資料（--train）
├── voices/                # 角色語音素材與 TTS 參考音檔，見 voices/README.md
├── tts_server/            # GPT-SoVITS（TTS server）的安裝與啟動腳本，見 tts_server/README.md
├── docs/images/           # 舊版訓練產生的曲線圖
├── config.example.yaml    # 設定範本，複製成 config.yaml 再修改
├── pyproject.toml         # 套件資訊、相依套件（chat / train / dev）、pytest 與 ruff 設定
└── README.md
```

執行後才會產生、不納入版控的檔案也都放在 repo 根目錄：`config.yaml`、`.env`、`pyvts_token.txt`（VTube Studio token）、
`saved_model/` 與 `outputs/`（訓練輸出）、`GPT-SoVITS/`（setup 腳本下載的上游原始碼與模型），以及 `tts_server/tts_infer.yaml`。

## 架構

```
             ┌────────────── aivtuber（src/vtuber/cli.py：參數解析、模式分派）──────────────┐
             │                                                                              │
 語音聊天    麥克風 ─► asr.py ─► llm.py ──────► sentences.py ─► tts.py ───────────► audio.py ─► 喇叭
（chat.py、        Google      Groq 串流回覆      按句切分      GPT-SoVITS api_v2    PyAudio
 pipeline.py）     語音辨識    （gpt-oss-20b）    與清理        POST /tts，一次一句
                                                                   │  回傳 WAV
                                                                   ▼
                                      GPT-SoVITS api_v2.py（另一個程序，可在別台電腦；見 tts_server/）

 --classify   文字 ─► sentiment.py（DistilBERT SST-2）─► vts.py ─► VTube Studio（WebSocket :8001）
 --generate   文字 ─► generation.py（微調後的 GPT-2）＋ llm.py（Groq，一次拿整段）─► 分別印出
 --train      data/*.txt ─► training.py（清理 → 切 block → Trainer）─► saved_model/、outputs/
```

### 語音回覆管線：邊合成邊播放

使用者說完話後，不再「等 LLM 整段回完 → 整段合成 → 播放」，而是由三個 task 以佇列串接（`src/vtuber/pipeline.py`）：

```
LLM 串流 ─► 切句、清理 ─► 文字佇列 ─► 合成（一次一句）─► 音訊佇列（最多 2 句）─► 依序播放
```

- 句子一完整就送去合成，播放第 N 句的同時合成第 N+1 句；開口前只要等「第一句的 LLM + 第一句的 TTS」，
  而不是整段回覆的 LLM 與 TTS。
- 切句依據 `. ! ? …`、中文句尾標點與換行，不會在縮寫（Mr. e.g. U.S.）、小數（3.14）、引號或括號收尾處誤切；
  比 `chat.min_sentence_chars` 短的句子併入下一句，超過 `chat.max_sentence_chars` 還沒有句尾時在逗號或空白處切開。
- 朗讀前去掉 markdown 記號與 emoji（只有真正行首的 `-`、`#`、`>` 才算條列或標題記號）；數字一律保留，例如編號 `1.` 會照念。沒有字母或數字的片段不送去合成。
- 某一句合成失敗只會跳過那一句。按 Ctrl+C 會立刻停止播放，所有 task 與連線都會釋放。
- 每輪結束時 log 會記錄完整回覆，以及從辨識出使用者的話到「第一句文字出現」與「開始出聲」（第一段聲音開始寫入輸出裝置）各花了幾秒；這一輪中途出錯時，也會記錄已經產生的回覆。
- 播放器整個 session 共用一個 PyAudio，連續幾句格式相同時重用同一個輸出 stream；所有 PyAudio 呼叫都在一條專屬的背景執行緒。

| 檔案 | 職責 |
|---|---|
| `src/vtuber/cli.py`、`__main__.py` | 命令列（`aivtuber`／`python -m vtuber`，保留舊版 `run_3.py` 的全部參數）、模式分派 |
| `src/vtuber/config.py` | 所有設定的 dataclass，從 `config.yaml` 載入 |
| `src/vtuber/chat.py` | 聊天主迴圈：聽 → 串流回覆、邊合成邊播放；單輪出錯不會中斷 |
| `src/vtuber/pipeline.py` | 語音回覆管線：LLM 串流 → 切句 → 逐句合成 → 依序播放 |
| `src/vtuber/sentences.py` | 增量式切句器與朗讀前清理（markdown、emoji） |
| `src/vtuber/asr.py` | 語音辨識（啟動時校正一次環境噪音，每輪重新開麥克風） |
| `src/vtuber/llm.py` | Groq 客戶端（共用一個連線；聊天用串流，`--generate` 一次拿整段；失敗時念備援訊息） |
| `src/vtuber/tts.py` | TTS 介面 `TextToSpeech` 與 GPT-SoVITS 客戶端（api_v2，另保留舊版 api.py） |
| `src/vtuber/audio.py` | 直接播放記憶體中的 WAV，不寫暫存檔；整個 session 共用 PyAudio 與輸出 stream |
| `src/vtuber/vts.py` | VTube Studio 連線、token 認證、切換表情 |
| `src/vtuber/sentiment.py`、`generation.py`、`training.py` | 情緒分類、GPT-2 生成、GPT-2 微調（torch 只在這些模式才載入） |

想換 TTS 引擎時，新增一個實作 `TextToSpeech`（`prepare()`、`synthesize(text) -> WAV bytes` 與 `aclose()`）的類別，
並在 `src/vtuber/tts.py` 的 `create_tts()` 加上對應的 `tts.engine` 即可，聊天流程不用改。

## 安裝

需求：Python 3.9 以上、麥克風與喇叭、[Groq API key](https://console.groq.com/keys)。
以下以「主程式和 GPT-SoVITS server 在同一台電腦」為例。

### 1. 主程式

在 repo 根目錄執行：

```bash
python -m venv .venv
source .venv/bin/activate              # Windows：.venv\Scripts\activate
pip install -e ".[chat]"               # 語音聊天、VTube Studio
pip install -e ".[train]"              # 需要 --train / --generate / --classify 時再裝
```

- 相依套件寫在 `pyproject.toml`，依用途分成幾組：`chat`（語音聊天與 VTube Studio）、`train`（GPT-2 微調／生成與情緒分類）、
  `dev`（測試與靜態檢查）。可以一次裝多組，例如 `--classify --api` 需要 `pip install -e ".[chat,train]"`。
- 請用 `-e`（editable）安裝：設定檔、語音素材與訓練資料都在 repo 裡，程式以 repo 根目錄為基準找這些檔案。安裝後改程式碼不必重裝。
- **PyAudio** 需要系統的 PortAudio：macOS 先 `brew install portaudio`；Ubuntu/Debian 先 `sudo apt install portaudio19-dev`；Windows 直接 `pip install` 即可。
- **torch** 請依平台（CUDA / CPU / Apple Silicon）照 [PyTorch 官網](https://pytorch.org/get-started/locally/) 安裝。
- 訓練可以在另一台有 GPU 的機器上進行，那台只需要 `pip install -e ".[train]"`（不需要 PortAudio）。

### 2. GPT-SoVITS（TTS server）

TTS server 是 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的 `api_v2.py`，不放在這個 repo 裡，用自己的 Python 環境另外安裝。
完整步驟、需要的模型與疑難排解見 [`tts_server/README.md`](tts_server/README.md)，摘要如下（三選一）：

- **macOS／Linux**：先建立並啟用 conda 環境（`conda create -n GPTSoVits python=3.10 -y && conda activate GPTSoVits`），再執行
  `bash tts_server/setup_gpt_sovits.sh --device CU128`（沒有 NVIDIA 顯示卡用 `CPU`，Mac 用 `MPS`，推論只用 CPU）。
- **Windows（conda）**：同樣先啟用 conda 環境，再執行 `pwsh -ExecutionPolicy Bypass -File tts_server\setup_gpt_sovits.ps1 -Device CU128`。
- **Windows 整合包**：下載官方整合包解壓即可，不必執行 setup 腳本；用環境變數 `GPT_SOVITS_DIR` 告訴啟動腳本它的位置，
  例如 `setx GPT_SOVITS_DIR D:\GPT-SoVITS-v2pro-20250604`。

setup 腳本會把上游鎖定在 commit `48b1a01`、clone 到 repo 根目錄的 `GPT-SoVITS/`（已被 `.gitignore` 忽略），並下載 v2ProPlus 零樣本合成需要的模型（約 1.3 GB），不需要訓練角色模型：音色來自參考音檔。

## 設定

在 repo 根目錄執行：

```bash
cp config.example.yaml config.yaml   # Windows：copy config.example.yaml config.yaml
```

- `config.yaml` 只需要寫想改的項目，其餘使用預設值；沒有這個檔案時全部使用預設值。每個設定的說明都在 `config.example.yaml` 裡。
- `config.yaml` 已被 `.gitignore` 忽略。也可以用 `--config 路徑` 指定其他設定檔。
- 設定檔中本機的相對路徑（`training.data_dir`、`training.model_save_path`、`training.output_dir`、`vts.token_path`）一律以專案根目錄（repo 根目錄）為基準，從哪個目錄執行都一樣。
- `tts.ref_audio_path`（參考音檔）是由 TTS server 讀取的路徑：相對路徑同樣以專案根目錄為基準，轉成這台電腦上的絕對路徑後送出；
  絕對路徑原樣送出（server 在別台電腦時使用），見下方〈參考音檔〉。
- 模型欄位 `training.base_model`、`sentiment.model` 可填 Hugging Face 模型 ID 或本機路徑：以 `./`、`../` 開頭、絕對路徑，
  或專案根目錄底下確實存在的路徑視為本機路徑（同樣以專案根目錄為基準），其餘（例如 `gpt2-medium`）視為模型 ID。
- 命令列參數優先於設定檔：`--device_index` 覆蓋 `asr.device_index`，`--data_dir`、`--model_save_path` 覆蓋 `training` 的同名設定（命令列的相對路徑以目前目錄為基準）。

常改的設定：

| 設定 | 預設值 | 說明 |
|---|---|---|
| `llm.model` | `openai/gpt-oss-20b` | Groq 模型；舊的 `llama3-8b-8192` 已下架 |
| `llm.reasoning_effort`、`include_reasoning` | `low`、`false` | 推理越少，第一句越快出來；推理內容不傳回。換成非推理模型時兩個都要改成 `null`（各系列的設法見 `config.example.yaml`） |
| `tts.url` | `http://127.0.0.1:9880` | GPT-SoVITS api_v2 的位址 |
| `tts.ref_audio_path`、`prompt_text`、`prompt_lang` | `voices/firefly/ref_firefly_01.wav` 與其內容 | 角色聲音的參考音檔、音檔裡說的話與語言 |
| `tts.text_lang` | `en` | 要合成的文字的語言（小寫代碼，可用的值見 `config.example.yaml`） |
| `chat.min_sentence_chars`、`max_sentence_chars` | `12`、`120` | 切句的最短與最長長度（字元） |
| `asr.language` | `en-US` | 語音辨識語言 |
| `vts.positive_expression` / `negative_expression` | `Happy` / `Sad` | 表情檔名（自動補 `.exp3.json`） |

### Groq API key

API key 只從環境變數讀取，不要寫進設定檔：

```bash
export GROQ_API_KEY="gsk_..."          # Windows PowerShell：$env:GROQ_API_KEY="gsk_..."
```

或在 repo 根目錄的 `.env` 寫一行 `GROQ_API_KEY=gsk_...`（`.env` 已被 `.gitignore` 忽略）。
沒有 key 時程式仍會執行，但 AI 只會念 `llm.fallback_message`。

## 啟動 TTS server

聊天前先在同一台電腦開另一個終端機啟動 GPT-SoVITS 的 `api_v2.py`（選項與疑難排解見 [`tts_server/README.md`](tts_server/README.md)）：

```bash
conda activate GPTSoVits
bash tts_server/start_tts_server.sh     # Windows：點兩下 tts_server\start_tts_server.bat
```

- 載入模型需要幾十秒，載入完成後 server 才開始接受連線；`GET /docs` 回應 200 就代表已就緒：
  `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:9880/docs`
- 主程式啟動時會先等 server 就緒（最多 `tts.ready_timeout` 秒，預設 60），再合成一句短句暖機（`tts.warmup_text`，不播放），然後才開始聽。
  等不到 server，或暖機失敗（例如參考音檔路徑錯誤、語言代碼不支援）時，程式會說明原因並結束（結束碼 1），不會等到第一輪對話才失敗。
- 沒有開 TTS server、只想測試語音辨識與 LLM 時，在 `config.yaml` 設 `tts.wait_for_server: false` 與 `tts.warmup_text: ""`（每句合成都會失敗並被跳過）。
- 舊版 GPT-SoVITS 的 `api.py` 仍可使用：設 `tts.engine: gpt-sovits-v1`（設定說明見 `config.example.yaml`）。

### 參考音檔

角色的音色來自參考音檔（3–10 秒、只有一個人說話）。每次合成時，主程式都會把參考音檔的路徑和內容一起送給 server：

```yaml
tts:
  ref_audio_path: voices/firefly/ref_firefly_01.wav  # 預設值
  prompt_text: "I understand. Article 4 of Glamoth military regulations."  # 音檔裡說的話，要逐字相符
  prompt_lang: en
```

- **相對路徑**以專案根目錄為基準，轉成這台電腦上的絕對路徑後送出。server 在同一台電腦時用這種寫法；找不到檔案時，啟動時會發出警告。
- **絕對路徑**原樣送出，POSIX 與 Windows 寫法都可以（例如 `/srv/voices/ref.wav`、`D:/voices/ref.wav`）。server 在別台電腦時用這種寫法。
- 預設參考音檔的來源與切法見 [`voices/README.md`](voices/README.md)。

### 改成連到別台電腦的 TTS server

1. 在 server 那台以 `TTS_HOST=0.0.0.0` 啟動，讓區網可以連線（預設只監聽 `127.0.0.1`），並在防火牆開放 9880 埠：

   ```bash
   TTS_HOST=0.0.0.0 bash tts_server/start_tts_server.sh   # Windows：start_tts_server.bat -BindHost 0.0.0.0
   ```

   api_v2 沒有任何驗證機制，只能在信任的區網開放。
2. 把參考音檔放到 server 那台電腦上，再改主程式這台的 `config.yaml`：

   ```yaml
   tts:
     url: http://192.168.1.106:9880                           # server 那台的 IP
     ref_audio_path: D:/AIVtuber/voices/firefly/ref_firefly_01.wav  # 參考音檔在 server 那台電腦上的絕對路徑
   ```

   參考音檔是由 server 讀取的。相對路徑會被轉成主程式這台電腦上的路徑，server 在別台電腦時讀不到（啟動時會提醒）。

## 使用方式

安裝後在任何目錄都可以執行 `aivtuber`（等同 `python -m vtuber`），參數與舊版 `run_3.py` 完全相同。

```bash
aivtuber                                   # 語音聊天：說 "exit"（大小寫、句尾標點不拘）或按 Ctrl+C 結束
aivtuber --api                             # 語音聊天並連線 VTube Studio（--vts_control 相同）
aivtuber --device_index 2                  # 指定麥克風
aivtuber --train                           # 微調 GPT-2（可加 --data_dir、--model_save_path）
aivtuber --generate --input_text "AI: Hi! How are you today?"
aivtuber --classify --input_text "I love this!" --api
aivtuber --config my.yaml --log-level DEBUG
```

- 列出麥克風編號：`python -c "import speech_recognition as sr; print(list(enumerate(sr.Microphone.list_microphone_names())))"`
- **VTube Studio**：在 VTS 設定中開啟 API（預設埠 8001）。第一次連線時 VTS 會跳出授權視窗，按「Allow」後 token 會存到 repo 根目錄的 `pyvts_token.txt`，之後不再詢問。
  表情名稱就是模型資料夾裡 `.exp3.json` 的檔名（`Happy` 會自動變成 `Happy.exp3.json`）。
- **訓練輸出**：模型與 tokenizer 存到 `saved_model/`（`--generate` 從這裡載入）；checkpoint、訓練曲線
  `train_loss.png`、`learning_rate.png`、`perplexity.png` 與 `training_metrics.json` 存到 `outputs/`。兩者都已被 `.gitignore` 忽略。
- **結束碼**：`0` 正常；`1` 執行錯誤，訊息會說明原因（例如 TTS server 沒開、GPT-2 生成失敗、`--classify --api` 沒能切換表情）；
  `2` 設定錯誤，包括設定檔格式或數值錯誤，以及執行時才發現的設定問題（例如 `device: cuda` 但這台沒有 CUDA）；`130` 被 Ctrl+C 中斷。

### 從舊的 `aiVtuber/` 目錄結構升級

程式碼改成標準的 src layout（`src/vtuber/`），原本放在 `aiVtuber/` 底下的內容都移到 repo 根目錄：

- 安裝改用 `pip install -e ".[chat]"`：相依套件改寫在 `pyproject.toml`，`requirements*.txt` 已移除。舊的 `aiVtuber/.venv` 請在 repo 根目錄重建。
- `run_3.py` 已移除，改用 `aivtuber`（或 `python -m vtuber`），參數完全相同。
- 不在版控內的本機檔案請從 `aiVtuber/` 搬到 repo 根目錄的相同位置：`config.yaml`、`.env`、`pyvts_token.txt`、`saved_model/`、`outputs/`、
  `GPT-SoVITS/`、`tts_server/tts_infer.yaml`。設定檔裡的相對路徑不用改（基準目錄跟著變成 repo 根目錄）；
  寫成絕對路徑的設定要拿掉路徑中的 `aiVtuber/`，例如 server 在別台電腦時的 `tts.ref_audio_path`。
- `image/` 的訓練曲線圖移到 `docs/images/`。

### 與舊版 `run_3.py` 的差異

- Groq 模型改為 `openai/gpt-oss-20b`（`llama3-8b-8192` 已下架），聊天時不再白跑 GPT-2。
- 回覆改成串流：LLM 一邊產生，一邊按句合成與播放，不必等整段回覆合成完才開口。
- TTS 改用 GPT-SoVITS 的 `api_v2.py`（舊的 `api.py` 可用 `tts.engine: gpt-sovits-v1`），參考音檔改用 repo 內的 `voices/firefly/ref_firefly_01.wav`。
- TTS 與參考音檔等設定改到 `config.yaml`；語音直接在記憶體中播放，不再寫 `output.wav`。
- VTube Studio 外掛名稱改為 `AIVtuber`、token 存在 repo 根目錄的 `pyvts_token.txt`，第一次使用需要在 VTS 重新按一次 Allow。
- `--classify` 改用已微調的 SST-2 情緒模型（舊版的分類頭是隨機初始化的）。
- `--train` 會保留大小寫與標點；Trainer 的中間產物改放 `outputs/`（舊版是 `./results`、`./logs`）。舊的 `saved_model/` 可以直接給 `--generate` 使用。

## 測試

測試不需要網路、麥克風、GPU 或 torch。在 repo 根目錄執行：

```bash
uv venv .venv                                   # 或 python -m venv .venv
uv pip install --python .venv -e ".[dev]"       # 或 .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                      # Windows：.venv\Scripts\python -m pytest
uvx ruff check .                                # 或 .venv/bin/ruff check .
python -m compileall -q src tests
```

測試的對象是安裝在 `.venv` 裡的套件（src layout 的慣例），所以要先安裝；用 `-e` 安裝後改程式碼不必重裝。
