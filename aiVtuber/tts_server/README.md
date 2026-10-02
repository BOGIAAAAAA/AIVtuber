# TTS server（GPT-SoVITS api_v2）

aiVtuber 的語音由另一個程序合成：[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的 `api_v2.py`，主程式用 HTTP 呼叫它（預設 `http://127.0.0.1:9880`）。

- GPT-SoVITS 不放在這個 repo 裡。setup 腳本會把上游 clone 到 `aiVtuber/GPT-SoVITS`（已加入 .gitignore），並鎖定在 commit [`48b1a01`](https://github.com/RVC-Boss/GPT-SoVITS/commit/48b1a0169a28582a8984402f82cf438d3bfa6aca)（2026-08-18）。
- 預設使用 v2ProPlus 底模做零樣本合成：不需要訓練，音色來自每次請求帶的參考音檔。

| 檔案 | 用途 |
|---|---|
| `setup_gpt_sovits.sh`／`.ps1` | 下載上游原始碼與模型，並執行上游的安裝腳本（macOS・Linux／Windows conda） |
| `start_tts_server.sh`／`.bat`／`.ps1` | 啟動 server（`.bat` 可以直接點兩下，實際工作由 `.ps1` 完成） |
| `tts_infer.example.yaml` | server 設定範本。第一次啟動時會複製成 `tts_infer.yaml`，之後改那份 |

## 1. 安裝（三選一）

### A. Windows 整合包（最簡單）

1. 從 Hugging Face 的 [`lj1995/GPT-SoVITS-windows-package`](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/tree/main) 下載 `GPT-SoVITS-v2pro-20250604.7z`（約 8.2 GB）。**RTX 50 系列**請改下載 `GPT-SoVITS-v2pro-20250604-nvidia50.7z`。
2. 用 7-Zip 解壓到**不含空白和中文**的路徑，例如 `D:\GPT-SoVITS-v2pro-20250604`。整合包已內含 Python 環境和所有模型，不需要執行 setup 腳本。
3. 告訴啟動腳本整合包的位置（擇一）：
   - 設定環境變數（只需要做一次；設定後開啟的新視窗才會生效）：`setx GPT_SOVITS_DIR D:\GPT-SoVITS-v2pro-20250604`
   - 或每次啟動時加上參數：`start_tts_server.bat -GsvDir D:\GPT-SoVITS-v2pro-20250604`

> 整合包裡的程式是 2025-06-04 的版本，比鎖定的 48b1a01 舊，但 api_v2 的用法相同。
> 沒有 NVIDIA 顯示卡時，`tts_infer.yaml` 一定要改成 `device: cpu`、`is_half: false`，因為舊版不會自動關閉半精度。

### B. Windows（conda）

需要 [Git](https://git-scm.com/download/win)、[Miniforge](https://github.com/conda-forge/miniforge)（或 Miniconda），以及 **PowerShell 7**（`winget install Microsoft.PowerShell`）。上游的安裝腳本在內建的 Windows PowerShell 5.1 下可能中途停止。

在 repo 根目錄開啟 Miniforge Prompt，執行：

```powershell
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits
pwsh -ExecutionPolicy Bypass -File aiVtuber\tts_server\setup_gpt_sovits.ps1 -Device CU128
```

`-Device` 的選法：NVIDIA 顯示卡用 `CU128`（顯示卡驅動較舊時改用 `CU126`），沒有 NVIDIA 顯示卡用 `CPU`。

### C. macOS／Linux（conda）

需要 git、curl 和 [Miniforge](https://github.com/conda-forge/miniforge)。macOS 另外需要 Xcode Command Line Tools，沒有的話上游安裝腳本會跳出安裝視窗。

在 repo 根目錄執行：

```bash
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits
bash aiVtuber/tts_server/setup_gpt_sovits.sh --device CU128   # Linux + NVIDIA（也可以用 CU126、ROCM、CPU）
bash aiVtuber/tts_server/setup_gpt_sovits.sh --device MPS     # macOS（會安裝 CPU 版 PyTorch，推論只用 CPU）
```

### setup 腳本做的事

1. 把上游 clone 到 `aiVtuber/GPT-SoVITS`（可以用環境變數 `GPT_SOVITS_DIR` 改位置），並切換到 48b1a01。
2. 只下載 v2ProPlus 推論需要的模型（約 1.3 GB，見第 2 節）。這樣上游安裝腳本會跳過 4.6 GB 的完整模型包。
3. 執行上游的 `install.sh`／`install.ps1`：安裝 FFmpeg、PyTorch、Python 套件，並下載 G2PW 中文模型（約 590 MB）、NLTK 資料和日文辭典。

常用選項（括號內是 PowerShell 版的寫法）：

| 選項 | 作用 |
|---|---|
| `--source HF-Mirror`、`--source ModelScope`（`-Source`） | 中國大陸網路用；選 ModelScope 時會改成下載完整模型包 |
| `--full-models`（`-FullModels`） | 下載全部底模（約 4.6 GB） |
| `--skip-install`（`-SkipInstall`） | 只更新原始碼和模型，不執行上游安裝腳本 |

要改用其他 commit，可以設定環境變數 `GPT_SOVITS_COMMIT`。例如環境需要 transformers 4.50 以下時，可以改用 `d523079fc05d9a8028d6085bffe4a2757c32abb6`：程式碼和 48b1a01 相同，只差 transformers 的版本範圍。

## 2. 需要的模型（約 1.3 GB）

setup 腳本會從 Hugging Face 的 [`lj1995/GPT-SoVITS`](https://huggingface.co/lj1995/GPT-SoVITS)（固定在 2025-06-04 的版本 `336b2ec`）下載以下模型，放到 GPT-SoVITS 的 `GPT_SoVITS/pretrained_models/`。Windows 整合包已經內含這些模型。

| 檔案 | 大小 | 用途 |
|---|---|---|
| `s1v3.ckpt` | 155 MB | GPT 模型（把文字轉成語意 token） |
| `v2Pro/s2Gv2ProPlus.pth` | 200 MB | SoVITS v2ProPlus 底模 |
| `sv/pretrained_eres2netv2w24s4ep4.ckpt` | 108 MB | 說話人特徵模型（v2Pro 系列需要） |
| `chinese-hubert-base/` | 189 MB | 從參考音檔抽取特徵 |
| `chinese-roberta-wwm-ext-large/` | 651 MB | 中文 BERT（只合成英文也需要，啟動時就會載入） |

## 3. 啟動

- **macOS／Linux**：
  ```bash
  conda activate GPTSoVits
  bash aiVtuber/tts_server/start_tts_server.sh
  ```
- **Windows**：點兩下 `aiVtuber\tts_server\start_tts_server.bat`。
  - 整合包：照第 1 節設定好 `GPT_SOVITS_DIR` 就可以直接用。
  - conda：要在已經執行過 `conda activate GPTSoVits` 的視窗裡執行它。若想直接點兩下，可以用 `GPT_SOVITS_PYTHON` 指向該環境的 `python.exe`（例如 `%USERPROFILE%\miniforge3\envs\GPTSoVits\python.exe`）。

第一次啟動時，`tts_infer.example.yaml` 會被複製成 `tts_infer.yaml`。要換模型或改 `device`，都改這份（說明寫在檔案裡），改完後重啟 server。每次啟動時，腳本會把它再複製一份到 `runtime/tts_infer.yaml` 交給 server，因為 api_v2 會改寫它拿到的設定檔。

| 環境變數 | `.bat`／`.ps1` 參數 | 預設值 | 說明 |
|---|---|---|---|
| `GPT_SOVITS_DIR` | `-GsvDir` | `aiVtuber/GPT-SoVITS` | GPT-SoVITS 所在的目錄 |
| `GPT_SOVITS_PYTHON` | `-Python` | 自動選擇 | 要使用的 Python |
| `TTS_HOST` | `-BindHost` | `127.0.0.1` | 監聽位址；設成 `0.0.0.0` 就能從區網連線 |
| `TTS_PORT` | `-Port` | `9880` | 連接埠 |
| `GPT_SOVITS_CONDA_ENV` | （只有 `.sh` 有） | `GPTSoVits` | conda 環境名稱 |

沒有指定 Python 時的選擇順序：
- `.sh`：已啟用的 GPTSoVits 環境 → `conda run -n GPTSoVits` → `python`
- `.ps1`：整合包的 `runtime\python.exe` → `python`

**確認已就緒**：載入模型需要幾十秒，載入完成後 server 才開始接受連線。`GET /docs` 回應 200 就代表可以使用：

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:9880/docs   # 印出 200 就是已就緒
```

也可以用瀏覽器開啟 <http://127.0.0.1:9880/docs>，在 `POST /tts` 按 **Try it out** 試著合成（Windows 建議用這個方法）。在 macOS／Linux 也可以用指令測試（GPT-SoVITS 放在預設位置時）：

```bash
curl -X POST http://127.0.0.1:9880/tts -H "Content-Type: application/json" -o test.wav -d '{
  "text": "Hello! Nice to meet you.", "text_lang": "en",
  "ref_audio_path": "../voices/firefly/ref_firefly_01.wav",
  "prompt_text": "I understand. Article 4 of Glamoth military regulations.", "prompt_lang": "en"}'
```

成功時回傳 WAV（HTTP 200）；參數錯誤時回傳 HTTP 400 和 JSON 錯誤訊息。主程式的連線設定請見 `aiVtuber/config.example.yaml` 的 `tts` 區段。

## 4. 參考音檔

預設使用 [`aiVtuber/voices/firefly/ref_firefly_01.wav`](../voices/firefly/ref_firefly_01.wav)（5.09 秒，切法見 [`voices/README.md`](../voices/README.md)），搭配：

- `prompt_text`：`I understand. Article 4 of Glamoth military regulations.`
- `prompt_lang`：`en`

注意事項：
- 長度必須在 **3–10 秒**之間，否則 api_v2 會回傳 400。
- `prompt_text` 必須和音檔內容逐字相符，頭尾也不能多出別的字，否則合成結果的開頭容易出現多餘的聲音。
- 參考音檔是由 server 讀取的。**直接呼叫 API**（例如上面的 curl）時，相對路徑以 GPT-SoVITS 目錄（server 的工作目錄）為基準：上例的 `../voices/...` 只有 GPT-SoVITS 放在預設位置時才正確，其他情況請寫絕對路徑。server 在另一台電腦上時，要寫那台電腦上的路徑。
- **主程式**的 `tts.ref_audio_path` 規則不同：相對路徑以 `aiVtuber/` 為基準，由主程式轉成絕對路徑後才送給 server（預設值 `voices/firefly/ref_firefly_01.wav` 就是這樣）；絕對路徑則原樣送出。server 在另一台電腦上時，同樣要寫那台電腦上的絕對路徑。詳見 `aiVtuber/config.example.yaml` 的 `tts` 區段。
- 自己準備參考音檔時，請選只有一個人說話、背景音樂和雜音越少越好的 3–10 秒片段，存成 WAV。

## 5. 疑難排解

| 狀況 | 原因與處理 |
|---|---|
| log 出現 `fall back to default ...` | `tts_infer.yaml` 的權重路徑寫錯（相對路徑以 GPT-SoVITS 目錄為基準），server 改用了底模。修正路徑後重啟。 |
| 回應是 HTTP 200，內容卻只有 1 秒的靜音（16 kHz） | 推論過程出錯了。api_v2 不會回傳錯誤碼，而是回傳一段靜音，請看 server 視窗裡的錯誤訊息。主程式會把這種回應當成失敗。 |
| 回應 HTTP 400 | 看 JSON 裡的訊息，常見原因：參考音檔不在 3–10 秒之間、路徑不存在、不支援的語言代碼。 |
| 另一台電腦連不到 server | server 預設只監聽 `127.0.0.1`。請用 `TTS_HOST=0.0.0.0`（或 `-BindHost 0.0.0.0`）啟動，並在防火牆開放 9880 埠。api_v2 沒有任何驗證機制，只能在信任的區網開放。 |
| 使用 GTX 10／16 系列顯示卡 | `tts_infer.yaml` 要設 `is_half: false`，這些卡用半精度會很慢或輸出異常。顯存小於 4 GB 時改用 `device: cpu`。 |
| 使用 Mac | 只能用 CPU（上游已經停用 MPS）：請設 `device: cpu`、`is_half: false`。合成速度會比 GPU 慢很多。 |
| 換了權重（改設定檔，或呼叫 `/set_gpt_weights`、`/set_sovits_weights`）後聲音不對 | api_v2 不會清除參考音檔的 prompt 快取，會繼續沿用舊模型算出的特徵。換權重後請重啟 server。 |
| `api_v2.py not found` | `GPT_SOVITS_DIR`／`-GsvDir` 沒有指到 GPT-SoVITS 目錄。使用 Windows 整合包時，要指到解壓後含有 `api_v2.py` 的那一層。 |
| `cannot import torch/fastapi` | 用到的 Python 不是 GPT-SoVITS 的環境。請先執行 `conda activate GPTSoVits`，或用 `GPT_SOVITS_PYTHON`／`-Python` 指定。 |
| 啟動時出現 `[WARN] Missing ...` | 預設設定需要的模型不存在，請重新執行 setup 腳本。如果改用自己的權重，可以忽略對應的警告。 |

## 6. 版權與授權

- **GPT-SoVITS**（48b1a01）採用 MIT 授權，Copyright (c) 2024 RVC-Boss，全文見 [`LICENSE-GPT-SoVITS`](LICENSE-GPT-SoVITS)（就是舊版 repo 內附 GPT-SoVITS 副本裡的那份 LICENSE）。它的原始碼不在這個 repo 裡，由 setup 腳本另外下載。Hugging Face 上的模型庫 `lj1995/GPT-SoVITS` 和 Windows 整合包 `lj1995/GPT-SoVITS-windows-package` 也都標示為 MIT。
- **流螢語音素材**（`aiVtuber/voices/`）擷取自遊戲《崩壞：星穹鐵道》的官方影片，版權屬於遊戲的版權方。公開直播、營利或散布用它合成的語音之前，請先自行評估授權風險。詳見 [`voices/README.md`](../voices/README.md)。
