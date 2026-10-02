# 語音素材

| 檔案 | 內容 | 原本在 repo 的位置 |
|---|---|---|
| `firefly/firefly.wav` | 流螢的英文配音，231.9 秒，44.1 kHz 立體聲。依檔案內嵌資訊，擷取自《崩壞：星穹鐵道》預告片〈Myriad Celestia Trailer — "Presently, Beneath a Shared Sky of Stars"〉（<https://www.youtube.com/watch?v=qsBBZRNoP8s>） | `aiVtuber/GPT-SoVITS/GPT_SoVITS/audio_files/firefly.wav` |
| `firefly/ref_firefly_01.wav` | TTS 參考音檔，從 `firefly.wav` 切出（見下節） | 新檔。取代原本的 `output/slicer_opt/firefly.wav_0001213120_0001383360.wav`，該檔從未放進 repo |
| `vocal.wav` | 日文台詞，242.2 秒，48 kHz 立體聲。依檔案內嵌資訊，來自《崩壊：スターレイル》PV「虚譚・浮世三千一刀繚断」（<https://www.youtube.com/watch?v=XbWbOIxbFMA>）。抽樣轉寫的內容是關於「出雲」和刀的旁白，**不是流螢**，原本的用途不明 | `aiVtuber/GPT-SoVITS/GPT_SoVITS/audio_files/vocal.wav` |

## 參考音檔 `ref_firefly_01.wav` 的切法

- 取 `firefly/firefly.wav` 的 **37.91–43.00 秒**（5.09 秒），把兩個聲道平均混成單聲道，存成 16-bit PCM、44.1 kHz。沒有重新取樣，也沒有調整音量。
- 起點和原作者使用的 `firefly.wav_0001213120_0001383360.wav` 相同。該檔名裡的數字是 32 kHz 的取樣點，換算成時間是 37.91–43.23 秒。
- 終點提前到 43.00 秒：43.08 秒之後已經是下一句（"Leaving the cockpit is strictly prohibited…"）的起音。
- 文字（prompt_text，語言 `en`）：`I understand. Article 4 of Glamoth military regulations.`
- 驗證：Google 語音辨識（en-US）的結果是 "I understand Article 4 of glamorous military regulations"。除了專有名詞 Glamoth（遊戲裡的星球名）被聽成 glamorous 之外，逐字相符。

## 版權

這些音檔擷取自遊戲《崩壞：星穹鐵道》的官方影片，版權屬於遊戲的版權方。請只用於個人研究與測試。公開直播、營利，或散布用它合成的語音之前，請先自行評估授權風險（包括聲優的權利）。
