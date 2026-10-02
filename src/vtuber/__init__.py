"""AI VTuber 核心套件。

語音辨識 → Groq LLM 串流回覆 → 按句以 GPT-SoVITS 合成、邊合成邊播放，
另含 GPT-2 微調／生成與情緒分類（驅動 VTube Studio 表情）。

執行方式：在專案根目錄 `pip install -e ".[chat]"` 安裝後執行 `aivtuber`（等同 `python -m vtuber`）。
"""
