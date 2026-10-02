"""情緒分類（--classify）：延遲載入已微調的二元情緒模型，回傳 positive / negative。"""

from __future__ import annotations

import logging
from typing import Any

from vtuber.device import select_device
from vtuber.errors import ConfigError

logger = logging.getLogger(__name__)

POSITIVE = "positive"
NEGATIVE = "negative"


def normalize_label(label: str) -> str:
    """把模型標籤（例如 SST-2 的 POSITIVE / NEGATIVE）轉成 positive / negative。

    未微調的分類頭只會有 LABEL_0 / LABEL_1 這類標籤，這裡會直接報錯，避免拿亂猜的結果去換表情。
    """
    key = label.strip().lower()
    if key.startswith("pos"):
        return POSITIVE
    if key.startswith("neg"):
        return NEGATIVE
    raise ConfigError(
        f"Sentiment model label {label!r} is not positive/negative; set sentiment.model to a fine-tuned "
        "binary sentiment model such as distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )


class SentimentClassifier:
    """第一次分類時才載入 tokenizer 與模型。

    classify() 會阻塞；CLI 在主執行緒直接呼叫，下載或載入模型時按 Ctrl+C 可以立刻中斷。
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self._model_name = model_name
        self._device_preference = device
        self._device = "cpu"
        self._tokenizer: Any = None
        self._model: Any = None

    def classify(self, text: str) -> str:
        import torch

        self._ensure_loaded()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True).to(self._device)
        with torch.inference_mode():
            logits = self._model(**inputs).logits
        label_id = int(logits.argmax(dim=-1)[0])
        return normalize_label(self._model.config.id2label[label_id])

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = select_device(self._device_preference)
        logger.info("Loading sentiment model %s on %s", self._model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._model = model.to(self._device).eval()
