"""微調後 GPT-2 的文字生成（--generate）：第一次生成時才載入模型。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vtuber.config import GenerationConfig
from vtuber.device import select_device
from vtuber.errors import ModelNotFoundError

logger = logging.getLogger(__name__)


class TextGenerator:
    """第一次生成時才載入 model_path 底下的模型與 tokenizer。

    generate() 會阻塞；CLI 在主執行緒直接呼叫，下載或載入模型時按 Ctrl+C 可以立刻中斷。
    """

    def __init__(self, model_path: Path, config: GenerationConfig, device: str = "auto") -> None:
        self._model_path = model_path
        self._config = config
        self._device_preference = device
        self._device = "cpu"
        self._tokenizer: Any = None
        self._model: Any = None

    def generate(self, prompt: str) -> str:
        """回傳 prompt 加上模型續寫的完整文字（與舊版相同）。"""
        self._ensure_loaded()
        import torch

        config = self._config
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        # GPT-2 沒有 pad token；明確指定可避免 generate() 每次都發出警告
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=pad_token_id,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not (self._model_path / "config.json").is_file():
            raise ModelNotFoundError(
                f"No fine-tuned model found at {self._model_path}; run --train first or pass --model_save_path"
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = select_device(self._device_preference)
        logger.info("Loading fine-tuned model from %s on %s", self._model_path, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
        model = AutoModelForCausalLM.from_pretrained(str(self._model_path))
        self._model = model.to(self._device).eval()
