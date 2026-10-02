"""Groq LLM 客戶端：整個程式共用一個 AsyncGroq 實例，失敗時回傳設定的備援訊息。

聊天用 stream_reply()（串流，邊產生邊交給切句與 TTS）；--generate 用 reply()（一次拿到整段）。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Optional

import httpx
from groq import AsyncGroq, GroqError

from vtuber.config import LLMConfig

logger = logging.getLogger(__name__)

# 回應結構不對時會出現的例外，例如被代理伺服器或 captive portal 回了 HTML、JSON 被截斷、choice 沒有 message
_MALFORMED_RESPONSE_ERRORS = (AttributeError, IndexError, KeyError, TypeError, ValueError)
# 串流途中的錯誤：SDK 只包裝建立請求時的錯誤，讀取串流時的網路錯誤會直接是 httpx 的例外
_STREAM_ERRORS = (GroqError, httpx.HTTPError, *_MALFORMED_RESPONSE_ERRORS)


class GroqChat:
    """把使用者的話送給 Groq，回傳要念出來的回覆文字（永遠回傳文字，不丟例外）。"""

    def __init__(self, config: LLMConfig, client: Optional[AsyncGroq] = None) -> None:
        self._config = config
        self._client = client if client is not None else _create_client(config)

    def build_request(self, prompt: str) -> dict[str, Any]:
        config = self._config
        request: dict[str, Any] = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            # Groq API 已把 max_tokens 標為 deprecated，改用 max_completion_tokens
            "max_completion_tokens": config.max_tokens,
        }
        # 兩個都是 None 時不送：非推理模型不支援這兩個參數
        if config.reasoning_effort is not None:
            request["reasoning_effort"] = config.reasoning_effort
        if config.include_reasoning is not None:
            # 推理內容不會念出來，就不必傳回來（gpt-oss 預設會放在 reasoning 欄位一起傳回）
            request["include_reasoning"] = config.include_reasoning
        return request

    async def reply(self, prompt: str) -> str:
        if self._client is None:
            return self._config.fallback_message
        try:
            completion = await self._client.chat.completions.create(**self.build_request(prompt))
            content, finish_reason = _extract_reply(completion)
        except GroqError as exc:
            logger.error("Groq API error: %s", exc)
            return self._config.fallback_message
        except _MALFORMED_RESPONSE_ERRORS as exc:
            logger.error(
                "Unexpected response from Groq (%s: %s); using the fallback message", type(exc).__name__, exc
            )
            return self._config.fallback_message
        if not content:
            # 推理模型若把 token 額度都用在推理上，content 會是空的
            logger.warning("Groq returned an empty reply (finish_reason=%s); using the fallback message", finish_reason)
            return self._config.fallback_message
        return content

    async def stream_reply(self, prompt: str) -> AsyncIterator[str]:
        """串流產生回覆，逐段輸出 delta.content（推理內容不輸出）；除了被取消之外不丟例外。

        - 第一段內容出來之前就失敗：輸出備援訊息
        - 中途失敗：記錄錯誤後停止，已經輸出的部分照常念完
        - 整段都沒有內容（例如 token 額度都用在推理上）：輸出備援訊息
        """
        if self._client is None:
            yield self._config.fallback_message
            return
        started = False  # 是否已輸出過非空白的內容
        finish_reason: Optional[str] = None
        try:
            stream = await self._client.chat.completions.create(**self.build_request(prompt), stream=True)
            async with stream:  # 提早結束（例如被取消）時也關閉 HTTP 回應
                async for chunk in stream:
                    content, reason = _extract_delta(chunk)
                    finish_reason = reason or finish_reason
                    if content:
                        started = started or not content.isspace()
                        yield content
        except _STREAM_ERRORS as exc:
            if started:
                logger.error(
                    "Groq stream broke off mid-reply (%s: %s); speaking the part received so far",
                    type(exc).__name__,
                    exc,
                )
                return
            logger.error("Groq request failed (%s: %s); using the fallback message", type(exc).__name__, exc)
            yield self._config.fallback_message
            return
        if not started:
            logger.warning("Groq returned an empty reply (finish_reason=%s); using the fallback message", finish_reason)
            yield self._config.fallback_message

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


def _extract_reply(completion: Any) -> tuple[str, Optional[str]]:
    """取出第一個 choice 的文字與 finish_reason；回應結構不對時會丟出 AttributeError、IndexError 或 TypeError。"""
    choice = completion.choices[0]
    content = choice.message.content
    if content is not None and not isinstance(content, str):
        raise TypeError(f"message.content is {type(content).__name__}, not str")
    return (content or "").strip(), choice.finish_reason


def _extract_delta(chunk: Any) -> tuple[str, Optional[str]]:
    """取出串流片段的 delta.content 與 finish_reason；推理內容（delta.reasoning）不取。

    沒有 choices 的片段（例如只帶用量統計的最後一段）回傳空字串。
    """
    if not chunk.choices:
        return "", None
    choice = chunk.choices[0]
    content = choice.delta.content
    if content is not None and not isinstance(content, str):
        raise TypeError(f"delta.content is {type(content).__name__}, not str")
    return content or "", choice.finish_reason


def _create_client(config: LLMConfig) -> Optional[AsyncGroq]:
    try:
        # 沒傳 api_key 時，SDK 會讀環境變數 GROQ_API_KEY；重試次數預設比 SDK 少，避免 429 時卡很久
        return AsyncGroq(timeout=config.timeout, max_retries=config.max_retries)
    except GroqError as exc:
        logger.error(
            "Cannot create the Groq client (%s); set GROQ_API_KEY. Replies will use the fallback message.", exc
        )
        return None
