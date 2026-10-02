from __future__ import annotations

import asyncio
import json
import logging
import types
from dataclasses import replace

import groq
import httpx
import pytest
from groq import AsyncGroq

from vtuber import llm as llm_module
from vtuber.config import LLMConfig
from vtuber.llm import GroqChat

FALLBACK = LLMConfig().fallback_message


def _completion(content: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-20b",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
    }


def _groq_client(handler) -> AsyncGroq:
    return AsyncGroq(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _reply(chat: GroqChat, *prompts: str) -> list[str]:
    async def run() -> list[str]:
        try:
            return [await chat.reply(prompt) for prompt in prompts]
        finally:
            await chat.aclose()

    return asyncio.run(run())


def test_reply_sends_configured_parameters_and_returns_content():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/v1/chat/completions"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_completion("  Hello! Nice to meet you.  "))

    replies = _reply(GroqChat(LLMConfig(), client=_groq_client(handler)), "Hi there")

    assert replies == ["Hello! Nice to meet you."]
    body = bodies[0]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["messages"] == [{"role": "user", "content": "Hi there"}]
    assert body["temperature"] == 0.5
    assert body["max_completion_tokens"] == 1024  # max_tokens 在 Groq API 已標為 deprecated
    assert "max_tokens" not in body
    assert body["reasoning_effort"] == "low"  # 預設 low：推理越少，第一句越快出來
    assert body["include_reasoning"] is False  # 推理內容不會念出來，不必傳回
    assert "stream" not in body  # reply() 一次拿整段（--generate 使用）


def test_reasoning_parameters_are_omitted_for_non_reasoning_models():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_completion("ok"))

    config = replace(LLMConfig(), model="llama-3.3-70b-versatile", reasoning_effort=None, include_reasoning=None)
    _reply(GroqChat(config, client=_groq_client(handler)), "Hi")

    assert "reasoning_effort" not in bodies[0] and "include_reasoning" not in bodies[0]  # 非推理模型會回 400


@pytest.mark.parametrize(
    ("effort", "include", "expected"),
    [
        ("none", None, {"reasoning_effort": "none"}),  # qwen3 系列：不推理
        ("default", False, {"reasoning_effort": "default", "include_reasoning": False}),
        (None, False, {"include_reasoning": False}),  # 推理強度用模型預設值，但不要傳回推理內容
        ("high", True, {"reasoning_effort": "high", "include_reasoning": True}),
    ],
)
def test_reasoning_parameters_are_set_independently(effort, include, expected):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_completion("ok"))

    config = replace(LLMConfig(), reasoning_effort=effort, include_reasoning=include)
    _reply(GroqChat(config, client=_groq_client(handler)), "Hi")

    sent = {key: bodies[0][key] for key in ("reasoning_effort", "include_reasoning") if key in bodies[0]}
    assert sent == expected


def test_reasoning_effort_is_sent_when_configured():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_completion("ok"))

    config = replace(LLMConfig(), reasoning_effort="low", model="openai/gpt-oss-120b", max_tokens=256)
    _reply(GroqChat(config, client=_groq_client(handler)), "Hi")

    assert bodies[0]["reasoning_effort"] == "low"
    assert bodies[0]["model"] == "openai/gpt-oss-120b"
    assert bodies[0]["max_completion_tokens"] == 256


def test_api_error_returns_configured_fallback_message(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        error = {"message": "The model `llama3-8b-8192` has been decommissioned", "code": "model_decommissioned"}
        return httpx.Response(400, json={"error": error})

    config = replace(LLMConfig(), fallback_message="Let me think about that later.")
    with caplog.at_level(logging.ERROR, logger="vtuber.llm"):
        replies = _reply(GroqChat(config, client=_groq_client(handler)), "Hi")

    assert replies == ["Let me think about that later."]
    assert "decommissioned" in caplog.text


def test_empty_content_returns_fallback_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("", finish_reason="length"))

    assert _reply(GroqChat(LLMConfig(), client=_groq_client(handler)), "Hi") == [FALLBACK]


def test_missing_api_key_falls_back_gracefully(monkeypatch, caplog):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with caplog.at_level(logging.ERROR, logger="vtuber.llm"):
        replies = _reply(GroqChat(LLMConfig()), "Hi", "Are you there?")

    assert replies == [FALLBACK, FALLBACK]
    assert "GROQ_API_KEY" in caplog.text


def test_one_client_is_created_and_reused_for_every_reply(monkeypatch):
    created = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("hi"))

    def fake_async_groq(**kwargs):
        created.append(kwargs)
        return _groq_client(handler)

    monkeypatch.setattr(llm_module, "AsyncGroq", fake_async_groq)

    replies = _reply(GroqChat(LLMConfig()), "one", "two", "three")

    assert replies == ["hi", "hi", "hi"]
    assert len(created) == 1
    assert created[0]["timeout"] == 30.0
    assert created[0]["max_retries"] == 1  # SDK 預設 2 次，遇到 429 會等很久


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>captive portal</html>", headers={"content-type": "text/html"}),
        httpx.Response(200, content=b'{"choices": [', headers={"content-type": "application/json"}),
        httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                  "choices": [{"index": 0, "finish_reason": "stop"}]}),
        httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": []}),
    ],
    ids=["html-page", "truncated-json", "choice-without-message", "no-choices"],
)
def test_malformed_responses_fall_back_instead_of_raising(response, caplog):
    with caplog.at_level(logging.ERROR, logger="vtuber.llm"):
        replies = _reply(GroqChat(LLMConfig(), client=_groq_client(lambda request: response)), "Hi")

    assert replies == [FALLBACK]
    assert "Unexpected response from Groq" in caplog.text


def test_reasoning_field_is_ignored_and_only_the_answer_is_spoken():
    message = {"role": "assistant", "content": "Hello!", "reasoning": "The user greets; answer briefly."}

    def handler(request: httpx.Request) -> httpx.Response:
        body = _completion("unused")
        body["choices"][0]["message"] = message
        return httpx.Response(200, json=body)

    assert _reply(GroqChat(LLMConfig(), client=_groq_client(handler)), "Hi") == ["Hello!"]


# ---- 串流（stream_reply） ----


def _chunk(content=None, reasoning=None, finish_reason=None, choices=True):
    """仿 groq 的 ChatCompletionChunk：只放 GroqChat 會用到的欄位。"""
    if not choices:
        return types.SimpleNamespace(choices=[])
    delta = types.SimpleNamespace(content=content, reasoning=reasoning)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta, finish_reason=finish_reason)])


class FakeStream:
    """假的 AsyncStream：依序吐出片段；項目是例外時就在那個位置丟出。"""

    def __init__(self, items) -> None:
        self._items = list(items)
        self.closed = False

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *exc_info) -> None:
        self.closed = True

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeGroqClient:
    """chat.completions.create() 回傳假的串流，或在建立請求時就丟出例外。"""

    def __init__(self, stream_or_error) -> None:
        self.result = stream_or_error
        self.requests: list[dict] = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def close(self) -> None:
        pass


def _stream(chat: GroqChat, prompt: str = "Hi") -> list[str]:
    async def run() -> list[str]:
        return [part async for part in chat.stream_reply(prompt)]

    return asyncio.run(run())


REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def test_stream_yields_only_content_and_skips_reasoning():
    stream = FakeStream(
        [
            _chunk(reasoning="The user greets me."),
            _chunk(content="Hello"),
            _chunk(reasoning=" Keep it short."),
            _chunk(content="! Nice"),
            _chunk(content=None),
            _chunk(content=" to meet you.", finish_reason="stop"),
            _chunk(choices=False),  # 最後只帶用量統計、沒有 choices 的片段
        ]
    )
    client = FakeGroqClient(stream)

    assert _stream(GroqChat(LLMConfig(), client=client)) == ["Hello", "! Nice", " to meet you."]
    assert client.requests[0]["stream"] is True
    assert client.requests[0]["include_reasoning"] is False
    assert stream.closed


@pytest.mark.parametrize(
    "error",
    [
        groq.APIConnectionError(request=REQUEST),
        groq.APIError("model_decommissioned", request=REQUEST, body=None),
    ],
    ids=["connection", "api-error"],
)
def test_error_before_the_request_starts_streaming_falls_back(error, caplog):
    with caplog.at_level(logging.ERROR, logger="vtuber.llm"):
        parts = _stream(GroqChat(LLMConfig(), client=FakeGroqClient(error)))

    assert parts == [FALLBACK]
    assert "using the fallback message" in caplog.text


def test_error_before_the_first_content_falls_back():
    stream = FakeStream([_chunk(reasoning="thinking"), _chunk(content="  "), httpx.ReadError("reset", request=REQUEST)])

    assert _stream(GroqChat(LLMConfig(), client=FakeGroqClient(stream))) == ["  ", FALLBACK]
    assert stream.closed


@pytest.mark.parametrize(
    "error",
    [httpx.ReadError("connection reset", request=REQUEST), groq.APIError("stream error", request=REQUEST, body=None)],
    ids=["network", "api-error-event"],
)
def test_error_mid_stream_keeps_what_was_said_and_stops(error, caplog):
    stream = FakeStream([_chunk(content="First sentence."), _chunk(content=" Second"), error, _chunk(content="never")])

    with caplog.at_level(logging.ERROR, logger="vtuber.llm"):
        parts = _stream(GroqChat(LLMConfig(), client=FakeGroqClient(stream)))

    assert parts == ["First sentence.", " Second"]  # 不念備援訊息，已經產生的部分照常念完
    assert "broke off mid-reply" in caplog.text
    assert stream.closed


def test_malformed_chunk_mid_stream_is_treated_like_an_error():
    stream = FakeStream([_chunk(content="Hello."), types.SimpleNamespace(), _chunk(content="never")])  # 沒有 choices

    assert _stream(GroqChat(LLMConfig(), client=FakeGroqClient(stream))) == ["Hello."]


@pytest.mark.parametrize(
    "items",
    [[], [_chunk(content=""), _chunk(content=None, finish_reason="length")], [_chunk(content=" \n ")]],
    ids=["nothing", "reasoning-used-all-tokens", "whitespace-only"],
)
def test_empty_stream_falls_back(items, caplog):
    with caplog.at_level(logging.WARNING, logger="vtuber.llm"):
        parts = _stream(GroqChat(LLMConfig(), client=FakeGroqClient(FakeStream(items))))

    assert "".join(parts).strip() == FALLBACK
    assert "empty reply" in caplog.text


def test_stream_without_api_key_falls_back(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert _stream(GroqChat(LLMConfig())) == [FALLBACK]


def test_closing_the_stream_early_closes_the_http_response():
    stream = FakeStream([_chunk(content="One."), _chunk(content=" Two.")])

    async def run() -> None:
        parts = GroqChat(LLMConfig(), client=FakeGroqClient(stream)).stream_reply("Hi")
        assert await parts.__anext__() == "One."
        await parts.aclose()  # 例如這一輪被取消

    asyncio.run(run())

    assert stream.closed


def _sse(*events: dict) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _sse_chunk(delta: dict, finish_reason=None) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "openai/gpt-oss-20b",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def test_stream_through_the_real_sdk():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        body = _sse(
            _sse_chunk({"role": "assistant", "content": ""}),
            _sse_chunk({"reasoning": "Greet back."}),
            _sse_chunk({"content": "Hi! "}),
            _sse_chunk({"content": "How are you?"}, finish_reason="stop"),
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async def run() -> list[str]:
        chat = GroqChat(LLMConfig(), client=_groq_client(handler))
        try:
            return [part async for part in chat.stream_reply("Hello")]
        finally:
            await chat.aclose()

    assert asyncio.run(run()) == ["Hi! ", "How are you?"]
    assert bodies[0]["stream"] is True
    assert (bodies[0]["reasoning_effort"], bodies[0]["include_reasoning"]) == ("low", False)


def test_http_error_before_streaming_through_the_real_sdk_falls_back():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Rate limit reached", "type": "tokens"}})

    async def run() -> list[str]:
        chat = GroqChat(LLMConfig(), client=_groq_client(handler))
        try:
            return [part async for part in chat.stream_reply("Hello")]
        finally:
            await chat.aclose()

    assert asyncio.run(run()) == [FALLBACK]
