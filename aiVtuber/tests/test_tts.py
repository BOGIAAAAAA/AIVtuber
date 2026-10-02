from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from vtuber.config import BASE_DIR, TTSConfig
from vtuber.errors import ConfigError, TTSError
from vtuber.tts import GPTSoVITSV1Client, GPTSoVITSV2Client, create_tts, resolve_ref_audio_path

V1 = replace(TTSConfig(), engine="gpt-sovits-v1")


def _client(config: TTSConfig, handler, **kwargs):
    cls = GPTSoVITSV2Client if config.engine == "gpt-sovits-v2" else GPTSoVITSV1Client
    kwargs.setdefault("poll_interval", 0.01)
    return cls(config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs)


def _run(client, method: str, *args):
    async def run():
        try:
            return await getattr(client, method)(*args)
        finally:
            await client.aclose()

    return asyncio.run(run())


def _recording_handler(response_for):
    """回傳 (handler, 收到的請求清單)；response_for(request) 決定回應。"""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response_for(request)

    return handler, requests


# ---- api_v2：payload ----


def test_v2_posts_every_field_to_tts_and_returns_the_wav(speech_wav):
    handler, requests = _recording_handler(lambda request: httpx.Response(200, content=speech_wav))

    audio = _run(_client(TTSConfig(), handler), "synthesize", "Hello there!")

    assert audio == speech_wav
    (request,) = requests
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:9880/tts"
    assert json.loads(request.content) == {
        "text": "Hello there!",
        "text_lang": "en",
        # 相對路徑以 aiVtuber/ 為基準轉成絕對路徑（server 在同一台電腦）
        "ref_audio_path": str(BASE_DIR / "voices" / "firefly" / "ref_firefly_01.wav"),
        "prompt_text": "I understand. Article 4 of Glamoth military regulations.",
        "prompt_lang": "en",
        "text_split_method": "cut0",
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "media_type": "wav",
        "streaming_mode": False,
    }


def test_v2_sends_configured_values(speech_wav, tmp_path):
    handler, requests = _recording_handler(lambda request: httpx.Response(200, content=speech_wav))
    config = replace(
        TTSConfig(),
        url="http://192.168.1.106:9880/",
        ref_audio_path="./my_voices/../my_voices/zora.wav",
        prompt_text="Hi.",
        prompt_lang="ja",
        text_lang="all_ko",
        text_split_method="cut5",
        speed_factor=1.2,
        fragment_interval=0.0,
        top_k=5,
        top_p=0.8,
        temperature=0.6,
    )

    _run(_client(config, handler, base_dir=tmp_path), "synthesize", "안녕")

    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "http://192.168.1.106:9880/tts"
    assert body["ref_audio_path"] == str(tmp_path / "my_voices" / "zora.wav")
    assert (body["text_lang"], body["prompt_lang"], body["prompt_text"]) == ("all_ko", "ja", "Hi.")
    assert (body["text_split_method"], body["speed_factor"], body["fragment_interval"]) == ("cut5", 1.2, 0.0)
    assert (body["top_k"], body["top_p"], body["temperature"]) == (5, 0.8, 0.6)


@pytest.mark.parametrize(
    "path",
    [
        "/srv/gpt-sovits/voices/ref.wav",
        "D:\\GPT-SoVITS\\voices\\ref.wav",
        "C:/Users/zora/voices/ref.wav",
        "\\\\nas\\voices\\ref.wav",
    ],
)
def test_absolute_reference_paths_are_sent_as_is(speech_wav, tmp_path, path):
    # server 在別台電腦（可能是另一種作業系統）時，填的是那台電腦上的絕對路徑
    handler, requests = _recording_handler(lambda request: httpx.Response(200, content=speech_wav))

    _run(_client(replace(TTSConfig(), ref_audio_path=path), handler, base_dir=tmp_path), "synthesize", "Hi")

    assert json.loads(requests[0].content)["ref_audio_path"] == path


def test_relative_reference_paths_resolve_against_base_dir_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert resolve_ref_audio_path("voices/a.wav") == str(BASE_DIR / "voices" / "a.wav")
    assert resolve_ref_audio_path("voices/a.wav", tmp_path / "base") == str(tmp_path / "base" / "voices" / "a.wav")
    assert resolve_ref_audio_path("~/a.wav") == os.path.expanduser("~/a.wav")


# ---- api_v2：錯誤回應 ----


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (400, {"message": "ref_audio_path is required"}, "HTTP 400: ref_audio_path is required"),
        (
            400,
            {"message": "tts failed", "Exception": "/x/ref.wav not exists"},
            "HTTP 400: tts failed: /x/ref.wav not exists",
        ),
        (400, {"message": "tts failed", "Exception": ""}, "HTTP 400: tts failed: no details; see the TTS server log"),
        (
            422,
            {"detail": [{"type": "int_parsing", "loc": ["body", "top_k"], "msg": "Input should be a valid integer"}]},
            "HTTP 422: body.top_k: Input should be a valid integer",
        ),
    ],
    ids=["missing-parameter", "failed-before-inference", "assertion-without-details", "validation-error"],
)
def test_error_responses_are_turned_into_clear_messages(status, body, expected):
    handler, _ = _recording_handler(lambda request: httpx.Response(status, json=body))

    with pytest.raises(TTSError) as excinfo:
        _run(_client(TTSConfig(), handler), "synthesize", "Hi")

    assert expected in str(excinfo.value)


def test_non_json_error_body_is_shown():
    handler, _ = _recording_handler(lambda request: httpx.Response(500, text="Internal Server Error"))

    with pytest.raises(TTSError, match="HTTP 500: Internal Server Error"):
        _run(_client(TTSConfig(), handler), "synthesize", "Hi")


def test_not_found_hints_at_an_engine_mismatch():
    handler, _ = _recording_handler(lambda request: httpx.Response(404, json={"detail": "Not Found"}))

    with pytest.raises(TTSError, match=r"Not Found.*tts.engine"):
        _run(_client(TTSConfig(), handler), "synthesize", "Hi")


def test_non_wav_response_raises_clear_error():
    handler, _ = _recording_handler(
        lambda request: httpx.Response(200, content=b"OggS\x00\x02fake-ogg-data", headers={"content-type": "audio/ogg"})
    )

    with pytest.raises(TTSError, match=r"not WAV.*api_v2"):
        _run(_client(TTSConfig(), handler), "synthesize", "Hi")


def test_silent_wav_from_a_failed_inference_is_an_error(make_wav):
    # api_v2 推論中途出錯時回 HTTP 200 加上 1 秒、16 kHz 的全靜音 WAV
    silent = make_wav(frames=16000, rate=16000)
    handler, _ = _recording_handler(lambda request: httpx.Response(200, content=silent))

    with pytest.raises(TTSError, match=r"1\.00s of silence \(16000 Hz\).*server log"):
        _run(_client(TTSConfig(), handler), "synthesize", "Hi")


def test_wav_without_any_frames_is_an_error(make_wav):
    handler, _ = _recording_handler(lambda request: httpx.Response(200, content=make_wav(frames=0)))

    with pytest.raises(TTSError, match="silence"):
        _run(_client(TTSConfig(), handler), "synthesize", "Hi")


def test_quiet_but_not_silent_audio_is_accepted(make_wav):
    quiet = make_wav(frames=16000, rate=16000, amplitude=1)
    handler, _ = _recording_handler(lambda request: httpx.Response(200, content=quiet))

    assert _run(_client(TTSConfig(), handler), "synthesize", "Hi") == quiet


def test_truncated_wav_is_an_error(speech_wav):
    handler, _ = _recording_handler(lambda request: httpx.Response(200, content=speech_wav[:30]))

    with pytest.raises(TTSError, match="not a playable WAV"):
        _run(_client(TTSConfig(), handler), "synthesize", "Hi")


def test_connection_failure_raises_tts_error_with_url():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    with pytest.raises(TTSError, match=r"127\.0\.0\.1:9880/tts.*ConnectError"):
        _run(_client(TTSConfig(), handler), "synthesize", "Hi")


def test_empty_text_is_rejected_without_calling_server():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - 不應被呼叫
        raise AssertionError("server must not be called")

    with pytest.raises(TTSError, match="empty"):
        _run(_client(TTSConfig(), handler), "synthesize", "   ")


# ---- 就緒檢查與暖機 ----


def _server(speech_wav, *, refuse_first: int = 0, docs_status: int = 200, tts_response=None):
    """假的 api_v2：前 refuse_first 次連線被拒（模型還在載入），之後 /docs 回 docs_status。"""
    state = {"connections": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["connections"] += 1
        if state["connections"] <= refuse_first:
            raise httpx.ConnectError("Connection refused", request=request)
        if request.url.path == "/docs":
            return httpx.Response(docs_status, text="<html>Swagger UI</html>")
        return tts_response or httpx.Response(200, content=speech_wav)

    return _recording_handler(respond)


def test_prepare_waits_for_the_server_then_warms_up(speech_wav, caplog):
    handler, requests = _server(speech_wav, refuse_first=3)

    with caplog.at_level(logging.INFO, logger="vtuber.tts"):
        _run(_client(TTSConfig(), handler), "prepare")

    # 前 3 次 GET /docs 連不上，第 4 次就緒，接著合成一句暖機
    assert [(request.method, request.url.path) for request in requests] == [("GET", "/docs")] * 4 + [("POST", "/tts")]
    assert json.loads(requests[-1].content)["text"] == TTSConfig().warmup_text
    assert "Waiting up to 60s" in caplog.text
    assert "is ready" in caplog.text and "warm-up done" in caplog.text


def test_server_that_never_becomes_ready_fails_with_instructions(speech_wav):
    handler, requests = _server(speech_wav, refuse_first=10**6)
    config = replace(TTSConfig(), ready_timeout=0.05)

    with pytest.raises(TTSError) as excinfo:
        _run(_client(config, handler), "prepare")

    message = str(excinfo.value)
    assert "not ready after 0.05s" in message and "ConnectError" in message
    assert "start_tts_server" in message and "tts.wait_for_server: false" in message
    assert all(request.url.path == "/docs" for request in requests)  # 沒就緒就不暖機
    assert len(requests) >= 2  # 有重試


def test_something_else_answering_on_the_port_fails_fast(speech_wav):
    handler, requests = _server(speech_wav, docs_status=404)

    with pytest.raises(TTSError, match=r"HTTP 404.*GPT-SoVITS"):
        _run(_client(TTSConfig(), handler), "prepare")

    assert len(requests) == 1


def test_warm_up_failure_is_reported_with_how_to_skip_it(speech_wav, make_wav):
    silent = httpx.Response(200, content=make_wav(frames=16000, rate=16000))
    handler, _ = _server(speech_wav, tts_response=silent)

    with pytest.raises(TTSError, match=r"warm-up failed.*silence.*tts.warmup_text"):
        _run(_client(TTSConfig(), handler), "prepare")


def test_startup_check_and_warm_up_can_be_turned_off(speech_wav):
    handler, requests = _server(speech_wav, refuse_first=10**6)
    config = replace(TTSConfig(), wait_for_server=False, warmup_text="")

    _run(_client(config, handler), "prepare")

    assert requests == []


def test_warm_up_runs_even_without_the_readiness_check(speech_wav):
    handler, requests = _server(speech_wav)

    _run(_client(replace(TTSConfig(), wait_for_server=False), handler), "prepare")

    assert [request.url.path for request in requests] == ["/tts"]


@pytest.mark.parametrize(
    ("url", "path", "warning"),
    [
        ("http://127.0.0.1:9880", "voices/missing.wav", "Reference audio not found"),
        ("http://localhost:9880", "voices/missing.wav", "Reference audio not found"),
        ("http://127.0.0.1:9880", "voices/ref.wav", None),
        ("http://192.168.1.106:9880", "voices/ref.wav", "is relative"),
        ("http://192.168.1.106:9880", "D:/voices/ref.wav", None),
    ],
)
def test_reference_audio_warnings(tmp_path, caplog, speech_wav, url, path, warning):
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "ref.wav").write_bytes(speech_wav)
    handler, _ = _server(speech_wav)
    config = replace(TTSConfig(), url=url, ref_audio_path=path, wait_for_server=False, warmup_text="")

    with caplog.at_level(logging.WARNING, logger="vtuber.tts"):
        _run(_client(config, handler, base_dir=tmp_path), "prepare")

    if warning is None:
        assert caplog.records == []
    else:
        assert warning in caplog.text


def test_default_reference_audio_exists_in_the_repo():
    assert Path(resolve_ref_audio_path(TTSConfig().ref_audio_path)).is_file()


# ---- 舊版 api.py（gpt-sovits-v1） ----


def test_v1_maps_settings_to_the_old_field_names(speech_wav):
    handler, requests = _recording_handler(lambda request: httpx.Response(200, content=speech_wav))

    audio = _run(_client(V1, handler), "synthesize", "Hello there!")

    assert audio == speech_wav
    request = requests[0]
    assert request.url.raw_path == b"/"  # api.py 的推理端點是根路徑
    assert json.loads(request.content) == {
        "refer_wav_path": str(BASE_DIR / "voices" / "firefly" / "ref_firefly_01.wav"),
        "prompt_text": "I understand. Article 4 of Glamoth military regulations.",
        "prompt_language": "en",
        "text": "Hello there!",
        "text_language": "en",
    }  # 預設不送 cut_punc，維持舊版行為；v2 專用欄位也不送


def test_v1_cut_punc_is_sent_only_when_configured(speech_wav):
    handler, requests = _recording_handler(lambda request: httpx.Response(200, content=speech_wav))

    _run(_client(replace(V1, cut_punc=".?!"), handler), "synthesize", "One. Two?")

    assert json.loads(requests[0].content)["cut_punc"] == ".?!"


def test_v1_uses_the_server_default_reference_when_the_path_is_empty(speech_wav):
    handler, requests = _recording_handler(lambda request: httpx.Response(200, content=speech_wav))
    config = replace(V1, ref_audio_path="", prompt_text="", prompt_lang="")

    _run(_client(config, handler), "synthesize", "Hi")

    assert json.loads(requests[0].content) == {"text": "Hi", "text_language": "en"}


def test_v1_errors_and_non_wav_hint():
    error = {"code": 400, "message": "未指定参考音频"}
    handler, _ = _recording_handler(lambda request: httpx.Response(400, json=error))
    with pytest.raises(TTSError, match="HTTP 400: 未指定参考音频"):
        _run(_client(V1, handler), "synthesize", "Hi")

    handler, _ = _recording_handler(lambda request: httpx.Response(200, content=b"not a wav at all"))
    with pytest.raises(TTSError, match=r"not WAV.*-sm close"):
        _run(_client(V1, handler), "synthesize", "Hi")


# ---- 建立與釋放 ----


def test_aclose_closes_the_http_client():
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    tts = GPTSoVITSV2Client(TTSConfig(), client=http_client)

    asyncio.run(tts.aclose())

    assert http_client.is_closed


@pytest.mark.parametrize(
    ("engine", "expected"), [("gpt-sovits-v2", GPTSoVITSV2Client), ("gpt-sovits-v1", GPTSoVITSV1Client)]
)
def test_create_tts_builds_the_client_for_the_engine(engine, expected):
    async def run() -> None:
        tts = create_tts(replace(TTSConfig(), engine=engine))
        try:
            assert type(tts) is expected
        finally:
            await tts.aclose()

    asyncio.run(run())


def test_default_http_client_ignores_system_proxy_settings(monkeypatch):
    # 系統若設了 HTTP(S)_PROXY，本機／區網的 TTS 請求不應被繞到代理伺服器
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")

    async def run() -> bool:
        tts = GPTSoVITSV2Client(TTSConfig())
        try:
            return tts._client.trust_env
        finally:
            await tts.aclose()

    assert asyncio.run(run()) is False


# ---- 網址與路徑出錯時不噴 traceback ----


def test_server_that_accepts_but_never_answers_does_not_stall_startup():
    # 用真的 socket：MockTransport 不會套用逾時。每次 GET /docs 最多等 _PROBE_TIMEOUT，不是整個合成逾時（120 秒）
    async def run() -> float:
        async def accept_and_ignore(reader, writer) -> None:
            await asyncio.sleep(3600)

        server = await asyncio.start_server(accept_and_ignore, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        config = replace(TTSConfig(), url=f"http://127.0.0.1:{port}", ready_timeout=0.3, warmup_text="")
        tts = GPTSoVITSV2Client(config, poll_interval=0.01)
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            with pytest.raises(TTSError, match=r"not ready after 0\.3s.*ReadTimeout"):
                await asyncio.wait_for(tts.prepare(), 5)
        finally:
            await tts.aclose()
            server.close()
        return loop.time() - started

    assert asyncio.run(run()) < 2


def _bypass_validation(config: TTSConfig, **changes) -> TTSConfig:
    """模擬沒有經過設定驗證的 TTSConfig（例如程式內部建出來的），確認 client 本身也不會噴出 traceback。"""
    for key, value in changes.items():
        object.__setattr__(config, key, value)
    return config


@pytest.mark.parametrize("url", ["http://[::1", "http://xn--:9880"])
def test_invalid_urls_that_slip_past_validation_become_tts_errors(url):
    config = _bypass_validation(replace(TTSConfig(), warmup_text=""), url=url)

    async def run() -> None:
        tts = GPTSoVITSV2Client(config)
        try:
            with pytest.raises(TTSError, match="Request to the TTS server"):
                await tts.synthesize("Hi")
            with pytest.raises(TTSError, match=r"tts\.url is not a valid URL"):
                await tts.prepare()
        finally:
            await tts.aclose()

    asyncio.run(run())


def test_reference_path_of_an_unknown_user_is_a_config_error():
    with pytest.raises(ConfigError, match=r"tts\.ref_audio_path"):
        resolve_ref_audio_path("~nosuchuser/ref.wav")


def test_long_error_messages_are_truncated():
    handler, _ = _recording_handler(
        lambda request: httpx.Response(400, json={"message": "tts failed", "Exception": "x" * 5000})
    )

    with pytest.raises(TTSError) as excinfo:
        _run(_client(TTSConfig(), handler), "synthesize", "Hi")

    assert len(str(excinfo.value)) < 300
