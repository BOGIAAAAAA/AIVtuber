from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

import pytest

import vtuber.asr
import vtuber.audio
import vtuber.llm
import vtuber.tts
from vtuber.chat import is_exit_command, run_chat, run_chat_loop
from vtuber.config import AppConfig
from vtuber.errors import TTSError


class ScriptExhausted(BaseException):
    """腳本用完卻還在聽：用 BaseException，聊天迴圈的 except Exception 攔不到，測試會立刻失敗而不是卡住。"""


class ScriptedListener:
    """依序回傳預先寫好的辨識結果；項目可以是文字、None 或要丟出的例外。"""

    def __init__(self, script) -> None:
        self.script = list(script)
        self.calibrated = False
        self.closed = False

    async def calibrate(self) -> None:
        self.calibrated = True

    async def listen(self):
        if not self.script:
            raise ScriptExhausted("loop kept listening after the script ended")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True


class BlockingListener:
    def __init__(self) -> None:
        self.closed = False

    async def calibrate(self) -> None:
        pass

    async def listen(self):
        await asyncio.Event().wait()  # 一直等，模擬沒人說話時被 Ctrl+C

    async def aclose(self) -> None:
        self.closed = True


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.closed = False

    async def stream_reply(self, prompt: str):
        self.prompts.append(prompt)
        for part in ("reply ", "to ", prompt):
            yield part

    async def aclose(self) -> None:
        self.closed = True


class FakeTTS:
    def __init__(self, fail_texts=(), prepare_error=None) -> None:
        self.texts: list[str] = []
        self.fail_texts = set(fail_texts)
        self.prepare_error = prepare_error
        self.prepared = False
        self.closed = False

    async def prepare(self) -> None:
        if self.prepare_error is not None:
            raise self.prepare_error
        self.prepared = True

    async def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        if text in self.fail_texts:
            raise TTSError("GPT-SoVITS server is down")
        return b"WAV:" + text.encode()

    async def aclose(self) -> None:
        self.closed = True


class FakePlayer:
    def __init__(self, fail_first: bool = False) -> None:
        self.played: list[bytes] = []
        self.fail_first = fail_first
        self.closed = False

    async def play(self, wav_bytes: bytes, *, on_start=None) -> None:
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("audio device busy")
        if on_start is not None:
            on_start()
        self.played.append(wav_bytes)

    async def aclose(self) -> None:
        self.closed = True


def _run(listener, llm=None, tts=None, player=None, **kwargs):
    llm, tts, player = llm or FakeLLM(), tts or FakeTTS(), player or FakePlayer()
    asyncio.run(run_chat_loop(listener, llm, tts, player, retry_delay=0, **kwargs))
    return llm, tts, player


@pytest.mark.parametrize("text", ["exit", "Exit.", "  EXIT!  ", "exit。", "Exit?!"])
def test_exit_command_ignores_case_and_trailing_punctuation(text):
    assert is_exit_command(text, ["exit"])


@pytest.mark.parametrize("configured", ["bye!", " BYE。 ", "Bye?"])
def test_configured_exit_commands_are_normalized_the_same_way(configured):
    assert is_exit_command("bye", [configured])


@pytest.mark.parametrize("text", ["exit please", "exiting", "next", "", "!"])
def test_other_sentences_are_not_exit_commands(text):
    assert not is_exit_command(text, ["exit"])


def test_exit_stops_the_loop_without_calling_the_llm():
    llm, tts, _ = _run(ScriptedListener(["Exit."]))

    assert llm.prompts == []
    assert tts.texts == []


def test_custom_exit_commands_are_supported():
    llm, _, _ = _run(ScriptedListener(["hello", "Bye!"]), exit_commands=("quit", "bye"))

    assert llm.prompts == ["hello"]


def test_one_turn_flows_from_speech_to_llm_to_tts_to_playback():
    llm, tts, player = _run(ScriptedListener(["Hello Zora", "exit"]))

    assert llm.prompts == ["Hello Zora"]
    assert tts.texts == ["reply to Hello Zora"]
    assert player.played == [b"WAV:reply to Hello Zora"]


def test_silence_and_unrecognised_speech_are_skipped():
    llm, _, _ = _run(ScriptedListener([None, "", "Hi", None, "EXIT"]))

    assert llm.prompts == ["Hi"]


def test_failed_tts_turn_is_logged_and_next_turn_still_works(caplog):
    tts = FakeTTS(fail_texts={"reply to first"})

    with caplog.at_level(logging.ERROR):
        llm, _, player = _run(ScriptedListener(["first", "second", "exit"]), tts=tts)

    assert llm.prompts == ["first", "second"]
    assert player.played == [b"WAV:reply to second"]
    assert "GPT-SoVITS server is down" in caplog.text


def test_each_turn_logs_the_full_reply_and_latency(caplog):
    with caplog.at_level(logging.INFO, logger="vtuber.chat"):
        _run(ScriptedListener(["Hello", "exit"]))

    assert "AI response: reply to Hello" in caplog.text
    assert "Latency after recognition: first sentence 0." in caplog.text
    assert "first audio 0." in caplog.text and "1 of 1 sentence(s) spoken" in caplog.text


def test_long_replies_are_split_into_sentences_with_the_configured_limits():
    _, tts, player = _run(
        ScriptedListener(["Hi. Ok. This is a longer one", "exit"]), min_sentence_chars=5, max_sentence_chars=40
    )

    assert tts.texts == ["reply to Hi.", "Ok. This is a longer one"]  # "Ok." 太短，併入下一句
    assert len(player.played) == 2


def test_unexpected_error_in_a_turn_does_not_stop_the_loop(caplog):
    with caplog.at_level(logging.INFO, logger="vtuber.chat"):
        _, _, player = _run(ScriptedListener(["first", "second", "exit"]), player=FakePlayer(fail_first=True))

    assert player.played == [b"WAV:reply to second"]
    assert "audio device busy" in caplog.text
    # 播放失敗的那一輪，已經產生的回覆仍然會記錄下來
    assert "AI response (interrupted after 0 of 1 sentence(s)): reply to first" in caplog.text


def test_speech_input_failure_is_retried():
    llm, _, _ = _run(ScriptedListener([OSError("microphone unplugged"), "Hi", "exit"]))

    assert llm.prompts == ["Hi"]


def test_cancellation_is_not_swallowed_by_the_loop():
    async def run() -> None:
        task = asyncio.ensure_future(run_chat_loop(BlockingListener(), FakeLLM(), FakeTTS(), FakePlayer()))
        await asyncio.sleep(0.01)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())


@pytest.fixture
def fake_components(monkeypatch):
    """把 run_chat 會建立的實際元件換成假物件，並記錄建構時收到的參數。"""
    created: dict = {}

    def install(listener) -> dict:
        created.update(listener=listener, llm=FakeLLM(), tts=FakeTTS(), player=FakePlayer(), args={})

        def make(name):
            def factory(argument):
                created["args"][name] = argument
                return created[name]

            return factory

        monkeypatch.setattr(vtuber.asr, "SpeechRecognizer", make("listener"))
        monkeypatch.setattr(vtuber.llm, "GroqChat", make("llm"))
        monkeypatch.setattr(vtuber.tts, "create_tts", make("tts"))
        monkeypatch.setattr(vtuber.audio, "PyAudioPlayer", make("player"))
        return created

    return install


def _custom_config() -> AppConfig:
    config = AppConfig()
    return replace(
        config,
        asr=replace(config.asr, device_index=3),
        audio=replace(config.audio, output_device_index=7),
        chat=replace(config.chat, exit_commands=("bye!",)),
    )


def test_run_chat_wires_components_from_config_and_closes_them_on_exit(fake_components):
    created = fake_components(ScriptedListener(["Hello", "Bye"]))
    config = _custom_config()

    asyncio.run(run_chat(config))

    assert created["args"]["listener"] == config.asr
    assert created["args"]["llm"] == config.llm
    assert created["args"]["tts"] == config.tts
    assert created["args"]["player"] == 7  # 播放裝置用 audio.output_device_index，不是麥克風的編號
    assert created["listener"].calibrated
    assert created["tts"].prepared  # 聊天開始前先確認 TTS server 就緒並暖機
    assert created["player"].played == [b"WAV:reply to Hello"]  # 說 "Bye" 就結束：exit_commands 有傳進迴圈
    assert created["listener"].closed and created["llm"].closed and created["tts"].closed
    assert created["player"].closed


def test_run_chat_closes_everything_when_cancelled(fake_components):
    created = fake_components(BlockingListener())

    async def run() -> None:
        task = asyncio.ensure_future(run_chat(AppConfig()))
        await asyncio.sleep(0.01)
        task.cancel()  # asyncio.run 在 Ctrl+C 時也是取消主 task
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

    assert created["listener"].closed and created["llm"].closed and created["tts"].closed
    assert created["player"].closed


def test_run_chat_stops_before_listening_when_the_tts_server_is_unavailable(fake_components):
    created = fake_components(ScriptedListener([]))  # 聽到任何東西都會讓測試失敗
    created["tts"].prepare_error = TTSError("TTS server at http://127.0.0.1:9880 is not ready after 60s")

    with pytest.raises(TTSError, match="not ready"):
        asyncio.run(run_chat(AppConfig()))

    assert not created["listener"].calibrated
    assert created["listener"].closed and created["tts"].closed
