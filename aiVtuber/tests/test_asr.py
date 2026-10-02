from __future__ import annotations

import asyncio
import logging
import time
import types
from dataclasses import replace
from typing import Optional

import pytest
import speech_recognition as sr

from vtuber.asr import POLL_SECONDS, SpeechRecognizer
from vtuber.config import ASRConfig
from vtuber.errors import AudioDeviceError

SILENCE = sr.WaitTimeoutError("listening timed out while waiting for phrase to start")


class FakeMicrophone:
    """與 sr.AudioSource 相同：進入時開串流（stream 不為 None），離開時關閉。"""

    def __init__(self) -> None:
        self.stream: Optional[object] = None
        self.opened = 0
        self.closed = 0

    def __enter__(self) -> FakeMicrophone:
        self.opened += 1
        self.stream = object()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closed += 1
        self.stream = None


class FakeRecognizer:
    """heard 可以是錄到的音訊、要丟出的例外，或依序使用的清單；delay 模擬每次聆聽花費的時間。"""

    def __init__(self, heard=b"audio", transcript="Hello Zora", delay: float = 0.0) -> None:
        self.energy_threshold = 300.0
        self.operation_timeout = None
        self.heard = heard
        self.transcript = transcript
        self.delay = delay
        self.calibrations: list[float] = []
        self.listen_calls: list[tuple] = []
        self.languages: list[str] = []

    def adjust_for_ambient_noise(self, source, duration):
        assert source.stream is not None
        self.calibrations.append(duration)

    def listen(self, source, timeout, phrase_time_limit):
        assert source.stream is not None
        self.listen_calls.append((timeout, phrase_time_limit))
        time.sleep(self.delay)
        result = self.heard.pop(0) if isinstance(self.heard, list) else self.heard
        if isinstance(result, BaseException):
            raise result
        return result

    def recognize_google(self, audio, language):
        self.languages.append(language)
        if isinstance(self.transcript, Exception):
            raise self.transcript
        return self.transcript


def _make(config: Optional[ASRConfig] = None, **recognizer_kwargs):
    recognizer = FakeRecognizer(**recognizer_kwargs)
    microphone = FakeMicrophone()
    asr = SpeechRecognizer(config or ASRConfig(), recognizer=recognizer, microphone=microphone)
    return asr, recognizer, microphone


async def _listen_times(asr: SpeechRecognizer, times: int) -> list:
    return [await asr.listen() for _ in range(times)]


def test_noise_is_calibrated_once_and_each_turn_reopens_the_microphone():
    asr, recognizer, microphone = _make()

    async def run() -> list:
        await asr.calibrate()
        return [await asr.listen(), await asr.listen()]

    assert asyncio.run(run()) == ["Hello Zora", "Hello Zora"]
    assert recognizer.calibrations == [1.0]
    assert microphone.opened == microphone.closed == 3  # 校正 1 次 + 每輪聆聽各重開 1 次


def test_listen_polls_in_short_chunks_and_uses_configured_language_and_limits():
    config = replace(ASRConfig(), language="zh-TW", timeout=3.0, phrase_time_limit=7.0, request_timeout=9.0)
    asr, recognizer, _ = _make(config, heard=[SILENCE, SILENCE, b"audio"])

    assert asyncio.run(asr.listen()) == "Hello Zora"
    assert recognizer.listen_calls == [(POLL_SECONDS, 7.0)] * 3
    assert recognizer.languages == ["zh-TW"]
    assert recognizer.operation_timeout == 9.0


def test_silence_for_the_whole_timeout_returns_none_after_the_configured_total_wait():
    asr, recognizer, _ = _make(replace(ASRConfig(), timeout=2.5), heard=SILENCE)

    assert asyncio.run(asr.listen()) is None
    assert [call[0] for call in recognizer.listen_calls] == [1.0, 1.0, 0.5]


def test_defaults_keep_the_original_five_second_limits():
    asr, recognizer, _ = _make(heard=SILENCE)

    asyncio.run(asr.listen())

    assert sum(timeout for timeout, _ in recognizer.listen_calls) == pytest.approx(5.0)
    assert {limit for _, limit in recognizer.listen_calls} == {5.0}


def test_silence_is_handled_quietly(caplog):
    asr, recognizer, _ = _make(heard=SILENCE)

    with caplog.at_level(logging.DEBUG, logger="vtuber.asr"):
        results = asyncio.run(_listen_times(asr, 3))

    assert results == [None, None, None]
    assert recognizer.languages == []  # 沒錄到聲音就不呼叫辨識服務
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert sum("Listening" in record.getMessage() and record.levelno == logging.INFO for record in caplog.records) == 1


def test_unintelligible_speech_and_service_errors_return_none(caplog):
    unclear, _, _ = _make(transcript=sr.UnknownValueError())
    offline, _, _ = _make(transcript=sr.RequestError("no internet"))

    with caplog.at_level(logging.INFO, logger="vtuber.asr"):
        assert asyncio.run(unclear.listen()) is None
        assert asyncio.run(offline.listen()) is None

    assert "no internet" in caplog.text


def test_cancelled_listen_does_not_make_asyncio_run_wait_for_the_recording_thread():
    # 每次聆聽 1 秒且永遠沒人說話、timeout 為 null（一直等）：模擬按下 Ctrl+C 時的狀況
    asr, recognizer, microphone = _make(replace(ASRConfig(), timeout=None), heard=SILENCE, delay=1.0)

    async def run() -> None:
        task = asyncio.ensure_future(asr.listen())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    started = time.monotonic()
    asyncio.run(run())
    assert time.monotonic() - started < 0.6  # 用預設 executor 的話，asyncio.run 會等滿這 1 秒

    # 錄音執行緒聽完手上這 1 秒就會停止，並在自己的執行緒關閉麥克風
    deadline = time.monotonic() + 3
    while microphone.closed == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert microphone.closed == 1
    assert len(recognizer.listen_calls) <= 2


def test_aclose_waits_for_the_recording_thread_to_close_the_microphone():
    asr, _, microphone = _make(replace(ASRConfig(), timeout=None), heard=SILENCE, delay=0.3)

    async def run() -> None:
        task = asyncio.ensure_future(asr.listen())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asr.aclose()
        assert microphone.closed == 1

    asyncio.run(run())


def test_missing_pyaudio_is_reported_as_a_clear_error(monkeypatch):
    def microphone_without_pyaudio(device_index=None):
        raise AttributeError("Could not find PyAudio; check installation")

    monkeypatch.setattr(sr, "Microphone", microphone_without_pyaudio)

    with pytest.raises(AudioDeviceError, match="PyAudio is not installed"):
        SpeechRecognizer(ASRConfig(), recognizer=FakeRecognizer())


def test_invalid_microphone_index_is_reported_with_the_index(monkeypatch):
    def out_of_range(device_index=None):
        raise AssertionError("Device index out of range (2 devices available)")

    monkeypatch.setattr(sr, "Microphone", out_of_range)

    with pytest.raises(AudioDeviceError, match=r"device_index=7.*out of range"):
        SpeechRecognizer(replace(ASRConfig(), device_index=7), recognizer=FakeRecognizer())


@pytest.fixture
def pyaudio_that_cannot_open_input(monkeypatch):
    """假 pyaudio：裝置列舉正常，但 open() 失敗（例如 --device_index 指到只有輸出的裝置）。"""
    module = types.ModuleType("pyaudio")
    module.paInt16 = 8
    module.get_sample_size = lambda fmt: 2
    calls = {"open": 0, "terminate": 0}

    class PyAudio:
        def get_device_count(self) -> int:
            return 3

        def get_default_input_device_info(self) -> dict:
            return {"defaultSampleRate": 44100.0}

        def get_device_info_by_index(self, index: int) -> dict:
            return {"defaultSampleRate": 44100.0}

        def open(self, **kwargs):
            calls["open"] += 1
            raise OSError(-9998, "Invalid number of channels")

        def terminate(self) -> None:
            calls["terminate"] += 1

    module.PyAudio = PyAudio
    monkeypatch.setitem(__import__("sys").modules, "pyaudio", module)
    return calls


def test_real_microphone_that_fails_to_open_raises_audio_device_error(pyaudio_that_cannot_open_input):
    # 真的 sr.Microphone 會吞掉 open() 的 OSError 並留下 stream=None
    asr = SpeechRecognizer(replace(ASRConfig(), device_index=1))

    with pytest.raises(AudioDeviceError, match="device_index=1") as calibrate_error:
        asyncio.run(asr.calibrate())
    with pytest.raises(AudioDeviceError, match="input stream"):
        asyncio.run(asr.listen())

    assert not isinstance(calibrate_error.value.__context__, AttributeError)
    calls = pyaudio_that_cannot_open_input
    # 建構時列舉裝置 1 次 + 兩次開啟各 1 次；不會因為 __exit__ 再 terminate 一次
    assert calls["open"] == 2
    assert calls["terminate"] == 3


def test_microphone_read_errors_become_audio_device_errors():
    asr, _, microphone = _make(heard=OSError(-9999, "Unanticipated host error"))

    with pytest.raises(AudioDeviceError, match="Microphone read failed"):
        asyncio.run(asr.listen())

    assert microphone.closed == 1
