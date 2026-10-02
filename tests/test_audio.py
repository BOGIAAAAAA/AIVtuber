from __future__ import annotations

import asyncio
import logging
import threading
import time
import types
import wave

import pytest

from vtuber import audio as audio_module
from vtuber.audio import PyAudioPlayer
from vtuber.errors import AudioDeviceError


class FakeStream:
    """仿 PyAudio 的輸出 stream：open 後就是啟動狀態；停止後要 start_stream() 才能再寫。"""

    def __init__(self, threads: set, write_delay: float = 0.0, fail_on_write: bool = False, block=None) -> None:
        self.threads = threads
        self.chunks: list[bytes] = []
        self.started = self.stopped = 0
        self.running = True
        self.closed = False
        self._write_delay = write_delay
        self._fail_on_write = fail_on_write
        self._block = block

    def write(self, data: bytes) -> None:
        self.threads.add(threading.get_ident())
        if self._block is not None:
            self._block.wait()  # 模擬卡住的輸出裝置
        if self._fail_on_write:
            raise OSError("output device unplugged")
        if self.closed or not self.running:
            raise OSError("Stream not open or not started")
        time.sleep(self._write_delay)
        self.chunks.append(data)

    def stop_stream(self) -> None:
        self.threads.add(threading.get_ident())
        self.stopped += 1
        self.running = False

    def start_stream(self) -> None:
        self.threads.add(threading.get_ident())
        self.started += 1
        self.running = True

    def close(self) -> None:
        self.threads.add(threading.get_ident())
        self.closed = True


def fake_pyaudio_module(open_error=None, **stream_kwargs):
    """回傳 (假 pyaudio 模組, 建立過的 PyAudio 實例清單)；每個實例記錄呼叫它的執行緒。"""
    instances = []

    class FakePyAudio:
        def __init__(self) -> None:
            self.threads = {threading.get_ident()}
            self.opens: list[dict] = []
            self.streams: list[FakeStream] = []
            self.terminated = False
            instances.append(self)

        def get_format_from_width(self, width: int) -> str:
            self.threads.add(threading.get_ident())
            return f"int{width * 8}"

        def open(self, **kwargs) -> FakeStream:
            self.threads.add(threading.get_ident())
            self.opens.append(kwargs)
            if open_error is not None:
                raise open_error
            stream = FakeStream(self.threads, **stream_kwargs)
            self.streams.append(stream)
            return stream

        def terminate(self) -> None:
            self.threads.add(threading.get_ident())
            self.terminated = True

    return types.SimpleNamespace(PyAudio=FakePyAudio), instances


def _play_all(player: PyAudioPlayer, *wavs: bytes, close: bool = True) -> None:
    async def run() -> None:
        try:
            for wav in wavs:
                await player.play(wav)
        finally:
            if close:
                await player.aclose()

    asyncio.run(run())


def _wait_until(condition, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


def test_play_writes_every_frame_on_one_background_thread_and_waits_for_the_buffer(make_wav):
    module, instances = fake_pyaudio_module()
    player = PyAudioPlayer(output_device_index=3, pyaudio_module=module)

    _play_all(player, make_wav(frames=5000, rate=32000), close=False)

    (pa,) = instances
    assert pa.opens == [{"format": "int16", "channels": 1, "rate": 32000, "output": True, "output_device_index": 3}]
    (stream,) = pa.streams
    assert sum(len(chunk) for chunk in stream.chunks) == 5000 * 2
    assert stream.stopped == 1  # 播完時等緩衝區清空才返回
    assert not stream.closed and not pa.terminated  # stream 與 PyAudio 留給下一句用

    _play_all(player)  # 只關閉

    assert stream.closed and pa.terminated
    # 所有 PyAudio 呼叫都在同一條背景執行緒，不在 event loop 的執行緒
    assert len(pa.threads) == 1 and threading.get_ident() not in pa.threads


def test_consecutive_audio_with_the_same_format_reuses_the_stream(make_wav):
    module, instances = fake_pyaudio_module()

    _play_all(PyAudioPlayer(pyaudio_module=module), make_wav(frames=3000), make_wav(frames=1000), make_wav(frames=10))

    (pa,) = instances  # 整個 session 只有一個 PyAudio
    (stream,) = pa.streams  # 只開一次 stream
    assert len(pa.opens) == 1
    assert sum(len(chunk) for chunk in stream.chunks) == (3000 + 1000 + 10) * 2
    assert (stream.stopped, stream.started) == (3, 2)  # 每句播完都等緩衝區清空，下一句再啟動


def test_a_different_format_reopens_the_stream(make_wav):
    module, instances = fake_pyaudio_module()

    _play_all(PyAudioPlayer(pyaudio_module=module), make_wav(rate=32000), make_wav(rate=16000), make_wav(rate=16000))

    (pa,) = instances
    assert [kwargs["rate"] for kwargs in pa.opens] == [32000, 16000]
    first, second = pa.streams
    assert first.closed and second.closed
    assert len(second.chunks) > 0 and second.started == 1


def test_cancelling_playback_stops_immediately_and_discards_the_buffer(make_wav):
    module, instances = fake_pyaudio_module(write_delay=0.01)
    player = PyAudioPlayer(pyaudio_module=module)
    long_wav = make_wav(frames=1024 * 200)  # 200 個區塊，約 2 秒的假寫入

    async def run() -> float:
        task = asyncio.ensure_future(player.play(long_wav))
        await asyncio.sleep(0.05)
        started = time.monotonic()
        task.cancel()  # 例如 Ctrl+C
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - started

    assert asyncio.run(run()) < 0.1  # 取消不必等播放執行緒

    (pa,) = instances
    (stream,) = pa.streams
    assert _wait_until(lambda: stream.closed)
    assert len(stream.chunks) < 200
    assert stream.stopped == 0  # 中斷時不等緩衝區播完，直接 close 丟棄
    assert not pa.terminated  # PyAudio 留著，下一句重開 stream 即可

    _play_all(player, make_wav(frames=100))

    assert len(pa.streams) == 2 and pa.streams[1].chunks and pa.terminated


def test_play_releases_device_when_write_fails(make_wav):
    module, instances = fake_pyaudio_module(fail_on_write=True)
    player = PyAudioPlayer(pyaudio_module=module)

    with pytest.raises(OSError, match="unplugged"):
        _play_all(player, make_wav(), close=False)

    (pa,) = instances
    assert pa.streams[0].closed and pa.terminated  # 裝置出錯就全部釋放
    with pytest.raises(OSError):
        _play_all(player, make_wav())
    assert len(instances) == 2  # 下一句重新初始化 PyAudio（也會重新列舉裝置）


def test_invalid_audio_is_rejected_before_opening_the_device():
    module, instances = fake_pyaudio_module()

    with pytest.raises(wave.Error, match="RIFF"):
        _play_all(PyAudioPlayer(pyaudio_module=module), b"definitely not a wav file")

    assert instances == []


def test_unavailable_output_device_is_reported_clearly_and_released(make_wav):
    module, instances = fake_pyaudio_module(open_error=OSError(-9996, "Invalid output device"))

    with pytest.raises(AudioDeviceError, match="output_device_index=9"):
        _play_all(PyAudioPlayer(output_device_index=9, pyaudio_module=module), make_wav())

    (pa,) = instances
    assert pa.terminated


def test_aclose_without_playing_does_nothing():
    module, instances = fake_pyaudio_module()

    _play_all(PyAudioPlayer(pyaudio_module=module))

    assert instances == []


def test_aclose_gives_up_on_a_stuck_device_instead_of_hanging(make_wav, monkeypatch, caplog):
    monkeypatch.setattr(audio_module, "CLOSE_TIMEOUT", 0.1)
    stuck = threading.Event()
    module, instances = fake_pyaudio_module(block=stuck)
    player = PyAudioPlayer(pyaudio_module=module)

    async def run() -> float:
        task = asyncio.ensure_future(player.play(make_wav()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        started = time.monotonic()
        await player.aclose()
        return time.monotonic() - started

    try:
        with caplog.at_level(logging.WARNING, logger="vtuber.audio"):
            assert asyncio.run(run()) < 1.0
        assert "did not close" in caplog.text
    finally:
        stuck.set()  # 讓卡住的 daemon 執行緒結束
    (pa,) = instances
    assert _wait_until(lambda: pa.terminated)


def test_on_start_is_called_on_the_event_loop_once_per_sentence_when_output_begins(make_wav):
    module, _ = fake_pyaudio_module()
    player = PyAudioPlayer(pyaudio_module=module)
    calls: list[int] = []

    async def run() -> None:
        try:
            for frames in (3000, 1000):
                await player.play(make_wav(frames=frames), on_start=lambda: calls.append(threading.get_ident()))
        finally:
            await player.aclose()

    asyncio.run(run())

    assert calls == [threading.get_ident()] * 2  # 每一句一次，而且在 event loop 的執行緒上呼叫


def test_on_start_is_not_called_when_playback_is_cancelled_before_it_begins(make_wav):
    stuck = threading.Event()
    module, _ = fake_pyaudio_module(block=stuck)
    player = PyAudioPlayer(pyaudio_module=module)
    calls: list[str] = []

    async def run() -> None:
        first = asyncio.ensure_future(player.play(make_wav()))  # 卡在寫入，佔住播放執行緒
        await asyncio.sleep(0.05)
        second = asyncio.ensure_future(player.play(make_wav(), on_start=lambda: calls.append("second")))
        await asyncio.sleep(0.05)
        for task in (second, first):
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        stuck.set()
        await player.aclose()

    asyncio.run(run())

    assert calls == []


def test_aclose_during_playback_stops_it_immediately(make_wav):
    module, instances = fake_pyaudio_module(write_delay=0.01)
    player = PyAudioPlayer(pyaudio_module=module)
    long_wav = make_wav(frames=1024 * 200)  # 200 個區塊，約 2 秒的假寫入

    async def run() -> float:
        playing = asyncio.ensure_future(player.play(long_wav))
        await asyncio.sleep(0.05)
        started = time.monotonic()
        await player.aclose()  # 沒有取消 play()，直接關閉
        elapsed = time.monotonic() - started
        await asyncio.wait_for(playing, 1)  # play() 也會結束，不會卡住
        return elapsed

    assert asyncio.run(run()) < 0.5  # 不必等整段播完，也遠低於 CLOSE_TIMEOUT

    (pa,) = instances
    (stream,) = pa.streams
    assert stream.closed and stream.stopped == 0  # 直接丟棄緩衝區，不等它播完
    assert len(stream.chunks) < 200
    assert pa.terminated
