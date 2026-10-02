"""音訊播放：直接播放記憶體中的 WAV bytes，不寫暫存檔。

PyAudio 只在一條專屬的 daemon 執行緒上使用（建立、開關 stream、寫入、terminate 都在這條執行緒）：
- PortAudio 不保證跨執行緒安全，Windows 的 WASAPI 等後端也要求在同一條執行緒使用；
- 阻塞的寫入不佔用 event loop；
- 用 daemon 執行緒而不是 asyncio.to_thread：預設 executor 的執行緒會讓 asyncio.run 結束時等它跑完，
  輸出裝置卡住時 Ctrl+C 就會卡住（同 asr.py 的做法）。

整個 session 共用一個 PyAudio；連續幾段音訊的格式相同時重用同一個輸出 stream，
避免每句重新開關 stream 造成的延遲與句間空檔，格式不同才重開。聊天結束時一定要呼叫 aclose()。
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import queue
import threading
import wave
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Optional, Protocol, Union

from vtuber.errors import AudioDeviceError

logger = logging.getLogger(__name__)

# aclose() 最多等播放執行緒這麼久；裝置卡住時直接放手（daemon 執行緒不會讓程式結束時卡住）
CLOSE_TIMEOUT = 2.0


class AudioPlayer(Protocol):
    """播放器介面。"""

    async def play(self, wav_bytes: bytes, *, on_start: Optional[Callable[[], None]] = None) -> None:
        """播放一段完整的 WAV，播完才返回；被取消時立刻停止。

        on_start 在聲音開始寫入輸出裝置時於 event loop 上呼叫一次（用來量測延遲）。
        """
        ...

    async def aclose(self) -> None:
        """釋放播放裝置。"""
        ...


@dataclass(frozen=True)
class _AudioFormat:
    channels: int
    sample_width: int
    rate: int

    @property
    def frame_size(self) -> int:
        return self.channels * self.sample_width


class _Job:
    """交給播放執行緒的工作；結果透過 event loop 的 future 交回。"""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.done: asyncio.Future = loop.create_future()

    def settle(self, error: Optional[BaseException]) -> None:
        """在播放執行緒呼叫：把結果交回 event loop。"""

        def apply() -> None:
            if self.done.done():  # 呼叫端已取消或不再等待
                return
            if error is None:
                self.done.set_result(None)
            else:
                self.done.set_exception(error)

        # event loop 已關閉（程式正在結束）時會丟 RuntimeError；此時結果已不再需要
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(apply)


class _PlayJob(_Job):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        audio_format: _AudioFormat,
        frames: bytes,
        on_start: Optional[Callable[[], None]],
    ) -> None:
        super().__init__(loop)
        self.format = audio_format
        self.frames = frames
        self.stop = threading.Event()
        self._on_start = on_start

    def started(self) -> None:
        """在播放執行緒呼叫：第一個區塊要寫入裝置了。"""
        if self._on_start is None:
            return

        def notify() -> None:
            if not self.done.done():  # 已被取消就不必通知
                self._on_start()

        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(notify)


class _CloseJob(_Job):
    pass


class PyAudioPlayer:
    """用 PyAudio 播放 WAV；第一次播放時才啟動播放執行緒並 import pyaudio。"""

    def __init__(
        self,
        output_device_index: Optional[int] = None,
        *,
        chunk_frames: int = 1024,
        pyaudio_module: Optional[ModuleType] = None,
    ) -> None:
        self._output_device_index = output_device_index
        self._chunk_frames = chunk_frames
        # None 表示在播放執行緒裡才 import pyaudio（測試可注入假模組）
        self._pyaudio = pyaudio_module
        self._worker: Optional[_AudioWorker] = None

    async def play(self, wav_bytes: bytes, *, on_start: Optional[Callable[[], None]] = None) -> None:
        # 先在這裡解析 WAV：格式錯誤時直接丟出 wave.Error，不會動到播放裝置
        audio_format, frames = _read_wav(wav_bytes)
        if not frames:
            return
        job = _PlayJob(asyncio.get_running_loop(), audio_format, frames, on_start)
        if self._worker is None:
            self._worker = _AudioWorker(self._pyaudio, self._output_device_index, self._chunk_frames)
        self._worker.submit(job)
        try:
            await job.done
        except asyncio.CancelledError:
            # 例如 Ctrl+C：通知播放執行緒在下一個區塊停止，並丟棄緩衝區裡還沒播出的聲音
            job.stop.set()
            raise

    async def aclose(self) -> None:
        """關閉輸出 stream 並釋放 PyAudio；之後再播放會重新初始化。"""
        worker, self._worker = self._worker, None
        if worker is None:
            return
        job = _CloseJob(asyncio.get_running_loop())
        worker.close(job)
        try:
            await asyncio.wait_for(job.done, CLOSE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Audio output did not close within %.0fs; abandoning it", CLOSE_TIMEOUT)


class _AudioWorker:
    """擁有 PyAudio 與輸出 stream 的 daemon 執行緒；所有 PyAudio 呼叫都在這條執行緒上。

    執行緒內不寫 log，避免直譯器結束時 daemon 執行緒還握著輸出串流的鎖（同 asr.py）。
    """

    def __init__(
        self, pyaudio_module: Optional[ModuleType], output_device_index: Optional[int], chunk_frames: int
    ) -> None:
        self._pyaudio_module = pyaudio_module
        self._output_device_index = output_device_index
        self._chunk_frames = chunk_frames
        self._jobs: queue.Queue[Union[_PlayJob, _CloseJob]] = queue.Queue()
        self._closing = threading.Event()
        # 以下只在播放執行緒存取
        self._pa: Any = None
        self._stream: Any = None
        self._stream_format: Optional[_AudioFormat] = None
        self._stream_running = False
        threading.Thread(target=self._run, name="vtuber-audio", daemon=True).start()

    def submit(self, job: _PlayJob) -> None:
        self._jobs.put(job)

    def close(self, job: _CloseJob) -> None:
        """停止正在播放與排隊中的音訊，然後釋放裝置並結束執行緒。"""
        self._closing.set()
        self._jobs.put(job)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if isinstance(job, _CloseJob):
                self._release()
                job.settle(None)
                return
            try:
                self._play(job)
            except BaseException as exc:  # 交回 event loop 端，由呼叫端處理
                # 裝置出錯（例如被拔掉）：全部釋放，下次播放重新初始化 PyAudio，也會重新列舉裝置
                self._release()
                job.settle(exc)
            else:
                job.settle(None)

    def _play(self, job: _PlayJob) -> None:
        def stopped() -> bool:
            return job.stop.is_set() or self._closing.is_set()

        if stopped():  # 還沒輪到就被取消
            return
        stream = self._open(job.format)
        if not self._stream_running:
            stream.start_stream()
            self._stream_running = True
        job.started()
        step = self._chunk_frames * job.format.frame_size
        for offset in range(0, len(job.frames), step):
            if stopped():
                # 被中斷：close() 直接丟棄緩衝區裡的聲音（stop_stream() 會等它播完）
                self._close_stream()
                return
            stream.write(job.frames[offset : offset + step])
        # 等緩衝區播完才算播完；下一句格式相同時 start_stream() 就能接著播，不必重開 stream
        stream.stop_stream()
        self._stream_running = False

    def _open(self, audio_format: _AudioFormat) -> Any:
        if self._stream is not None and self._stream_format != audio_format:
            self._close_stream()
        if self._stream is None:
            if self._pa is None:
                self._pa = (self._pyaudio_module or _import_pyaudio()).PyAudio()
            try:
                self._stream = self._pa.open(
                    format=self._pa.get_format_from_width(audio_format.sample_width),
                    channels=audio_format.channels,
                    rate=audio_format.rate,
                    output=True,
                    output_device_index=self._output_device_index,
                )
            except OSError as exc:  # 例如裝置編號不存在或裝置被占用
                raise AudioDeviceError(
                    f"Cannot open audio output (output_device_index={self._output_device_index}): {exc}"
                ) from exc
            self._stream_format = audio_format
            self._stream_running = True  # open() 預設會立刻啟動 stream
        return self._stream

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        self._stream_format = None
        self._stream_running = False
        if stream is not None:
            with contextlib.suppress(Exception):  # 裝置已經壞掉時 close 也可能失敗；釋放是盡力而為
                stream.close()

    def _release(self) -> None:
        self._close_stream()
        pa, self._pa = self._pa, None
        if pa is not None:
            with contextlib.suppress(Exception):
                pa.terminate()


def _read_wav(wav_bytes: bytes) -> tuple[_AudioFormat, bytes]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        audio_format = _AudioFormat(wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
        return audio_format, wav.readframes(wav.getnframes())


def _import_pyaudio() -> Any:
    try:
        import pyaudio
    except ImportError as exc:
        raise AudioDeviceError("PyAudio is not installed; install PortAudio and PyAudio first (see README)") from exc
    return pyaudio
