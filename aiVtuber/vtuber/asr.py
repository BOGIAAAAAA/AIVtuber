"""語音辨識：SpeechRecognition 錄音 + Google Web Speech API 轉文字。

阻塞的錄音與辨識都放在 daemon 執行緒：asyncio.to_thread 用的預設 executor 會讓
asyncio.run 結束時等執行緒跑完，按 Ctrl+C 後可能要等很久，甚至卡住。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Iterator
from typing import Any, Callable, Optional, TypeVar

import speech_recognition as sr

from vtuber.config import ASRConfig
from vtuber.errors import AudioDeviceError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 錄音執行緒每次最多等這麼久就回頭檢查是否該停止；沒人說話時也以此為單位累計等待時間
POLL_SECONDS = 1.0


class SpeechRecognizer:
    """共用一個 Recognizer；環境噪音只在啟動時校正一次，之後每輪重新開麥克風串流來聽。"""

    def __init__(self, config: ASRConfig, *, recognizer: Any = None, microphone: Any = None) -> None:
        self._config = config
        self._recognizer = recognizer if recognizer is not None else sr.Recognizer()
        self._recognizer.operation_timeout = config.request_timeout
        # Microphone 物件只保存裝置設定，每次進入時才開新的串流：
        # 每輪都是乾淨的緩衝區，不會錄到播放期間殘留的 AI 聲音
        self._microphone = microphone if microphone is not None else _create_microphone(config.device_index)
        self._announce_listening = True
        self._stop = threading.Event()
        self._microphone_idle = threading.Event()
        self._microphone_idle.set()

    async def calibrate(self) -> None:
        """依環境噪音調整音量門檻（約 ambient_noise_duration 秒）。"""
        logger.info("Calibrating for ambient noise (%.1fs), please stay quiet...", self._config.ambient_noise_duration)
        threshold = await self._use_microphone(self._calibrate_blocking)
        logger.info("Microphone ready (energy threshold %.0f)", threshold)

    async def listen(self) -> Optional[str]:
        """聽一句話並轉成文字；沒人說話、聽不懂或辨識服務失敗時回傳 None。"""
        if self._announce_listening:
            logger.info("Listening...")
            self._announce_listening = False
        audio = await self._use_microphone(self._record)
        if audio is None:
            # 沒人說話是常態，不算錯誤；也不重複印 "Listening..."
            logger.debug("No speech within %ss; listening again", self._config.timeout)
            return None
        self._announce_listening = True
        try:
            text = await run_in_daemon_thread(
                self._recognizer.recognize_google, audio, language=self._config.language
            )
        except sr.UnknownValueError:
            logger.info("Could not understand the audio")
            return None
        except sr.RequestError as exc:
            logger.error("Google Speech Recognition request failed: %s", exc)
            return None
        logger.info("You said: %s", text)
        return text

    async def aclose(self) -> None:
        """請錄音執行緒停止，並等它自己關閉麥克風（最多約 POLL_SECONDS 秒）。"""
        self._stop.set()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + POLL_SECONDS + 0.5
        while not self._microphone_idle.is_set() and loop.time() < deadline:
            await asyncio.sleep(0.05)

    async def _use_microphone(self, func: Callable[[], T]) -> T:
        """在 daemon 執行緒使用麥克風；被取消時通知該執行緒停止（串流只在它自己的執行緒關閉）。"""
        self._microphone_idle.clear()

        def job() -> T:
            try:
                return func()
            finally:
                self._microphone_idle.set()

        try:
            return await run_in_daemon_thread(job)
        except asyncio.CancelledError:
            self._stop.set()
            raise

    def _calibrate_blocking(self) -> float:
        with open_input_stream(self._microphone, self._config.device_index) as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=self._config.ambient_noise_duration)
        return float(self._recognizer.energy_threshold)

    def _record(self) -> Any:
        """錄一句話；沒人說話超過 asr.timeout（None 表示一直等）或被要求停止時回傳 None。

        每次只等 POLL_SECONDS 秒，逾時就檢查停止旗標並累計等待時間，所以取消後約 1 秒內就會結束。
        """
        timeout = self._config.timeout
        waited = 0.0
        with open_input_stream(self._microphone, self._config.device_index) as source:
            while not self._stop.is_set() and (timeout is None or waited < timeout):
                chunk = POLL_SECONDS if timeout is None else min(POLL_SECONDS, timeout - waited)
                try:
                    return self._recognizer.listen(
                        source, timeout=chunk, phrase_time_limit=self._config.phrase_time_limit
                    )
                except sr.WaitTimeoutError:
                    waited += chunk
        return None


@contextlib.contextmanager
def open_input_stream(microphone: Any, device_index: Optional[int]) -> Iterator[Any]:
    """進入 sr.Microphone，串流打不開或讀取失敗時丟出 AudioDeviceError。

    SpeechRecognition 的 Microphone.__enter__ 會吞掉 PyAudio.open() 的錯誤並留下 stream=None，
    之後 listen() 會 AssertionError、__exit__ 又會在 stream.close() 丟 AttributeError。
    因此先檢查 stream；沒開成功時也不呼叫 __exit__（__enter__ 已自行 terminate PyAudio）。
    """
    where = f"device_index={device_index}"
    try:
        source = microphone.__enter__()
    except OSError as exc:
        raise AudioDeviceError(f"Cannot open the microphone ({where}): {exc}") from exc
    if getattr(source, "stream", None) is None:
        raise AudioDeviceError(
            f"Cannot open the microphone input stream ({where}); make sure it is an input device, "
            "not used by another program, and that this app has microphone permission"
        )
    try:
        yield source
    except OSError as exc:  # 例如錄音中裝置被拔掉
        raise AudioDeviceError(f"Microphone read failed ({where}): {exc}") from exc
    finally:
        microphone.__exit__(None, None, None)


async def run_in_daemon_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """在 daemon 執行緒執行阻塞呼叫並等待結果。

    與 asyncio.to_thread 不同，程式結束（例如 Ctrl+C）時不會等這個執行緒跑完。
    呼叫端被取消時，執行緒仍會跑完手上的工作，結果直接丟棄。執行緒內不寫 log，
    避免直譯器結束時 daemon 執行緒還握著輸出串流的鎖。
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def worker() -> None:
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:  # 交回 event loop 端由呼叫者處理
            _settle_threadsafe(loop, future, None, exc)
        else:
            _settle_threadsafe(loop, future, result, None)

    threading.Thread(target=worker, name="vtuber-asr", daemon=True).start()
    return await future


def _settle_threadsafe(
    loop: asyncio.AbstractEventLoop, future: asyncio.Future, result: Any, error: Optional[BaseException]
) -> None:
    def settle() -> None:
        if future.done():  # 呼叫端已取消
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    # event loop 已關閉（程式正在結束）時會丟 RuntimeError；此時結果已不再需要
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(settle)


def _create_microphone(device_index: Optional[int]) -> Any:
    try:
        return sr.Microphone(device_index=device_index)
    except AttributeError as exc:  # SpeechRecognition 找不到 PyAudio 時丟的是 AttributeError
        raise AudioDeviceError(
            "PyAudio is not installed; install PortAudio and PyAudio first (see README)"
        ) from exc
    except (AssertionError, OSError) as exc:  # 裝置編號超出範圍或裝置無法使用
        raise AudioDeviceError(f"Cannot open microphone (device_index={device_index}): {exc}") from exc
