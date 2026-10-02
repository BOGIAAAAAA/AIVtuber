"""文字轉語音：TTS 介面與 GPT-SoVITS 的 HTTP 客戶端。

- gpt-sovits-v2（預設）：GPT-SoVITS 的 api_v2.py，POST JSON 到 /tts
- gpt-sovits-v1：舊版 GPT-SoVITS 的 api.py，POST JSON 到根路徑

更換 TTS 引擎時，只要新增一個符合 TextToSpeech 介面的類別，
並在 create_tts() 加上對應的 tts.engine 分支；聊天流程完全不用改。
"""

from __future__ import annotations

import asyncio
import io
import ipaddress
import json
import logging
import os
import wave
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Optional, Protocol

import httpx

from vtuber.config import BASE_DIR, TTSConfig
from vtuber.errors import ConfigError, TTSError

logger = logging.getLogger(__name__)

# 就緒檢查：每次 GET /docs 最多等幾秒、失敗後隔多久再試
_PROBE_TIMEOUT = 5.0
_POLL_INTERVAL = 1.0
# 網址本身有問題時 httpx 丟出的例外（不是 httpx.HTTPError）：格式錯誤，或主機名稱不合法（idna.IDNAError）
_INVALID_URL_ERRORS = (httpx.InvalidURL, UnicodeError)


class TextToSpeech(Protocol):
    """TTS 引擎介面。"""

    async def prepare(self) -> None:
        """聊天開始前呼叫一次（例如確認 server 已就緒並暖機）；無法使用時丟出 TTSError。"""
        ...

    async def synthesize(self, text: str) -> bytes:
        """把文字合成為語音，回傳完整的 WAV 檔內容（RIFF/WAVE bytes）；失敗時丟出 TTSError。"""
        ...

    async def aclose(self) -> None:
        """釋放連線等資源；沒有資源要釋放的引擎可以什麼都不做。"""
        ...


def is_wav(data: bytes) -> bool:
    """檢查是否為 RIFF/WAVE 檔頭。"""
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def resolve_ref_audio_path(path: str, base_dir: Path = BASE_DIR) -> str:
    """決定要送給 server 的參考音檔路徑。

    絕對路徑原樣送出（POSIX 或 Windows 寫法都算，因為 server 可能在另一種作業系統上）；
    相對路徑（含 ~ 開頭）以 base_dir（專案根目錄）為基準轉成這台電腦上的絕對路徑。
    """
    if _is_absolute_anywhere(path):
        return path
    try:
        expanded = Path(path).expanduser()
    except RuntimeError as exc:  # ~someone 但這台電腦沒有這個使用者；載入設定時已檢查過，這裡是保險
        raise ConfigError(f"Invalid path {path!r} in tts.ref_audio_path: {exc}") from exc
    return os.path.normpath(base_dir / expanded)


class _GPTSoVITSClient:
    """GPT-SoVITS 各版 API 共用的部分：HTTP client、回應檢查、就緒檢查與暖機。"""

    # 子類別設定：推理端點（相對於 tts.url）與回應不是 WAV 時的提示
    _endpoint_path = ""
    _not_wav_hint = ""

    def __init__(
        self,
        config: TTSConfig,
        client: Optional[httpx.AsyncClient] = None,
        *,
        base_dir: Path = BASE_DIR,
        poll_interval: float = _POLL_INTERVAL,
    ) -> None:
        self._config = config
        # connect 逾時刻意設短：server 沒開時要盡快回報，而不是等滿整個合成逾時。
        # trust_env=False：TTS server 在本機或區網，不該被系統的 HTTP(S)_PROXY 繞到代理伺服器
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout, connect=5.0), trust_env=False
        )
        base_url = config.url.rstrip("/")
        self._endpoint = base_url + self._endpoint_path if self._endpoint_path else config.url
        self._docs_url = base_url + "/docs"
        self._ref_audio_path = resolve_ref_audio_path(config.ref_audio_path, base_dir) if config.ref_audio_path else ""
        self._base_dir = base_dir
        self._poll_interval = poll_interval

    def build_payload(self, text: str) -> dict[str, Any]:
        raise NotImplementedError

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise TTSError("Refusing to synthesize empty text")
        try:
            response = await self._client.post(self._endpoint, json=self.build_payload(text))
        except (httpx.HTTPError, *_INVALID_URL_ERRORS) as exc:
            raise TTSError(
                f"Request to the TTS server at {self._endpoint} failed: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code != 200:
            hint = ""
            if response.status_code in (404, 405):
                hint = " (does tts.engine match the server? api_v2.py: gpt-sovits-v2, api.py: gpt-sovits-v1)"
            raise TTSError(f"TTS server returned HTTP {response.status_code}: {_error_detail(response)}{hint}")
        audio = response.content
        if not is_wav(audio):
            raise TTSError(
                f"TTS response is not WAV audio (content-type={response.headers.get('content-type')!r}, "
                f"{len(audio)} bytes, starts with {audio[:12]!r}); {self._not_wav_hint}"
            )
        _check_not_silent(audio)
        logger.debug("Synthesized %d bytes of WAV audio", len(audio))
        return audio

    async def prepare(self) -> None:
        """檢查參考音檔、等 server 就緒並暖機（依設定）；失敗時丟出 TTSError。"""
        self._warn_about_reference_audio()
        if self._config.wait_for_server:
            await self.wait_until_ready()
        if self._config.warmup_text.strip():
            await self.warm_up()

    async def wait_until_ready(self) -> None:
        """等到 GET {url}/docs 回 200（server 載完模型後才開始接受連線），最多 tts.ready_timeout 秒。"""
        loop = asyncio.get_running_loop()
        timeout = self._config.ready_timeout
        deadline = loop.time() + timeout
        announced = False
        while True:
            problem = await self._probe(deadline - loop.time())
            if problem is None:
                logger.info("TTS server at %s is ready", self._config.url)
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TTSError(
                    f"TTS server at {self._config.url} is not ready after {timeout:g}s ({problem}). "
                    "Start it with tts_server/start_tts_server.sh (start_tts_server.bat on Windows) and wait "
                    f"until GET {self._docs_url} returns 200; to start without waiting set "
                    "tts.wait_for_server: false and tts.warmup_text: \"\""
                )
            if not announced:
                logger.info("Waiting up to %gs for the TTS server at %s (%s)...", timeout, self._config.url, problem)
                announced = True
            await asyncio.sleep(min(self._poll_interval, remaining))

    async def _probe(self, remaining: float) -> Optional[str]:
        """GET /docs 一次：回 200 時傳回 None，否則傳回原因；回 4xx 表示連到的不是 GPT-SoVITS，直接丟出 TTSError。"""
        try:
            # 每次最多等 _PROBE_TIMEOUT 秒：server 接受連線卻不回應時，才不會等滿整個合成逾時
            response = await self._client.get(self._docs_url, timeout=max(min(_PROBE_TIMEOUT, remaining), 0.5))
        except _INVALID_URL_ERRORS as exc:  # 重試也沒用
            raise TTSError(f"tts.url is not a valid URL ({exc}): {self._config.url!r}") from exc
        except httpx.HTTPError as exc:
            return f"{type(exc).__name__}: {exc}"
        if response.status_code == 200:
            return None
        if 400 <= response.status_code < 500:
            raise TTSError(
                f"GET {self._docs_url} returned HTTP {response.status_code}; "
                "is tts.url pointing at a GPT-SoVITS API server?"
            )
        return f"GET /docs returned HTTP {response.status_code}"

    async def warm_up(self) -> None:
        """合成 tts.warmup_text（不播放）：server 會先處理並快取參考音檔，第一句回覆就不必多等。"""
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            audio = await self.synthesize(self._config.warmup_text)
        except TTSError as exc:
            raise TTSError(f"TTS warm-up failed: {exc}. To skip the warm-up set tts.warmup_text: \"\"") from exc
        logger.info("TTS warm-up done in %.1fs (%.1fs of audio)", loop.time() - started, _duration(audio))

    def _warn_about_reference_audio(self) -> None:
        """同一台電腦時確認參考音檔存在；server 在別台電腦卻填相對路徑時提醒（相對路徑是在這台電腦上解析的）。"""
        configured = self._config.ref_audio_path
        if not configured:
            return
        if _is_local_url(self._config.url):
            if not os.path.isfile(self._ref_audio_path):
                logger.warning(
                    "Reference audio not found: %s (tts.ref_audio_path=%r; relative paths are resolved against %s). "
                    "The TTS server on this computer will not be able to read it.",
                    self._ref_audio_path,
                    configured,
                    self._base_dir,
                )
        elif not _is_absolute_anywhere(configured):
            logger.warning(
                "tts.ref_audio_path %r is relative, so it was resolved on this computer (%s), but the TTS server "
                "%s seems to be another computer; if so, set the absolute path of the file on the server machine.",
                configured,
                self._ref_audio_path,
                self._config.url,
            )

    async def aclose(self) -> None:
        await self._client.aclose()


class GPTSoVITSV2Client(_GPTSoVITSClient):
    """呼叫 GPT-SoVITS 的 api_v2.py（POST /tts，每次都帶參考音檔，回傳完整 WAV）。"""

    _endpoint_path = "/tts"
    _not_wav_hint = (
        "the client asks for media_type=wav with streaming_mode=false; is tts.url pointing at GPT-SoVITS api_v2.py?"
    )

    def build_payload(self, text: str) -> dict[str, Any]:
        config = self._config
        return {
            "text": text,
            "text_lang": config.text_lang,
            # 由 server 讀取；api_v2 沒有預設參考音檔，每次都要帶
            "ref_audio_path": self._ref_audio_path,
            "prompt_text": config.prompt_text,
            "prompt_lang": config.prompt_lang,
            "text_split_method": config.text_split_method,
            "speed_factor": config.speed_factor,
            "fragment_interval": config.fragment_interval,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "temperature": config.temperature,
            # 串流模式回傳的是 data 長度為 0 的檔頭加上 raw PCM，不是標準 WAV，所以每句都明確關掉
            "media_type": "wav",
            "streaming_mode": False,
        }


class GPTSoVITSV1Client(_GPTSoVITSClient):
    """呼叫舊版 GPT-SoVITS 的 api.py（POST JSON 到根路徑，回傳完整 WAV）；把設定對應到 v1 的欄位名稱。"""

    _not_wav_hint = "start api.py with the default stream mode (-sm close) and media type (-mt wav)"

    def build_payload(self, text: str) -> dict[str, Any]:
        config = self._config
        payload = {"text": text, "text_language": config.text_lang}
        if self._ref_audio_path:
            # 留空則改用 server 啟動時 -dr/-dt/-dl 指定的預設參考音檔
            payload.update(
                refer_wav_path=self._ref_audio_path,
                prompt_text=config.prompt_text,
                prompt_language=config.prompt_lang,
            )
        if config.cut_punc is not None:
            payload["cut_punc"] = config.cut_punc
        return payload


def create_tts(config: TTSConfig) -> TextToSpeech:
    """依 tts.engine 建立 TTS 實作；新增引擎時在這裡加一個分支。"""
    if config.engine == "gpt-sovits-v2":
        return GPTSoVITSV2Client(config)
    if config.engine == "gpt-sovits-v1":
        return GPTSoVITSV1Client(config)
    raise ConfigError(f"Unsupported TTS engine: {config.engine!r}")


def _check_not_silent(audio: bytes) -> None:
    """api_v2 推論中途出錯時不回錯誤碼，而是回 HTTP 200 加上一段全靜音的 WAV（1 秒、16 kHz），
    純標點的文字也一樣；所以全部 sample 都是 0（或根本沒有聲音）時視為失敗。"""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            width, rate = wav.getsampwidth(), wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise TTSError(f"TTS response is not a playable WAV file: {exc}") from exc
    silence = b"\x80" if width == 1 else b"\x00"  # 8-bit PCM 的靜音是 128
    if frames.count(silence) == len(frames):
        raise TTSError(
            f"TTS server returned {_duration(audio):.2f}s of silence ({rate} Hz) instead of speech: synthesis failed "
            "on the server (see the TTS server log for the error) or the text has nothing to pronounce"
        )


def _duration(audio: bytes) -> float:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        return wav.getnframes() / wav.getframerate() if wav.getframerate() else 0.0


def _error_detail(response: httpx.Response) -> str:
    """把錯誤回應整理成一行：api_v2 的 message 與 Exception、FastAPI 型別錯誤（422）的 detail，或原始內容。"""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200].strip() or "<empty body>"
    if isinstance(payload, dict):
        parts = []
        if payload.get("message"):
            parts.append(str(payload["message"]))
        if "Exception" in payload:
            # 例如 assert 失敗時 Exception 是空字串，細節只印在 server 的 log
            parts.append(str(payload["Exception"]) or "no details; see the TTS server log")
        if payload.get("detail"):
            parts.append(_format_validation_detail(payload["detail"]))
        if parts:
            return ": ".join(parts)[:200]
    return json.dumps(payload, ensure_ascii=False)[:200]


def _format_validation_detail(detail: Any) -> str:
    """FastAPI 的 422 detail：[{"loc": ["body", "top_k"], "msg": "..."}] → "body.top_k: ..."。"""
    if not isinstance(detail, list):
        return str(detail)
    messages = []
    for item in detail:
        if isinstance(item, dict):
            where = ".".join(str(part) for part in item.get("loc") or ())
            message = str(item.get("msg", item))
            messages.append(f"{where}: {message}" if where else message)
        else:
            messages.append(str(item))
    return "; ".join(messages)


def _is_absolute_anywhere(path: str) -> bool:
    """POSIX（/home/...）或 Windows（C:\\...、C:/...、\\\\server\\share）寫法的絕對路徑。"""
    return PurePosixPath(path).is_absolute() or bool(PureWindowsPath(path).drive)


def _is_local_url(url: str) -> bool:
    """TTS server 是否在這台電腦（localhost 或 loopback 位址）。"""
    try:
        host = httpx.URL(url).host
    except _INVALID_URL_ERRORS:
        return False
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified
