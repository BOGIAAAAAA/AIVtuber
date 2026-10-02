"""聊天主迴圈：聽 → LLM 串流回覆 → 按句合成 → 邊合成邊播放（管線見 pipeline.py）。

迴圈只依賴下面的介面（Protocol），元件由呼叫端注入，測試時可以換成假物件。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterable
from typing import Optional, Protocol

from vtuber.audio import AudioPlayer
from vtuber.config import AppConfig, ChatConfig
from vtuber.errors import VTuberError
from vtuber.pipeline import ReplyTurn, StreamingReplyGenerator, TurnResult
from vtuber.sentences import SentenceSplitter
from vtuber.tts import TextToSpeech

logger = logging.getLogger(__name__)

_TRAILING_PUNCTUATION = " \t.,!?;:…。，！？；：、~"


class SpeechListener(Protocol):
    async def listen(self) -> Optional[str]:
        """聽一句話；沒有可用的文字（沒人說話、聽不懂）時回傳 None。"""
        ...


def _normalize_command(text: str) -> str:
    return text.strip().rstrip(_TRAILING_PUNCTUATION).casefold()


def is_exit_command(text: str, exit_commands: Iterable[str]) -> bool:
    """忽略大小寫、前後空白與句尾標點判斷是否為離開指令（例如 "Exit." 也算）；設定值也做同樣處理。"""
    return _normalize_command(text) in {_normalize_command(command) for command in exit_commands}


async def respond(
    user_text: str,
    llm: StreamingReplyGenerator,
    tts: TextToSpeech,
    player: AudioPlayer,
    splitter: SentenceSplitter,
    *,
    started_at: Optional[float] = None,
) -> TurnResult:
    """處理一輪對話：串流產生回覆、邊合成邊播放；結束後記錄完整回覆與延遲。

    延遲從 started_at 算起；「開始出聲」是第一段聲音開始寫入輸出裝置的時間（不含裝置本身的緩衝延遲）。
    """
    turn = ReplyTurn(llm, tts, player, splitter, started_at=started_at)
    try:
        result = await turn.run(user_text)
    except Exception:
        # 例如播放裝置失效：仍把已經產生的回覆記下來（Ctrl+C 的取消不是 Exception，不會記錄）
        partial = turn.result()
        if partial.text:
            logger.info(
                "AI response (interrupted after %d of %d sentence(s)): %s",
                partial.spoken,
                partial.sentences,
                partial.text,
            )
        raise
    logger.info("AI response: %s", result.text)
    logger.info(
        "Latency after recognition: first sentence %s, first audio %s; %d of %d sentence(s) spoken%s",
        _seconds(result.first_sentence_latency),
        _seconds(result.first_audio_latency),
        result.spoken,
        result.sentences,
        f", {result.failed} skipped (TTS failed)" if result.failed else "",
    )
    return result


def _seconds(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


async def run_chat_loop(
    listener: SpeechListener,
    llm: StreamingReplyGenerator,
    tts: TextToSpeech,
    player: AudioPlayer,
    *,
    exit_commands: Iterable[str] = ChatConfig.exit_commands,
    min_sentence_chars: int = ChatConfig.min_sentence_chars,
    max_sentence_chars: int = ChatConfig.max_sentence_chars,
    retry_delay: float = 1.0,
) -> None:
    """持續對話直到聽到離開指令；單輪出錯只記錄並進入下一輪，不會讓程式結束。"""
    commands = tuple(exit_commands)
    logger.info("Entering interactive mode. Say %s to quit.", " / ".join(repr(command) for command in commands))
    debug = logger.isEnabledFor(logging.DEBUG)
    loop = asyncio.get_running_loop()
    while True:
        try:
            user_text = await listener.listen()
        except Exception as exc:
            # 例如麥克風被拔掉；稍等再試，避免錯誤時空轉狂刷 log
            logger.error("Speech input failed: %s; retrying in %.0fs", exc, retry_delay, exc_info=debug)
            await asyncio.sleep(retry_delay)
            continue
        # 延遲從辨識出使用者的話開始算（辨識本身的時間不在內）
        heard_at = loop.time()
        if not user_text:
            continue
        if is_exit_command(user_text, commands):
            logger.info("Exit command received; leaving interactive mode")
            return
        splitter = SentenceSplitter(min_chars=min_sentence_chars, max_chars=max_sentence_chars)
        try:
            await respond(user_text, llm, tts, player, splitter, started_at=heard_at)
        except VTuberError as exc:
            logger.error("Could not answer this turn: %s", exc, exc_info=debug)
        except Exception:
            logger.exception("Unexpected error while answering this turn; continuing")


async def run_chat(config: AppConfig) -> None:
    """建立實際元件並開始聊天；結束時（包括 Ctrl+C 與啟動失敗）一定關閉麥克風、播放裝置與 HTTP client。"""
    from vtuber.asr import SpeechRecognizer
    from vtuber.audio import PyAudioPlayer
    from vtuber.llm import GroqChat
    from vtuber.tts import create_tts

    async with contextlib.AsyncExitStack() as cleanup:
        listener = SpeechRecognizer(config.asr)
        cleanup.push_async_callback(listener.aclose)
        tts = create_tts(config.tts)
        cleanup.push_async_callback(tts.aclose)
        # 等 TTS server 就緒並暖機；server 沒開或設定錯誤時在這裡就丟出 TTSError，
        # 程式以清楚的訊息結束，而不是到第一輪對話才失敗
        await tts.prepare()
        llm = GroqChat(config.llm)
        cleanup.push_async_callback(llm.aclose)
        player = PyAudioPlayer(config.audio.output_device_index)
        cleanup.push_async_callback(player.aclose)
        await listener.calibrate()
        chat = config.chat
        await run_chat_loop(
            listener,
            llm,
            tts,
            player,
            exit_commands=chat.exit_commands,
            min_sentence_chars=chat.min_sentence_chars,
            max_sentence_chars=chat.max_sentence_chars,
        )
