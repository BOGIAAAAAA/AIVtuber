"""串流語音回覆管線：LLM 串流 → 按句切分與清理 → 逐句合成 → 依序播放。

    LLM 串流 ─► 切句、清理 ─► 文字佇列 ─► 合成（一次一句）─► 音訊佇列（有上限）─► 播放

三個 task 以佇列串接，播放第 N 句的同時就在合成第 N+1 句，使用者說完話到 AI 開始出聲的時間
從「整段 LLM + 整段 TTS」縮短成「第一句 LLM + 第一句 TTS」。合成一次只送一句
（GPT-SoVITS server 本來就一次只處理一個請求），所以播放順序一定和句子順序相同。

- 某一句合成失敗（TTSError）只記錄並跳過，繼續下一句。
- 其他錯誤（例如播放裝置失效）或被取消（Ctrl+C）時，先取消所有 task、等它們結束（播放會立刻停止、
  LLM 串流與 HTTP 連線會關閉），再把例外往外丟。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from vtuber.audio import AudioPlayer
from vtuber.errors import TTSError
from vtuber.sentences import Sentence, SentenceSplitter, clean_for_speech
from vtuber.tts import TextToSpeech

logger = logging.getLogger(__name__)

# 音訊佇列的上限：合成最多領先播放這麼多句（另外還有正在合成的一句），避免中斷時白白合成太多
AUDIO_QUEUE_SIZE = 2


class StreamingReplyGenerator(Protocol):
    def stream_reply(self, prompt: str) -> AsyncIterator[str]:
        """逐段產生要念出來的回覆文字；錯誤自行處理（例如改輸出備援訊息），不丟例外。"""
        ...


@dataclass(frozen=True)
class TurnResult:
    """一輪回覆的結果。延遲的單位是秒、從 started_at 算起；沒有發生時為 None。"""

    text: str
    sentences: int  # 送去合成的句數
    spoken: int  # 播完的句數
    failed: int  # 合成失敗、被跳過的句數
    first_sentence_latency: Optional[float]
    first_audio_latency: Optional[float]  # 第一段聲音開始寫入輸出裝置的時間
    duration: float


async def speak_reply(
    prompt: str,
    llm: StreamingReplyGenerator,
    tts: TextToSpeech,
    player: AudioPlayer,
    splitter: SentenceSplitter,
    *,
    started_at: Optional[float] = None,
    audio_queue_size: int = AUDIO_QUEUE_SIZE,
) -> TurnResult:
    """串流產生回覆並邊合成邊播放，全部播完才返回。參數見 ReplyTurn。"""
    turn = ReplyTurn(llm, tts, player, splitter, started_at=started_at, audio_queue_size=audio_queue_size)
    return await turn.run(prompt)


class ReplyTurn:
    """一輪語音回覆的三個 task 與共用的狀態（只在 event loop 的執行緒上使用）。

    started_at 是計算延遲的起點（event loop 的 time()，例如辨識出使用者說的話的時間），預設為建立的時間。
    run() 正常結束、出錯或被取消之後，都可以用 result() 看已經完成的部分。
    """

    def __init__(
        self,
        llm: StreamingReplyGenerator,
        tts: TextToSpeech,
        player: AudioPlayer,
        splitter: SentenceSplitter,
        *,
        started_at: Optional[float] = None,
        audio_queue_size: int = AUDIO_QUEUE_SIZE,
    ) -> None:
        self._llm, self._tts, self._player, self._splitter = llm, tts, player, splitter
        self._loop = asyncio.get_running_loop()
        self._start = self._loop.time() if started_at is None else started_at
        # None 表示上游已經結束
        self._texts: asyncio.Queue = asyncio.Queue()
        self._audios: asyncio.Queue = asyncio.Queue(maxsize=audio_queue_size)
        self._parts: list[str] = []
        self._sentences = self._spoken = self._failed = 0
        self._first_sentence: Optional[float] = None
        self._first_audio: Optional[float] = None

    async def run(self, prompt: str) -> TurnResult:
        await _run_together(self._produce(prompt), self._synthesize(), self._play())
        return self.result()

    def result(self) -> TurnResult:
        return TurnResult(
            text="".join(self._parts).strip(),
            sentences=self._sentences,
            spoken=self._spoken,
            failed=self._failed,
            first_sentence_latency=self._first_sentence,
            first_audio_latency=self._first_audio,
            duration=self._elapsed(),
        )

    async def _produce(self, prompt: str) -> None:
        """LLM 串流 → 切句 → 清理 → 文字佇列。"""
        stream = self._llm.stream_reply(prompt)
        try:
            async for part in stream:
                self._parts.append(part)
                for sentence in self._splitter.feed(part):
                    self._enqueue(sentence)
        finally:
            # 自己出錯時串流還停在 yield；明確關閉才會立刻釋放 HTTP 連線（例外的 traceback 會讓它一直被參考著）
            await _aclose(stream)
        for sentence in self._splitter.flush():
            self._enqueue(sentence)
        self._texts.put_nowait(None)

    def _enqueue(self, sentence: Sentence) -> None:
        text = clean_for_speech(sentence.text, line_start=sentence.line_start)
        if not text:
            logger.debug("Nothing to pronounce in %r; skipped", sentence.text)
            return
        if self._first_sentence is None:
            self._first_sentence = self._elapsed()
        self._sentences += 1
        logger.debug("Sentence %d ready after %.2fs: %s", self._sentences, self._elapsed(), text)
        self._texts.put_nowait(text)

    async def _synthesize(self) -> None:
        """一次合成一句；音訊佇列滿了就等播放趕上。"""
        number = 0
        while True:
            text = await self._texts.get()
            if text is None:
                break
            number += 1
            try:
                audio = await self._tts.synthesize(text)
            except TTSError as exc:
                self._failed += 1
                logger.error("Skipping sentence %d because TTS failed: %s (text: %r)", number, exc, text)
                continue
            await self._audios.put(audio)
        await self._audios.put(None)

    async def _play(self) -> None:
        """依序播放，播完一句才拿下一句。"""
        while True:
            audio = await self._audios.get()
            if audio is None:
                return
            await self._player.play(audio, on_start=self._audio_started)
            self._spoken += 1

    def _audio_started(self) -> None:
        # 由播放器在聲音真正開始輸出時呼叫，所以包含第一次開啟音訊裝置的時間
        if self._first_audio is None:
            self._first_audio = self._elapsed()

    def _elapsed(self) -> float:
        return self._loop.time() - self._start


async def _run_together(*coroutines: Coroutine[Any, Any, None]) -> None:
    """同時執行並等全部完成。任一個失敗或被取消、或自己被取消時，先取消其餘並等它們結束，再往外丟。

    Python 3.9 沒有 asyncio.TaskGroup，這裡做同樣的事：不留下還在跑的 task，例外也都有人接，
    不會在結束時印出 "Task exception was never retrieved"。
    """
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    try:
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.cancelled():
                    raise asyncio.CancelledError
                error = task.exception()
                if error is not None:
                    raise error
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _aclose(stream: Any) -> None:
    """關閉 async generator（例如 LLM 串流），讓它內部的 HTTP 連線立刻釋放，而不是等垃圾回收。"""
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()
