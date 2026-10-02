from __future__ import annotations

import asyncio
import logging

import pytest

from vtuber.errors import TTSError
from vtuber.pipeline import AUDIO_QUEUE_SIZE, ReplyTurn, speak_reply
from vtuber.sentences import SentenceSplitter

# Windows 上 time.monotonic()（event loop 的時鐘）的解析度約 15.6 ms；時間的下限要留這個容差
TICK = 0.016


class Timeline:
    """記錄每個事件（名稱、句子、event loop 時間），用來斷言順序與重疊。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, float]] = []

    def mark(self, name: str, text: str = "") -> None:
        self.events.append((name, text, asyncio.get_running_loop().time()))

    def time_of(self, name: str, text: str = "") -> float:
        return next(time for event, label, time in self.events if event == name and label == text)

    def names(self, name: str) -> list[str]:
        return [label for event, label, _ in self.events if event == name]


class FakeLLM:
    """依序吐出 parts，每段之間等 delay 秒；記錄串流是否被關閉。"""

    def __init__(self, timeline: Timeline, parts, delay: float = 0.0) -> None:
        self.timeline = timeline
        self.parts = list(parts)
        self.delay = delay
        self.closed = False
        # 保留產生過的串流：確保是管線自己關閉它，而不是垃圾回收順便關掉
        self.streams: list = []

    def stream_reply(self, prompt: str):
        stream = self._stream()
        self.streams.append(stream)
        return stream

    async def _stream(self):
        try:
            for part in self.parts:
                await asyncio.sleep(self.delay)
                yield part
            self.timeline.mark("llm_done")
        finally:
            self.closed = True


class FakeTTS:
    """合成一句要 seconds 秒；回傳的「音訊」就是文字本身，方便斷言。"""

    def __init__(self, timeline: Timeline, seconds=0.0, fail_texts=()) -> None:
        self.timeline = timeline
        self.seconds = seconds
        self.fail_texts = set(fail_texts)
        self.texts: list[str] = []
        self.cancelled = False

    async def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        self.timeline.mark("synth_start", text)
        try:
            await asyncio.sleep(self.seconds(text) if callable(self.seconds) else self.seconds)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if text in self.fail_texts:
            raise TTSError(f"server could not synthesize {text!r}")
        self.timeline.mark("synth_end", text)
        return text.encode()


class FakePlayer:
    """播放一句要 seconds 秒（開始輸出前先等 warmup 秒）；stuck=True 時第一句永遠播不完（直到被取消）。"""

    def __init__(self, timeline: Timeline, seconds: float = 0.0, stuck: bool = False, warmup: float = 0.0) -> None:
        self.timeline = timeline
        self.seconds = seconds
        self.stuck = stuck
        self.warmup = warmup
        self.played: list[str] = []
        self.cancelled = False

    async def play(self, audio: bytes, *, on_start=None) -> None:
        text = audio.decode()
        try:
            await asyncio.sleep(self.warmup)  # 例如第一次開啟音訊裝置
            self.timeline.mark("play_start", text)
            if on_start is not None:
                on_start()
            if self.stuck:
                await asyncio.Event().wait()
            await asyncio.sleep(self.seconds)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.played.append(text)
        self.timeline.mark("play_end", text)


SENTENCES = ["First sentence here.", "Second one is a bit longer.", "Third.", "And the fourth one ends it."]


def _speak(llm, tts, player, *, min_chars: int = 0, max_chars: int = 200, **kwargs):
    splitter = SentenceSplitter(min_chars=min_chars, max_chars=max_chars)
    return speak_reply("Hi", llm, tts, player, splitter, **kwargs)


def _run(coroutine, timeout: float = 10.0):
    """實作若有問題卡住，在這裡就失敗，而不是讓整個測試卡住。"""
    return asyncio.run(asyncio.wait_for(coroutine, timeout))


async def _cancel(task: asyncio.Task) -> None:
    """取消 task，並確認它在 5 秒內以 CancelledError 結束（卡住時會變成 TimeoutError，測試失敗）。"""
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)


def _tokens(text: str) -> list[str]:
    """把文字切成像 LLM token 的小段（每段 3 個字元）。"""
    return [text[index : index + 3] for index in range(0, len(text), 3)]


def test_sentences_are_played_in_order_even_when_synthesis_times_differ():
    timeline = Timeline()
    durations = {SENTENCES[0]: 0.03, SENTENCES[1]: 0.0, SENTENCES[2]: 0.02, SENTENCES[3]: 0.0}
    llm = FakeLLM(timeline, _tokens(" ".join(SENTENCES)))
    tts = FakeTTS(timeline, seconds=durations.__getitem__)
    player = FakePlayer(timeline, seconds=0.01)

    result = _run(_speak(llm, tts, player))

    assert tts.texts == SENTENCES
    assert player.played == SENTENCES
    assert (result.sentences, result.spoken, result.failed) == (4, 4, 0)
    assert result.text == " ".join(SENTENCES)


def test_next_sentence_is_synthesized_while_the_current_one_is_playing():
    timeline = Timeline()
    llm = FakeLLM(timeline, _tokens(" ".join(SENTENCES)))
    player = FakePlayer(timeline, seconds=0.08)

    _run(_speak(llm, FakeTTS(timeline, seconds=0.02), player))

    for current, following in zip(SENTENCES, SENTENCES[1:]):
        # 第 N+1 句在第 N 句播完之前就開始合成，而且在第 N 句播完之前就合成好了
        assert timeline.time_of("synth_start", following) < timeline.time_of("play_end", current)
        assert timeline.time_of("synth_end", following) < timeline.time_of("play_end", current)
        # 播放仍然一句接一句，不會重疊
        assert timeline.time_of("play_end", current) <= timeline.time_of("play_start", following)


def test_first_audio_starts_before_the_llm_has_finished():
    timeline = Timeline()
    llm = FakeLLM(timeline, [f"{sentence} " for sentence in SENTENCES], delay=0.05)

    result = _run(_speak(llm, FakeTTS(timeline, seconds=0.01), FakePlayer(timeline, seconds=0.01)))

    first_audio = timeline.time_of("play_start", SENTENCES[0])
    assert first_audio < timeline.time_of("llm_done")  # 兩者相差約 0.14 秒
    assert result.first_sentence_latency <= result.first_audio_latency < 0.15
    assert result.duration >= 0.2 - TICK  # LLM 本身要 4 × 0.05 秒


def test_first_audio_is_marked_when_output_starts_not_when_play_is_called_or_finishes():
    timeline = Timeline()
    llm = FakeLLM(timeline, ["Hello there."])
    # 開始輸出前要 0.1 秒（例如第一次開啟音訊裝置），播放本身 0.3 秒
    player = FakePlayer(timeline, seconds=0.3, warmup=0.1)

    result = _run(_speak(llm, FakeTTS(timeline), player))

    assert 0.1 - TICK <= result.first_audio_latency < 0.3


def test_latency_is_measured_from_started_at():
    async def run():
        timeline = Timeline()
        started_at = asyncio.get_running_loop().time() - 1.0  # 一秒前辨識出使用者的話
        llm = FakeLLM(timeline, ["Hello there."])
        return await _speak(llm, FakeTTS(timeline), FakePlayer(timeline), started_at=started_at)

    result = _run(run())

    assert 1.0 <= result.first_sentence_latency <= result.first_audio_latency < 1.5


def test_a_failed_sentence_is_skipped_and_the_rest_are_played(caplog):
    timeline = Timeline()
    llm = FakeLLM(timeline, _tokens(" ".join(SENTENCES)))
    tts = FakeTTS(timeline, fail_texts={SENTENCES[1]})
    player = FakePlayer(timeline)

    with caplog.at_level(logging.ERROR, logger="vtuber.pipeline"):
        result = _run(_speak(llm, tts, player))

    assert tts.texts == SENTENCES  # 失敗後繼續合成下一句
    assert player.played == [SENTENCES[0], SENTENCES[2], SENTENCES[3]]
    assert (result.sentences, result.spoken, result.failed) == (4, 3, 1)
    assert "Skipping sentence 2" in caplog.text and "could not synthesize" in caplog.text


def test_markdown_is_cleaned_and_unpronounceable_pieces_are_not_sent_to_tts():
    timeline = Timeline()
    text = "😊 ...\n**Hello there!** Read [the guide](https://example.com/guide).\n---\n🎉🎉"
    llm = FakeLLM(timeline, _tokens(text))
    tts = FakeTTS(timeline)

    result = _run(_speak(llm, tts, FakePlayer(timeline)))

    assert tts.texts == ["Hello there!", "Read the guide."]
    assert result.text == text.strip()  # 記錄的完整回覆是原始文字


def test_numbers_survive_the_whole_pipeline():
    timeline = Timeline()
    text = "What's 50 times 2? 100. Easy, right? Count down with me. 3. 2. 1. Go!\n- 5 is a bullet."
    tts = FakeTTS(timeline)

    _run(_speak(FakeLLM(timeline, _tokens(text)), tts, FakePlayer(timeline), min_chars=12, max_chars=120))

    assert tts.texts == [
        "What's 50 times 2?",
        "100. Easy, right?",
        "Count down with me.",
        "3. 2. 1. Go!",
        "5 is a bullet.",  # 真正行首的 "- " 是條列符號
    ]


def test_whatever_the_llm_produced_before_stopping_is_spoken():
    # GroqChat 串流中途出錯時會停止輸出；最後一句即使沒有句尾也要念出來
    timeline = Timeline()
    llm = FakeLLM(timeline, ["Here is the first part. And then the", " connection dro"])
    tts = FakeTTS(timeline)

    _run(_speak(llm, tts, FakePlayer(timeline)))

    assert tts.texts == ["Here is the first part.", "And then the connection dro"]


def test_audio_queue_is_bounded():
    timeline = Timeline()
    llm = FakeLLM(timeline, [f"Sentence number {index}. " for index in range(10)])
    tts = FakeTTS(timeline)
    player = FakePlayer(timeline, stuck=True)  # 第一句一直播不完

    async def run() -> None:
        task = asyncio.ensure_future(_speak(llm, tts, player))
        await asyncio.sleep(0.1)
        await _cancel(task)

    _run(run())

    # 播放中 1 句 + 佇列裡 AUDIO_QUEUE_SIZE 句 + 合成好卻等著放進佇列的 1 句
    assert len(tts.texts) == 1 + AUDIO_QUEUE_SIZE + 1


def test_cancelling_stops_everything_and_leaves_no_tasks_behind():
    timeline = Timeline()
    llm = FakeLLM(timeline, [f"{sentence} " for sentence in SENTENCES * 3], delay=0.03)
    tts = FakeTTS(timeline, seconds=0.05)
    player = FakePlayer(timeline, seconds=10)

    async def run() -> set:
        task = asyncio.ensure_future(_speak(llm, tts, player))
        # 等到第一句開始播放、第二句正在合成（LLM 也還在串流）時再取消
        deadline = asyncio.get_running_loop().time() + 5
        while not (timeline.names("play_start") and len(timeline.names("synth_start")) >= 2):
            assert asyncio.get_running_loop().time() < deadline, "playback never started"
            await asyncio.sleep(0.005)
        await _cancel(task)  # 例如 Ctrl+C
        return asyncio.all_tasks() - {asyncio.current_task()}

    leftover = asyncio.run(run())  # 不用 _run：外層的 wait_for 本身也是一個 task

    assert leftover == set()
    assert player.cancelled  # 播放立刻停止
    assert tts.cancelled  # 正在合成的下一句也取消了
    assert llm.closed  # LLM 串流已關閉
    assert "llm_done" not in [name for name, _, _ in timeline.events]  # LLM 還沒說完就被中斷
    assert player.played == []


class BrokenPlayer(FakePlayer):
    """播到第 fail_at 句時丟出錯誤（例如播放裝置被拔掉）。"""

    def __init__(self, timeline: Timeline, fail_at: int = 1) -> None:
        super().__init__(timeline)
        self.fail_at = fail_at
        self.calls = 0

    async def play(self, audio: bytes, *, on_start=None) -> None:
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("audio device busy")
        await super().play(audio, on_start=on_start)


def test_player_error_cancels_the_rest_of_the_turn():
    timeline = Timeline()
    llm = FakeLLM(timeline, [f"{sentence} " for sentence in SENTENCES], delay=0.05)
    tts = FakeTTS(timeline, seconds=0.01)

    async def run() -> set:
        with pytest.raises(RuntimeError, match="audio device busy"):
            await asyncio.wait_for(_speak(llm, tts, BrokenPlayer(timeline)), 5)
        return asyncio.all_tasks() - {asyncio.current_task()}

    assert asyncio.run(run()) == set()
    assert llm.closed and len(tts.texts) < len(SENTENCES)


def test_progress_after_an_error_counts_only_sentences_that_finished_playing():
    timeline = Timeline()
    llm = FakeLLM(timeline, _tokens(" ".join(SENTENCES)))

    async def run():
        splitter = SentenceSplitter(min_chars=0, max_chars=200)
        turn = ReplyTurn(llm, FakeTTS(timeline), BrokenPlayer(timeline, fail_at=2), splitter)
        with pytest.raises(RuntimeError, match="audio device busy"):
            await turn.run("Hi")
        return turn.result()

    partial = _run(run())

    assert partial.spoken == 1  # 第二句沒播出來，不算
    assert partial.first_audio_latency is not None
    assert partial.text.startswith(SENTENCES[0])


def test_llm_stream_is_closed_right_away_when_the_turn_fails_inside_the_producer():
    class ExplodingSplitter(SentenceSplitter):
        def feed(self, text: str):
            raise RuntimeError("splitter bug")

    timeline = Timeline()
    llm = FakeLLM(timeline, ["First part. ", "Second part."])

    async def run() -> bool:
        splitter = ExplodingSplitter(min_chars=0, max_chars=100)
        with pytest.raises(RuntimeError, match="splitter bug"):
            await speak_reply("Hi", llm, FakeTTS(timeline), FakePlayer(timeline), splitter)
        # 串流停在 yield 時出錯：管線要自己關閉它，不能等到 event loop 結束時才被收拾
        return llm.closed

    assert _run(run())


def test_empty_reply_finishes_without_audio():
    timeline = Timeline()

    result = _run(_speak(FakeLLM(timeline, []), FakeTTS(timeline), FakePlayer(timeline)))

    assert (result.text, result.sentences, result.spoken) == ("", 0, 0)
    assert result.first_sentence_latency is None and result.first_audio_latency is None
