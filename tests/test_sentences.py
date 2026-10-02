from __future__ import annotations

import random
import time

import pytest

from vtuber.sentences import Sentence, SentenceSplitter, clean_for_speech


def split_sentences(chunks, *, min_chars: int = 0, max_chars: int = 1000) -> list[Sentence]:
    """依序 feed 每一段，最後 flush，回傳所有 Sentence。"""
    splitter = SentenceSplitter(min_chars=min_chars, max_chars=max_chars)
    sentences = []
    for chunk in [chunks] if isinstance(chunks, str) else chunks:
        sentences += splitter.feed(chunk)
    return sentences + splitter.flush()


def split(chunks, **limits) -> list[str]:
    """同 split_sentences，只回傳文字。"""
    return texts(split_sentences(chunks, **limits))


def texts(sentences) -> list[str]:
    return [sentence.text for sentence in sentences]


def speak(text: str, *, min_chars: int = 12, max_chars: int = 120) -> list[str]:
    """切句加清理，回傳實際會送去合成的文字（與管線相同；預設長度同 chat 的預設值）。"""
    sentences = split_sentences(text, min_chars=min_chars, max_chars=max_chars)
    spoken = [clean_for_speech(sentence.text, line_start=sentence.line_start) for sentence in sentences]
    return [line for line in spoken if line]


# ---- 句尾 ----


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello there. How are you? I'm fine!", ["Hello there.", "How are you?", "I'm fine!"]),
        ("Really?! Yes!!! Sure…  Fine.", ["Really?!", "Yes!!!", "Sure…", "Fine."]),
        ("你好。我是小明！你呢？好的", ["你好。", "我是小明！", "你呢？", "好的"]),
        ("OK!我們走吧。", ["OK!", "我們走吧。"]),  # 英文句尾直接接中文
        ("First line\nsecond line\n\nThird.", ["First line", "second line", "Third."]),  # 換行一定切
        ("Done.\nnext line", ["Done.", "next line"]),
        ("No trailing punctuation", ["No trailing punctuation"]),
        ("", []),
        ("   \n  ", []),
    ],
)
def test_sentence_boundaries(text, expected):
    assert split(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Mr. Smith met Dr. Jones today.",
        "Ask Prof. Lee or St. John about it.",
        "Bring fruit, e.g. Apples or Pears.",
        "Pick one, i.e. The red one.",
        "We need eggs, milk, etc. And bread.",  # etc. 後面接大寫也不切（寧可合併，不要切錯）
        "Call me at 5 a.m. Tomorrow works.",
        "She moved to the U.S. Last year.",
        "J. K. Rowling wrote it.",
        "He has a Ph.D. In physics.",
    ],
)
def test_abbreviations_do_not_end_a_sentence(text):
    assert split(text) == [text]


@pytest.mark.parametrize(
    "text",
    [
        "Pi is about 3.14 or so.",
        "It costs 1,000 dollars.",
        "Version 2.0.1 is out.",
        "Visit example.com for details.",
        "Read the docs, e.g., the manual.",
        "Hmm... let me think about it.",  # 省略號後面是小寫：同一句
        'She asked "why?" and left.',  # 引號裡的問號後面是小寫：同一句
        '"Wow!" she said.',
        "價格是１．５元，版本是２．０。",  # 全形小數點夾在數字之間
        "１．５",
    ],
)
def test_numbers_ellipses_and_lowercase_continuations_are_not_split(text):
    assert split(text) == [text]


def test_full_width_full_stop_still_ends_a_sentence_after_text():
    assert split("今天很好．明天也好．２．０版．") == ["今天很好．", "明天也好．", "２．０版．"]


def test_so_did_i_still_ends_a_sentence():
    # 單一大寫字母的名字縮寫規則不包含常見的 I、A
    assert split("So did I. Then we left. I got an A. Great.") == [
        "So did I.",
        "Then we left.",
        "I got an A.",
        "Great.",
    ]


def test_ellipsis_is_one_boundary():
    assert split("Wait... What? I don't know… Maybe.") == ["Wait...", "What?", "I don't know…", "Maybe."]


def test_closing_quotes_and_brackets_stay_with_their_sentence():
    assert split('He said "Hello!" Then he left. (This is great!) **Done.** Next.') == [
        'He said "Hello!"',
        "Then he left.",
        "(This is great!)",
        "**Done.**",
        "Next.",
    ]
    assert split("「你好。」他說。") == ["「你好。」", "他說。"]


# ---- 編號與數字（寧可多念，不可漏念）----


def test_numbered_list_markers_are_not_sentence_ends():
    assert split("Here are tips:\n1. Drink water.\n2. Sleep well.") == [
        "Here are tips:",
        "1. Drink water.",
        "2. Sleep well.",
    ]
    assert split("I have 3. You have 4.") == ["I have 3.", "You have 4."]  # 一般位置的數字照常斷句
    assert split("**Steps:** 1. Open it. 2. Close it.") == ["**Steps:** 1. Open it.", "2. Close it."]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What's 50 times 2? 100. Easy, right?", ["What's 50 times 2?", "100. Easy, right?"]),
        (
            "Good question! 42. That's the answer to everything.",
            ["Good question!", "42. That's the answer to everything."],
        ),
        ("Count down with me. 3. 2. 1. Go!", ["Count down with me.", "3. 2. 1. Go!"]),
        (
            "Here are some tips: 1. Practice daily. 2. Read aloud. 3. Use apps.",
            ["Here are some tips: 1. Practice daily.", "2. Read aloud.", "3. Use apps."],
        ),
        ("It is really cold today. - 5 degrees outside.", ["It is really cold today.", "- 5 degrees outside."]),
    ],
)
def test_numbers_are_never_dropped(text, expected):
    # 句子開頭的 "100."、"- 5" 不在真正的行首，不是條列記號，要原樣念出來
    assert speak(text) == expected


def test_line_start_is_only_true_at_the_start_of_the_reply_or_after_a_newline():
    sentences = split_sentences("Hello there friend. Next one here.\n- Item one is here. Then more text.")

    assert [(sentence.text, sentence.line_start) for sentence in sentences] == [
        ("Hello there friend.", True),
        ("Next one here.", False),
        ("- Item one is here.", True),
        ("Then more text.", False),
    ]
    assert speak("Hello there friend. Next one here.\n- Item one is here.", min_chars=0) == [
        "Hello there friend.",
        "Next one here.",
        "Item one is here.",  # 真正行首的條列符號才會被清掉
    ]


# ---- 需要往後看的時候先等 ----


def test_undecided_boundaries_wait_for_more_text():
    splitter = SentenceSplitter(min_chars=0, max_chars=1000)

    assert splitter.feed("Pi is 3.") == []  # 可能是 3.14
    assert texts(splitter.feed("14. Next")) == ["Pi is 3.14."]
    assert texts(splitter.feed(" one. Mr.")) == ["Next one."]
    assert splitter.feed(" ") == []  # Mr. 後面還不知道是誰
    assert splitter.feed("Smith left. ") == []  # left. 後面還不知道是不是小寫
    assert splitter.feed("and") == []
    assert splitter.feed(" came back.") == []  # 句點後面還沒有字
    assert texts(splitter.flush()) == ["Mr. Smith left. and came back."]


def test_cjk_ending_waits_only_for_a_possible_closing_quote():
    splitter = SentenceSplitter(min_chars=0, max_chars=1000)

    assert splitter.feed("你好。") == []  # 後面可能是 」
    assert texts(splitter.feed("我")) == ["你好。"]
    assert splitter.feed("很好！」") == []
    assert texts(splitter.feed("嗯")) == ["我很好！」"]


def test_newline_after_an_ending_is_decided_without_the_next_word():
    splitter = SentenceSplitter(min_chars=0, max_chars=1000)

    assert splitter.feed("Hello.\n") == [Sentence("Hello.", True)]


def test_a_long_gap_or_run_of_marks_after_an_ending_is_decided_without_waiting():
    splitter = SentenceSplitter(min_chars=0, max_chars=1000)

    assert splitter.feed("Hello." + " " * 15) == []  # 空白還不夠長，可能接著小寫字
    assert texts(splitter.feed(" ")) == ["Hello."]  # 空白夠長就直接算句尾
    assert texts(splitter.feed("Wow" + "!" * 20)) == ["Wow" + "!" * 16]  # 一長串標點最多算 16 個
    assert splitter.flush() == [Sentence("!!!!", False)]  # 剩下的標點念不出來，清理後會被略過
    assert clean_for_speech("!!!!") == ""


# ---- 任意分段結果都相同 ----

CORPUS = [
    "Sure! I'd be happy to help. Mr. Smith said the meeting is at 3 p.m. in Room 2.5, e.g. near the lobby. "
    "Isn't that great?! Well... maybe not. \"Really?\" she asked. (Yes.) The U.S. team won 3.14 points.",
    "Here are a few tips:\n1. **Drink water** every day.\n2. Sleep at least 8 hours…\n- Exercise, e.g. walking.\n"
    "That's it! 😊 Let me know if you need more help.",
    "你好！我是小明。今天天氣很好，我們去公園吧？好啊……走吧。「真的嗎？」他問。OK!我們出發。價格１．５元。",
    "This sentence just keeps going, and going, and going without ever ending properly because the model "
    "decided that punctuation is optional; still it should be split somewhere reasonable before it gets too long "
    "and then it finally ends.",
    "Hi! Ok. Sure thing. Absolutely! That works for me. Bye.",
    "Call Dr. J. K. Rowling at 5 a.m. Tomorrow, i.e. early. Version 2.0.1 costs $1,000.50 etc. Done!!",
    "Line one\nline two\n\n\nLine three. \n   \nEnd",
    "Supercalifragilisticexpialidocious" * 6 + " then a normal ending.",
    "What's 50 times 2? 100. Easy, right? Count down with me. 3. 2. 1. Go! Tips: 1. Read. 2. Write.",
    "Wow" + "!" * 40 + " so" + " " * 30 + "much space." + "*" * 20 + " Done.",
]
LIMITS = [(0, 1000), (12, 120), (5, 40), (20, 60)]


@pytest.mark.parametrize("text", CORPUS)
@pytest.mark.parametrize(("min_chars", "max_chars"), LIMITS)
def test_any_chunking_gives_the_same_sentences(text, min_chars, max_chars):
    limits = {"min_chars": min_chars, "max_chars": max_chars}
    expected = split_sentences(text, **limits)  # 文字與 line_start 都要相同

    assert split_sentences(list(text), **limits) == expected  # 一次一個字元
    for position in range(len(text) + 1):  # 每個字元位置都切一次
        assert split_sentences([text[:position], text[position:]], **limits) == expected, position
    rng = random.Random(f"{text}-{min_chars}-{max_chars}")
    for _ in range(30):  # 多種隨機切法，包含空字串片段
        cuts = sorted(rng.choices(range(len(text) + 1), k=rng.randint(1, 25)))
        pieces = [text[start:end] for start, end in zip([0, *cuts], [*cuts, len(text)])]
        assert split_sentences(pieces, **limits) == expected, pieces


def test_random_texts_are_chunking_invariant_and_lose_nothing():
    alphabet = [*"aB .!?…。！？．\n\"')*_,;:09", "Mr", "e.g", "U.S", " the ", "你好", "...", "1. ", "😊", "１", "   "]
    rng = random.Random(20261002)
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        min_chars = rng.choice([0, 3, 12])
        limits = {"min_chars": min_chars, "max_chars": rng.choice([min_chars + 1, 25, 120])}
        expected = split_sentences(text, **limits)
        cuts = sorted(rng.choices(range(len(text) + 1), k=rng.randint(1, 10)))
        pieces = [text[start:end] for start, end in zip([0, *cuts], [*cuts, len(text)])]

        assert split_sentences(pieces, **limits) == expected == split_sentences(list(text), **limits), text
        # 只會丟掉空白，文字內容與順序都不變
        assert "".join("".join(texts(expected)).split()) == "".join(text.split())


def test_random_word_texts_never_produce_short_sentences_before_the_last_one():
    words = ["hello", "a", "big", "world.", "ok!", "really?", "fine,", "x", "done;", "3.", "e.g.", "wow"]
    rng = random.Random(7)
    for _ in range(300):
        text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 60)))
        min_chars = rng.choice([0, 5, 12, 20])
        sentences = split(text, min_chars=min_chars, max_chars=rng.choice([min_chars + 10, 40, 120]))

        assert all(len(sentence) >= min_chars for sentence in sentences[:-1]), (text, sentences)


# ---- 最短與最長長度 ----


def test_short_sentences_are_merged_with_the_next_one():
    text = "Hi! How are you today? Sure. That works for me. Ok."

    assert split(text, min_chars=12) == ["Hi! How are you today?", "Sure. That works for me.", "Ok."]
    assert split(text, min_chars=0) == ["Hi!", "How are you today?", "Sure.", "That works for me.", "Ok."]


def test_the_first_sentence_is_not_too_short_either():
    splitter = SentenceSplitter(min_chars=12, max_chars=120)

    assert splitter.feed("Sure. ") == []
    assert splitter.feed("Let me check. ") == []
    assert texts(splitter.feed("Ok")) == ["Sure. Let me check."]
    assert texts(splitter.flush()) == ["Ok"]  # 最後剩下的短句照樣念出來


def test_short_lines_are_merged_too():
    sentences = split_sentences("## Tips\n- Water\n- Sleep well every night.\nDone.", min_chars=12)

    assert texts(sentences) == ["## Tips\n- Water", "- Sleep well every night.", "Done."]
    assert clean_for_speech(sentences[0].text, line_start=sentences[0].line_start) == "Tips, Water"


def test_long_text_is_cut_at_the_last_clause_break_within_the_limit():
    text = "Well, this goes on and on, and it keeps going; it never stops at all until here"

    sentences = split(text, min_chars=5, max_chars=50)

    assert sentences[0] == "Well, this goes on and on, and it keeps going;"
    assert all(len(sentence) <= 50 for sentence in sentences)
    assert " ".join(sentences) == text


def test_long_text_without_clause_breaks_is_cut_at_a_space_then_hard():
    words = "one two three four five six seven eight nine ten eleven twelve"
    assert split(words, min_chars=5, max_chars=20) == [
        "one two three four",
        "five six seven eight",
        "nine ten eleven",
        "twelve",
    ]
    assert split("x" * 45, min_chars=5, max_chars=20) == ["x" * 20, "x" * 20, "x" * 5]
    assert split("一二三四五六七八九十，一二三四五六七八九十一二三", min_chars=3, max_chars=15) == [
        "一二三四五六七八九十，",
        "一二三四五六七八九十一二三",
    ]


def test_forced_cuts_respect_the_minimum_length():
    # 太前面的逗號與空白（短於 min_chars）不能當切點，寧可直接切在 max_chars
    assert split("Hi, " + "x" * 30, min_chars=10, max_chars=20) == ["Hi, " + "x" * 16, "x" * 14]
    assert split("Hi there " + "y" * 30, min_chars=12, max_chars=20) == ["Hi there " + "y" * 11, "y" * 19]


def test_commas_inside_numbers_are_not_cut_points():
    assert split("It costs 1,000,000 dollars for everything we want", min_chars=5, max_chars=25) == [
        "It costs 1,000,000",
        "dollars for everything we",
        "want",
    ]


def test_a_long_first_sentence_is_released_as_soon_as_the_limit_is_reached():
    splitter = SentenceSplitter(min_chars=5, max_chars=30)

    assert splitter.feed("This keeps going, and going") == []
    assert texts(splitter.feed(" and going on")) == ["This keeps going,"]


def test_flush_resets_the_splitter():
    splitter = SentenceSplitter(min_chars=0, max_chars=100)
    splitter.feed("Unfinished")

    assert splitter.flush() == [Sentence("Unfinished", True)]
    assert splitter.feed("New. Turn") == [Sentence("New.", True)]  # 新的一輪從行首開始


@pytest.mark.parametrize(("min_chars", "max_chars"), [(-1, 10), (10, 10), (20, 10)])
def test_invalid_limits_are_rejected(min_chars, max_chars):
    with pytest.raises(ValueError):
        SentenceSplitter(min_chars=min_chars, max_chars=max_chars)


# ---- 效能：還不能確定的狀態不會讓每次 feed 都重掃整段 ----


@pytest.mark.parametrize(
    ("prefix", "filler", "suffix", "max_chars", "size"),
    [
        ("Hello.", " ", "Next one.", 120, 20_000),  # 句尾後面一長串空白
        ("Hello", "!", " Next one.", 120, 20_000),  # 一長串驚嘆號
        ("Hello.", "*", " Next one.", 120, 20_000),  # 一長串收尾符號
        ("Start", "a.b", " end.", 120, 20_000),  # 很多不是句尾的句點，也沒有空白
        # max_chars 設得很大時，每次 feed 都從句首重掃會變成 O(n²)；要記住已經掃過的位置
        ("Start", "a.b", " end.", 4_000, 8_000),
        ("", "word ", "end.", 4_000, 8_000),
    ],
)
def test_feeding_one_character_at_a_time_stays_linear(prefix, filler, suffix, max_chars, size):
    text = prefix + filler * (size // len(filler)) + suffix
    splitter = SentenceSplitter(min_chars=12, max_chars=max_chars)

    started = time.perf_counter()
    sentences = [sentence for char in text for sentence in splitter.feed(char)] + splitter.flush()
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert "".join("".join(texts(sentences)).split()) == "".join(text.split())


# ---- 朗讀前清理 ----


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**Bold**, *italic*, __strong__, _em_ and ~~old~~ text.", "Bold, italic, strong, em and old text."),
        ("Run `pip install` first.", "Run pip install first."),
        ("## Heading", "Heading"),
        ("- item one", "item one"),
        ("* item two", "item two"),
        ("1. First step.", "1. First step."),  # 數字編號保留：寧可多念，不可漏念
        ("2) Second step.", "2) Second step."),
        ("> quoted - text", "quoted - text"),
        ("See [the docs](https://example.com/a.b) now.", "See the docs now."),
        ("![a cat](cat.png) Cute.", "a cat Cute."),
        ("snake_case stays.", "snake_case stays."),
        ("Great job! 😊👍🏽 Keep going 🇹🇼 ❤️ ✨", "Great job! Keep going"),
        ("Family: 👨‍👩‍👧 fun.", "Family: fun."),
        ("Hello   world \t again .", "Hello world again."),
        ("你好😊。我很好！", "你好。我很好！"),
        ("## Summary\nThe main point is clear.", "Summary, The main point is clear."),
        ("- Apples\n- Pears\n- Plums", "Apples, Pears, Plums"),
        ("Tips:\n- Drink water", "Tips: Drink water"),
        ("標題\n內容。", "標題，內容。"),
        ("2*3=6 and 4 * 5 = 20.", "2*3=6 and 4 * 5 = 20."),  # 不是強調記號的 * 保留原樣
        ("Answer: 7 * 8 = 56.", "Answer: 7 * 8 = 56."),
        ("2*3*4 = 24", "2*3*4 = 24"),
        ("a **bold** move and ***both*** and 2 * 3", "a bold move and both and 2 * 3"),
        ("**Important: do this.", "Important: do this."),  # 粗體被切成兩句時落單的 **
        ("Title\n---\nText", "Title, Text"),
        ("Is 3 > 2? Yes.", "Is 3 > 2? Yes."),
    ],
)
def test_clean_for_speech(text, expected):
    assert clean_for_speech(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("- 5 degrees outside.", "- 5 degrees outside."),
        ("> 2 is true.", "> 2 is true."),
        ("# 1 rule: be kind.", "# 1 rule: be kind."),
        ("- 5 degrees\n- 10 degrees", "- 5 degrees, 10 degrees"),  # 第二行在真正的行首
    ],
)
def test_markers_are_kept_when_the_sentence_does_not_start_a_line(text, expected):
    assert clean_for_speech(text, line_start=False) == expected


@pytest.mark.parametrize("text", ["...", "!?", "…。", "---", "* * *", "😊🎉", "** **", "()", "  \n "])
def test_text_with_nothing_to_pronounce_becomes_empty(text):
    assert clean_for_speech(text) == ""
