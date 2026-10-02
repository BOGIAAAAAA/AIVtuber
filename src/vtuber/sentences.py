"""回覆文字的按句切分與朗讀前清理。

LLM 的串流輸出是一小段一小段進來的：SentenceSplitter 把它切成完整的句子，讓 TTS 可以邊收邊合成；
clean_for_speech() 再把每一句整理成適合念出來的文字。原則是寧可多念，不可漏念：數字一律保留。

切句規則：
- 句尾：. ! ? …、中日文句尾標點（。！？），以及換行。連續的句尾標點（"?!"、"..."）與緊接在後的
  引號、括號、markdown 強調記號算同一個句尾。英文句尾後面要接空白（或中日韓文字）才算，
  而且下一個字是小寫時視為同一句（例如 "Well... maybe"、'"Wow!" she said'）。
- 不切：常見縮寫（Mr. Dr. etc.）、帶點的縮寫（e.g. i.e. a.m. U.S.）、名字縮寫（J. K.）、
  小數與數字（3.14、1,000、１．５），以及編號：句子開頭、換行、冒號或另一個句尾之後的 "1."
  （例如 "tips: 1. Practice daily."、"Count down with me. 3. 2. 1. Go!"）。
- 要看後面的字才能決定時（例如句點後面還沒有下一個字），先保留，等下一段文字進來再決定；
  已經確定的部分不再重掃。因此不論文字怎麼分段送進來，切出來的結果都和一次送完整段相同。
- 比 min_chars 短的句子併入下一句；一直沒有句尾、超過 max_chars 時，在 max_chars 以內
  最後一個逗號、分號或冒號處切開，沒有的話切在空白處，都沒有才直接切斷。
"""

from __future__ import annotations

import re
from typing import NamedTuple, Optional

# 句尾標點；連續出現時（"?!"、"..."）視為同一個句尾
SENTENCE_ENDINGS = frozenset(".!?…。！？．｡")
# 中日文的句尾標點後面不需要空白
_CJK_ENDINGS = frozenset("。！？．｡")
# 緊接在句尾標點後面、屬於同一句的收尾符號：引號、括號，以及 markdown 的粗體／斜體記號
_CLOSER_CHARS = "\"'”’)]}」』）】》〉*_"
_CLOSERS = frozenset(_CLOSER_CHARS)
# 太長時優先在子句標點後面切開；英文標點後面要接空白，避免切開 1,000 這種數字
_CLAUSE_BREAKS = frozenset(",;:")
_CJK_CLAUSE_BREAKS = frozenset("，；：、—")
# 數字前面是這些字（或句子開頭）時，"1." 是編號而不是句尾
_BEFORE_NUMBERING = frozenset("\n:：") | SENTENCE_ENDINGS
# 句尾標點加收尾符號、句尾後面的空白，各自最多看這麼多個字，超過就直接算句尾：
# 避免一長串 "!!!..." 或空白讓「還不能確定」的狀態一直持續，每次 feed 都要重看一遍
_MAX_ENDING_RUN = 16
_MAX_GAP = 16
# 後面接大寫字也不算句尾的縮寫（不分大小寫、不含句點）；帶點的縮寫與名字縮寫另外判斷
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "mx", "dr", "prof", "sr", "jr", "st", "mt", "rev", "hon", "pres",
        "gen", "gov", "capt", "col", "lt", "sgt", "vs", "etc", "cf", "approx", "dept", "fig",
        "vol", "inc", "ltd", "corp", "co", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
        "sep", "sept", "oct", "nov", "dec",
    }
)  # fmt: skip


class Sentence(NamedTuple):
    """切出來的一句。line_start 表示這句從真正的行首開始（回覆開頭或換行之後），
    只有那裡的 "- "、"# " 才是 markdown 記號（見 clean_for_speech）。"""

    text: str
    line_start: bool


class SentenceSplitter:
    """增量式切句器：feed() 回傳已經確定完整的句子，flush() 在文字結束時回傳剩下的部分。

    長度以字元計（去掉頭尾空白），中日文一個字算一個字元。
    """

    def __init__(self, *, min_chars: int, max_chars: int) -> None:
        if not 0 <= min_chars < max_chars:
            raise ValueError(f"need 0 <= min_chars < max_chars, got {min_chars} and {max_chars}")
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._reset()

    def feed(self, text: str) -> list[Sentence]:
        self._buffer += text
        return self._take_sentences(final=False)

    def flush(self) -> list[Sentence]:
        sentences = self._take_sentences(final=True)
        rest = self._buffer.strip()
        if rest:
            sentences.append(Sentence(rest, self._line_start))
        self._reset()
        return sentences

    def _reset(self) -> None:
        self._buffer = ""
        # 這一句已經掃過、判斷不會再改變的位置；下次從這裡繼續
        self._scanned = 0
        # 緩衝區的開頭是不是真正的行首
        self._line_start = True

    def _take_sentences(self, *, final: bool) -> list[Sentence]:
        sentences = []
        while True:
            if self._scanned == 0:  # 一句的開頭：去掉前面的空白，順便看有沒有換行
                stripped = self._buffer.lstrip()
                if "\n" in self._buffer[: len(self._buffer) - len(stripped)]:
                    self._line_start = True
                self._buffer = stripped
            cut = self._find_cut(final)
            if cut is None:
                return sentences
            sentences.append(Sentence(self._buffer[:cut].rstrip(), self._line_start))
            self._buffer = self._buffer[cut:]
            self._scanned = 0
            self._line_start = False

    def _find_cut(self, final: bool) -> Optional[int]:
        """回傳這一句的結束位置；還不能確定，或剩下的文字裡沒有夠長的句子時回傳 None。

        只依據已經確定的判斷做決定，所以結果與文字的分段方式無關。
        """
        text = self._buffer
        index = self._scanned
        while index < min(len(text), self._max_chars):
            if text[index] == "\n":
                decision: Optional[tuple[bool, int, int]] = (True, index, index + 1)
            elif text[index] in SENTENCE_ENDINGS:
                decision = _ending_at(text, index, final)
                if decision is None:
                    self._scanned = index  # 要看後面的字才能確定；下次從這裡繼續
                    return None
            else:
                index += 1
                continue
            is_end, end, index = decision
            # 太短的句子不切，繼續找下一個句尾（等於和下一句合併）
            if is_end and len(text[:end].rstrip()) >= max(self._min_chars, 1):
                return end
        self._scanned = index
        if len(text) > self._max_chars:
            return self._forced_cut(text)
        return None

    def _forced_cut(self, text: str) -> int:
        """max_chars 以內沒有句尾時的切點；只看前 max_chars + 1 個字，所以也與分段方式無關。

        切出來的句子不會短於 min_chars（除非前 max_chars 個字幾乎都是空白）。
        """
        clause = space = None
        for cut in range(max(self._min_chars, 1), self._max_chars + 1):
            before, after = text[cut - 1], text[cut]
            if before in _CJK_CLAUSE_BREAKS or (before in _CLAUSE_BREAKS and after.isspace()):
                clause = cut
            if after.isspace() and not before.isspace():
                space = cut
        if clause is not None:
            return clause
        return space if space is not None else self._max_chars


def _ending_at(text: str, index: int, final: bool) -> Optional[tuple[bool, int, int]]:
    """判斷從 text[index] 開始的句尾標點是不是真的句尾。

    回傳（是否為句尾, 句子結束位置, 下一個要檢查的位置）；要看後面的字才能確定時回傳 None。
    """
    size = len(text)
    run_limit = min(size, index + _MAX_ENDING_RUN)
    end = index
    while end < run_limit and text[end] in SENTENCE_ENDINGS:
        end += 1
    marks = text[index:end]
    while end < run_limit and text[end] in _CLOSERS:
        end += 1
    if end - index == _MAX_ENDING_RUN:  # 異常長的一串標點：就切在這裡，剩下的標點念不出來，之後會被略過
        return True, end, end
    if end == size:  # 後面可能還有標點、引號或下一句
        return (True, end, end) if final else None
    following = text[end]
    if marks == "．" and text[index - 1 : index].isdigit() and following.isdigit():  # 全形小數點：１．５
        return False, end, end
    if any(mark in _CJK_ENDINGS for mark in marks) or _is_cjk(following):
        return True, end, end
    if not following.isspace():  # 3.14、e.g.,、U.S.A、example.com
        return False, end, end
    word_start = end
    gap_limit = min(size, end + _MAX_GAP)
    while word_start < gap_limit and text[word_start].isspace():
        word_start += 1
    if "\n" in text[end:word_start] or word_start - end == _MAX_GAP:
        return True, end, end
    if word_start == size:
        return (True, end, end) if final else None
    if text[word_start].islower():  # 同一句的延續："Well... maybe"、"e.g. this"
        return False, end, end
    if marks == "." and _is_abbreviation(text, index):
        return False, end, end
    return True, end, end


def _is_abbreviation(text: str, dot: int) -> bool:
    """句點前面是縮寫、名字縮寫或編號時回傳 True。"""
    start = dot
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "."):
        start -= 1
    word = text[start:dot]
    if not word:
        return False
    if word.isdigit():
        # 編號：數字在這一句的開頭，或緊接在換行、冒號、另一個句尾之後（"tips: 1. Practice"、
        # "me. 3. 2. 1. Go!"、"2? 100. Easy"）；其他位置照常斷句（"I have 3. You have 4."）
        before = text[:start].rstrip(" \t").rstrip(_CLOSER_CHARS)
        return not before or before[-1] in _BEFORE_NUMBERING
    if word.lower() in _ABBREVIATIONS:
        return True
    if "." in word:  # e.g、i.e、a.m、U.S、Ph.D
        return all(part.isalpha() and len(part) <= 2 for part in word.split("."))
    # 名字縮寫（J. K. Rowling）；I 和 A 常是句子的最後一個字，不算
    return len(word) == 1 and word.isupper() and word not in ("I", "A")


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF  # 平假名、片假名
        or 0x3400 <= code <= 0x9FFF  # 中日韓統一表意文字（含擴充 A）
        or 0xAC00 <= code <= 0xD7AF  # 韓文音節
        or 0xF900 <= code <= 0xFAFF  # 相容表意文字
    )


# 連結與圖片 [文字](網址)：只留文字
_LINK = re.compile(r"!?\[([^\]\n]*)\]\([^)\n]*\)")
# 行首的標題、引用與條列符號（可以疊在一起，例如 "> - "）；數字編號保留，念出來也沒關係
_LINE_MARKERS = re.compile(r"\A[ \t]*(?:(?:#{1,6}|>+|[-*+•])[ \t]+)+")
# 成對、緊貼文字的 *斜體*、**粗體**；數字之間或前後有空白的 *（2*3=6、4 * 5）不是記號，保留原樣
_PAIRED_EMPHASIS = re.compile(r"(?<![\w*])(\*{1,3})(?=[^\s*])(.+?)(?<=[^\s*])\1(?![\w*])")
# 其他 markdown 記號：落單的 ** 與 ***（例如粗體被切成兩句）、行內程式碼、刪除線，
# 以及單字頭尾的底線（snake_case 中間的保留）
_MARKUP = re.compile(r"\*{2,}|`+|~~|(?<!\w)_+|_+(?!\w)")
# emoji 與圖形符號（含膚色、旗幟、組合用的 ZWJ 與變體選擇符）
_EMOJI = re.compile(
    "[‍⃣⌀-⏿☀-➿⬀-⯿︎️\U0001f000-\U0001faff\U000e0020-\U000e007f]"
)
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+(?=[,.!?;:…，。！？；：、])")
_CJK_PUNCTUATION = frozenset("，。！？；：、「」『』（）《》【】…")


def clean_for_speech(text: str, *, line_start: bool = True) -> str:
    """把一句回覆整理成要念出來的文字：去掉 markdown 記號與 emoji、壓縮空白；數字一律保留。

    line_start=False 表示 text 是從一行中間開始的（切句器的 Sentence.line_start），
    這時第一行開頭的 "- "、"# "、"> " 不是 markdown 記號，照原樣保留（例如 "- 5 degrees"）。
    沒有任何字母或數字時回傳空字串（純標點送進 GPT-SoVITS 只會得到靜音）。
    """
    text = _EMOJI.sub("", _LINK.sub(r"\1", text))
    spoken = ""
    for number, line in enumerate(text.split("\n")):
        if number or line_start:
            line = _LINE_MARKERS.sub("", line)
        line = " ".join(_MARKUP.sub("", _PAIRED_EMPHASIS.sub(r"\2", line)).split())
        if any(char.isalnum() for char in line):  # 分隔線、只有 emoji 或標點的行略過
            spoken = _join_lines(spoken, line) if spoken else line
    return _SPACE_BEFORE_PUNCTUATION.sub("", spoken)


def _join_lines(left: str, right: str) -> str:
    """併在一起的短行（標題、條列項目）之間補個逗號，念起來才有停頓；中日文之間不加空白。"""
    cjk = _is_cjk(left[-1])
    if left[-1].isalnum():
        left += "，" if cjk else ","
    wide = (cjk or left[-1] in _CJK_PUNCTUATION) and _is_cjk(right[0])
    return left + ("" if wide else " ") + right
