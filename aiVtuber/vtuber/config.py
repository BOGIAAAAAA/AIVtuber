"""應用程式設定：以 dataclass 定義所有可調參數，並從 YAML 載入。

- 設定檔預設為 aiVtuber/config.yaml；不存在時使用預設值。
- 設定檔裡「本機」的相對路徑（型別為 Path 的欄位）一律以 aiVtuber/ 為基準解析，
  所以不論從哪個目錄執行結果都一樣。
- tts.ref_audio_path 是要送給 TTS server 的路徑，由 TTS client 在送出前處理（見 tts.resolve_ref_audio_path）：
  相對路徑同樣以 aiVtuber/ 為基準轉成絕對路徑，絕對路徑則原樣送出。
- API key 不放在設定檔：Groq SDK 會自行讀取環境變數 GROQ_API_KEY。
"""

from __future__ import annotations

import dataclasses
import logging
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import httpx
import yaml

from vtuber.errors import ConfigError

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.yaml"

DEVICES = ("auto", "cpu", "cuda", "mps")
# Groq SDK 接受的值：gpt-oss 系列用 low / medium / high，qwen3 系列用 none / default
REASONING_EFFORTS = (None, "none", "default", "low", "medium", "high")
# gpt-sovits-v2：GPT-SoVITS 的 api_v2.py；gpt-sovits-v1：舊版的 api.py
TTS_ENGINES = ("gpt-sovits-v2", "gpt-sovits-v1")
# 語言代碼（小寫）。v2 依 GPT-SoVITS 48b1a01 的 TTS_infer_pack/TTS.py（TTS_Config.v2_languages），
# v1 依舊版 api.py 的 dict_language（中文別名不列出）
TTS_LANGUAGES = {
    "gpt-sovits-v2": ("auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"),
    "gpt-sovits-v1": ("auto", "en", "zh", "ja", "all_zh", "all_ja"),
}
# api_v2 的 text_split_method（TTS_infer_pack/text_segmentation_method.py）
TEXT_SPLIT_METHODS = ("cut0", "cut1", "cut2", "cut3", "cut4", "cut5")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _check_optional_positive(value: Optional[float], key: str) -> None:
    _check(value is None or value > 0, f"{key} must be a positive number or null, got {value}")


def _expand_user(value: str, key: str) -> Path:
    """展開 ~ 開頭的路徑；~someone 但這台電腦沒有這個使用者時丟出 ConfigError（pathlib 會丟 RuntimeError）。"""
    try:
        return Path(value).expanduser()
    except RuntimeError as exc:
        raise ConfigError(f"Invalid path {value!r} in {key}: {exc}") from exc


def _model_ref(default: str) -> Any:
    """模型欄位：可填 Hugging Face 模型 ID 或本機路徑；本機路徑以 aiVtuber/ 為基準（見 resolve_model_reference）。"""
    return field(default=default, metadata={"model_ref": True})


# 注意：欄位型別請用 Optional[...]，不要寫 X | None。
# 載入時會用 typing.get_type_hints() 在執行期解析型別，Python 3.9 不支援 X | None。


@dataclass(frozen=True)
class LLMConfig:
    """Groq 對話模型設定。"""

    # llama3-8b-8192 已於 2025-08-30 下架；官方建議的後繼 llama-3.1-8b-instant
    # 也已於 2026-08-16（free/developer 方案）下架，建議改用 openai/gpt-oss-20b。
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.5
    max_tokens: int = 1024
    # 只有推理模型支援（gpt-oss 系列：low / medium / high；qwen3 系列：none / default）；None 表示不送。
    # 預設 low：推理越少，第一句越快出來
    reasoning_effort: Optional[str] = "low"
    # 要不要 API 傳回推理內容；推理內容不會念出來，所以預設 False。None 表示不送（非推理模型請用 None）
    include_reasoning: Optional[bool] = False
    timeout: float = 30.0
    # 失敗（連線錯誤、429、5xx）時的重試次數；SDK 預設 2 次，429 時最壞要等很久
    max_retries: int = 1
    # 會被送去 TTS 念出來，語言要和 tts.text_lang 一致
    fallback_message: str = "Sorry, I can't generate a reply right now."

    def __post_init__(self) -> None:
        _check(0 <= self.temperature <= 2, f"llm.temperature must be between 0 and 2, got {self.temperature}")
        _check(self.max_tokens > 0, "llm.max_tokens must be positive")
        _check(
            self.reasoning_effort in REASONING_EFFORTS,
            "llm.reasoning_effort must be one of none / default / low / medium / high or null, "
            f"got {self.reasoning_effort!r}",
        )
        _check(self.timeout > 0, "llm.timeout must be positive")
        _check(self.max_retries >= 0, "llm.max_retries must not be negative")


@dataclass(frozen=True)
class TTSConfig:
    """TTS 引擎設定：GPT-SoVITS 的 api_v2.py（預設）或舊版 api.py。欄位名稱以 api_v2 為準。"""

    engine: str = "gpt-sovits-v2"
    url: str = "http://127.0.0.1:9880"
    # 參考音檔：相對路徑以 aiVtuber/ 為基準轉成絕對路徑後送出，絕對路徑原樣送出（server 在別台電腦時用）
    ref_audio_path: str = "voices/firefly/ref_firefly_01.wav"
    prompt_text: str = "I understand. Article 4 of Glamoth military regulations."
    prompt_lang: str = "en"
    text_lang: str = "en"
    # ---- 以下只有 gpt-sovits-v2 使用 ----
    # client 已經按句送出，server 不必再切（cut0）：每句一次推論，語氣連貫，也不會多出句中的靜音
    text_split_method: str = "cut0"
    speed_factor: float = 1.0
    # server 在每段音訊後面補的靜音秒數（最後一段也補）；cut0 時每句只有一段，就是句子之間的停頓
    fragment_interval: float = 0.3
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    # ---- 只有 gpt-sovits-v1 使用：None 表示不送，由 server 的 -cp 預設值決定 ----
    cut_punc: Optional[str] = None
    # 單句合成的逾時秒數
    timeout: float = 120.0
    # 聊天開始前先等 server 就緒（GET {url}/docs 回 200），最多等 ready_timeout 秒
    wait_for_server: bool = True
    ready_timeout: float = 60.0
    # 就緒後先合成這句話暖機（不播放）；空字串表示不暖機
    warmup_text: str = "Hello, nice to meet you."

    def __post_init__(self) -> None:
        _check(self.engine in TTS_ENGINES, f"tts.engine must be one of {', '.join(TTS_ENGINES)}, got {self.engine!r}")
        _check_server_url(self.url)
        if self.ref_audio_path.startswith("~"):
            _expand_user(self.ref_audio_path, "tts.ref_audio_path")
        _check(self.timeout > 0, "tts.timeout must be positive")
        _check(self.ready_timeout > 0, "tts.ready_timeout must be positive")
        self._check_reference_and_languages()
        _check(
            self.text_split_method in TEXT_SPLIT_METHODS,
            f"tts.text_split_method must be one of {', '.join(TEXT_SPLIT_METHODS)}, got {self.text_split_method!r}",
        )
        _check(self.speed_factor > 0, f"tts.speed_factor must be positive, got {self.speed_factor}")
        _check(self.fragment_interval >= 0, f"tts.fragment_interval must not be negative, got {self.fragment_interval}")
        # 範圍同 GPT-SoVITS 的取樣程式：temperature 0 等同只取機率最高的 token
        _check(self.top_k >= 1, f"tts.top_k must be at least 1, got {self.top_k}")
        _check(0 <= self.top_p <= 1, f"tts.top_p must be between 0 and 1, got {self.top_p}")
        _check(self.temperature >= 0, f"tts.temperature must not be negative, got {self.temperature}")

    def _check_reference_and_languages(self) -> None:
        languages = TTS_LANGUAGES[self.engine]
        allowed = ", ".join(languages)
        _check(
            self.text_lang in languages,
            f"tts.text_lang must be one of {allowed} for {self.engine}, got {self.text_lang!r}",
        )
        if self.engine == "gpt-sovits-v2":
            # api_v2 沒有 server 端的預設參考音檔，每次請求都要帶
            _check(bool(self.ref_audio_path), "tts.ref_audio_path is required for gpt-sovits-v2")
        elif self.ref_audio_path:
            # 舊版 api.py 只要參考音檔、文字、語言三者缺一，就會悄悄改用 server 預設的參考音檔
            _check(bool(self.prompt_text), "tts.prompt_text is required when tts.ref_audio_path is set")
        if self.ref_audio_path:
            _check(
                self.prompt_lang in languages,
                f"tts.prompt_lang must be one of {allowed} for {self.engine}, got {self.prompt_lang!r}",
            )


def _check_server_url(url: str) -> None:
    """用 TTS client 實際使用的 httpx 解析 tts.url，格式錯誤在載入設定時就報錯，而不是連線時才噴出 traceback。"""
    try:
        parsed = httpx.URL(url)
        host, port = parsed.host, parsed.port  # host 會解碼國際化網域名稱，不合法時也在這裡出錯
    except (httpx.InvalidURL, UnicodeError) as exc:  # 例如 port 不是數字、IPv6 少了 ]、主機名稱不合法
        raise ConfigError(f"tts.url is not a valid URL ({exc}): {url!r}") from exc
    _check(parsed.scheme in ("http", "https"), f"tts.url must start with http:// or https://, got {url!r}")
    _check(bool(host), f"tts.url has no host name: {url!r}")
    _check(port is None or 1 <= port <= 65535, f"tts.url port must be between 1 and 65535: {url!r}")
    _check(not parsed.query and not parsed.fragment, f"tts.url must not contain ? or #: {url!r}")


@dataclass(frozen=True)
class AudioConfig:
    """播放設定。"""

    # None 為系統預設輸出裝置；要接虛擬音源線（給 VTS 口型用）時可指定裝置編號
    output_device_index: Optional[int] = None

    def __post_init__(self) -> None:
        index = self.output_device_index
        _check(index is None or index >= 0, f"audio.output_device_index must not be negative, got {index}")


@dataclass(frozen=True)
class ASRConfig:
    """語音辨識（SpeechRecognition + Google Web Speech API）設定。"""

    device_index: Optional[int] = None
    language: str = "en-US"
    # 等待開口的秒數；逾時視為沒人說話，安靜地進入下一輪
    timeout: Optional[float] = 5.0
    phrase_time_limit: Optional[float] = 5.0
    # 只在啟動時校正一次環境噪音
    ambient_noise_duration: float = 1.0
    # 呼叫 Google 辨識 API 的逾時秒數（SpeechRecognition 預設是無限等待）
    request_timeout: Optional[float] = 15.0

    def __post_init__(self) -> None:
        index = self.device_index
        _check(index is None or index >= 0, f"asr.device_index must not be negative, got {index}")
        for key in ("timeout", "phrase_time_limit", "request_timeout"):
            _check_optional_positive(getattr(self, key), f"asr.{key}")
        _check(self.ambient_noise_duration >= 0, "asr.ambient_noise_duration must not be negative")


@dataclass(frozen=True)
class ChatConfig:
    """聊天迴圈設定。"""

    # 比對時忽略大小寫與句尾標點
    exit_commands: tuple[str, ...] = ("exit",)
    # 回覆按句切分後逐句合成；比這短的句子併入下一句（GPT-SoVITS 合成很短的片段品質差）
    min_sentence_chars: int = 12
    # 一直沒有句尾時，超過這個長度就在逗號、分號或空白處切開，避免第一句等太久
    max_sentence_chars: int = 120

    def __post_init__(self) -> None:
        _check(bool(self.exit_commands), "chat.exit_commands must not be empty")
        _check(self.min_sentence_chars >= 0, "chat.min_sentence_chars must not be negative")
        _check(
            self.max_sentence_chars > self.min_sentence_chars,
            f"chat.max_sentence_chars ({self.max_sentence_chars}) must be greater than "
            f"chat.min_sentence_chars ({self.min_sentence_chars})",
        )


@dataclass(frozen=True)
class VTSConfig:
    """VTube Studio API 設定。"""

    host: str = "localhost"
    port: int = 8001
    # VTS 要求 3–32 字元；換名稱後需要在 VTS 重新授權一次
    plugin_name: str = "AIVtuber"
    plugin_developer: str = "AIVtuber"
    token_path: Path = Path("pyvts_token.txt")
    # 表情檔名；沒寫副檔名時自動補上 .exp3.json
    positive_expression: str = "Happy"
    negative_expression: str = "Sad"
    fade_time: float = 0.25

    def __post_init__(self) -> None:
        for key in ("plugin_name", "plugin_developer"):
            value = getattr(self, key)
            _check(
                3 <= len(value) <= 32,
                f"vts.{key} must be 3-32 characters (VTube Studio requirement), got {value!r}",
            )
        _check(0 < self.port < 65536, f"vts.port is out of range: {self.port}")
        _check(0 <= self.fade_time <= 2, f"vts.fade_time must be between 0 and 2 seconds, got {self.fade_time}")


@dataclass(frozen=True)
class SentimentConfig:
    """情緒分類模型（需為已微調、標籤含 positive/negative 的分類模型）。"""

    model: str = _model_ref("distilbert/distilbert-base-uncased-finetuned-sst-2-english")


@dataclass(frozen=True)
class GenerationConfig:
    """微調後 GPT-2 的取樣參數（--generate）。"""

    max_new_tokens: int = 150
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2

    def __post_init__(self) -> None:
        _check(self.max_new_tokens > 0, "generation.max_new_tokens must be positive")
        # do_sample=True 時 temperature 必須大於 0，top_p 必須在 (0, 1]
        _check(self.temperature > 0, f"generation.temperature must be greater than 0, got {self.temperature}")
        _check(0 < self.top_p <= 1, f"generation.top_p must be in (0, 1], got {self.top_p}")
        _check(self.top_k >= 0, "generation.top_k must not be negative")
        _check(self.repetition_penalty > 0, "generation.repetition_penalty must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    """GPT-2 微調設定（--train）。"""

    base_model: str = _model_ref("gpt2-medium")
    data_dir: Path = Path("data")
    # 相對於 data_dir
    data_files: tuple[str, ...] = (
        "dialogue_data_proficiency.txt",
        "dialogue_data.txt",
        "personality_data.txt",
        "zora.txt",
    )
    # 微調後模型的存放處；--generate 也從這裡載入
    model_save_path: Path = Path("saved_model")
    # Trainer 的 checkpoint 與訓練曲線圖
    output_dir: Path = Path("outputs")
    epochs: float = 10.0
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    block_size: int = 512
    validation_split: float = 0.1
    logging_steps: int = 5
    seed: int = 42
    # 例如 "tensorboard"；預設不送到任何追蹤服務
    report_to: str = "none"

    def __post_init__(self) -> None:
        _check(bool(self.data_files), "training.data_files must not be empty")
        _check(self.epochs > 0, "training.epochs must be positive")
        _check(self.learning_rate > 0, "training.learning_rate must be positive")
        _check(0 <= self.warmup_ratio < 1, "training.warmup_ratio must be in [0, 1)")
        _check(0 < self.validation_split < 1, "training.validation_split must be between 0 and 1")
        for key in ("batch_size", "gradient_accumulation_steps", "block_size", "logging_steps"):
            _check(getattr(self, key) > 0, f"training.{key} must be positive")


@dataclass(frozen=True)
class AppConfig:
    """全部設定的根節點，對應 config.yaml 的頂層。"""

    # 推論裝置：auto 依序選 cuda > mps > cpu
    device: str = "auto"
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    vts: VTSConfig = field(default_factory=VTSConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        _check(self.device in DEVICES, f"device must be one of {DEVICES}, got {self.device!r}")


def load_config(path: Optional[Path] = None, base_dir: Path = BASE_DIR) -> AppConfig:
    """讀取設定檔並回傳 AppConfig。

    path 為 None 時讀預設的 aiVtuber/config.yaml，不存在就用預設值；
    明確指定的檔案不存在則視為錯誤（避免打錯檔名卻默默用預設值）。
    """
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path).expanduser()
    if config_path.is_file():
        raw = _read_yaml(config_path)
        logger.info("Loaded config from %s", config_path)
    elif path is None:
        logger.info(
            "No config file at %s; using built-in defaults (copy config.example.yaml to config.yaml to customise)",
            config_path,
        )
        raw = {}
    else:
        raise ConfigError(f"Config file not found: {config_path}")
    return resolve_paths(_build(AppConfig, raw, ""), base_dir)


def resolve_paths(config: Any, base_dir: Path) -> Any:
    """把 Path 欄位的相對路徑改成以 base_dir 為基準的絕對路徑；模型欄位則交給 resolve_model_reference。"""
    changes: dict[str, Any] = {}
    for item in dataclasses.fields(config):
        value = getattr(config, item.name)
        if dataclasses.is_dataclass(value):
            changes[item.name] = resolve_paths(value, base_dir)
        elif isinstance(value, Path) and not value.is_absolute():
            changes[item.name] = base_dir / value
        elif item.metadata.get("model_ref"):
            changes[item.name] = resolve_model_reference(value, base_dir)
    return dataclasses.replace(config, **changes) if changes else config


def resolve_model_reference(value: str, base_dir: Path) -> str:
    """模型可填 Hugging Face 模型 ID 或本機路徑。

    看起來像本機路徑（./、../、~ 開頭或絕對路徑），或 base_dir 底下確實有這個路徑時，
    轉成以 base_dir 為基準的絕對路徑；其餘（例如 gpt2-medium、distilbert/xxx）視為模型 ID 原樣保留。
    """
    path = _expand_user(value, "a model setting")
    if path.is_absolute():
        return str(path)
    if value.startswith(".") or (base_dir / path).exists():
        return os.path.normpath(base_dir / path)
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _build(cls: Any, raw: Any, where: str) -> Any:
    """依 dataclass 欄位型別，把 YAML 的 dict 轉成（巢狀的）設定物件。"""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"'{where or 'top level'}' must be a mapping, got {type(raw).__name__}")
    names = {item.name for item in dataclasses.fields(cls)}
    unknown = sorted(str(key) for key in raw if key not in names)
    if unknown:
        raise ConfigError(
            f"Unknown setting(s) in '{where or 'top level'}': {', '.join(unknown)} "
            f"(valid keys: {', '.join(sorted(names))})"
        )
    hints = typing.get_type_hints(cls)
    values = {}
    for name, value in raw.items():
        key = f"{where}.{name}" if where else name
        hint = hints[name]
        values[name] = _build(hint, value, key) if dataclasses.is_dataclass(hint) else _coerce(value, hint, key)
    return cls(**values)


def _coerce(value: Any, hint: Any, key: str) -> Any:
    """把 YAML 值轉成欄位宣告的型別；型別不符時丟出指出欄位名稱的 ConfigError。"""
    origin = typing.get_origin(hint)
    if origin is Union:  # Optional[X]
        if value is None:
            return None
        inner = next(arg for arg in typing.get_args(hint) if arg is not type(None))
        return _coerce(value, inner, key)
    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{key} must be a list, got {value!r}")
        item_hint = typing.get_args(hint)[0]
        return tuple(_coerce(item, item_hint, f"{key}[{index}]") for index, item in enumerate(value))

    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if hint is bool and isinstance(value, bool):
        return value
    if hint is int and is_number and float(value).is_integer():
        return int(value)
    if hint is float:
        if is_number:
            return float(value)
        if isinstance(value, str):
            # PyYAML 會把沒有小數點的 5e-5 讀成字串
            try:
                return float(value)
            except ValueError:
                pass
    if hint is str and (isinstance(value, str) or is_number):
        return str(value)
    if hint is str and isinstance(value, bool):
        raise ConfigError(
            f"{key} was read as the boolean {value} because YAML treats unquoted yes/no/on/off/true/false "
            'as booleans; put the value in quotes, for example "off"'
        )
    if hint is Path and isinstance(value, str) and value.strip():
        return _expand_user(value, key)
    raise ConfigError(f"{key} has an invalid value {value!r} (expected {getattr(hint, '__name__', hint)})")
