from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest
import yaml

from vtuber import config as config_module
from vtuber.config import AppConfig, TrainingConfig, load_config
from vtuber.errors import ConfigError


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_missing_default_file_uses_defaults_and_logs(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")

    with caplog.at_level(logging.INFO, logger="vtuber.config"):
        config = load_config(base_dir=tmp_path)

    assert "using built-in defaults" in caplog.text
    assert config.llm.model == "openai/gpt-oss-20b"
    assert config.llm.temperature == 0.5
    assert config.llm.max_tokens == 1024
    assert config.llm.reasoning_effort == "low"  # 推理越少，第一句越快出來
    assert config.llm.fallback_message.isascii()  # 預設英文，配合 text_lang=en
    assert config.tts.engine == "gpt-sovits-v2"
    assert config.tts.url == "http://127.0.0.1:9880"
    assert config.tts.ref_audio_path == "voices/firefly/ref_firefly_01.wav"
    assert (config.tts.text_lang, config.tts.prompt_lang) == ("en", "en")
    assert (config.tts.text_split_method, config.tts.fragment_interval) == ("cut0", 0.3)
    assert config.tts.cut_punc is None
    assert (config.chat.min_sentence_chars, config.chat.max_sentence_chars) == (12, 120)
    assert (config.asr.language, config.asr.timeout, config.asr.phrase_time_limit) == ("en-US", 5.0, 5.0)
    assert config.chat.exit_commands == ("exit",)
    assert config.training.learning_rate == 5e-5
    assert config.training.model_save_path == tmp_path / "saved_model"
    assert config.training.data_dir == tmp_path / "data"
    assert config.vts.token_path == tmp_path / "pyvts_token.txt"


def test_load_config_yaml_overrides_only_given_keys(tmp_path):
    path = _write(
        tmp_path / "custom.yaml",
        """
llm:
  model: some/other-model
  reasoning_effort: low
tts:
  url: http://192.168.1.106:9880
  cut_punc: ".?!"
asr:
  device_index: 2
  phrase_time_limit: null
training:
  learning_rate: 3e-5      # PyYAML 會讀成字串，必須轉成 float
  epochs: 3
  data_dir: my_data
  data_files: [a.txt, b.md]
chat:
  exit_commands: [exit, quit]
""",
    )

    config = load_config(path, base_dir=tmp_path)

    assert config.llm.model == "some/other-model"
    assert config.llm.reasoning_effort == "low"
    assert config.llm.temperature == 0.5  # 未覆寫的維持預設
    assert config.tts.url == "http://192.168.1.106:9880"
    assert config.tts.cut_punc == ".?!"
    assert config.asr.device_index == 2
    assert config.asr.phrase_time_limit is None
    assert config.training.learning_rate == pytest.approx(3e-5)
    assert isinstance(config.training.learning_rate, float)
    assert config.training.epochs == 3.0
    assert config.training.data_dir == tmp_path / "my_data"
    assert config.training.data_files == ("a.txt", "b.md")
    assert config.chat.exit_commands == ("exit", "quit")


def test_relative_paths_resolve_against_base_dir_not_cwd(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    base = tmp_path / "project"
    path = _write(tmp_path / "c.yaml", "training:\n  model_save_path: models/zora\n  output_dir: /abs/outputs\n")

    config = load_config(path, base_dir=base)

    assert config.training.model_save_path == base / "models" / "zora"
    assert config.training.output_dir == Path("/abs/outputs")


def test_ref_audio_path_is_kept_as_written_for_the_tts_client_to_resolve(tmp_path):
    # 相對路徑由 TTS client 在送出前轉成絕對路徑（見 test_tts.py）；設定值本身保持原樣
    path = _write(tmp_path / "c.yaml", "tts:\n  ref_audio_path: voices/zora/ref.wav\n")

    config = load_config(path, base_dir=tmp_path)

    assert config.tts.ref_audio_path == "voices/zora/ref.wav"


def test_explicit_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", base_dir=tmp_path)


def test_unknown_key_raises_with_its_location(tmp_path):
    path = _write(tmp_path / "c.yaml", "llm:\n  temprature: 0.9\n")

    with pytest.raises(ConfigError, match=r"'llm'.*temprature"):
        load_config(path, base_dir=tmp_path)


def test_wrong_type_raises_with_key_name(tmp_path):
    path = _write(tmp_path / "c.yaml", "asr:\n  timeout: soon\n")

    with pytest.raises(ConfigError, match=r"asr\.timeout"):
        load_config(path, base_dir=tmp_path)


def test_invalid_value_is_rejected(tmp_path):
    path = _write(tmp_path / "c.yaml", "device: tpu\n")

    with pytest.raises(ConfigError, match="device"):
        load_config(path, base_dir=tmp_path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        # api_v2 沒有 server 端的預設參考音檔
        ("tts:\n  ref_audio_path: ''\n", "ref_audio_path is required for gpt-sovits-v2"),
        # 舊版 api.py 缺一項就會悄悄改用 server 預設的參考音檔
        ("tts:\n  engine: gpt-sovits-v1\n  prompt_text: ''\n", "prompt_text is required"),
    ],
)
def test_incomplete_reference_audio_settings_are_rejected(tmp_path, text, message):
    path = _write(tmp_path / "c.yaml", text)

    with pytest.raises(ConfigError, match=message):
        load_config(path, base_dir=tmp_path)


def test_v1_may_leave_the_reference_to_the_server_default(tmp_path):
    path = _write(tmp_path / "c.yaml", "tts:\n  engine: gpt-sovits-v1\n  ref_audio_path: ''\n  prompt_text: ''\n")

    assert load_config(path, base_dir=tmp_path).tts.ref_audio_path == ""


@pytest.mark.parametrize(
    ("engine", "language", "valid"),
    [
        ("gpt-sovits-v2", "ko", True),
        ("gpt-sovits-v2", "all_yue", True),
        ("gpt-sovits-v2", "auto_yue", True),
        ("gpt-sovits-v1", "all_ja", True),
        ("gpt-sovits-v1", "ko", False),  # 舊版 api.py 不支援韓文與粵語
        ("gpt-sovits-v1", "yue", False),
        ("gpt-sovits-v2", "EN", False),  # api_v2 的 POST 不會轉小寫
        ("gpt-sovits-v2", "english", False),
    ],
)
def test_language_codes_depend_on_the_engine(tmp_path, engine, language, valid):
    for key in ("text_lang", "prompt_lang"):
        path = _write(tmp_path / "c.yaml", f"tts:\n  engine: {engine}\n  {key}: {language}\n")
        if valid:
            assert getattr(load_config(path, base_dir=tmp_path).tts, key) == language
        else:
            with pytest.raises(ConfigError, match=rf"tts\.{key} must be one of auto, .* for {engine}"):
                load_config(path, base_dir=tmp_path)


def test_old_v1_setting_names_are_reported_as_unknown(tmp_path):
    path = _write(tmp_path / "c.yaml", "tts:\n  refer_wav_path: ref.wav\n  text_language: en\n")

    with pytest.raises(ConfigError, match=r"'tts'.*refer_wav_path, text_language.*valid keys:.*ref_audio_path"):
        load_config(path, base_dir=tmp_path)


def test_empty_yaml_file_means_defaults(tmp_path):
    path = _write(tmp_path / "c.yaml", "# only comments\n")

    config = load_config(path, base_dir=tmp_path)

    assert config.tts.url == "http://127.0.0.1:9880"


def test_example_config_documents_every_setting_with_the_default_values(tmp_path, monkeypatch):
    example_path = config_module.BASE_DIR / "config.example.yaml"
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml")

    raw = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    for item in dataclasses.fields(AppConfig):
        value = getattr(AppConfig(), item.name)
        if dataclasses.is_dataclass(value):
            assert set(raw[item.name]) == {sub.name for sub in dataclasses.fields(value)}, item.name
        else:
            assert item.name in raw
    assert load_config(example_path, base_dir=tmp_path) == load_config(base_dir=tmp_path)


def test_base_dir_is_the_project_root_holding_the_default_files():
    # 相對路徑的預設值以 BASE_DIR 為基準，要指到 repo 裡實際存在的檔案（參考音檔見 test_tts.py）
    base = config_module.BASE_DIR
    training = TrainingConfig()

    assert (base / "pyproject.toml").is_file()
    assert all((base / training.data_dir / name).is_file() for name in training.data_files)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("llm:\n  reasoning_effort: extreme\n", "llm.reasoning_effort"),
        ("llm:\n  temperature: 3\n", "llm.temperature"),
        ("llm:\n  temperature: .nan\n", "llm.temperature"),
        ("llm:\n  max_retries: -1\n", "llm.max_retries"),
        ("generation:\n  temperature: 0\n", "generation.temperature"),
        ("generation:\n  top_p: 0\n", "generation.top_p"),
        ("generation:\n  top_p: 1.5\n", "generation.top_p"),
        ("asr:\n  timeout: -1\n", "asr.timeout"),
        ("asr:\n  device_index: -1\n", "asr.device_index"),
        ("vts:\n  fade_time: 5\n", "vts.fade_time"),
        ("tts:\n  engine: gpt-sovits\n", "tts.engine"),
        ("tts:\n  text_split_method: cut9\n", "tts.text_split_method"),
        ("tts:\n  speed_factor: 0\n", "tts.speed_factor"),
        ("tts:\n  fragment_interval: -0.1\n", "tts.fragment_interval"),
        ("tts:\n  top_k: 0\n", "tts.top_k"),
        ("tts:\n  top_p: 1.5\n", "tts.top_p"),
        ("tts:\n  temperature: -1\n", "tts.temperature"),
        ("tts:\n  ready_timeout: 0\n", "tts.ready_timeout"),
        ("chat:\n  min_sentence_chars: -1\n", "chat.min_sentence_chars"),
        ("chat:\n  min_sentence_chars: 50\n  max_sentence_chars: 50\n", "chat.max_sentence_chars"),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, text, message):
    path = _write(tmp_path / "c.yaml", text)

    with pytest.raises(ConfigError, match=message.replace(".", r"\.")):
        load_config(path, base_dir=tmp_path)


@pytest.mark.parametrize("effort", ["none", "default", "low", "medium", "high"])
def test_supported_reasoning_efforts_are_accepted(tmp_path, effort):
    path = _write(tmp_path / "c.yaml", f"llm:\n  reasoning_effort: {effort}\n")

    assert load_config(path, base_dir=tmp_path).llm.reasoning_effort == effort


@pytest.mark.parametrize(("value", "expected"), [("null", None), ("true", True), ("false", False)])
def test_include_reasoning_can_be_set_on_its_own(tmp_path, value, expected):
    path = _write(tmp_path / "c.yaml", f"llm:\n  reasoning_effort: null\n  include_reasoning: {value}\n")

    config = load_config(path, base_dir=tmp_path).llm

    assert (config.reasoning_effort, config.include_reasoning) == (None, expected)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:abc",  # port 不是數字
        "http://[::1",  # IPv6 少了 ]
        "http://127.0.0.1:98800",  # port 超出範圍（httpx 解析得過，連線時才會出錯）
        "http://127.0.0.1:0",
        "http://:9880",  # 沒有主機名稱
        "http://xn--:9880",  # 不合法的國際化網域名稱
        "ftp://127.0.0.1:9880",
        "http://127.0.0.1:9880/?debug=1",
    ],
)
def test_malformed_tts_urls_are_rejected_when_loading(tmp_path, url):
    path = _write(tmp_path / "c.yaml", f'tts:\n  url: "{url}"\n')

    with pytest.raises(ConfigError, match=r"tts\.url"):
        load_config(path, base_dir=tmp_path)


@pytest.mark.parametrize(
    "url", ["HTTP://LOCALHOST:9880", "https://tts.example.com", "http://[::1]:9880", "http://192.168.1.106:9880/gsv/"]
)
def test_valid_tts_urls_are_accepted(tmp_path, url):
    path = _write(tmp_path / "c.yaml", f'tts:\n  url: "{url}"\n')

    assert load_config(path, base_dir=tmp_path).tts.url == url


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ('tts:\n  ref_audio_path: "~nosuchuser/ref.wav"\n', "tts.ref_audio_path"),
        ('vts:\n  token_path: "~nosuchuser/token.txt"\n', "vts.token_path"),
        ('training:\n  base_model: "~nosuchuser/model"\n', "a model setting"),
    ],
)
def test_paths_of_unknown_users_are_config_errors_not_crashes(tmp_path, text, key):
    # pathlib 對不存在的使用者會丟 RuntimeError；要轉成指出欄位的 ConfigError
    path = _write(tmp_path / "c.yaml", text)

    with pytest.raises(ConfigError, match=f"~nosuchuser.*{key}|{key}.*~nosuchuser"):
        load_config(path, base_dir=tmp_path)


def test_home_relative_reference_path_is_accepted(tmp_path):
    path = _write(tmp_path / "c.yaml", 'tts:\n  ref_audio_path: "~/voices/ref.wav"\n')

    assert load_config(path, base_dir=tmp_path).tts.ref_audio_path == "~/voices/ref.wav"


def test_unquoted_yaml_booleans_in_text_settings_get_a_quoting_hint(tmp_path):
    path = _write(tmp_path / "c.yaml", "chat:\n  exit_commands: [exit, stop, off]\n")

    with pytest.raises(ConfigError, match=r"exit_commands\[2\].*quotes"):
        load_config(path, base_dir=tmp_path)


def test_llm_retries_default_to_one(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml")

    assert load_config(base_dir=tmp_path).llm.max_retries == 1


def test_model_settings_accept_hub_ids_and_local_paths_relative_to_base_dir(tmp_path):
    (tmp_path / "my_models" / "sst2").mkdir(parents=True)
    path = _write(
        tmp_path / "c.yaml",
        "training:\n  base_model: ./models/gpt2\nsentiment:\n  model: my_models/sst2\n",
    )

    config = load_config(path, base_dir=tmp_path)

    assert config.training.base_model == str(tmp_path / "models" / "gpt2")  # ./ 開頭：本機路徑
    assert config.sentiment.model == str(tmp_path / "my_models" / "sst2")  # base_dir 底下存在：本機路徑


def test_hub_model_ids_are_left_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "missing.yaml")

    config = load_config(base_dir=tmp_path)

    assert config.training.base_model == "gpt2-medium"
    assert config.sentiment.model == "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
