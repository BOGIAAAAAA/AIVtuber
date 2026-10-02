from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
import sys
import sysconfig
import threading
from pathlib import Path
from typing import ClassVar

import pytest

import vtuber.chat
import vtuber.generation
import vtuber.llm
import vtuber.sentiment
import vtuber.vts
from vtuber import cli
from vtuber import config as config_module
from vtuber.config import AppConfig
from vtuber.errors import ConfigError

BASE_DIR = Path(__file__).resolve().parent.parent
LEGACY_FLAGS = (
    "--train", "--generate", "--classify", "--input_text", "--api",
    "--vts_control", "--device_index", "--data_dir", "--model_save_path",
)  # fmt: skip


@pytest.fixture(autouse=True)
def no_local_config(tmp_path, monkeypatch):
    """不讓開發者本機的 config.yaml / .env 影響測試結果，並還原 cli.main() 調整過的 log 等級。"""
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "missing-config.yaml")
    monkeypatch.setattr(cli, "load_dotenv_file", lambda: None)
    package_logger = logging.getLogger("vtuber")
    level = package_logger.level
    yield
    package_logger.setLevel(level)


def test_parser_accepts_every_legacy_flag():
    argv = ["--train", "--generate", "--classify", "--input_text", "I love this!", "--api", "--vts_control"]
    argv += ["--device_index", "2", "--data_dir", "my_data", "--model_save_path", "my_model"]

    args = cli.build_parser().parse_args(argv)

    assert args.train and args.generate and args.classify and args.api and args.vts_control
    assert args.input_text == "I love this!"
    assert args.device_index == 2
    assert (args.data_dir, args.model_save_path) == ("my_data", "my_model")


def test_defaults_match_the_legacy_script():
    args = cli.build_parser().parse_args([])

    assert not any((args.train, args.generate, args.classify, args.api, args.vts_control))
    assert args.input_text == "Hello, I'm a language model"
    assert args.device_index is None
    assert args.config is None
    assert args.log_level == "INFO"


@pytest.mark.parametrize(
    ("argv", "mode"),
    [
        ([], "chat"),
        (["--api"], "chat"),
        (["--vts_control"], "chat"),
        (["--classify", "--api"], "classify"),
        (["--generate", "--classify"], "generate"),
        (["--train", "--generate", "--classify", "--api"], "train"),
    ],
)
def test_mode_precedence_is_the_same_as_before(argv, mode):
    assert cli.resolve_mode(cli.build_parser().parse_args(argv)) == mode


def test_log_level_is_case_insensitive():
    assert cli.build_parser().parse_args(["--log-level", "debug"]).log_level == "DEBUG"


def test_cli_values_override_config_and_paths_are_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    argv = ["--data_dir", "corpus", "--model_save_path", "out/zora", "--device_index", "4"]
    args = cli.build_parser().parse_args(argv)

    config = cli.apply_cli_overrides(AppConfig(), args)

    assert config.training.data_dir == tmp_path.resolve() / "corpus"
    assert config.training.model_save_path == tmp_path.resolve() / "out" / "zora"
    assert config.asr.device_index == 4


def test_without_cli_paths_the_config_defaults_are_kept():
    config = cli.apply_cli_overrides(AppConfig(), cli.build_parser().parse_args([]))

    assert config == AppConfig()


class FakeVTSSession:
    def __init__(self, controller=None) -> None:
        self.controller = controller
        self.entered = self.exited = 0

    @contextlib.asynccontextmanager
    async def __call__(self, config):
        self.entered += 1
        try:
            yield self.controller
        finally:
            self.exited += 1


class FakeController:
    def __init__(self, succeed: bool = True) -> None:
        self.switches: list[tuple[str, list[str]]] = []
        self.succeed = succeed

    async def switch_expression(self, expression: str, *, replaces) -> bool:
        self.switches.append((expression, list(replaces)))
        return self.succeed


def _record_chat(monkeypatch, error: BaseException | None = None) -> list:
    calls = []

    async def fake_run_chat(config):
        calls.append(config)
        if error is not None:
            raise error

    monkeypatch.setattr(vtuber.chat, "run_chat", fake_run_chat)
    return calls


def test_no_flags_starts_chat_without_vts(monkeypatch):
    calls = _record_chat(monkeypatch)
    session = FakeVTSSession()
    monkeypatch.setattr(vtuber.vts, "vts_session", session)

    assert cli.main([]) == 0
    assert len(calls) == 1
    assert session.entered == 0


@pytest.mark.parametrize("flag", ["--api", "--vts_control"])
def test_api_flags_wrap_the_chat_in_a_vts_session(monkeypatch, flag):
    calls = _record_chat(monkeypatch)
    session = FakeVTSSession()
    monkeypatch.setattr(vtuber.vts, "vts_session", session)

    assert cli.main([flag, "--device_index", "1"]) == 0
    assert calls[0].asr.device_index == 1
    assert (session.entered, session.exited) == (1, 1)


def test_ctrl_c_exits_cleanly_with_code_130(monkeypatch):
    _record_chat(monkeypatch, error=KeyboardInterrupt())
    session = FakeVTSSession()
    monkeypatch.setattr(vtuber.vts, "vts_session", session)

    assert cli.main(["--api"]) == 130
    assert session.exited == 1  # VTS 連線在 Ctrl+C 時也關閉


def test_train_mode_runs_training_with_cli_paths(monkeypatch, tmp_path):
    # 舊版 --train 會在 main() 內因 UnboundLocalError 直接崩潰
    import vtuber.training

    calls = []
    monkeypatch.setattr(vtuber.training, "train", lambda config, device: calls.append((config, device)))
    monkeypatch.chdir(tmp_path)

    assert cli.main(["--train", "--data_dir", "data", "--model_save_path", "saved_model", "--api"]) == 0

    ((training_config, device),) = calls
    assert training_config.data_dir == tmp_path.resolve() / "data"
    assert training_config.model_save_path == tmp_path.resolve() / "saved_model"
    assert device == "auto"


def test_invalid_config_returns_exit_code_2(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm:\n  modle: typo\n", encoding="utf-8")

    assert cli.main(["--config", str(bad)]) == 2


class FakeGenerator:
    threads: ClassVar[list] = []

    def __init__(self, model_path, config, device) -> None:
        self.model_path = model_path

    def generate(self, prompt: str) -> str:
        FakeGenerator.threads.append(threading.current_thread())
        return f"GPT-2 says: {prompt} and more"


class FakeGroqChat:
    instances: ClassVar[list] = []

    def __init__(self, config) -> None:
        self.closed = False
        FakeGroqChat.instances.append(self)

    async def reply(self, prompt: str) -> str:
        return f"Groq says hi to {prompt}"

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_generation(monkeypatch):
    FakeGenerator.threads, FakeGroqChat.instances = [], []
    monkeypatch.setattr(vtuber.generation, "TextGenerator", FakeGenerator)
    monkeypatch.setattr(vtuber.llm, "GroqChat", FakeGroqChat)


def test_generate_prints_finetuned_and_groq_results_separately(fake_generation, capsys):
    assert cli.main(["--generate", "--input_text", "Hi Zora"]) == 0

    out = capsys.readouterr().out
    assert "=== Fine-tuned GPT-2" in out and "GPT-2 says: Hi Zora and more" in out
    assert "=== Groq (openai/gpt-oss-20b) ===" in out and "Groq says hi to Hi Zora" in out
    assert out.index("Fine-tuned GPT-2") < out.index("=== Groq")
    # 模型在主執行緒執行（Ctrl+C 能立刻中斷下載／載入），Groq client 用完一定關閉
    assert FakeGenerator.threads == [threading.main_thread()]
    assert [chat.closed for chat in FakeGroqChat.instances] == [True]


def test_generate_without_a_trained_model_still_shows_groq(monkeypatch, capsys, tmp_path):
    FakeGroqChat.instances = []
    monkeypatch.setattr(vtuber.llm, "GroqChat", FakeGroqChat)

    code = cli.main(["--generate", "--model_save_path", str(tmp_path / "not-trained")])

    out = capsys.readouterr().out
    assert code == 1
    assert "(failed: No fine-tuned model found" in out
    assert "Groq says hi to Hello, I'm a language model" in out
    assert [chat.closed for chat in FakeGroqChat.instances] == [True]


class FakeClassifier:
    threads: ClassVar[list] = []

    def __init__(self, model_name, device) -> None:
        self.model_name = model_name

    def classify(self, text: str) -> str:
        FakeClassifier.threads.append(threading.current_thread())
        return "negative" if "sad" in text else "positive"


@pytest.mark.parametrize(
    ("text", "expression", "other"), [("I am so sad", "Sad", "Happy"), ("What a great day", "Happy", "Sad")]
)
def test_classify_with_api_switches_to_the_matching_expression(monkeypatch, capsys, text, expression, other):
    FakeClassifier.threads = []
    controller = FakeController()
    session = FakeVTSSession(controller)
    monkeypatch.setattr(vtuber.sentiment, "SentimentClassifier", FakeClassifier)
    monkeypatch.setattr(vtuber.vts, "vts_session", session)

    assert cli.main(["--classify", "--api", "--input_text", text]) == 0

    # 先停用另一個情緒表情再啟用，避免兩個表情疊加
    assert controller.switches == [(expression, [other])]
    assert (session.entered, session.exited) == (1, 1)  # --classify 也一定關閉 VTS 連線
    assert "Sentiment:" in capsys.readouterr().out
    assert FakeClassifier.threads == [threading.main_thread()]


@pytest.mark.parametrize("controller", [FakeController(succeed=False), None], ids=["switch-failed", "vts-unavailable"])
def test_classify_with_api_returns_non_zero_when_the_expression_was_not_set(monkeypatch, controller):
    monkeypatch.setattr(vtuber.sentiment, "SentimentClassifier", FakeClassifier)
    monkeypatch.setattr(vtuber.vts, "vts_session", FakeVTSSession(controller))

    assert cli.main(["--classify", "--api", "--input_text", "What a great day"]) == 1


def test_configuration_problems_found_at_runtime_exit_with_code_2(monkeypatch):
    class MisconfiguredClassifier(FakeClassifier):
        def classify(self, text: str) -> str:
            raise ConfigError("device 'cuda' is not available on this machine")

    monkeypatch.setattr(vtuber.sentiment, "SentimentClassifier", MisconfiguredClassifier)

    assert cli.main(["--classify"]) == 2


def test_classify_without_api_does_not_touch_vts(monkeypatch, capsys):
    session = FakeVTSSession(FakeController())
    monkeypatch.setattr(vtuber.sentiment, "SentimentClassifier", FakeClassifier)
    monkeypatch.setattr(vtuber.vts, "vts_session", session)

    assert cli.main(["--classify", "--input_text", "I am so sad"]) == 0

    assert session.entered == 0
    assert "Sentiment: negative" in capsys.readouterr().out


def test_chat_path_modules_do_not_import_heavy_packages():
    code = (
        "import sys\n"
        "import vtuber.cli, vtuber.chat, vtuber.llm, vtuber.tts, vtuber.audio, vtuber.asr, vtuber.vts\n"
        "import vtuber.training, vtuber.generation, vtuber.sentiment\n"
        "heavy = ('torch', 'transformers', 'datasets', 'matplotlib', 'pyvts', 'cv2', 'pyaudio')\n"
        "print(sorted(name for name in heavy if name in sys.modules))\n"
    )

    result = subprocess.run([sys.executable, "-c", code], cwd=BASE_DIR, capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[]"


def test_python_dash_m_vtuber_runs(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "vtuber", "--help"], cwd=BASE_DIR, capture_output=True, text=True, check=True
    )

    assert result.stdout.startswith("usage: python -m vtuber")
    assert all(flag in result.stdout for flag in (*LEGACY_FLAGS, "--config", "--log-level"))


def test_aivtuber_console_script_works_from_any_directory(tmp_path):
    script = shutil.which("aivtuber", path=sysconfig.get_path("scripts"))
    assert script, 'aivtuber is not installed in this environment; run pip install -e ".[dev]"'

    result = subprocess.run([script, "--help"], cwd=tmp_path, capture_output=True, text=True, check=True)

    assert result.stdout.startswith("usage: aivtuber")
    assert all(flag in result.stdout for flag in LEGACY_FLAGS)


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ('url: "http://127.0.0.1:abc"', "tts.url is not a valid URL"),
        ('url: "http://[::1"', "tts.url is not a valid URL"),
        ('url: "http://127.0.0.1:98800"', "tts.url port must be between 1 and 65535"),
        ('ref_audio_path: "~nosuchuser/x.wav"', "Invalid path '~nosuchuser/x.wav' in tts.ref_audio_path"),
    ],
)
def test_malformed_tts_settings_exit_with_code_2_and_a_single_error_line(tmp_path, caplog, capsys, setting, expected):
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"tts:\n  {setting}\n", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        code = cli.main(["--config", str(bad)])

    problems = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert code == 2
    assert len(problems) == 1 and problems[0].exc_info is None  # 只有一行錯誤，沒有 traceback
    message = problems[0].getMessage()
    assert message.startswith("Invalid configuration: ") and expected in message and "\n" not in message
    assert "Traceback" not in capsys.readouterr().err
