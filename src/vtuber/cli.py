"""命令列入口（`aivtuber` / `python -m vtuber`）：保留舊版 run_3.py 的所有參數與語意，另外新增 --config 與 --log-level。

模式優先順序與舊版相同：--train > --generate > --classify > 聊天（沒給任何模式旗標時也是聊天）。
--api / --vts_control 會連上 VTube Studio，供聊天模式與 --classify 使用。
各模式需要的模組在分派時才 import，因此聊天模式不會載入 torch。
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from vtuber.config import BASE_DIR, AppConfig, VTSConfig, load_config
from vtuber.errors import ConfigError, VTuberError

logger = logging.getLogger("vtuber")

DEFAULT_INPUT_TEXT = "Hello, I'm a language model"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    """prog 為 None 時由 argparse 依執行的檔名決定（例如 aivtuber）。"""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="AI VTuber: voice chat (speech recognition -> Groq LLM -> GPT-SoVITS TTS), "
        "GPT-2 fine-tuning/generation and sentiment-driven VTube Studio expressions. "
        "Without a mode flag it starts the voice chat.",
    )
    modes = parser.add_argument_group("modes (legacy run_3.py flags)")
    modes.add_argument("--train", action="store_true", help="Fine-tune GPT-2 on the training data.")
    modes.add_argument(
        "--generate", action="store_true", help="Generate text with the fine-tuned GPT-2 and with Groq, print both."
    )
    modes.add_argument(
        "--classify",
        action="store_true",
        help="Classify the sentiment of --input_text (switches the VTube Studio expression with --api).",
    )
    modes.add_argument(
        "--input_text", type=str, default=DEFAULT_INPUT_TEXT, help="Input text for --generate / --classify."
    )
    modes.add_argument(
        "--api", action="store_true", help="Connect to VTube Studio (voice chat unless another mode is given)."
    )
    modes.add_argument("--vts_control", action="store_true", help="Same as --api.")
    modes.add_argument(
        "--device_index", type=int, default=None, help="Microphone device index (overrides asr.device_index)."
    )
    modes.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Training data directory, relative to the current directory (default: training.data_dir = data).",
    )
    modes.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Where --train saves and --generate loads the fine-tuned model, relative to the current directory "
        "(default: training.model_save_path = saved_model).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (default: config.yaml in the project root, if it exists).",
    )
    parser.add_argument(
        "--log-level", type=str.upper, default="INFO", choices=LOG_LEVELS, help="Logging level (default: INFO)."
    )
    return parser


def resolve_mode(args: argparse.Namespace) -> str:
    if args.train:
        return "train"
    if args.generate:
        return "generate"
    if args.classify:
        return "classify"
    return "chat"


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """命令列參數優先於設定檔。路徑照一般 CLI 慣例，相對於目前工作目錄解析。"""
    training = config.training
    if args.data_dir is not None:
        training = dataclasses.replace(training, data_dir=Path(args.data_dir).expanduser().resolve())
    if args.model_save_path is not None:
        training = dataclasses.replace(training, model_save_path=Path(args.model_save_path).expanduser().resolve())
    asr = config.asr
    if args.device_index is not None:
        asr = dataclasses.replace(asr, device_index=args.device_index)
    return dataclasses.replace(config, training=training, asr=asr)


def configure_logging(level: str) -> None:
    # --log-level 控制本程式的 log；第三方套件（httpx 每個請求、matplotlib 字型快取…）的 INFO 很吵，
    # 只有 DEBUG 時才一起顯示
    root_level = level if level == "DEBUG" else "WARNING"
    logging.basicConfig(
        level=root_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    logger.setLevel(level)


def load_dotenv_file(path: Path = BASE_DIR / ".env") -> None:
    """有安裝 python-dotenv 且專案根目錄的 .env 存在時載入（不覆蓋已存在的環境變數）。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if path.is_file():
        load_dotenv(path, override=False)
        logger.debug("Loaded environment variables from %s", path)


def main(argv: Optional[Sequence[str]] = None, *, prog: Optional[str] = None) -> int:
    args = build_parser(prog).parse_args(argv)
    configure_logging(args.log_level)
    load_dotenv_file()
    try:
        config = apply_cli_overrides(load_config(args.config), args)
    except ConfigError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2

    mode = resolve_mode(args)
    use_vts = args.api or args.vts_control
    if use_vts and mode in ("train", "generate"):
        logger.warning("--api/--vts_control has no effect in %s mode; VTube Studio is not connected", mode)
    try:
        return _run_mode(mode, config, args.input_text, use_vts)
    except KeyboardInterrupt:
        logger.info("Interrupted by user; shut down cleanly")
        return 130
    except ConfigError as exc:  # 執行到一半才發現的設定問題，例如指定的 device 不存在
        logger.error("Invalid configuration: %s", exc)
        return 2
    except VTuberError as exc:
        logger.error("%s", exc)
        return 1
    except ImportError as exc:
        logger.error(
            'Missing dependency: %s. Install the extras: pip install -e ".[chat]" for voice chat, '
            'pip install -e ".[train]" for --train / --generate / --classify.',
            exc,
        )
        return 1


def _run_mode(mode: str, config: AppConfig, input_text: str, use_vts: bool) -> int:
    if mode == "train":
        from vtuber.training import train

        train(config.training, device=config.device)
        return 0
    if mode == "generate":
        return _generate(config, input_text)
    if mode == "classify":
        return _classify(config, input_text, use_vts)
    asyncio.run(_chat(config, use_vts))
    return 0


def _generate(config: AppConfig, prompt: str) -> int:
    """先用微調 GPT-2 生成、再問 Groq，分別標示輸出；GPT-2 失敗也照樣顯示 Groq 的結果。

    模型的載入與生成都在主執行緒同步執行，第一次下載或載入模型時按 Ctrl+C 會立刻生效。
    """
    from vtuber.generation import TextGenerator

    logger.info("Generating text for: %s", prompt)
    generator = TextGenerator(config.training.model_save_path, config.generation, config.device)
    try:
        finetuned: object = generator.generate(prompt)
    except Exception as exc:  # 顯示錯誤後照常詢問 Groq
        finetuned = exc
    _print_result(f"Fine-tuned GPT-2 ({config.training.model_save_path})", finetuned)
    _print_result(f"Groq ({config.llm.model})", asyncio.run(_ask_groq(config, prompt)))
    if isinstance(finetuned, ConfigError):
        return 2
    return 1 if isinstance(finetuned, Exception) else 0


async def _ask_groq(config: AppConfig, prompt: str) -> str:
    from vtuber.llm import GroqChat

    llm = GroqChat(config.llm)
    try:
        return await llm.reply(prompt)
    finally:
        await llm.aclose()


def _print_result(title: str, result: object) -> None:
    print(f"=== {title} ===", flush=True)
    if isinstance(result, VTuberError):
        print(f"(failed: {result})\n", flush=True)
    elif isinstance(result, ImportError):
        print(f'(failed: {result}; install the train extra: pip install -e ".[train]")\n', flush=True)
    elif isinstance(result, BaseException):
        logger.error("%s failed", title, exc_info=result)
        print(f"(failed: {type(result).__name__}: {result})\n", flush=True)
    else:
        print(f"{result}\n", flush=True)


def _classify(config: AppConfig, text: str, use_vts: bool) -> int:
    """情緒分類在主執行緒同步執行（第一次下載模型時按 Ctrl+C 會立刻生效），再視需要切換 VTS 表情。"""
    from vtuber.sentiment import POSITIVE, SentimentClassifier

    sentiment = SentimentClassifier(config.sentiment.model, config.device).classify(text)
    print(f"Sentiment: {sentiment}", flush=True)
    if not use_vts:
        return 0
    positive, negative = config.vts.positive_expression, config.vts.negative_expression
    expression, other = (positive, negative) if sentiment == POSITIVE else (negative, positive)
    return asyncio.run(_show_expression(config.vts, expression, other))


async def _show_expression(config: VTSConfig, expression: str, other: str) -> int:
    """切換成 expression（先停用另一個情緒表情，避免疊加）；VTS 連不上或切換失敗時回傳 1。"""
    from vtuber.vts import vts_session

    async with vts_session(config) as vts:
        if vts is None:
            return 1
        return 0 if await vts.switch_expression(expression, replaces=[other]) else 1


async def _chat(config: AppConfig, use_vts: bool) -> None:
    from vtuber.chat import run_chat

    if not use_vts:
        await run_chat(config)
        return
    from vtuber.vts import vts_session

    # 聊天期間保持 VTS 連線；結束或 Ctrl+C 時由 vts_session 關閉
    async with vts_session(config.vts):
        await run_chat(config)
