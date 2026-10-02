"""GPT-2 微調（--train）：資料清理 → 切成等長 block → Trainer 訓練 → 存模型與訓練曲線。

torch / transformers / datasets / matplotlib 只在實際訓練或畫圖時才 import，
資料清理等純 Python 函式可以在沒有這些套件的環境下使用與測試。
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from vtuber.config import TrainingConfig
from vtuber.errors import TrainingDataError

logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^#{1,6}\s+")
_LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1")
_HORIZONTAL_RULE = re.compile(r"^(?:[-*_]\s*){3,}$")

# ---------------------------------------------------------------- 資料清理


def clean_markdown_line(line: str) -> str:
    """去掉一行的 markdown 標記（標題 #、清單符號、粗體 **/__、分隔線），保留大小寫與標點。"""
    text = line.strip()
    if _HORIZONTAL_RULE.match(text):
        return ""
    text = _HEADING.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _EMPHASIS.sub(r"\2", text)
    return text.replace("**", "").strip()


def clean_markdown_text(text: str) -> str:
    """逐行清理；空行（或分隔線）視為段落分隔，回傳以一個空行分段的文字。"""
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = clean_markdown_line(raw_line)
        if line:
            current.append(line)
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return "\n\n".join(paragraphs)


def json_to_text(data: Any) -> str:
    """依出現順序把 JSON 的字串值轉成文字行；頂層清單的每個元素各成一段。

    物件裡的字串寫成 "key: value"，保留 "AI: ..." 這類說話者標籤，格式與其他對話檔一致；
    清單裡的字串直接成行；數字、布林不是訓練文字，略過。
    """
    items = data if isinstance(data, list) else [data]
    paragraphs = ("\n".join(_iter_lines(item)) for item in items)
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _iter_lines(node: Any, key: Optional[str] = None) -> Iterator[str]:
    if isinstance(node, str):
        text = node.strip()
        if text:
            yield f"{key}: {text}" if key else text
    elif isinstance(node, dict):
        for child_key, value in node.items():
            yield from _iter_lines(value, str(child_key))
    elif isinstance(node, list):
        for value in node:
            yield from _iter_lines(value)


def load_text_file(path: Path) -> str:
    """讀取一個訓練檔：內容是 JSON（不論副檔名）就展開成文字，否則當作 markdown 清理。"""
    raw = path.read_text(encoding="utf-8-sig")
    is_json_file = path.suffix.lower() == ".json"
    if is_json_file or raw.lstrip()[:1] in ("[", "{"):
        try:
            return json_to_text(json.loads(raw))
        except json.JSONDecodeError as exc:
            if is_json_file:
                raise TrainingDataError(f"Invalid JSON in training file {path}: {exc}") from exc
            logger.warning(
                "%s looks like JSON but cannot be parsed (%s); it will be used as plain text, fix the JSON if "
                "that is not intended",
                path,
                exc,
            )
    return clean_markdown_text(raw)


def load_and_clean_text_files(file_paths: Iterable[Path]) -> list[str]:
    """讀取並清理所有訓練檔，每個檔案回傳一份文件字串（原始檔案不會被修改）。"""
    documents = []
    for path in file_paths:
        if not path.is_file():
            raise TrainingDataError(f"Training data file not found: {path}")
        text = load_text_file(path)
        if text:
            documents.append(text)
        else:
            logger.warning("Training file %s contains no text after cleaning; skipped", path)
    if not documents:
        raise TrainingDataError("No training text found in the configured data files")
    return documents


def group_into_blocks(
    token_ids: Iterable[Sequence[int]], block_size: int, eos_token_id: Optional[int]
) -> list[list[int]]:
    """標準 causal LM 前處理：各文件 token 串接（文件之間插入 EOS）後切成等長 block。

    尾端不足一個 block 的 token 直接捨棄，因此完全不需要 padding。
    """
    stream: list[int] = []
    for ids in token_ids:
        stream.extend(ids)
        if eos_token_id is not None:
            stream.append(eos_token_id)
    usable = len(stream) - len(stream) % block_size
    return [stream[start : start + block_size] for start in range(0, usable, block_size)]


# ---------------------------------------------------------------- 訓練紀錄與圖表


@dataclass
class TrainingHistory:
    """訓練過程中記錄的 (optimizer step, 值)，用來畫訓練曲線。"""

    train_loss: list[tuple[int, float]] = field(default_factory=list)
    eval_loss: list[tuple[int, float]] = field(default_factory=list)
    learning_rate: list[tuple[int, float]] = field(default_factory=list)

    def record(self, step: int, logs: Mapping[str, Any]) -> None:
        for key, series in (
            ("loss", self.train_loss),
            ("eval_loss", self.eval_loss),
            ("learning_rate", self.learning_rate),
        ):
            if key in logs:
                series.append((step, float(logs[key])))


def _perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 700 else math.inf


def _as_perplexity(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    return [(step, _perplexity(loss)) for step, loss in points]


# 圖表配色：dataviz 參考色盤（亮色模式）
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_TRAIN_COLOR = "#2a78d6"
_EVAL_COLOR = "#eb6834"


def save_training_plots(history: TrainingHistory, output_dir: Path) -> list[Path]:
    """把 loss、學習率、困惑度曲線存成 PNG（Agg backend，不開視窗），並輸出原始數值 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    charts: dict[str, tuple[str, str, dict[str, list[tuple[int, float]]]]] = {
        "train_loss.png": (
            "Training and validation loss",
            "Loss",
            {"train": history.train_loss, "validation": history.eval_loss},
        ),
        "learning_rate.png": ("Learning rate", "Learning rate", {"learning rate": history.learning_rate}),
        "perplexity.png": (
            "Perplexity (exp of loss)",
            "Perplexity",
            {"train": _as_perplexity(history.train_loss), "validation": _as_perplexity(history.eval_loss)},
        ),
    }
    saved = []
    for filename, (title, ylabel, series) in charts.items():
        non_empty = {label: points for label, points in series.items() if points}
        if not non_empty:
            logger.warning("No data recorded for %s; chart skipped", filename)
            continue
        _plot_lines(output_dir / filename, title, ylabel, non_empty)
        saved.append(output_dir / filename)

    metrics = {"train_loss": history.train_loss, "eval_loss": history.eval_loss, "learning_rate": history.learning_rate}
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    saved.append(metrics_path)
    return saved


def _plot_lines(path: Path, title: str, ylabel: str, series: dict[str, list[tuple[int, float]]]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import StrMethodFormatter

    figure = Figure(figsize=(8, 4.5), dpi=150, facecolor=_SURFACE)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    axes.set_facecolor(_SURFACE)
    for label, points in series.items():
        steps, values = zip(*points)
        axes.plot(
            steps,
            values,
            label=label,
            color=_EVAL_COLOR if label == "validation" else _TRAIN_COLOR,
            linewidth=1.25,
            solid_capstyle="round",
            solid_joinstyle="round",
            # 點不多時（例如每個 epoch 一次的驗證）加上圓點，外圈用底色隔開線條
            marker="o" if len(points) <= 30 else None,
            markersize=4.5,
            markeredgecolor=_SURFACE,
            markeredgewidth=1,
        )
    axes.set_title(title, color=_INK_PRIMARY, loc="left", fontsize=12)
    axes.set_xlabel("Optimizer step", color=_INK_SECONDARY)
    axes.set_ylabel(ylabel, color=_INK_SECONDARY)
    axes.tick_params(colors=_INK_MUTED, labelcolor=_INK_SECONDARY)
    axes.yaxis.get_offset_text().set_color(_INK_SECONDARY)
    if min(value for points in series.values() for _, value in points) >= 1000:
        axes.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))  # 例如訓練初期的困惑度
    axes.grid(axis="y", color=_GRID, linewidth=0.75)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_BASELINE)
    if len(series) > 1:
        axes.legend(frameon=False, labelcolor=_INK_SECONDARY)
    figure.tight_layout()
    figure.savefig(path, facecolor=_SURFACE)


# ---------------------------------------------------------------- 訓練


def training_arguments_kwargs(
    config: TrainingConfig, *, transformers_version: str, use_cpu: bool, use_cuda: bool
) -> dict[str, Any]:
    """組出 TrainingArguments 參數（相容 transformers 4.41+ 與 5.x）。

    使用 Trainer 內建的 AdamW 與 linear scheduler，學習率與 warmup 只在這裡設定一次；
    每個 epoch 評估與存檔一次，最後載回驗證 loss 最低的 checkpoint。
    """
    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": "linear",
        "weight_decay": config.weight_decay,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_strategy": "steps",
        "logging_steps": config.logging_steps,
        "logging_first_step": True,
        "report_to": config.report_to,
        "seed": config.seed,
        "use_cpu": use_cpu,
        "fp16": use_cuda,
        "dataloader_pin_memory": use_cuda,
    }
    # transformers 5.0 起棄用 warmup_ratio，改由 warmup_steps 接受 [0, 1) 的比例。
    # 5.0 的棄用訊息寫「v5.2 移除」，實際上 5.2–5.14 仍保留這個欄位，到 5.15.0 才移除。
    major = int(transformers_version.split(".")[0])
    kwargs["warmup_steps" if major >= 5 else "warmup_ratio"] = config.warmup_ratio
    return kwargs


def train(config: TrainingConfig, device: str = "auto") -> Path:
    """微調 config.base_model，把最佳模型與 tokenizer 存到 config.model_save_path 並回傳該路徑。"""
    from vtuber.device import select_device

    # 指定的裝置不存在時（例如 device: cuda 但沒有 CUDA）直接報錯，不默默改用其他裝置
    resolved_device = select_device(device)
    logger.info("Training on %s", resolved_device)

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

    documents = load_and_clean_text_files(config.data_dir / name for name in config.data_files)
    logger.info("Loaded %d training documents from %s", len(documents), config.data_dir)

    logger.info("Loading base model %s", config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForCausalLM.from_pretrained(config.base_model)
    block_size = _fit_block_size(config.block_size, model.config)
    splits = _build_dataset(documents, tokenizer, block_size).train_test_split(
        test_size=config.validation_split, seed=config.seed
    )
    logger.info(
        "Training on %d blocks, validating on %d blocks (%d tokens per block)",
        len(splits["train"]), len(splits["test"]), block_size,
    )

    args = TrainingArguments(
        **training_arguments_kwargs(
            config,
            transformers_version=transformers.__version__,
            use_cpu=resolved_device == "cpu",
            use_cuda=resolved_device == "cuda",
        )
    )
    history = TrainingHistory()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        data_collator=default_data_collator,
        callbacks=[_history_callback(history)],
    )
    trainer.train()

    best = trainer.state.best_metric
    if best is not None:
        logger.info("Best validation loss %.4f (perplexity %.2f)", best, _perplexity(best))
    config.model_save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.model_save_path))
    tokenizer.save_pretrained(str(config.model_save_path))
    logger.info("Saved fine-tuned model to %s", config.model_save_path)
    for path in save_training_plots(history, config.output_dir):
        logger.info("Saved training chart/metrics: %s", path)
    return config.model_save_path


def _fit_block_size(requested: int, model_config: Any) -> int:
    limit = getattr(model_config, "n_positions", None) or getattr(model_config, "max_position_embeddings", None)
    if limit and requested > limit:
        logger.warning("block_size %d exceeds the model context (%d); using %d", requested, limit, limit)
        return int(limit)
    return requested


def build_lm_examples(documents: Sequence[str], tokenizer: Any, block_size: int) -> dict[str, list[list[int]]]:
    """tokenize 後切成等長 block，回傳 Dataset.from_dict 用的欄位。

    labels 與 input_ids 完全相同：causal LM 模型會在內部把 labels 位移一格，這裡不可先位移。
    """
    # verbose=False：整份文件一次 tokenize 會超過 model_max_length，但之後會切 block，不需要警告
    encoded = tokenizer(list(documents), verbose=False)["input_ids"]
    blocks = group_into_blocks(encoded, block_size, tokenizer.eos_token_id)
    if len(blocks) < 2:
        total = sum(len(ids) for ids in encoded)
        raise TrainingDataError(
            f"Training text has only {total} tokens, giving {len(blocks)} block(s) of {block_size}; "
            "at least 2 are needed. Lower training.block_size or add more data."
        )
    return {
        "input_ids": blocks,
        "attention_mask": [[1] * block_size for _ in blocks],
        "labels": [list(block) for block in blocks],
    }


def _build_dataset(documents: list[str], tokenizer: Any, block_size: int) -> Any:
    from datasets import Dataset

    return Dataset.from_dict(build_lm_examples(documents, tokenizer, block_size))


def _history_callback(history: TrainingHistory) -> Any:
    from transformers import TrainerCallback

    class HistoryCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict] = None, **kwargs: Any) -> None:
            if logs:
                history.record(state.global_step, logs)

    return HistoryCallback()
