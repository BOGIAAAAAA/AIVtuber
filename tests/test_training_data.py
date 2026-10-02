from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from vtuber.config import TrainingConfig
from vtuber.errors import ConfigError, TrainingDataError
from vtuber.training import (
    TrainingHistory,
    build_lm_examples,
    clean_markdown_text,
    group_into_blocks,
    json_to_text,
    load_and_clean_text_files,
    train,
    training_arguments_kwargs,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_clean_markdown_keeps_case_punctuation_and_contractions():
    text = "AI: Hi! What can I get you today?\nStudent: I'm fine, thanks. I’d like a latte, please."

    assert clean_markdown_text(text) == text


def test_clean_markdown_strips_headings_bold_and_list_markers():
    text = (
        "### Beginner Level - Simple Greetings\n"
        "**Scenario:** Beginner learners need encouragement.\n"
        "1. **Ordering Coffee at a Cafe**\n"
        "   - **情境**: 一個學生在咖啡店點飲品。\n"
        "# Name: Zora\n"
        "* Bullet with __underline bold__ text\n"
    )

    assert clean_markdown_text(text).splitlines() == [
        "Beginner Level - Simple Greetings",
        "Scenario: Beginner learners need encouragement.",
        "Ordering Coffee at a Cafe",
        "情境: 一個學生在咖啡店點飲品。",
        "Name: Zora",
        "Bullet with underline bold text",
    ]


def test_clean_markdown_does_not_eat_leading_numbers_or_dashes_that_are_not_markers():
    assert clean_markdown_text("-5 degrees today.\n1.5 hours later!") == "-5 degrees today.\n1.5 hours later!"


def test_clean_markdown_splits_paragraphs_on_blank_or_rule_lines():
    text = "\n\nAI: Hello!\nStudent: Hi.\n   \n\n---\nAI: Bye!\n\n"

    assert clean_markdown_text(text) == "AI: Hello!\nStudent: Hi.\n\nAI: Bye!"


def test_json_to_text_keeps_keys_such_as_the_speaker_and_skips_non_text():
    data = [
        {"trait": "Patient", "score": 5, "characteristics": ["Waits.", "Listens."], "active": True},
        {"dialogues": [{"scenario": "Mistake", "AI": "It's okay to make mistakes!"}]},
    ]

    assert json_to_text(data) == (
        "trait: Patient\nWaits.\nListens.\n\nscenario: Mistake\nAI: It's okay to make mistakes!"
    )


def test_broken_json_in_a_txt_file_is_reported_with_the_file_name(tmp_path, caplog):
    broken = tmp_path / "personality.txt"
    broken.write_text('[{"trait": "Kind",}]', encoding="utf-8")  # 尾逗號

    with caplog.at_level(logging.WARNING, logger="vtuber.training"):
        documents = load_and_clean_text_files([broken])

    assert documents == ['[{"trait": "Kind",}]']
    assert "personality.txt looks like JSON but cannot be parsed" in caplog.text


def test_load_files_detects_json_by_content_even_with_txt_suffix(tmp_path):
    md = tmp_path / "dialogue.txt"
    md.write_text("## Title\nAI: How's the weather?\n", encoding="utf-8")
    js = tmp_path / "personality.txt"
    js.write_text("\n" + json.dumps([{"trait": "Kind", "description": "Very kind."}]), encoding="utf-8")

    documents = load_and_clean_text_files([md, js])

    assert documents == ["Title\nAI: How's the weather?", "trait: Kind\ndescription: Very kind."]


def test_load_files_missing_file_raises(tmp_path):
    with pytest.raises(TrainingDataError, match="not found"):
        load_and_clean_text_files([tmp_path / "missing.txt"])


def test_real_training_files_are_cleaned_without_destroying_text():
    files = [DATA_DIR / name for name in TrainingConfig().data_files]

    documents = load_and_clean_text_files(files)

    joined = "\n".join(documents)
    assert len(documents) == 4
    assert "AI: Hi! How are you today?" in joined  # 標點與大小寫保留
    assert "I'm good, thank you." in joined  # 縮寫保留
    assert "**" not in joined
    assert not any(line.startswith(("#", "- ")) for line in joined.splitlines())
    # personality_data.txt 是 JSON：要展開成文字，而不是原始的 JSON 語法
    assert "trait: Friendly and Encouraging" in documents[2]
    assert "AI: It's okay to make mistakes, [Student's Name]." in documents[2]  # 說話者標籤保留
    assert '"trait"' not in documents[2]
    assert "{" not in documents[2]


def test_group_into_blocks_concatenates_with_eos_and_drops_remainder():
    blocks = group_into_blocks([[1, 2, 3], [4, 5]], block_size=3, eos_token_id=0)

    assert blocks == [[1, 2, 3], [0, 4, 5]]  # 剩下的 [0] 不足一個 block，捨棄


def test_group_into_blocks_without_eos():
    assert group_into_blocks([[1, 2], [3, 4, 5]], block_size=2, eos_token_id=None) == [[1, 2], [3, 4]]


class FakeTokenizer:
    """每個字元一個 token（ord 值），EOS 為 0。"""

    eos_token_id = 0

    def __call__(self, texts, verbose=True):
        return {"input_ids": [[ord(character) for character in text] for text in texts]}


def test_build_lm_examples_joins_documents_with_eos_and_keeps_labels_unshifted():
    examples = build_lm_examples(["abc", "de"], FakeTokenizer(), block_size=3)

    assert examples["input_ids"] == [[97, 98, 99], [0, 100, 101]]  # 文件之間插入 EOS
    assert examples["labels"] == examples["input_ids"]  # 模型內部會位移，這裡不可先位移
    assert examples["attention_mask"] == [[1, 1, 1], [1, 1, 1]]


def test_build_lm_examples_needs_at_least_two_blocks():
    with pytest.raises(TrainingDataError, match="block_size"):
        build_lm_examples(["abc"], FakeTokenizer(), block_size=3)


def test_training_history_records_train_eval_and_lr():
    history = TrainingHistory()

    history.record(5, {"loss": 3.5, "learning_rate": 1e-5, "grad_norm": 2.0})
    history.record(10, {"eval_loss": 3.0, "epoch": 1.0})
    history.record(10, {"train_loss": 3.2})  # 訓練結束時的平均 loss，不是逐步記錄

    assert history.train_loss == [(5, 3.5)]
    assert history.eval_loss == [(10, 3.0)]
    assert history.learning_rate == [(5, 1e-5)]


@pytest.mark.parametrize(
    ("version", "expected_key", "absent_key"),
    [("4.44.2", "warmup_ratio", "warmup_steps"), ("5.18.0", "warmup_steps", "warmup_ratio")],
)
def test_training_arguments_use_epoch_strategy_and_version_specific_warmup(version, expected_key, absent_key):
    config = TrainingConfig()

    kwargs = training_arguments_kwargs(config, transformers_version=version, use_cpu=False, use_cuda=False)

    assert kwargs["eval_strategy"] == "epoch"
    assert kwargs["save_strategy"] == "epoch"
    assert "evaluation_strategy" not in kwargs  # transformers 4.51 起已移除
    assert "logging_dir" not in kwargs  # transformers 5.15 起已移除
    assert kwargs["learning_rate"] == 5e-5
    assert kwargs[expected_key] == pytest.approx(0.1)
    assert absent_key not in kwargs
    # 保留驗證 loss「最低」的 checkpoint
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_loss"
    assert kwargs["greater_is_better"] is False


@pytest.mark.parametrize(
    ("use_cpu", "use_cuda"), [(True, False), (False, False), (False, True)], ids=["cpu", "mps", "cuda"]
)
def test_mixed_precision_is_only_enabled_on_cuda(use_cpu, use_cuda):
    kwargs = training_arguments_kwargs(
        TrainingConfig(), transformers_version="5.18.0", use_cpu=use_cpu, use_cuda=use_cuda
    )

    assert kwargs["use_cpu"] is use_cpu
    assert kwargs["fp16"] is use_cuda
    assert kwargs["dataloader_pin_memory"] is use_cuda


def test_training_validates_the_configured_device_before_anything_else(monkeypatch, tmp_path):
    import vtuber.device

    requested = []

    def unavailable(preference: str) -> str:
        requested.append(preference)
        raise ConfigError(f"device '{preference}' is not available on this machine")

    monkeypatch.setattr(vtuber.device, "select_device", unavailable)

    with pytest.raises(ConfigError, match="cuda"):
        train(TrainingConfig(data_dir=tmp_path), device="cuda")

    assert requested == ["cuda"]
