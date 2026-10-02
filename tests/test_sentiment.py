from __future__ import annotations

import pytest

from vtuber.config import SentimentConfig
from vtuber.errors import ConfigError
from vtuber.sentiment import NEGATIVE, POSITIVE, normalize_label


@pytest.mark.parametrize(
    ("label", "expected"),
    [("POSITIVE", POSITIVE), ("NEGATIVE", NEGATIVE), ("pos", POSITIVE), ("Negative", NEGATIVE)],
)
def test_sst2_style_labels_map_to_positive_or_negative(label, expected):
    assert normalize_label(label) == expected


@pytest.mark.parametrize("label", ["LABEL_0", "LABEL_1", "neutral"])
def test_labels_of_an_untrained_or_non_binary_head_are_rejected(label):
    # distilbert-base-uncased 加上隨機初始化的分類頭只會有 LABEL_0 / LABEL_1
    with pytest.raises(ConfigError, match=r"sentiment\.model"):
        normalize_label(label)


def test_default_sentiment_model_is_the_finetuned_sst2_checkpoint():
    assert SentimentConfig().model == "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
