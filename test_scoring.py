"""Unit tests for the pure scoring helpers. No torch, GPU or model weights needed."""
import os

import pytest

from scoring import LABELS, MAX_LEN, encode_ids, label_token_ids, verdict


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids


class _FakeTok:
    """Whitespace tokenizer: each word maps to a stable integer id, so ids are deterministic and distinct."""

    def __init__(self):
        self._vocab = {}

    def _id(self, word):
        return self._vocab.setdefault(word, len(self._vocab) + 1)

    def __call__(self, text, add_special_tokens=True):
        return _Enc([self._id(w) for w in text.split()])


def test_label_token_ids_distinct():
    ids = label_token_ids(_FakeTok())
    assert len(ids) == len(LABELS)
    assert len(set(ids)) == len(LABELS)


def test_label_token_ids_rejects_collision():
    class _Collide(_FakeTok):
        def __call__(self, text, add_special_tokens=True):
            return _Enc([7, 99])  # same leading id for every label

    with pytest.raises(AssertionError):
        label_token_ids(_Collide())


def test_encode_ids_short_text_keeps_prompt_and_text():
    tok = _FakeTok()
    ids = encode_ids(tok, "rates held steady")
    assert ids == tok("Classify the financial sentiment of this text as Bearish, Bullish, or Neutral.\n"
                      "Text: rates held steady\nSentiment:").input_ids


def test_encode_ids_truncates_text_not_prompt():
    tok = _FakeTok()
    long_text = " ".join(f"w{i}" for i in range(500))
    ids = encode_ids(tok, long_text)
    assert len(ids) <= MAX_LEN
    suffix = tok("Sentiment:").input_ids
    # The final "Sentiment:" cue survives: truncation removes text tokens, never the prompt tail.
    assert ids[-len(suffix):] == suffix


def test_verdict_label_changed():
    # argmax clean -> Bullish (index 1), attacked -> Bearish (index 0)
    c, a, line = verdict([0.1, 0.8, 0.1], [0.7, 0.2, 0.1])
    assert (c, a) == ("Bullish", "Bearish")
    assert "Label changed." in line


def test_verdict_label_unchanged():
    c, a, line = verdict([0.2, 0.1, 0.7], [0.3, 0.2, 0.5])
    assert (c, a) == ("Neutral", "Neutral")
    assert "Label unchanged." in line


@pytest.mark.skipif(os.environ.get("HF_HUB_OFFLINE") == "1", reason="offline")
def test_real_qwen_leading_tokens_distinct():
    """The scorer assumption on the real tokenizer: the three label leading sub-tokens are distinct."""
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except Exception:
        pytest.skip("tokenizers/huggingface_hub not installed")
    try:
        path = hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "tokenizer.json")
    except Exception:
        pytest.skip("tokenizer not reachable")
    tk = Tokenizer.from_file(path)
    leading = [tk.encode(" " + label, add_special_tokens=False).ids[0] for label in LABELS]
    assert len(set(leading)) == len(LABELS)
