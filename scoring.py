"""Pure scoring helpers for the fin-LoRA attack demo.

Kept free of torch/gradio/spaces/peft so they can be unit-tested without a GPU or the model weights.
app.py builds the tensors and runs the model; the label logic lives here and mirrors the published
evaluation (finlora.common) so the demo scores exactly the way the panel's numbers were measured.
"""

LABELS = ["Bearish", "Bullish", "Neutral"]
# Same prompt the adapters were trained with (finlora.common.text_prompt).
PROMPT = "Classify the financial sentiment of this text as Bearish, Bullish, or Neutral.\nText: {text}\nSentiment:"
MAX_LEN = 256


def label_token_ids(tok):
    """Leading sub-token id of each class, the token the eval reads at the last prompt position.

    Same scorer as finlora.common.label_token_ids. Verified distinct under the Qwen2.5 tokenizer
    (" Bearish"/" Bullish" split after "Bear"/"Bull", " Neutral" is a single token), so scoring the
    first sub-token is unambiguous. The assert guards against a tokenizer change ever colliding them.
    """
    ids = [tok(" " + label, add_special_tokens=False).input_ids[0] for label in LABELS]
    assert len(set(ids)) == len(LABELS), "label leading sub-tokens must be distinct"
    return ids


def encode_ids(tok, text):
    """Prompt token ids for `text`, truncating the text (never the prompt) to the MAX_LEN window."""
    # Cutting from the right would drop the final "Sentiment:" cue, so truncate the text only.
    full = tok(PROMPT.format(text=text), add_special_tokens=False).input_ids
    if len(full) > MAX_LEN:
        pre, suf = PROMPT.split("{text}")
        p = tok(pre, add_special_tokens=False).input_ids
        s = tok(suf, add_special_tokens=False).input_ids
        full = p + tok(text, add_special_tokens=False).input_ids[:MAX_LEN - len(p) - len(s)] + s
    return full


def verdict(clean_probs, attacked_probs):
    """Picked label before/after planting and a one-line verdict string."""
    c = LABELS[clean_probs.index(max(clean_probs))]
    a = LABELS[attacked_probs.index(max(attacked_probs))]
    changed = "Label changed." if c != a else "Label unchanged."
    return c, a, f"Clean: **{c}**. With planted sentence: **{a}**. " + changed
