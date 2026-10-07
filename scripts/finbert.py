"""Baseline 3: FinBERT (ProsusAI/finbert), off the shelf.

FinBERT predicts positive / negative / neutral. Mapping: positive -> bullish, negative -> bearish.
The mapping is read from the model config by label name, never by position.
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bbg.data import load_split
from bbg.predictions import write_predictions
from bbg.runtime import gpu_info, timed, write_runtime

NAME_TO_OURS = {"positive": 1, "negative": 0, "neutral": 2}


def label_map(model) -> dict[int, int]:
    names = {i: n.lower() for i, n in model.config.id2label.items()}
    if set(names.values()) != set(NAME_TO_OURS):
        raise ValueError(f"unexpected FinBERT labels: {names}")
    return {int(i): NAME_TO_OURS[n] for i, n in names.items()}


@torch.no_grad()
def predict(model, tok, texts: list[str], batch_size: int, mapping: dict[int, int]) -> list[int]:
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(model.device)
        out += [mapping[j] for j in model(**enc).logits.argmax(-1).tolist()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="ProsusAI/finbert")
    ap.add_argument("--name", default="finbert")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id).to(device).eval()
    mapping = label_map(model)
    print(f"label mapping (finbert id -> ours): {mapping}")

    for split in ("val", "test"):
        df = load_split(split)
        with timed() as t:
            preds = predict(model, tok, df["text"].tolist(), args.batch_size, mapping)
        write_predictions(args.name, split, df["id"], preds)
        write_runtime(args.name, split, len(df), t["seconds"], batch_size=args.batch_size,
                      model_id=args.model_id, **gpu_info())
        print(f"{split}: {len(df)} tweets in {t['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
