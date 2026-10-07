"""Baseline 4: a small Qwen model, zero-shot, constrained to the three answers.

    python scripts/zero_shot_qwen.py                          # Qwen/Qwen3-1.7B
    python scripts/zero_shot_qwen.py --model-id Qwen/Qwen3-1.7B --batch-size 16
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bbg.data import load_split
from bbg.llm_scoring import predict
from bbg.predictions import save_scores, write_predictions
from bbg.runtime import gpu_info, timed, write_runtime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--name", default="qwen3-1.7b-zeroshot")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available: run scripts/check_gpu.py first.")
        return 1

    tok = AutoTokenizer.from_pretrained(args.model_id)
    tok.padding_side = "left"
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(args.device).eval()

    for split in ("val", "test"):
        df = load_split(split)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with timed() as t:
            preds, scores = predict(model, tok, df["text"].tolist(), args.batch_size)
        write_predictions(args.name, split, df["id"], preds)
        save_scores(args.name, split, df["id"], scores)
        write_runtime(args.name, split, len(df), t["seconds"], batch_size=args.batch_size,
                      model_id=args.model_id, **gpu_info())
        print(f"{split}: {len(df)} tweets in {t['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
