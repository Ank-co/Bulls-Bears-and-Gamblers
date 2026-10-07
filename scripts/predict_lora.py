"""Write val and test predictions of a trained LoRA run, under its final model name.

    python scripts/predict_lora.py --run sel-lr2e-4-balanced-s0 --as qwen3-1.7b-lora-s0

Predictions go through the same scoring function as the zero-shot baseline. As a consistency
check, the dev macro-F1 is recomputed and compared with the value logged during training.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from peft import PeftModel

from bbg.data import load_split
from bbg.llm_scoring import predict
from bbg.lora import checkpoint_dir, load_base, training_log_path
from bbg.metrics import macro_f1
from bbg.predictions import save_scores, write_predictions
from bbg.runtime import gpu_info, timed, write_runtime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run name")
    ap.add_argument("--as", dest="final", required=True, help="model name in the results")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    log = json.loads(training_log_path(args.run).read_text(encoding="utf-8"))
    if not log.get("done"):
        print(f"run {args.run} is not finished")
        return 1
    model, tok = load_base(log["config"]["model_id"], args.device)
    model = PeftModel.from_pretrained(model, checkpoint_dir(args.run) / "best").eval()
    use_amp = args.device == "cuda"

    for split in ("val", "test"):
        df = load_split(split)
        if use_amp:
            torch.cuda.reset_peak_memory_stats()
        with timed() as t, torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            preds, scores = predict(model, tok, df["text"].tolist(), args.batch_size)
        write_predictions(args.final, split, df["id"], preds)
        save_scores(args.final, split, df["id"], scores)
        write_runtime(args.final, split, len(df), t["seconds"], batch_size=args.batch_size,
                      training_run=args.run, **gpu_info())
        if split == "val":
            again = macro_f1(df["label"].to_numpy(), np.array(preds))
            logged = log["best"]["dev_macro_f1"]
            status = "OK" if abs(again - logged) < 0.01 else "MISMATCH"
            print(f"dev macro-F1 recomputed {again:.4f} vs logged {logged:.4f}: {status}")
            if status != "OK":
                return 1
    print(f"{args.final}: predictions written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
