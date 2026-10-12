"""Part 3, step 2: label every sampled headline with the Part 1 models, unchanged.

    python scripts/part3_classify.py --period dev --model all
    python scripts/part3_classify.py --period test --model qwen3-1.7b-lora-s0 --batch-size 32

Each model is first checked against Part 1: it re-labels the Part 1 dev tweets and must agree with
the stored Part 1 predictions on at least 99% of them. Labels are written in chunks to
results/part3/predictions/<period>/<model>.csv (hashed ids only); a stopped run resumes where it
left off. Run with llama-server stopped: the GPU has 8 GB.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bbg import config, part3
from bbg.data import load_split
from bbg.predictions import read_predictions
from bbg.runtime import gpu_info, timed, write_runtime

SCRIPTS = Path(__file__).resolve().parent
CHUNK = 4096
MIN_AGREEMENT = 0.99


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"bbg_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_classifier(name: str, device: str, batch_size: int):
    """Returns a function texts -> label ids, built exactly as in Part 1."""
    if name == "tfidf-logreg":
        bc = load_script("baselines_classic")
        sel = json.loads((config.RESULTS_DIR / "selection" / "tfidf-logreg.json").read_text())["selected"]
        train = load_split("train")
        pipe = bc.make_pipeline(sel["features"], sel["C"], sel["class_weight"]).fit(train["text"], train["label"])
        return lambda texts: [int(x) for x in pipe.predict(texts)]

    import torch
    if name == "finbert":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        fb = load_script("finbert")
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device).eval()
        mapping = fb.label_map(model)
        return lambda texts: fb.predict(model, tok, texts, batch_size, mapping)

    from bbg.llm_scoring import predict
    from bbg.lora import checkpoint_dir, load_base, training_log_path
    if name == "qwen3-1.7b-zeroshot":
        model, tok = load_base("Qwen/Qwen3-1.7B", device)
        model.eval()
        return lambda texts: predict(model, tok, texts, batch_size)[0]
    if name == "qwen3-1.7b-lora-s0":
        from peft import PeftModel
        run = json.loads((config.RESULTS_DIR / "runtime" / f"{name}.json").read_text())["test"]["training_run"]
        log = json.loads(training_log_path(run).read_text(encoding="utf-8"))
        model, tok = load_base(log["config"]["model_id"], device)
        model = PeftModel.from_pretrained(model, checkpoint_dir(run) / "best").eval()

        def run_lora(texts):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                return predict(model, tok, texts, batch_size)[0]
        return run_lora
    raise ValueError(name)


def check_against_part1(name: str, classify) -> float:
    val = load_split("val").sort_values("id")
    stored = read_predictions(config.PREDICTIONS_DIR / "val" / f"{name}.csv", "val").sort_values("id")
    again = np.array(classify(val["text"].tolist()))
    return float(np.mean(again == stored["y_pred"].to_numpy()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(config.PART3_PERIODS), required=True)
    ap.add_argument("--model", choices=part3.MODELS + ["all"], default="all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    heads_path = part3.work_dir(args.period) / "headlines.parquet"
    if not heads_path.exists():
        print(f"{heads_path} not found: run scripts/part3_prepare.py --period {args.period} first")
        return 1
    heads = pd.read_parquet(heads_path, columns=["id", "title"])

    for name in (part3.MODELS if args.model == "all" else [args.model]):
        path = part3.labels_path(args.period, name)
        done = part3.done_ids(path)
        todo = heads[~heads["id"].isin(done)]
        print(f"{name}: {len(done):,} headlines already labelled, {len(todo):,} to go", flush=True)
        if todo.empty:
            continue
        classify = make_classifier(name, args.device, args.batch_size)
        agreement = check_against_part1(name, classify)
        print(f"  agreement with the Part 1 dev predictions: {agreement:.4f}", flush=True)
        if agreement < MIN_AGREEMENT:
            print(f"  below {MIN_AGREEMENT}: this is not the Part 1 model, stopping")
            return 1
        with timed() as t:
            for i in range(0, len(todo), CHUNK):
                part = todo.iloc[i:i + CHUNK]
                part3.append_labels(path, part["id"], classify(part["title"].tolist()))
                print(f"  {min(i + CHUNK, len(todo)):,}/{len(todo):,}", flush=True)
        write_runtime(name, f"part3-{args.period}", len(todo), t["seconds"], batch_size=args.batch_size,
                      part1_agreement=agreement, **gpu_info())
        print(f"  done: {1000 * t['seconds'] / len(todo):.1f} ms per headline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
