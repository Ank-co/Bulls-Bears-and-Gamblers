"""Part 3, control C1: does the base model remember the returns of these stocks?

    python scripts/part3_probe.py --period dev

Asks the base Qwen3-1.7B, with no headline, whether a stock closed higher or lower between two
dates (ticker and dates only), for a fixed random sample of ticker-dates. The score is the
log-probability of "higher" minus that of "lower", read from the model's scores of the two
answers (no generated text). If the model does not remember these returns, the score carries no
information about them; part3_analyze.py measures it with the same daily IC as the hypotheses.
Writes results/part3/<period>/probe_scores.csv (ticker, date, score). Resumes if stopped.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from bbg import config, part3
from bbg.runtime import gpu_info, timed, write_runtime

CHUNK = 2048


def make_probe_scorer(device: str, batch_size: int):
    """Returns texts -> list of scores (log-prob "higher" minus log-prob "lower")."""
    import torch
    from bbg.llm_scoring import score_batch
    from bbg.lora import load_base
    model, tok = load_base("Qwen/Qwen3-1.7B", device)
    model.eval()

    def score(texts):
        out = []
        for i in range(0, len(texts), batch_size):
            lp = score_batch(model, tok, texts[i:i + batch_size], part3.PROBE_ANSWERS, part3.probe_messages)
            out += (lp[:, 0] - lp[:, 1]).tolist()
        return out
    _ = torch  # torch is needed by the model; imported here to keep the module importable without it
    return score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(config.PART3_PERIODS), required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ret_path = part3.work_dir(args.period) / "returns.parquet"
    if not ret_path.exists():
        print(f"{ret_path} not found: run scripts/part3_prepare.py --period {args.period} first")
        return 1
    sample = part3.probe_sample(pd.read_parquet(ret_path), config.PART3_PROBE_SIZE, config.PART3_SEED + 1)
    path = part3.probe_path(args.period)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        prev = pd.read_csv(path, parse_dates=["date"])
        done = set(zip(prev["ticker"], prev["date"]))
    todo = sample[[(t, d) not in done for t, d in zip(sample["ticker"], sample["date"])]]
    print(f"memory probe: {len(sample) - len(todo):,} done, {len(todo):,} to go", flush=True)
    if todo.empty:
        return 0
    score = make_probe_scorer(args.device, args.batch_size)
    with timed() as t:
        for i in range(0, len(todo), CHUNK):
            part = todo.iloc[i:i + CHUNK]
            texts = [part3.probe_question(r.ticker, r.t0, r.t1) for r in part.itertuples()]
            out = pd.DataFrame({"ticker": part["ticker"], "date": part["date"].dt.strftime("%Y-%m-%d"),
                                "score": score(texts)})
            out.to_csv(path, mode="a", header=not path.exists(), index=False)
            print(f"  {min(i + CHUNK, len(todo)):,}/{len(todo):,}", flush=True)
    write_runtime("qwen3-1.7b-memory-probe", f"part3-{args.period}", len(todo), t["seconds"],
                  batch_size=args.batch_size, **gpu_info())
    return 0


if __name__ == "__main__":
    sys.exit(main())
