"""Part 2 campaign: the same 48 scenarios x 5 seeds asked to each model served by llama.cpp.

    python scripts/run_advice.py
    python scripts/run_advice.py --model base=qwen3.6-35b-base-i1 --model abliterated=qwen3.6-35b-abliterated

Each --model is <short name>=<model name known by the llama.cpp router>. Models are run one
after the other (the router keeps one model in memory). Safe to re-run: answers are cached.
Environment: LLAMA_URL, LLAMA_API_KEY (see docs/setup-wsl.md).
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import requests

from bbg import config
from bbg.advice import build_scenarios, run_campaign
from bbg.runtime import write_runtime

DEFAULT_MODELS = ["base=qwen3.6-35b-base-i1", "abliterated=qwen3.6-35b-abliterated"]
SEEDS = [0, 1, 2, 3, 4]
OUT = config.RESULTS_DIR / "advice"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", help="short=server_name (repeatable)")
    args = ap.parse_args()
    models = [m.split("=", 1) for m in (args.model or DEFAULT_MODELS)]

    url = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080/v1/chat/completions")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['LLAMA_API_KEY']}"

    scenarios = build_scenarios()
    meta = pd.DataFrame(scenarios).drop(columns="text")
    frames = []
    for short, server_name in models:
        print(f"{short} ({server_name}): {len(scenarios)} scenarios x {len(SEEDS)} seeds", flush=True)
        try:
            recs = run_campaign(url, headers, server_name, scenarios, SEEDS, OUT / "raw" / f"{short}.jsonl")
        except requests.ConnectionError:
            print(f"Cannot reach {url}: is llama-server running? (docs/setup-wsl.md)")
            return 1
        df = pd.DataFrame(recs).merge(meta, on="scenario")
        df.insert(0, "model", short)
        frames.append(df)
        write_runtime(f"advice-{short}", "advice", len(df), float(df["seconds"].sum()),
                      server_model=server_name, note="sequential requests")
    answers = pd.concat(frames, ignore_index=True)
    answers.drop(columns="seconds").to_csv(OUT / "answers.csv", index=False)
    print(f"{len(answers)} answers written to {OUT / 'answers.csv'}")
    print("Next: python scripts/analyze_advice.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
