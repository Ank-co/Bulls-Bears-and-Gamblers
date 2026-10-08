"""Baseline 5: the 35B MoE served by llama.cpp on the same laptop, zero-shot, grammar-constrained.

Configuration through environment variables (never commit the API key):
    LLAMA_URL      default http://127.0.0.1:8080/v1/chat/completions
    LLAMA_MODEL    model name known by the server, default qwen3.6-35b-base
    LLAMA_API_KEY  the server API key, if any

    python scripts/zero_shot_llamacpp.py --name qwen3.6-35b-a3b-zeroshot
"""
from __future__ import annotations

import argparse
import os
import sys

import requests

from bbg import config
from bbg.advice import warm_up
from bbg.data import load_split
from bbg.llamacpp import Client, run_split
from bbg.predictions import write_predictions
from bbg.runtime import write_runtime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="qwen3.6-35b-a3b-zeroshot")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()

    url = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080/v1/chat/completions")
    model = os.environ.get("LLAMA_MODEL", "qwen3.6-35b-base")
    client = Client(url, model, os.environ.get("LLAMA_API_KEY"))

    try:
        print(f"loading {model} on the server (can take several minutes)...", flush=True)
        print(f"ready after {warm_up(url, client.headers, model):.0f}s", flush=True)
        client.classify("Apple shares rise after record iPhone sales.")
    except (requests.ConnectionError, requests.Timeout):
        print(f"Cannot reach {url}.\n"
              "1. Is llama-server running? On Windows: Invoke-RestMethod http://127.0.0.1:8080/health\n"
              "2. If it answers on Windows but not here, the Windows firewall blocks WSL:\n"
              "   enable mirrored networking (docs/setup-wsl.md) and use the default LLAMA_URL.")
        return 1
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 401:
            print("The server refused the API key: check LLAMA_API_KEY.")
            return 1
        raise

    for split in args.splits:
        df = load_split(split)
        cache = config.RESULTS_DIR / "raw" / args.name / f"{split}.jsonl"
        preds, latencies = run_split(client, df["id"].tolist(), df["text"].tolist(), cache)
        write_predictions(args.name, split, df["id"], preds)
        write_runtime(args.name, split, len(df), sum(latencies), server_model=model,
                      note="sequential requests, one tweet at a time")
        print(f"{split}: {len(df)} tweets, {1000 * sum(latencies) / len(df):.0f} ms per tweet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
