"""Part 4: ask the models served by llama.cpp (reading measure, stated scenarios, history scenarios).

    python scripts/part4_run.py --split dev
    python scripts/part4_run.py --split dev --model base=qwen3.6-35b-base-i1 --what reading

Each --model is <short name>=<model name known by the llama.cpp router>; models run one after the
other (the router keeps one model in memory). Safe to re-run: answers are cached and an interrupted
run resumes. Writes results/part4/<split>/answers/<short>/{reading,stated,history}.jsonl.
Environment: LLAMA_URL, LLAMA_API_KEY (see docs/setup-wsl.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

from bbg import config, fictional
from bbg.advice import warm_up
from bbg.part4_llm import reading_items, run, scenario_jobs
from bbg.runtime import write_runtime

DEFAULT_MODELS = ["base=qwen3.6-35b-base-i1", "abliterated=qwen3.6-35b-abliterated"]
WHAT = ("reading", "stated", "history")


def answers_dir(split: str, short: str):
    return config.RESULTS_DIR / "part4" / split / "answers" / short


def load_scenarios(split: str) -> list[dict]:
    path = config.DATA_DIR / "part4" / split / "scenarios.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run scripts/part4_generate.py --split {split}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(fictional.SPLITS), required=True)
    ap.add_argument("--model", action="append", help="short=server_name (repeatable)")
    ap.add_argument("--what", action="append", choices=WHAT, help="default: all three")
    args = ap.parse_args()
    models = [m.split("=", 1) for m in (args.model or DEFAULT_MODELS)]
    what = args.what or list(WHAT)

    url = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080/v1/chat/completions")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['LLAMA_API_KEY']}"

    scenarios = load_scenarios(args.split)
    jobs = {"reading": reading_items(), "stated": scenario_jobs(scenarios, "stated"),
            "history": scenario_jobs(scenarios, "history")}
    for short, server_name in models:
        print(f"{short} ({server_name}): " + ", ".join(f"{len(jobs[w])} {w}" for w in what), flush=True)
        try:
            print(f"  loading {server_name} on the server (can take several minutes)...", flush=True)
            print(f"  ready after {warm_up(url, headers, server_name):.0f}s", flush=True)
            for w in what:
                kind = "reading" if w == "reading" else "probability"
                print(f"  {w}:", flush=True)
                recs = run(url, headers, server_name, kind, jobs[w], answers_dir(args.split, short) / f"{w}.jsonl")
                fails = sum(r["error"] is not None for r in recs)
                seconds = sum(r["seconds"] for r in recs)
                tokens = [r["prompt_tokens"] for r in recs if r["prompt_tokens"] is not None]
                print(f"  {w}: {len(recs)} answers, {fails} failures, {seconds / max(len(recs), 1):.2f}s each", flush=True)
                write_runtime(f"part4-{short}", f"{args.split}-{w}", len(recs), seconds, server_model=server_name,
                              failures=fails, mean_prompt_tokens=round(sum(tokens) / len(tokens), 1) if tokens else None,
                              note="sequential requests")
        except requests.ConnectionError:
            print(f"Cannot reach {url}: is llama-server running? (docs/setup-wsl.md)")
            return 1
        except requests.Timeout:
            print("The server did not answer in time. Look at the llama-server window for an error "
                  "(out of memory, model file not found) and send it.")
            return 1
        except requests.HTTPError as exc:
            detail = exc.response.text[:500] if exc.response is not None else ""
            print(f"The server refused the request: {exc}\n{detail}")
            return 1
    print(f"Next: python scripts/part4_tone.py --split {args.split}, then "
          f"python scripts/part4_dev_report.py" if args.split == "dev" else "Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
