"""Replication test runs: Part 2, the addendum and Part 4, for one pair, with the original requests.

    python scripts/replication_run.py --pair gemma
    python scripts/replication_run.py --pair llama --what part2 --what addendum

Refused until docs/replication-preregistration.md is committed, unchanged and pushed (contained in
origin/main), and until results/replication/power.json exists. Each model of the pair answers every
experiment while it is loaded. The requests, schemas, sampling and validation are the original ones:
bbg.advice (Part 2, 48 scenarios x 5 seeds), bbg.llamacpp (addendum, 2,388 test tweets) and
bbg.part4_llm (Part 4, the 4,000 frozen test scenarios, 500 history versions, 120 reading prompts).
Answers are cached: an interrupted run resumes. Nothing here computes a result.
Writes results/replication/advice/raw/, results/replication/raw/ and predictions/ (addendum) and
results/part4/test/answers/<model>/ (Part 4, next to the original pair's answers).
Environment: LLAMA_URL, LLAMA_API_KEY (docs/setup-wsl.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

from bbg import advice, config, llamacpp
from bbg import replication as R
from bbg.data import load_split
from bbg.part4_llm import reading_items, run, scenario_jobs
from bbg.predictions import write_predictions
from bbg.runtime import write_runtime

WHAT = ("part2", "addendum", "part4")
SEEDS = [0, 1, 2, 3, 4]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_part2(url, headers, short, server):
    print("  Part 2:", flush=True)
    recs = advice.run_campaign(url, headers, server, advice.build_scenarios(), SEEDS, R.ADVICE_RAW / f"{short}.jsonl")
    write_runtime(f"advice-{short}", "advice", len(recs), sum(r["seconds"] for r in recs), server_model=server,
                  note="sequential requests, replication")
    print(f"  Part 2: {len(recs)} answers", flush=True)


def run_addendum(url, server, name):
    print("  addendum:", flush=True)
    df = load_split("test")
    client = llamacpp.Client(url, server, os.environ.get("LLAMA_API_KEY"))
    preds, lat = llamacpp.run_split(client, df["id"].tolist(), df["text"].tolist(), R.OUT / "raw" / name / "test.jsonl")
    write_predictions(name, "test", df["id"], preds, root=R.PREDICTIONS)
    write_runtime(name, "test", len(df), sum(lat), server_model=server, note="sequential requests, replication")
    print(f"  addendum: {len(df)} tweets, {1000 * sum(lat) / len(df):.0f} ms per tweet", flush=True)


def run_part4(url, headers, short, server, scenarios):
    jobs = {"reading": reading_items(), "stated": scenario_jobs(scenarios, "stated"),
            "history": scenario_jobs(scenarios, "history")}
    for w, js in jobs.items():
        kind = "reading" if w == "reading" else "probability"
        print(f"  Part 4 {w}:", flush=True)
        recs = run(url, headers, server, kind, js, config.RESULTS_DIR / "part4" / "test" / "answers" / short / f"{w}.jsonl")
        fails = sum(r["error"] is not None for r in recs)
        seconds = sum(r["seconds"] for r in recs)
        write_runtime(f"part4-{short}", f"test-{w}", len(recs), seconds, server_model=server, failures=fails,
                      note="sequential requests, replication")
        print(f"  Part 4 {w}: {len(recs)} answers, {fails} failures", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=R.REPLICATIONS, required=True)
    ap.add_argument("--what", action="append", choices=WHAT, help="default: all three")
    args = ap.parse_args()
    what = args.what or list(WHAT)

    ok, detail = R.preregistration_status()
    if not ok:
        print(f"Refused: {detail}.")
        return 1
    if not (R.OUT / "power.json").exists():
        print("Refused: results/replication/power.json is missing (scripts/replication_power.py).")
        return 1
    url = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080/v1/chat/completions")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['LLAMA_API_KEY']}"
    scenarios = read_jsonl(config.DATA_DIR / "part4" / "test" / "scenarios.jsonl")
    R.log_test_access(f"replication_run pair={args.pair} what={what} preregistration={detail[:7]}")

    pair = R.PAIRS[args.pair]
    for role in R.ROLES:
        short, server = pair["short"][role], pair["server"][role]
        print(f"{short} ({server}):", flush=True)
        try:
            print(f"  ready after {advice.warm_up(url, headers, server):.0f}s", flush=True)
            if "part2" in what:
                run_part2(url, headers, short, server)
            if "addendum" in what:
                run_addendum(url, server, pair["addendum"][role])
            if "part4" in what:
                run_part4(url, headers, short, server, scenarios)
        except requests.ConnectionError:
            print(f"Cannot reach {url}: is llama-server running, and is LLAMA_URL set? (docs/setup-wsl.md)")
            return 1
        except (advice.InvalidAdvice, llamacpp.InvalidAnswer) as exc:
            print(f"Stopped on an invalid answer, as in the original campaigns: {exc}")
            return 1
    print(f"Done for {args.pair}. Next: the other pair, then python scripts/replication_analyze.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
