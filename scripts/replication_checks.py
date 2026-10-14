"""Replication, before the test: format checks that look at no test item.

    python scripts/replication_checks.py

Part 2 has no dev run, and asking the 48 scenarios to check the format would mean reading test
answers. Each model is therefore asked one control scenario written for this check, outside the 48
(another amount, another wording), with the 5 seeds and the exact Part 2 request. The addendum is
checked on the first 20 tweets of the dev split (val), with the exact Part 1 request.
Each answer goes through the same validation as the campaigns (bbg.advice.parse_advice,
bbg.llamacpp.parse_answer); a failure is recorded with its reason instead of stopping.
Writes results/replication/checks.json. Environment: LLAMA_URL, LLAMA_API_KEY (docs/setup-wsl.md).
"""
from __future__ import annotations

import json
import os
import sys

import requests

from bbg import advice, llamacpp
from bbg import replication as R
from bbg.data import load_split

CONTROL_AMOUNT = "€15,000"
CONTROL = {
    "scenario": "control-neutral-solid-open",
    "text": ("Quick question before our meeting. "
             + advice.REQUESTS["open"].replace(advice.AMOUNT, CONTROL_AMOUNT) + " "
             + advice.PROFILES["neutral"] + " " + advice.SITUATIONS["solid"] + advice.INSTRUCTION),
}
SEEDS = [0, 1, 2, 3, 4]
TWEETS = 20


def check_model(url: str, headers: dict, server: str, tweets) -> dict:
    out = {"advice": [], "addendum": []}
    for seed in SEEDS:
        body = advice.post(url, headers, advice.payload(server, CONTROL, seed))
        usage = body.get("usage") or {}
        try:
            ans = advice.parse_advice(body)
            out["advice"].append({"seed": seed, "error": None, "risky_allocation_pct": ans["risky_allocation_pct"],
                                  "leverage": ans["leverage"], "advice_words": len(ans["advice"].split()),
                                  "completion_tokens": usage.get("completion_tokens")})
        except advice.InvalidAdvice as exc:
            out["advice"].append({"seed": seed, "error": str(exc), "completion_tokens": usage.get("completion_tokens")})
    client = llamacpp.Client(url, server, os.environ.get("LLAMA_API_KEY"))
    for i, text in tweets:
        r = requests.post(client.url, headers=client.headers, json=llamacpp.request_payload(server, text), timeout=120)
        r.raise_for_status()
        try:
            label = llamacpp.parse_answer(r.json())
            out["addendum"].append({"id": i, "error": None, "label": int(label)})
        except llamacpp.InvalidAnswer as exc:
            out["addendum"].append({"id": i, "error": str(exc)})
    return out


def main() -> int:
    url = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080/v1/chat/completions")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['LLAMA_API_KEY']}"
    dev = load_split("val").head(TWEETS)
    tweets = list(zip(dev["id"], dev["text"]))
    results = {"control_scenario": CONTROL, "seeds": SEEDS, "dev_tweets": [i for i, _ in tweets], "models": {}}
    try:
        for pair in R.REPLICATIONS:
            for role in R.ROLES:
                server = R.PAIRS[pair]["server"][role]
                print(f"{server}: loading...", flush=True)
                advice.warm_up(url, headers, server)
                res = check_model(url, headers, server, tweets)
                results["models"][server] = res
                fa = sum(r["error"] is not None for r in res["advice"])
                fb = sum(r["error"] is not None for r in res["addendum"])
                tokens = [r["completion_tokens"] for r in res["advice"] if r.get("completion_tokens")]
                print(f"  Part 2 control: {len(res['advice'])} answers, {fa} failures, at most "
                      f"{max(tokens) if tokens else '?'} tokens of {advice.MAX_TOKENS}; addendum: "
                      f"{len(res['addendum'])} tweets, {fb} failures", flush=True)
    except requests.ConnectionError:
        print(f"Cannot reach {url}: is llama-server running, and is LLAMA_URL set? (docs/setup-wsl.md)")
        return 1
    R.OUT.mkdir(parents=True, exist_ok=True)
    (R.OUT / "checks.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(r["error"] is not None for m in results["models"].values() for k in m for r in m[k])
    print(f"{total} failures in all. Written to {R.OUT / 'checks.json'}")
    return 0 if total == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
