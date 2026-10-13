"""Part 4: asking the models served by llama.cpp, and the reading measure.

Every request is greedy (temperature 0), with thinking disabled and the answer forced into a JSON
schema. An answer that still breaks the schema is recorded as a failure with its reason, not retried
and not mapped to a value: the design counts failures and studies them. Network timeouts are
retried by bbg.advice.post (the request is identical, so retrying cannot bias anything). Answers are
cached in JSONL files, so an interrupted run resumes where it stopped; a cache produced with another
model, prompt or schema is refused.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from bbg import fictional
from bbg.advice import post

SCHEMAS = {
    "probability": {"type": "object",
                    "properties": {"probability_up": {"type": "integer", "minimum": 0, "maximum": 100}},
                    "required": ["probability_up"], "additionalProperties": False},
    "reading": {"type": "object",
                "properties": {"reading": {"type": "string", "enum": ["good", "bad", "neither"]}},
                "required": ["reading"], "additionalProperties": False},
}
FIELD = {"probability": "probability_up", "reading": "reading"}
SAMPLING = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "presence_penalty": 0.0, "seed": 0}
MAX_TOKENS = 32

READING_SEED, READING_VARIANTS = 43, 3
READING_QUESTION = ("For the share price of {company}, {sector_with_article} company listed in {country}, is "
                    "this news good, bad or neither? Answer in JSON with one field, \"reading\": \"good\", "
                    "\"bad\" or \"neither\".")


def payload(model: str, system: str, user: str, kind: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        **SAMPLING,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": kind, "strict": True, "schema": SCHEMAS[kind]}},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse(body: dict, kind: str) -> tuple[object, str | None]:
    """(value, None) for a valid answer, (None, reason) otherwise. Never raises on the content."""
    choice = body["choices"][0]
    msg = choice["message"]
    if (msg.get("reasoning_content") or "").strip():
        return None, "reasoning trace present"
    if choice.get("finish_reason") == "length":
        return None, "truncated"
    try:
        data = json.loads(msg.get("content") or "")
    except json.JSONDecodeError:
        return None, "not JSON"
    if not isinstance(data, dict) or set(data) != {FIELD[kind]}:
        return None, "unexpected fields"
    value = data[FIELD[kind]]
    if kind == "probability":
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
            return None, "out of schema"
    elif value not in ("good", "bad", "neither"):
        return None, "out of schema"
    return value, None


def fingerprint(model: str, kind: str, jobs: list[dict]) -> str:
    blob = json.dumps({"model": model, "kind": kind, "schema": SCHEMAS[kind], "sampling": SAMPLING,
                       "max_tokens": MAX_TOKENS, "jobs": [[j["id"], j["system"], j["user"]] for j in jobs]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run(url: str, headers: dict, model: str, kind: str, jobs: list[dict], cache: Path,
        progress_every: int = 25) -> list[dict]:
    """Asks every job once (id, system, user), resuming from the cache. One record per job."""
    meta = cache.with_suffix(".meta.json")
    fp = fingerprint(model, kind, jobs)
    if meta.exists() and json.loads(meta.read_text())["fingerprint"] != fp:
        raise RuntimeError(f"{cache} was produced with another model, prompt or schema")
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"fingerprint": fp, "server_model": model, "kind": kind}))
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
    todo = [j for j in jobs if j["id"] not in done]
    with open(cache, "a", encoding="utf-8") as f:
        for n, job in enumerate(todo, 1):
            t0 = time.perf_counter()
            body = post(url, headers, payload(model, job["system"], job["user"], kind))
            value, error = parse(body, kind)
            usage = body.get("usage") or {}
            rec = {"id": job["id"], "value": value, "error": error,
                   "raw": (body["choices"][0]["message"].get("content") or "")[:200],
                   "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                   "seconds": round(time.perf_counter() - t0, 3)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[job["id"]] = rec
            if n % progress_every == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)
    return [done[j["id"]] for j in jobs]


def scenario_jobs(scenarios: list[dict], version: str) -> list[dict]:
    """Stated: every scenario. History: the scenarios of the history subset."""
    return [{"id": s["id"], "system": s["system"], "user": s[version]} for s in scenarios
            if version == "stated" or s["in_history_subset"]]


def reading_items(variants: int = READING_VARIANTS, seed: int = READING_SEED) -> list[dict]:
    """Every version of every kind of news, in neutral wording, filled with invented names, alone.

    Asked outside the scenarios, with no track record: it measures how a model reads the news. It is
    published and never used to select anything.
    """
    rng = np.random.default_rng(seed)
    banned = fictional.real_place_names()
    items = []
    for v in range(variants):
        names = fictional.invent_names(rng, banned)
        slots = {**names, "pct": int(rng.integers(2, 10)), "years": int(rng.integers(3, 12))}
        article = "an" if names["sector"][0] in "aeiou" else "a"
        question = READING_QUESTION.format(company=names["company"], country=names["country"],
                                           sector_with_article=f"{article} {names['sector']}")
        for fam in fictional.FAMILIES:
            for t in (1, -1):
                sentence = (fam.good if t == 1 else fam.bad).format(**slots)
                items.append({"id": f"reading-{fam.key}-{'good' if t == 1 else 'bad'}-v{v}", "family": fam.key,
                              "version": t, "variant": v, "system": fictional.SYSTEM,
                              "user": f"{sentence}\n\n{question}"})
    return items
