"""Client for a llama.cpp server (OpenAI-compatible endpoint), used for the 35B model.

Every answer is validated by code: it must be exactly one of the three labels and contain no
reasoning trace. Anything else stops the run instead of being silently mapped to a label.
Raw answers are cached in a JSONL file so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from .prompts import ANSWERS, GRAMMAR, INSTRUCTION, messages

LABEL_OF = {v: k for k, v in ANSWERS.items()}


class InvalidAnswer(RuntimeError):
    pass


def request_payload(model: str, text: str) -> dict:
    return {
        "model": model,
        "messages": messages(text),
        "temperature": 0,
        "max_tokens": 8,
        "grammar": GRAMMAR,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_answer(body: dict) -> int:
    msg = body["choices"][0]["message"]
    if (msg.get("reasoning_content") or "").strip():
        raise InvalidAnswer("the model produced a reasoning trace: thinking is not disabled")
    content = (msg.get("content") or "").strip()
    if content not in LABEL_OF:
        raise InvalidAnswer(f"answer {content!r} is not one of {sorted(LABEL_OF)}")
    return LABEL_OF[content]


class Client:
    def __init__(self, url: str, model: str, api_key: str | None = None, timeout: float = 120):
        self.url, self.model, self.timeout = url, model, timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def classify(self, text: str) -> tuple[int, str, float]:
        t0 = time.perf_counter()
        r = requests.post(self.url, headers=self.headers, json=request_payload(self.model, text),
                          timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        return parse_answer(body), body["choices"][0]["message"]["content"], time.perf_counter() - t0


def fingerprint(model: str) -> str:
    """Identifies everything that determines an answer: server model, prompt, grammar, sampling."""
    payload = json.dumps({"model": model, "instruction": INSTRUCTION, "grammar": GRAMMAR,
                          "temperature": 0}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_split(client: Client, ids: list[str], texts: list[str], cache: Path,
              progress_every: int = 100) -> tuple[list[int], list[float]]:
    """Classify every text, reusing cached answers. Returns predictions and per-item latencies.

    The cache is only reused if it was produced with the same model, prompt and grammar.
    """
    meta = cache.with_suffix(".meta.json")
    fp = fingerprint(client.model)
    if meta.exists() and json.loads(meta.read_text())["fingerprint"] != fp:
        raise RuntimeError(f"{cache} was produced with another model, prompt or grammar: "
                           "delete it or use another --name")
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"fingerprint": fp, "model": client.model}))
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
    with open(cache, "a", encoding="utf-8") as f:
        for n, (i, text) in enumerate(zip(ids, texts), 1):
            if i in done:
                continue
            label, raw, seconds = client.classify(text)
            rec = {"id": i, "label": label, "raw": raw, "seconds": round(seconds, 4)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[i] = rec
            if n % progress_every == 0:
                print(f"  {n}/{len(ids)}")
    return [done[i]["label"] for i in ids], [done[i]["seconds"] for i in ids]
