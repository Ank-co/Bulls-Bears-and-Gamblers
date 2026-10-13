"""Part 4: requests, answer validation, resume, reading measure, and the whole dev pipeline against a
fake llama.cpp server whose behaviour is planted, so the analysis must find it back."""
import importlib.util
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from bbg import config, fictional
from bbg import part4_llm as L

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(f"t_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def body(content, finish="stop", reasoning=None):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 8}}


PLANTED = {"base": (0.0, 1.0), "abliterated": (0.3, 0.8)}      # (shift, slope) on the ideal log-odds


def fake_answer(model: str, p: dict, counter: list) -> dict:
    user = p["messages"][-1]["content"]
    kind = p["response_format"]["json_schema"]["name"]
    counter.append(1)
    if model == "broken" and len(counter) % 5 == 0:
        return body('{"probability_up": 1', finish="length")
    if kind == "reading":
        _, t = fictional.read_sentence(user.split("\n\n")[0])
        return body(json.dumps({"reading": "good" if t == 1 else "bad"}))
    if "Over the past 20 weeks" in user:
        return body(json.dumps({"probability_up": 50}))
    shift, slope = PLANTED.get(model, (0.0, 1.0))
    ideal = fictional.parse_stated(user)["ideal"]
    x = shift + slope * fictional.logit(min(max(ideal, 1e-6), 1 - 1e-6))
    return body(json.dumps({"probability_up": int(round(100 / (1 + math.exp(-x))))}))


@pytest.fixture
def server():
    seen, counter = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(p)
            out = (body("OK") if p.get("max_tokens") == 4 else fake_answer(p["model"], p, counter))
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions", seen
    httpd.shutdown()


def test_payload_is_greedy_schema_bound_and_without_thinking():
    p = L.payload("m", "sys", "user", "probability")
    assert p["temperature"] == 0 and p["top_k"] == 1 and p["chat_template_kwargs"] == {"enable_thinking": False}
    schema = p["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["probability_up"] == {"type": "integer", "minimum": 0, "maximum": 100}
    assert L.payload("m", "s", "u", "reading")["response_format"]["json_schema"]["schema"]["properties"][
        "reading"]["enum"] == ["good", "bad", "neither"]


def test_parse_records_every_kind_of_failure_without_raising():
    assert L.parse(body('{"probability_up": 37}'), "probability") == (37, None)
    assert L.parse(body('{"reading": "neither"}'), "reading") == ("neither", None)
    cases = [(body('{"probability_up": 120}'), "out of schema"), (body('{"probability_up": 37.5}'), "out of schema"),
             (body('{"probability_up": true}'), "out of schema"), (body("maybe 40"), "not JSON"),
             (body('{"probability_up": 3, "why": "x"}'), "unexpected fields"),
             (body('{"probability_up": 3', finish="length"), "truncated"),
             (body('{"probability_up": 3}', reasoning="hmm"), "reasoning trace present")]
    for b, reason in cases:
        assert L.parse(b, "probability") == (None, reason)
    assert L.parse(body('{"reading": "great"}'), "reading") == (None, "out of schema")


def test_run_resumes_records_failures_and_refuses_a_foreign_cache(server, tmp_path):
    url, seen = server
    scs = fictional.scenarios("dev", 12)
    jobs = L.scenario_jobs(scs, "stated")
    cache = tmp_path / "broken.jsonl"
    recs = L.run(url, {}, "broken", "probability", jobs[:6], cache)
    assert len(recs) == 6 and sum(r["error"] == "truncated" for r in recs) == 1   # 5th request breaks
    assert all(r["prompt_tokens"] == 300 for r in recs)
    n = len(seen)
    recs = L.run(url, {}, "broken", "probability", jobs[:6], cache)
    assert len(seen) == n and len(recs) == 6                                      # nothing asked again
    with pytest.raises(RuntimeError):
        L.run(url, {}, "broken", "probability", jobs, cache)                      # other jobs: refused
    with pytest.raises(RuntimeError):
        L.run(url, {}, "other-model", "probability", jobs[:6], cache)


def test_history_jobs_are_the_subset_and_reading_items_cover_every_version():
    scs = fictional.scenarios("dev", 120)
    assert [j["id"] for j in L.scenario_jobs(scs, "history")] == [s["id"] for s in scs if s["in_history_subset"]]
    items = L.reading_items()
    assert len(items) == len(fictional.FAMILIES) * 2 * L.READING_VARIANTS
    assert len({i["id"] for i in items}) == len(items) and len({i["user"] for i in items}) == len(items)
    for it in items:
        sentence = it["user"].split("\n\n")[0]
        assert fictional.read_sentence(sentence) == (it["family"], it["version"])   # neutral, not dramatic
        assert "Track record" not in it["user"] and "suggest" not in it["user"]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    return tmp_path


def run(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"] + argv)
    return mod.main()


def test_dev_pipeline_finds_the_planted_behaviour(sandbox, server, monkeypatch):
    url, _ = server
    monkeypatch.setenv("LLAMA_URL", url)
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    assert run(load("part4_generate"), ["--split", "dev"], monkeypatch) == 0
    assert run(load("part4_run"), ["--split", "dev", "--model", "base=base", "--model", "abliterated=abliterated"],
               monkeypatch) == 0

    tone = load("part4_tone")

    def fake_finbert(device="cpu"):
        def classify(texts):
            out = []
            for s in texts:
                _, t = fictional.read_sentence(s)
                dramatic = any(s.startswith(w) for w in ("Euphoria", "Jubilation", "Alarm", "Shock"))
                out.append((t, t * (0.9 if dramatic else 0.5)))
            return out
        return classify
    monkeypatch.setattr(tone, "make_finbert", fake_finbert)
    assert run(tone, ["--split", "dev"], monkeypatch) == 0
    check = json.loads((sandbox / "results/part4/dev/tone_check.json").read_text())
    assert check["dramatic_more_polarized"] and check["mapping"]["fitted_on"] == "dev"

    assert run(load("part4_dev_report"), [], monkeypatch) == 0
    rep = json.loads((sandbox / "results/part4/dev/dev_report.json").read_text())
    base, abl = rep["models"]["base"], rep["models"]["abliterated"]
    assert base["stated"]["failures"] == 0 and base["reading"]["n"] == len(L.reading_items())
    assert base["reading_agreement"]["overall"] == {"agrees": len(L.reading_items())}
    w = base["stated_weights"]["weights"]                                  # answers = ideal, rounded
    for k in ("base_rate", "e_congruent", "e_reversed"):
        assert abs(w[k]["estimate"] - 1) < 0.08
    assert abs(w["distractor"]["estimate"]) < 0.05 and abs(w["dramatic_tone"]["estimate"]) < 0.05
    s = abl["shift_amplification_noise"]
    assert abs(s["shift"]["estimate"] - 0.3) < 0.08 and abs(s["amplification"]["estimate"] - 0.8) < 0.05
    d = rep["base_vs_abliterated"]["shift_and_slope_difference"]
    assert abs(d["shift_difference"]["estimate"] - 0.3) < 0.08 and abs(d["slope_difference"]["estimate"] + 0.2) < 0.06
    assert rep["models"]["base"]["history_distances"]["n"] == fictional.SPLITS["dev"]["history"]
    assert (sandbox / "results/part4/dev/dev_report.md").read_text().startswith("# Part 4, dev run")
