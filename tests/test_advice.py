"""Part 2 checks: scenario grid, strict answer validation, campaign resume, pre-registered tests."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from bbg.advice import (PROFILES, REQUESTS, SITUATIONS, InvalidAdvice, build_scenarios, parse_advice,
                        payload, run_campaign)
from bbg.advice_stats import run_all, scenario_means, h2_fragility


def body(content, finish="stop", reasoning=None):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": finish}]}


def test_grid_is_fully_crossed_and_unique():
    sc = build_scenarios()
    assert len(sc) == 4 * 2 * 2 * 3 == 48
    assert len({s["scenario"] for s in sc}) == 48 and len({s["text"] for s in sc}) == 48
    for s in sc:
        assert PROFILES[s["profile"]] in s["text"] and SITUATIONS[s["situation"]] in s["text"]
        assert REQUESTS[s["request"]] in s["text"] and "€20,000" in s["text"]


def test_payload_is_constrained_and_seeded():
    p = payload("m", build_scenarios()[0], seed=3)
    assert p["seed"] == 3 and p["chat_template_kwargs"] == {"enable_thinking": False}
    assert {k: p[k] for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty")} == {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
    schema = p["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["risky_allocation_pct"] == {"type": "integer", "minimum": 0, "maximum": 100}
    assert list(schema["properties"])[0] == "advice"        # the model explains before deciding


def test_parse_advice_strict():
    ok = parse_advice(body(json.dumps({"advice": "Keep it small.", "risky_allocation_pct": 10,
                                       "leverage": False})))
    assert ok["risky_allocation_pct"] == 10 and ok["leverage"] is False
    bad = [
        {"advice": "x", "risky_allocation_pct": 120, "leverage": False},
        {"advice": "x", "risky_allocation_pct": 10.5, "leverage": False},
        {"advice": "x", "risky_allocation_pct": True, "leverage": False},
        {"advice": "x", "risky_allocation_pct": 10, "leverage": "no"},
        {"advice": " ", "risky_allocation_pct": 10, "leverage": False},
        {"advice": "x", "risky_allocation_pct": 10},
        {"advice": "x", "risky_allocation_pct": 10, "leverage": False, "extra": 1},
    ]
    for d in bad:
        with pytest.raises(InvalidAdvice):
            parse_advice(body(json.dumps(d)))
    with pytest.raises(InvalidAdvice):
        parse_advice(body("not json"))
    with pytest.raises(InvalidAdvice):
        parse_advice(body('{"advice": "x", "risky_al', finish="length"))
    with pytest.raises(InvalidAdvice):
        parse_advice(body(json.dumps({"advice": "x", "risky_allocation_pct": 1, "leverage": False}),
                          reasoning="thinking..."))


@pytest.fixture
def server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(p)
            text = p["messages"][1]["content"]
            pct = 80 if "thrill" in text and p["model"] == "abl" else 10
            data = json.dumps(body(json.dumps({"advice": "ok", "risky_allocation_pct": pct,
                                               "leverage": pct > 50}))).encode()
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


def test_campaign_resumes_and_refuses_foreign_cache(server, tmp_path):
    url, seen = server
    sc = build_scenarios()[:4]
    cache = tmp_path / "abl.jsonl"
    run_campaign(url, {}, "abl", sc[:2], [0, 1], cache)
    assert len(seen) == 4
    with pytest.raises(RuntimeError):                      # grid changed: cache refused
        run_campaign(url, {}, "abl", sc, [0, 1], cache)
    recs = run_campaign(url, {}, "abl", sc[:2], [0, 1], cache)
    assert len(seen) == 4 and len(recs) == 4              # nothing re-asked


def synthetic(effect: float, seed: int = 0) -> pd.DataFrame:
    """Answers for both models; the abliterated model adds `effect` points on gambler/chasing."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in build_scenarios():
        base = {"cautious": 5, "neutral": 20, "gambler": 25, "chasing": 20}[s["profile"]]
        base -= 10 if s["situation"] == "fragile" else 0
        for model in ("base", "abliterated"):
            shift = effect if model == "abliterated" and s["profile"] in ("gambler", "chasing") else 0
            for seed_ in range(5):
                pct = int(np.clip(round(base + shift + rng.normal(0, 4)), 0, 100))
                rows.append({"model": model, **{k: s[k] for k in ("scenario", "profile", "situation",
                                                                   "request", "phrasing")},
                             "seed": seed_, "risky_allocation_pct": pct, "leverage": pct > 40,
                             "advice": "x"})
    return pd.DataFrame(rows)


def test_analysis_detects_a_real_effect_and_not_a_null_one():
    strong = run_all(synthetic(effect=25))["tests"]
    assert strong["H1_pressure"]["p_holm"] < 0.001 and strong["H1_pressure"]["mean_diff"] > 20
    assert strong["H4_leverage"]["abliterated_rate"] > strong["H4_leverage"]["base_rate"]
    null = run_all(synthetic(effect=0))["tests"]
    assert null["H1_pressure"]["p"] > 0.01
    assert abs(null["H2_fragility"]["mean_diff"]) < 3 and abs(null["H3_push"]["mean_diff"]) < 3


def test_contrast_units_are_correctly_paired():
    df = synthetic(effect=0)
    df.loc[(df["model"] == "abliterated") & (df["situation"] == "fragile"), "risky_allocation_pct"] += 7
    r = h2_fragility(scenario_means(df))
    assert r["n"] == 24 and r["mean_diff"] == pytest.approx(7, abs=1.5)
