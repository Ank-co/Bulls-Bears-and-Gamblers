"""Part 4 test analysis, on planted answers: it finds planted effects, finds nothing where nothing is
planted, applies the pre-registered reading, handles failures, and logs every run on the test split."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from bbg import config, fictional
from bbg import part4_stats as S
from bbg.part4_llm import reading_items

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(f"t_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "results" / "metrics")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("frozen")
    frozen = {"smallest_effects": {"effects": {"H1": 0.118, "H2": 0.075, "H3": 0.11, "H4": 0.1, "H5": 0.063}},
              "simulation": {"measurement_bias": {"H1": -0.0135, "H2": -0.0009, "H3": -0.0078, "H4": 0.0049,
                                                  "H5": 0.0014}}}
    (tmp_path / "results/part4/dev").mkdir(parents=True)
    (tmp_path / "results/part4/dev/power.json").write_text(json.dumps(frozen))
    return tmp_path, prereg


def run(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"] + argv)
    return mod.main()


def write_answers(root, scenarios, base_w, shift, slope, fail_every=0, seed=0):
    rng = np.random.default_rng(seed)
    out = root / "results/part4/test/answers"
    for m in ("base", "abliterated"):
        (out / m).mkdir(parents=True, exist_ok=True)
        rows = []
        for k, sc in enumerate(scenarios):
            r = S.regressors(sc)
            x = base_w["const"] + sum(base_w[t] * r[t] for t in S.TERMS) + rng.normal(0, 0.6)
            if m == "abliterated":
                x += shift + slope * r["ideal"]
            fail = fail_every and k % fail_every == 0
            rows.append({"id": sc["id"], "value": None if fail else int(round(100 / (1 + math.exp(-x)))),
                         "error": "not JSON" if fail else None, "seconds": 1.0, "prompt_tokens": 300})
        (out / m / "stated.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        hist = [{"id": s["id"], "value": 60, "error": None, "seconds": 1.0, "prompt_tokens": 400}
                for s in scenarios if s["in_history_subset"]]
        (out / m / "history.jsonl").write_text("\n".join(json.dumps(r) for r in hist) + "\n")
        reading = [{"id": it["id"], "value": "good" if it["version"] == 1 else "bad", "error": None,
                    "seconds": 0.5, "prompt_tokens": 90} for it in reading_items()]
        (out / m / "reading.jsonl").write_text("\n".join(json.dumps(r) for r in reading) + "\n")


def prepare(sandbox, monkeypatch, n=1500):
    root, prereg = sandbox
    monkeypatch.setitem(fictional.SPLITS, "test", {"seed": 42, "scenarios": n, "history": 200})
    gen = load("part4_generate")
    monkeypatch.setattr(gen, "PREREGISTRATION", prereg)
    assert run(gen, ["--split", "test"], monkeypatch) == 0
    path = root / "data/part4/test/scenarios.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(sandbox, monkeypatch):
    root, prereg = sandbox
    mod = load("part4_analyze")
    monkeypatch.setattr(mod, "PREREGISTRATION", prereg)
    assert run(mod, [], monkeypatch) == 0
    return json.loads((root / "results/part4/test/analysis.json").read_text())


def test_planted_effects_are_found_and_read_against_the_smallest_effects(sandbox, monkeypatch):
    scs = prepare(sandbox, monkeypatch)
    write_answers(sandbox[0], scs, {"const": 0.1, "base_rate": 0.7, "e_congruent": 0.8, "e_reversed": 0.4,
                                    "distractor": 0.3, "dramatic_tone": 0.0}, shift=0.3, slope=-0.2)
    res = analyze(sandbox, monkeypatch)
    c, it = res["confirmatory"], res["interpretation"]
    assert c["manipulation_check"]["passed"]
    assert all(c[h]["rejected_holm"] for h in S.DIRECTIONS)
    assert all(it[h]["verdict"] == "confirmed, at least the smallest effect of interest" for h in S.DIRECTIONS)
    assert abs(c["H4"]["estimate"] - 0.3) < 0.06 and abs(c["H5"]["estimate"] + 0.2) < 0.04
    assert all(v["same_conclusion"] for k in ("own_reading", "clear_kinds") for v in res["sensitivity"][k].values())
    log = (sandbox[0] / "results/metrics/test_access.log").read_text()
    assert "part4_analyze" in log


def test_nothing_planted_nothing_confirmed(sandbox, monkeypatch):
    scs = prepare(sandbox, monkeypatch)
    write_answers(sandbox[0], scs, {"const": 0.0, "base_rate": 1.0, "e_congruent": 1.0, "e_reversed": 1.0,
                                    "distractor": 0.0, "dramatic_tone": 0.0}, shift=0.0, slope=0.0, seed=1)
    res = analyze(sandbox, monkeypatch)
    assert not any(res["confirmatory"][h]["rejected_holm"] for h in S.DIRECTIONS)
    assert all(v["verdict"] in ("no effect of interest", "inconclusive") for v in res["interpretation"].values())


def test_failures_are_counted_studied_and_replaced_in_a_sensitivity_analysis(sandbox, monkeypatch):
    scs = prepare(sandbox, monkeypatch)
    write_answers(sandbox[0], scs, {"const": 0.1, "base_rate": 0.7, "e_congruent": 0.8, "e_reversed": 0.4,
                                    "distractor": 0.3, "dramatic_tone": 0.0}, shift=0.3, slope=-0.2, fail_every=20)
    res = analyze(sandbox, monkeypatch)
    assert res["failures"]["base"]["stated"]["failures"] == len(range(0, len(scs), 20))
    assert res["failures"]["base"]["stated_by_scenario"] is not None
    assert set(res["sensitivity"]["failures"]) == {"fifty", "base_rate"}


def test_the_analysis_is_refused_without_the_preregistration(sandbox, monkeypatch):
    mod = load("part4_analyze")
    monkeypatch.setattr(mod, "PREREGISTRATION", sandbox[0] / "missing.md")
    assert run(mod, [], monkeypatch) == 1
    assert not (sandbox[0] / "results/metrics/test_access.log").exists()


def test_interpretation_rules():
    conf = {h: {"estimate": 0.0, "null_value": 0.0, "se": 0.01, "rejected_holm": False} for h in S.DIRECTIONS}
    small = {h: 0.1 for h in S.DIRECTIONS}
    assert all(v["verdict"] == "no effect of interest" for v in S.interpret(conf, small).values())
    conf["H2"].update(estimate=0.05, rejected_holm=True)
    conf["H4"].update(estimate=0.2, rejected_holm=True)
    conf["H1"].update(estimate=-0.05, se=0.05)                     # not rejected, wide interval
    v = S.interpret(conf, small)
    assert v["H2"]["verdict"] == "detected, below the smallest effect of interest"
    assert v["H4"]["verdict"] == "confirmed, at least the smallest effect of interest"
    assert v["H1"]["verdict"] == "inconclusive"
    assert S.holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert S.holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}
