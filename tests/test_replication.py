"""Replication code on simulated data: it finds planted effects and finds nothing where nothing is
planted, applies Holm and the pre-registered criteria (direction, original sign, smallest effect),
refuses answers to different items or produced with other requests, uses the pre-registered units,
and refuses the test runs until the pre-registration is committed and pushed."""
import json
import math
import subprocess

import numpy as np
import pandas as pd
import pytest

from bbg import advice, config, fictional, llamacpp, metrics, predictions
from bbg import part4_stats as S
from bbg import replication as R
from bbg.part4_llm import fingerprint, reading_items, scenario_jobs

SESOI = {"H1": 0.118, "H2": 0.075, "H3": 0.11, "H4": 0.1, "H5": 0.063}
BIAS = {"H1": -0.0135, "H2": -0.0009, "H3": -0.0078, "H4": 0.0049, "H5": 0.0014}
PLANTED = {"const": 0.1, "base_rate": 0.7, "e_congruent": 0.8, "e_reversed": 0.4, "distractor": 0.3, "dramatic_tone": 0.0}
NULL = {"const": 0.0, "base_rate": 1.0, "e_congruent": 1.0, "e_reversed": 1.0, "distractor": 0.0, "dramatic_tone": 0.0}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "results" / "metrics")
    monkeypatch.setattr(R, "OUT", tmp_path / "results" / "replication")
    monkeypatch.setattr(R, "ADVICE_RAW", tmp_path / "results" / "replication" / "advice" / "raw")
    monkeypatch.setattr(R, "PREDICTIONS", tmp_path / "results" / "replication" / "predictions")
    return tmp_path


@pytest.fixture(scope="module")
def scenarios():
    banned = fictional.real_place_names()
    return [fictional.scenario("test", i, banned) for i in range(1500)]


@pytest.fixture(scope="module")
def mod():
    return R.load_script("replication_analyze")


@pytest.fixture(scope="module")
def frozen4():
    return R.load_script("part4_analyze")


# ---------------------------------------------------------------- Part 4


def write_part4(root, scs, pair, base_w, shift, slope, seed=0, drop=None):
    rng = np.random.default_rng(seed)
    jobs = {"reading": reading_items(), "stated": scenario_jobs(scs, "stated"), "history": scenario_jobs(scs, "history")}
    for role in R.ROLES:
        short, server = R.PAIRS[pair]["short"][role], R.PAIRS[pair]["server"][role]
        out = root / "results/part4/test/answers" / short
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for sc in scs:
            r = S.regressors(sc)
            x = base_w["const"] + sum(base_w[t] * r[t] for t in S.TERMS) + rng.normal(0, 0.6)
            if role == "abliterated":
                x += shift + slope * r["ideal"]
            if drop and role == "abliterated" and sc["id"] == drop:
                continue
            rows.append({"id": sc["id"], "value": int(round(100 / (1 + math.exp(-x)))), "error": None,
                         "seconds": 1.0, "prompt_tokens": 300})
        hist = [{"id": s["id"], "value": 60, "error": None, "seconds": 1.0, "prompt_tokens": 400}
                for s in scs if s["in_history_subset"]]
        reading = [{"id": it["id"], "value": "good" if it["version"] == 1 else "bad", "error": None,
                    "seconds": 0.5, "prompt_tokens": 90} for it in jobs["reading"]]
        for w, recs in (("stated", rows), ("history", hist), ("reading", reading)):
            (out / f"{w}.jsonl").write_text("\n".join(json.dumps(x) for x in recs) + "\n")
            kind = "reading" if w == "reading" else "probability"
            (out / f"{w}.meta.json").write_text(json.dumps({"fingerprint": fingerprint(server, kind, jobs[w])}))


def test_part4_planted_effects_meet_both_criteria(sandbox, scenarios, mod, frozen4):
    write_part4(sandbox, scenarios, "gemma", PLANTED, shift=0.3, slope=-0.2)
    res = mod.part4("gemma", scenarios, SESOI, BIAS, frozen4, tone={})
    assert res["confirmatory"]["manipulation_check"]["passed"]
    assert all(res["criteria"][h]["A"] and res["criteria"][h]["B"] for h in S.DIRECTIONS)
    # units: one row per scenario for H1 to H3, one cluster per scenario for H4 and H5
    assert res["units"] == {"scenarios": 1500, "clusters_h4_h5": 1500, "rows_h1_h3": 1500}
    # the replication runs the original code: same numbers as calling it directly
    frames = {role: S.stated_frame(scenarios, [json.loads(x) for x in (
        sandbox / "results/part4/test/answers" / R.PAIRS["gemma"]["short"][role] / "stated.jsonl").read_text().splitlines()])
        for role in R.ROLES}
    direct = S.confirmatory(frames["base"], frames["abliterated"], bias=BIAS)
    assert all(math.isclose(direct[h]["p"], res["confirmatory"][h]["p"]) for h in S.DIRECTIONS)


def test_part4_nothing_planted_nothing_replicated(sandbox, scenarios, mod, frozen4):
    write_part4(sandbox, scenarios, "llama", NULL, shift=0.0, slope=0.0, seed=1)
    res = mod.part4("llama", scenarios, SESOI, BIAS, frozen4, tone={})
    assert not any(res["confirmatory"][h]["rejected_holm"] for h in S.DIRECTIONS)
    assert not any(res["criteria"][h]["A"] or res["criteria"][h]["B"] for h in S.DIRECTIONS)


def test_part4_h5_needs_the_original_sign(sandbox, scenarios, mod, frozen4):
    write_part4(sandbox, scenarios, "gemma", PLANTED, shift=0.3, slope=+0.2)       # steeper, not flatter
    res = mod.part4("gemma", scenarios, SESOI, BIAS, frozen4, tone={})
    assert res["confirmatory"]["H5"]["rejected_holm"]
    assert not res["criteria"]["H5"]["A"] and not res["criteria"]["H5"]["B"]


def test_part4_refuses_different_items_or_requests(sandbox, scenarios, mod, frozen4):
    write_part4(sandbox, scenarios, "gemma", PLANTED, shift=0.3, slope=-0.2, drop=scenarios[7]["id"])
    with pytest.raises(ValueError, match="same items"):
        mod.part4("gemma", scenarios, SESOI, BIAS, frozen4, tone={})
    write_part4(sandbox, scenarios, "gemma", PLANTED, shift=0.3, slope=-0.2)
    meta = sandbox / "results/part4/test/answers/gemma-base/stated.meta.json"
    meta.write_text(json.dumps({"fingerprint": "0000000000000000"}))
    with pytest.raises(ValueError, match="other prompts"):
        mod.part4("gemma", scenarios, SESOI, BIAS, frozen4, tone={})


# ---------------------------------------------------------------- Part 2


def write_part2(pair, pressure=40.0, push=20.0, fragile_abl=-10.0, leverage=True, seed=0):
    rng = np.random.default_rng(seed)
    scs = advice.build_scenarios()
    R.ADVICE_RAW.mkdir(parents=True, exist_ok=True)
    for role in R.ROLES:
        rows = []
        for s in scs:
            for seed_ in (0, 1, 2, 3, 4):
                a = 10.0 - 10.0 * (s["situation"] == "fragile")
                if role == "abliterated":
                    a += pressure * (s["profile"] in ("gambler", "chasing")) + push * (s["request"] == "push")
                    a += (fragile_abl + 10.0) * (s["situation"] == "fragile")
                a = int(np.clip(round(a + 20 + rng.normal(0, 3)), 0, 100))
                lev = bool(leverage and role == "abliterated" and s["profile"] == "chasing" and s["situation"] == "fragile")
                rows.append({"scenario": s["scenario"], "seed": seed_, "advice": "x", "risky_allocation_pct": a,
                             "leverage": lev, "seconds": 1.0})
        short, server = R.PAIRS[pair]["short"][role], R.PAIRS[pair]["server"][role]
        (R.ADVICE_RAW / f"{short}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (R.ADVICE_RAW / f"{short}.meta.json").write_text(json.dumps({"fingerprint": advice.fingerprint(server, scs, [0, 1, 2, 3, 4])}))


def test_part2_planted_effects_and_units(sandbox, mod):
    write_part2("gemma")                                      # H2 has nothing planted
    res = mod.part2("gemma")
    a = {h: v["A"] for h, v in res["criteria"].items()}
    assert a == {"H1_pressure": True, "H2_fragility": False, "H3_push": True, "H4_leverage": True}
    assert {h: res["tests"][h]["n"] for h in mod.UNITS} == mod.UNITS


def test_part2_nothing_planted_or_wrong_sign(sandbox, mod):
    write_part2("llama", pressure=0.0, push=0.0, leverage=False, seed=3)
    assert not any(v["A"] for v in mod.part2("llama")["criteria"].values())
    write_part2("llama", pressure=-40.0, push=0.0, leverage=False, seed=3)  # rejected, opposite direction
    res = mod.part2("llama")
    assert res["tests"]["H1_pressure"]["p_holm"] < 0.05 and not res["criteria"]["H1_pressure"]["A"]


def test_part2_refuses_other_requests(sandbox, mod):
    write_part2("gemma")
    (R.ADVICE_RAW / "gemma-base.meta.json").write_text(json.dumps({"fingerprint": "x"}))
    with pytest.raises(ValueError, match="other prompts"):
        mod.part2("gemma")


# ---------------------------------------------------------------- addendum


@pytest.fixture
def fake_gold(monkeypatch):
    rng = np.random.default_rng(0)
    gold = pd.DataFrame({"id": [f"t{i:04d}" for i in range(600)], "text": "x", "label": rng.integers(0, 3, 600)})
    analysis = R.load_script("replication_analyze")
    monkeypatch.setattr(analysis, "load_split", lambda split: gold)
    monkeypatch.setattr(predictions, "load_split", lambda split, splits_dir=None: gold)
    return gold, analysis


def write_addendum(pair, gold, optimism):
    rng = np.random.default_rng(1)
    y = gold["label"].to_numpy()
    base = np.where(rng.random(len(y)) < 0.8, y, rng.integers(0, 3, len(y)))
    abl = base.copy()
    flip = (y != 1) & (rng.random(len(y)) < optimism)
    abl[flip] = 1
    for role, p in (("base", base), ("abliterated", abl)):
        name, server = R.PAIRS[pair]["addendum"][role], R.PAIRS[pair]["server"][role]
        predictions.write_predictions(name, "test", gold["id"], p, root=R.PREDICTIONS)
        meta = R.OUT / "raw" / name / "test.meta.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({"fingerprint": llamacpp.fingerprint(server), "model": server}))


def test_addendum_planted_optimism(sandbox, fake_gold):
    gold, mod_ = fake_gold
    write_addendum("gemma", gold, optimism=0.25)
    res = mod_.addendum("gemma", R.load_script("analyze_abliteration_sentiment"))
    assert all(v["A"] for v in res["criteria"].values())           # H3: lower macro-F1, the original sign


def test_addendum_nothing_planted(sandbox, fake_gold):
    gold, mod_ = fake_gold
    write_addendum("llama", gold, optimism=0.0)
    res = mod_.addendum("llama", R.load_script("analyze_abliteration_sentiment"))
    assert not any(v["A"] for v in res["criteria"].values())


# ---------------------------------------------------------------- criteria, Holm, power, guard


def test_holm_and_criteria():
    assert S.holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}
    adj = metrics.holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert adj == pytest.approx({"a": 0.03, "b": 0.04, "c": 0.04})
    assert R.criterion_a(True, -0.2, -1) and not R.criterion_a(True, 0.2, -1) and not R.criterion_a(False, -0.2, -1)
    assert "does not show" in R.classify(0) and "three families" in R.classify(2) and "depends" in R.classify(1)
    assert "not confirmed in the original" in R.classify(1, originally="not confirmed")


def test_power_helpers():
    assert R.mcnemar_power(2000, 300, 50, 1.0, 0.05 / 3) > 0.99
    assert R.mcnemar_power(2000, 300, 300, 1.0, 0.05 / 3) < 0.02          # no difference
    rng = np.random.default_rng(0)
    big = rng.normal(30, 10, 24)
    assert R.wilcoxon_power(big, 1.0, 0.0125, reps=300) > 0.95
    assert R.wilcoxon_power(rng.normal(0, 10, 24), 1.0, 0.0125, reps=300) < 0.1
    assert R.normal_power(-0.07, 0.01, 1.0, 0.05 / 3) > 0.99
    # exact McNemar power against a simulation of the same test
    from scipy.stats import binomtest
    n, a, b, alpha = 30, 13, 0, 0.0125
    sims = np.random.default_rng(1).multinomial(n, [a / n, b / n, 1 - (a + b) / n], 4000)
    sim = np.mean([(x + y) > 0 and binomtest(x, x + y, 0.5).pvalue < alpha and x > y for x, y, _ in sims])
    assert abs(R.mcnemar_power(n, a, b, 1.0, alpha) - sim) < 0.03


def git(repo, *args):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo, check=True,
                   capture_output=True)


def test_runs_refused_until_the_preregistration_is_pushed(tmp_path):
    repo, remote = tmp_path / "repo", tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    assert R.preregistration_status(repo)[0] is False                 # missing
    (repo / "docs").mkdir()
    (repo / R.PREREGISTRATION).write_text("plan")
    assert "not committed" in R.preregistration_status(repo)[1]
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "plan")
    git(repo, "remote", "add", "origin", str(remote))
    assert "push it first" in R.preregistration_status(repo)[1]
    git(repo, "push", "-q", "origin", "main")
    assert R.preregistration_status(repo)[0] is True
    (repo / R.PREREGISTRATION).write_text("plan, changed after the push")
    assert "modified" in R.preregistration_status(repo)[1]
