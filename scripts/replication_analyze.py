"""Replication: the pre-registered analysis of the test runs of both new pairs (run once).

    python scripts/replication_analyze.py

Refused until docs/replication-preregistration.md is committed, unchanged and pushed, and until every
answer of both pairs is in. Every run is logged in results/metrics/test_access.log. For each pair and
each experiment, the original tests run through the original code: bbg.advice_stats.run_all (Part 2),
run() of scripts/analyze_abliteration_sentiment.py (addendum), bbg.part4_stats.confirmatory with the
pair's measurement bias and the sensitivity analyses of scripts/part4_analyze.py (Part 4). Before any
test, the code checks that both models of a pair answered the same items with the same requests
(fingerprints of the caches) and that the units are those of the pre-registered tests.
Criterion A (statistical replication): rejected after Holm, in the pre-registered direction, with the
original sign for the two-sided tests. Criterion B (substantive, Part 4 only): A, and the effect
reaches the smallest effect of interest. A test that is not rejected does not show that the effect
is absent. Writes results/replication/analysis.json and analysis.md.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from bbg import advice, advice_stats, config, llamacpp
from bbg import part4_stats as S
from bbg import replication as R
from bbg.data import load_split
from bbg.part4_llm import fingerprint, reading_items, scenario_jobs
from bbg.predictions import prediction_path, read_predictions

SEEDS = [0, 1, 2, 3, 4]
UNITS = {"H1_pressure": 24, "H2_fragility": 24, "H3_push": 24, "H4_leverage": 30}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def part2(pair: str) -> dict:
    scenarios = advice.build_scenarios()
    for role in R.ROLES:
        short, server = R.PAIRS[pair]["short"][role], R.PAIRS[pair]["server"][role]
        R.check_fingerprint((R.ADVICE_RAW / f"{short}.jsonl").with_suffix(".meta.json"),
                            advice.fingerprint(server, scenarios, SEEDS), f"Part 2 {pair} {role}")
    df = R.advice_frame(pair)
    res = advice_stats.run_all(df)
    units = {h: res["tests"][h]["n"] for h in UNITS}
    if units != UNITS:
        raise ValueError(f"Part 2 {pair}: units {units} differ from the pre-registered {UNITS}")
    res["criteria"] = R.part2_criteria(res["tests"])
    return res


def addendum(pair: str, frozen) -> dict:
    gold = load_split("test").sort_values("id")
    ids, y = gold["id"].to_numpy(), gold["label"].to_numpy()
    preds = {}
    for role in R.ROLES:
        name, server = R.PAIRS[pair]["addendum"][role], R.PAIRS[pair]["server"][role]
        R.check_fingerprint(R.OUT / "raw" / name / "test.meta.json", llamacpp.fingerprint(server), f"addendum {pair} {role}")
        df = read_predictions(prediction_path(name, "test", R.PREDICTIONS), "test").sort_values("id")
        R.check_same_items(list(ids), list(df["id"]), f"addendum {pair} {role}")
        preds[role] = df["y_pred"].to_numpy()
    tests = frozen.run(y, preds["base"], preds["abliterated"])
    return {"n": int(len(y)), "tests": tests, "criteria": R.addendum_criteria(tests),
            "models": {role: frozen.describe(y, preds[role]) for role in R.ROLES}}


def part4(pair: str, scenarios: list[dict], sesoi: dict, bias: dict, frozen, tone: dict) -> dict:
    jobs = {"reading": reading_items(), "stated": scenario_jobs(scenarios, "stated"),
            "history": scenario_jobs(scenarios, "history")}
    answers = {}
    for role in R.ROLES:
        short, server = R.PAIRS[pair]["short"][role], R.PAIRS[pair]["server"][role]
        answers[role] = {}
        for w, js in jobs.items():
            path = config.RESULTS_DIR / "part4" / "test" / "answers" / short / f"{w}.jsonl"
            kind = "reading" if w == "reading" else "probability"
            R.check_fingerprint(path.with_suffix(".meta.json"), fingerprint(server, kind, js), f"Part 4 {pair} {role} {w}")
            recs = read_jsonl(path)
            R.check_same_items([j["id"] for j in js], [r["id"] for r in recs], f"Part 4 {pair} {role} {w}")
            answers[role][w] = recs
    frames = {role: S.stated_frame(scenarios, answers[role]["stated"]) for role in R.ROLES}
    R.check_same_items(list(frames["base"]["id"]), list(frames["abliterated"]["id"]), f"Part 4 {pair}")
    conf = S.confirmatory(frames["base"], frames["abliterated"], bias=bias)
    answered = set(frames["base"].dropna(subset=["answer_logodds"])["id"]) | set(
        frames["abliterated"].dropna(subset=["answer_logodds"])["id"])
    res = {"confirmatory": conf, "interpretation": S.interpret(conf, sesoi), "criteria": R.part4_criteria(conf, sesoi),
           "units": {"scenarios": len(scenarios), "clusters_h4_h5": len(answered),
                     "rows_h1_h3": int(frames["base"]["answer_logodds"].notna().sum())},
           "failures": {role: {w: S.failures(answers[role][w]) for w in jobs} for role in R.ROLES}}
    items = reading_items()
    agreement = {role: S.reading_agreement(items, answers[role]["reading"]) for role in R.ROLES}
    res["sensitivity"] = {
        "own_reading": frozen.sensitivity(conf, scenarios, answers["base"]["stated"], S.own_reading_map(agreement["base"])),
        "clear_kinds": frozen.sensitivity(conf, scenarios, answers["base"]["stated"], S.clear_kinds_reading())}
    if any(frames[r]["answer"].isna().any() for r in R.ROLES):
        res["sensitivity"]["failures"] = {}
        for how in ("fifty", "base_rate"):
            c = S.confirmatory(frozen.replaced(frames["base"], how), frozen.replaced(frames["abliterated"], how), bias=bias)
            res["sensitivity"]["failures"][how] = {"decisions": frozen.decisions(c),
                                                   "same_as_main": frozen.decisions(c) == frozen.decisions(conf)}
    res["descriptive"] = {}
    for role in R.ROLES:
        extra = {"tone_reader": frames[role]["id"].map(tone).to_numpy()} if tone else None
        res["descriptive"][role] = {"weights": S.grether(frames[role]),
                                    "shift_amplification_noise": S.shift_amplification_noise(frames[role]),
                                    "accuracy": S.accuracy(frames[role], extra),
                                    "calibration": S.calibration_table(frames[role]),
                                    "reading_agreement": agreement[role]}
    return res


def summary(results: dict) -> dict:
    out = {}
    for exp, hyps in R.ORIGINAL.items():
        out[exp] = {}
        for h, originally in hyps.items():
            a = sum(results[p][exp]["criteria"][h]["A"] for p in R.REPLICATIONS)
            row = {"originally": originally, "A": a, "A_reading": R.classify(a, len(R.REPLICATIONS), originally)}
            if exp == "part4":
                b = sum(results[p][exp]["criteria"][h]["B"] for p in R.REPLICATIONS)
                row.update({"B": b, "B_reading": R.classify(b, len(R.REPLICATIONS), originally)})
            out[exp][h] = row
    return out


def render(res: dict) -> str:
    yes = {True: "yes", False: "no"}
    lines = ["# Replication: pre-registered analysis of the test runs (run once)", "",
             "Criterion A: rejected after Holm in the pre-registered direction (original sign for two-sided tests). "
             "Criterion B (Part 4): A and the effect reaches the smallest effect of interest. "
             "A test that is not rejected does not show that the effect is absent.", ""]
    for pair in R.REPLICATIONS:
        r = res["pairs"][pair]
        lines += [f"## {R.PAIRS[pair]['family']}", "", "| Experiment | Test | Estimate | p (Holm) | A | B |", "|---|---|---|---|---|---|"]
        for h, t in r["part2"]["tests"].items():
            est = (f"{100 * t['abliterated_rate']:.0f}% vs {100 * t['base_rate']:.0f}%" if h == "H4_leverage"
                   else f"{t['mean_diff']:+.1f} points")
            lines.append(f"| Part 2 | {h} | {est} | {t['p_holm']:.3g} | {yes[r['part2']['criteria'][h]['A']]} | |")
        for h, t in r["addendum"]["tests"].items():
            est = f"{t['diff']:+.3f}" if h == "H3_macro_f1" else f"{100 * t['diff']:+.1f} points"
            lines.append(f"| Addendum | {h} | {est} | {t['p_holm']:.3g} | {yes[r['addendum']['criteria'][h]['A']]} | |")
        c4, k4 = r["part4"]["confirmatory"], r["part4"]["criteria"]
        for h in S.DIRECTIONS:
            lines.append(f"| Part 4 | {h} | {c4[h]['estimate']:+.3f} (null {c4[h]['null_value']:+.3f}) | "
                         f"{c4[h]['p']:.3g} ({'rejected' if c4[h]['rejected_holm'] else 'not rejected'}) | "
                         f"{yes[k4[h]['A']]} | {yes[k4[h]['B']]} (effect {k4[h]['effect']:.3f} / {k4[h]['smallest_effect']:.3f}) |")
        mc = c4["manipulation_check"]
        lines += ["", f"Part 4 manipulation check (base model uses the evidence): weight {mc['w_congruent']:.3f}, "
                  f"{'passed' if mc['passed'] else 'FAILED: H1 to H3 are uninterpretable for this pair'}.",
                  "Part 4 sensitivity: " + "; ".join(
                      f"{k.replace('_', ' ')}: " + ", ".join(
                          f"{h} {'same' if x['same_conclusion'] else 'CHANGES'}" for h, x in v.items())
                      for k, v in r["part4"]["sensitivity"].items() if k != "failures"), ""]
    lines += ["## Over the two new families", "", "| Experiment | Test | Original pair | Criterion A | Criterion B |", "|---|---|---|---|---|"]
    for exp, hyps in res["summary"].items():
        for h, row in hyps.items():
            lines.append(f"| {exp} | {h} | {row['originally']} | {row['A_reading']} | {row.get('B_reading', '')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ok, detail = R.preregistration_status()
    if not ok:
        print(f"Refused: {detail}.")
        return 1
    power = json.loads((R.OUT / "power.json").read_text(encoding="utf-8"))
    sesoi = power["smallest_effects"]
    scenarios = read_jsonl(config.DATA_DIR / "part4" / "test" / "scenarios.jsonl")
    tone_path = config.RESULTS_DIR / "part4" / "test" / "tone.jsonl"
    tone = {r["id"]: r["p_tone"] for r in read_jsonl(tone_path)} if tone_path.exists() else {}
    frozen4 = R.load_script("part4_analyze")
    frozen_add = R.load_script("analyze_abliteration_sentiment")
    R.log_test_access(f"replication_analyze pairs={list(R.REPLICATIONS)} preregistration={detail[:7]}")
    res = {"note": "pre-registered replication analysis, run once (results/metrics/test_access.log)",
           "preregistration_commit": detail, "pairs": {}}
    for pair in R.REPLICATIONS:
        bias = power["part4"][pair]["measurement_bias"]
        res["pairs"][pair] = {"part2": part2(pair), "addendum": addendum(pair, frozen_add),
                              "part4": part4(pair, scenarios, sesoi, bias, frozen4, tone)}
    res["summary"] = summary(res["pairs"])
    R.OUT.mkdir(parents=True, exist_ok=True)
    (R.OUT / "analysis.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    md = render(res)
    (R.OUT / "analysis.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
