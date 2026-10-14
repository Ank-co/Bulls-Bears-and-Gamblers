"""Part 4: the pre-registered analysis of the test run (docs/part4-preregistration.md).

    python scripts/part4_analyze.py

Run once, after the test answers are in. Refused before the pre-registration exists; every run on
the test split is logged in results/metrics/test_access.log. The measurement biases and the smallest
effects of interest are read from results/part4/dev/power.json, frozen before the test.
Writes results/part4/test/analysis.json and analysis.md.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from bbg import config, fictional
from bbg import part4_stats as S
from bbg.part4_llm import reading_items

PREREGISTRATION = config.ROOT / "docs" / "part4-preregistration.md"
MODELS = ("base", "abliterated")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def failure_regression(scenarios: list[dict], answers: list[dict]) -> dict | None:
    """Do failures depend on the scenario? Logistic regression on the number of news, the extremity of
    the ideal answer and the framing (descriptive)."""
    by_id = {a["id"]: a for a in answers}
    rows = [{"failed": float(by_id[s["id"]]["error"] is not None), "news": len(s["items"]),
             "extremity": abs(fictional.logit(s["ideal_stated"])), "volatile": float(s["framing"] == "volatile"),
             "calm": float(s["framing"] == "calm")} for s in scenarios if s["id"] in by_id]
    df = pd.DataFrame(rows)
    if df["failed"].sum() < 5:
        return None
    fit = sm.Logit(df["failed"], sm.add_constant(df.drop(columns="failed"))).fit(disp=0)
    return {k: {"estimate": float(fit.params[k]), "p": float(fit.pvalues[k])} for k in fit.params.index}


def replaced(df: pd.DataFrame, how: str) -> pd.DataFrame:
    d = df.copy()
    miss = d["answer"].isna()
    if how == "fifty":
        d.loc[miss, "answer"] = 50.0
    else:
        d.loc[miss, "answer"] = 100 / (1 + np.exp(-d.loc[miss, "base_rate"]))
    d["answer_logodds"] = [S.answer_logodds(v) for v in d["answer"]]
    return d


def decisions(conf: dict) -> dict:
    return {h: conf[h]["rejected_holm"] for h in S.DIRECTIONS}


def sensitivity(conf: dict, scenarios, base_answers, reading) -> dict:
    """H2 and H3 re-estimated with another reading, each compared with the threshold it had in Holm's
    procedure in the main analysis."""
    thr = S.holm_thresholds(conf)
    df = S.stated_frame(scenarios, base_answers, reading)
    g = S.grether(df)["weights"]
    d = df.dropna(subset=["answer_logodds"])
    fit = sm.OLS(d["answer_logodds"], sm.add_constant(d[S.TERMS + (["e_unread"] if (d["e_unread"] != 0).any() else [])])).fit(cov_type="HC1")
    cov = fit.cov_params()
    bias = {h: conf[h]["null_value"] for h in ("H2", "H3")}
    h3_est = fit.params["e_reversed"] - fit.params["e_congruent"]
    h3_se = float(np.sqrt(cov.loc["e_reversed", "e_reversed"] + cov.loc["e_congruent", "e_congruent"]
                          - 2 * cov.loc["e_reversed", "e_congruent"]))
    out = {}
    for h, est, se in (("H2", g["distractor"]["estimate"], g["distractor"]["se"]), ("H3", h3_est, h3_se)):
        z = (est - bias[h]) / se
        p = S._p_value(z, S.DIRECTIONS[h])
        out[h] = {"estimate": float(est), "se": float(se), "z": float(z), "p": p, "holm_threshold": thr[h],
                  "rejected": p <= thr[h], "same_conclusion": (p <= thr[h]) == conf[h]["rejected_holm"]}
    return out


def main() -> int:
    if not PREREGISTRATION.exists():
        print(f"Refused: {PREREGISTRATION.name} must exist before the test analysis.")
        return 1
    test_dir = config.RESULTS_DIR / "part4" / "test"
    scenarios = read_jsonl(config.DATA_DIR / "part4" / "test" / "scenarios.jsonl")
    answers = {m: {w: read_jsonl(test_dir / "answers" / m / f"{w}.jsonl") for w in ("reading", "stated", "history")}
               for m in MODELS}
    missing = [f"{m}/{w}" for m in MODELS for w in ("reading", "stated", "history") if not answers[m][w]]
    if not scenarios or missing:
        print(f"Missing test inputs: {'scenarios' if not scenarios else ', '.join(missing)}")
        return 1
    frozen = json.loads((config.RESULTS_DIR / "part4" / "dev" / "power.json").read_text(encoding="utf-8"))
    bias = frozen["simulation"]["measurement_bias"]
    smallest = frozen["smallest_effects"]["effects"]

    log = config.METRICS_DIR / "test_access.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} part4_analyze models={list(MODELS)}\n")

    tone = {r["id"]: r["p_tone"] for r in read_jsonl(test_dir / "tone.jsonl")}
    frames = {m: S.stated_frame(scenarios, answers[m]["stated"]) for m in MODELS}
    conf = S.confirmatory(frames["base"], frames["abliterated"], bias=bias)
    res = {"note": "pre-registered analysis of the test run, run once (results/metrics/test_access.log)",
           "frozen_from_dev": {"measurement_bias": bias, "smallest_effects": smallest},
           "scenarios": len(scenarios), "failures": {}, "confirmatory": conf,
           "interpretation": S.interpret(conf, smallest)}
    items = reading_items()
    for m in MODELS:
        res["failures"][m] = {w: S.failures(answers[m][w]) for w in ("reading", "stated", "history")}
        res["failures"][m]["stated_by_scenario"] = failure_regression(scenarios, answers[m]["stated"])

    # sensitivity of H2 and H3 to the reading of the news
    agreement = {m: S.reading_agreement(items, answers[m]["reading"]) for m in MODELS}
    res["sensitivity"] = {
        "own_reading": sensitivity(conf, scenarios, answers["base"]["stated"], S.own_reading_map(agreement["base"])),
        "clear_kinds": sensitivity(conf, scenarios, answers["base"]["stated"], S.clear_kinds_reading()),
    }
    # sensitivity to failures
    if any(frames[m]["answer"].isna().any() for m in MODELS):
        res["sensitivity"]["failures"] = {}
        for how in ("fifty", "base_rate"):
            c = S.confirmatory(replaced(frames["base"], how), replaced(frames["abliterated"], how), bias=bias)
            res["sensitivity"]["failures"][how] = {"decisions": decisions(c), "same_as_main": decisions(c) == decisions(conf)}

    # descriptive
    res["descriptive"] = {}
    for m in MODELS:
        extra = {"tone_reader": frames[m]["id"].map(tone).to_numpy()} if tone else None
        res["descriptive"][m] = {"weights": S.grether(frames[m]), "shift_amplification_noise": S.shift_amplification_noise(frames[m]),
                                 "accuracy": S.accuracy(frames[m], extra), "calibration": S.calibration_table(frames[m]),
                                 "reading_agreement": agreement[m]}
    res["descriptive"]["base_vs_abliterated"] = S.paired(frames["base"], frames["abliterated"])

    # exploratory
    res["exploratory"] = {}
    for m in MODELS:
        d = frames[m].dropna(subset=["answer_logodds"]).copy()
        d["residual"] = d["answer_logodds"] - d["ideal"]
        by_framing = d.merge(pd.DataFrame([{"id": s["id"], "framing": s["framing"]} for s in scenarios]), on="id")
        res["exploratory"][m] = {
            "history": S.history_distances(scenarios, answers[m]["history"]),
            "mean_gap_to_ideal_by_framing": by_framing.groupby("framing")["residual"].mean().to_dict(),
            "distractor_weight_by_topic": topic_weights(scenarios, answers[m]["stated"]),
        }
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "analysis.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    md = render(res)
    (test_dir / "analysis.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def topic_weights(scenarios, answers) -> dict:
    """Exploratory: the distractor term split into politics and international news against the rest."""
    df = S.stated_frame(scenarios, answers)
    by_id = {s["id"]: s for s in scenarios}
    far = []
    for i in df["id"]:
        far.append(sum(it["textbook"] for it in by_id[i]["items"]
                       if fictional.shown_version_pct(it) == (50, 50) and it["topic"] in ("politics", "international")))
    df["distractor_politics_international"] = far
    df["distractor_other"] = df["distractor"] - df["distractor_politics_international"]
    d = df.dropna(subset=["answer_logodds"])
    terms = ["base_rate", "e_congruent", "e_reversed", "distractor_politics_international", "distractor_other", "dramatic_tone"]
    fit = sm.OLS(d["answer_logodds"], sm.add_constant(d[terms])).fit(cov_type="HC1")
    return {k: {"estimate": float(fit.params[k]), "se": float(fit.bse[k])}
            for k in ("distractor_politics_international", "distractor_other")}


def render(res: dict) -> str:
    c, it = res["confirmatory"], res["interpretation"]
    names = {"H1": "Base rate underweighted (base model)", "H2": "Worthless news moves the answer (base model)",
             "H3": "Reversed track records followed less (base model)", "H4": "Abliterated model answers higher",
             "H5": "Abliterated model's slope differs"}
    lines = ["# Part 4, test analysis (pre-registered, run once)", "",
             f"Manipulation check (base model uses the evidence): weight {c['manipulation_check']['w_congruent']:.3f}, "
             f"{'passed' if c['manipulation_check']['passed'] else 'FAILED: results uninterpretable'}.", "",
             "| Hypothesis | Estimate (SE) | Null | p | Holm | Effect / smallest | Verdict |", "|---|---|---|---|---|---|---|"]
    for h in S.DIRECTIONS:
        r = c[h]
        lines.append(f"| {h}. {names[h]} | {r['estimate']:.3f} ({r['se']:.3f}) | {r['null_value']:.3f} | {r['p']:.2g} | "
                     f"{'rejected' if r['rejected_holm'] else 'not rejected'} | {it[h]['effect']:.3f} / "
                     f"{it[h]['smallest_effect']:.3f} | {it[h]['verdict']} |")
    lines += ["", "Sensitivity of H2 and H3:"]
    for k, v in res["sensitivity"].items():
        if k == "failures":
            lines.append(f"- failures replaced: " + ", ".join(f"{how}: {'same' if x['same_as_main'] else 'CHANGED'}" for how, x in v.items()))
        else:
            lines.append(f"- {k.replace('_', ' ')}: " + ", ".join(
                f"{h} {x['estimate']:.3f} (p {x['p']:.2g}, {'same conclusion' if x['same_conclusion'] else 'CONCLUSION CHANGES'})"
                for h, x in v.items()))
    lines += ["", "Failures: " + "; ".join(f"{m} {res['failures'][m]['stated']['failures']} stated, "
                                          f"{res['failures'][m]['history']['failures']} history" for m in MODELS)]
    for m in MODELS:
        dsc = res["descriptive"][m]
        w = dsc["weights"]["weights"]
        lines += ["", f"## {m} (descriptive)", "",
                  "- weights: " + ", ".join(f"{k} {w[k]['estimate']:.3f} ({w[k]['se']:.3f})" for k in ["const"] + S.TERMS),
                  f"- shift {dsc['shift_amplification_noise']['shift']['estimate']:.3f}, amplification "
                  f"{dsc['shift_amplification_noise']['amplification']['estimate']:.3f}, noise sd "
                  f"{dsc['shift_amplification_noise']['noise_sd']:.2f}",
                  "- Brier: " + ", ".join(f"{k} {v['brier']:.4f}" for k, v in dsc["accuracy"].items()),
                  f"- reading of the news: {dsc['reading_agreement']['overall']}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
