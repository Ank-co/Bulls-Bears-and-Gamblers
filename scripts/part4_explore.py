"""Part 4, exploratory (NOT pre-registered): readings written after the test results.

    python scripts/part4_explore.py

Written after the pre-registered analysis, to read it; nothing here changes a pre-registered conclusion.
Reads the test scenarios, the test answers and results/part4/test/analysis.json; every run is logged in
results/metrics/test_access.log. Writes results/part4/test/exploratory.json:
- the answers in probability points: each model's mean answer, the share of weeks up, and the paired
  difference between the two models (standard error over scenarios);
- the fitted answer when the ideal answer is 50%, from each model's shift in analysis.json;
- the share of the Brier-score gap between the base rate alone and the ideal observer that each model
  closes, from analysis.json;
- each effect's 90% interval next to its smallest effect of interest, from analysis.json;
- the framing, with the ideal answer held fixed: the answer's log-odds regressed on the ideal log-odds and
  one indicator per opening sentence (calm, volatile; no sentence is the reference), robust errors. The
  mean gap to the ideal by framing in analysis.json does not hold the ideal answer fixed;
- the history version against the stated version of the same 500 scenarios: how far the stated answers
  are from the ideal, next to how far the history answers are from each of their references.
All results are written, whatever their direction. None of the p-values here is confirmatory.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from bbg import config, fictional
from bbg import part4_stats as S

MODELS = ("base", "abliterated")
Z90 = stats.norm.ppf(0.95)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def points(frames: dict) -> dict:
    out = {}
    for m, df in frames.items():
        d = df.dropna(subset=["answer"])
        out[m] = {"n": int(len(d)), "mean_answer": float(d["answer"].mean()), "share_up": float(100 * d["y"].mean())}
    pair = frames["base"][["id", "answer"]].merge(frames["abliterated"][["id", "answer"]], on="id",
                                                   suffixes=("_base", "_abl")).dropna()
    diff = pair["answer_abl"] - pair["answer_base"]
    out["abliterated_minus_base"] = {"n": int(len(diff)), "mean": float(diff.mean()),
                                     "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                                     "share_higher": float(100 * (diff > 0).mean()),
                                     "share_equal": float(100 * (diff == 0).mean())}
    return out


def from_analysis(res: dict) -> dict:
    out = {"fitted_answer_at_ideal_50": {}, "brier_gap_closed": {}, "intervals_90": {}}
    for m in MODELS:
        shift = res["descriptive"][m]["shift_amplification_noise"]["shift"]["estimate"]
        out["fitted_answer_at_ideal_50"][m] = 100 * fictional.sigmoid(shift)
        acc = res["descriptive"][m]["accuracy"]
        gap = acc["base_rate_only"]["brier"] - acc["ideal"]["brier"]
        out["brier_gap_closed"][m] = (acc["base_rate_only"]["brier"] - acc["model"]["brier"]) / gap
    for h, it in res["interpretation"].items():
        se = res["confirmatory"][h]["se"]
        out["intervals_90"][h] = {"effect": it["effect"], "lower": it["effect"] - Z90 * se, "upper": it["upper_90"],
                                  "smallest_effect": it["smallest_effect"],
                                  "whole_interval_above_smallest": bool(it["effect"] - Z90 * se >= it["smallest_effect"])}
    return out


def framing(scenarios: list[dict], frames: dict) -> dict:
    fr = pd.DataFrame([{"id": s["id"], "framing": s["framing"]} for s in scenarios])
    out = {}
    for m, df in frames.items():
        d = df.dropna(subset=["answer_logodds"]).merge(fr, on="id")
        X = pd.DataFrame({"ideal": d["ideal"], "calm": (d["framing"] == "calm").astype(float),
                          "volatile": (d["framing"] == "volatile").astype(float)})
        fit = sm.OLS(d["answer_logodds"], sm.add_constant(X)).fit(cov_type="HC1")
        out[m] = {"n": int(len(d)),
                  "logodds_vs_no_sentence": {k: {"estimate": float(fit.params[k]), "se": float(fit.bse[k]),
                                                 "p_two_sided": float(fit.pvalues[k])} for k in ("calm", "volatile")},
                  "mean_answer_points": {k: float(v) for k, v in d.groupby("framing")["answer"].mean().items()},
                  "mean_ideal_points": {k: float(v) for k, v in (100 * d.groupby("framing")["ideal_p"].mean()).items()},
                  "scenarios": {k: int(v) for k, v in d.groupby("framing").size().items()}}
    return out


def history_vs_stated(scenarios: list[dict], answers: dict) -> dict:
    subset = [s for s in scenarios if s["in_history_subset"]]
    out = {}
    for m in MODELS:
        st = {a["id"]: a["value"] for a in answers[m]["stated"] if a["error"] is None}
        hi = {a["id"]: a["value"] for a in answers[m]["history"] if a["error"] is None}
        both = [s for s in subset if s["id"] in st and s["id"] in hi]
        clip = lambda v: min(max(v, S.CLIP[0]), S.CLIP[1])  # noqa: E731
        ans_st = np.array([clip(st[s["id"]]) for s in both], float)
        ideal = 100 * np.array([s["ideal_stated"] for s in both])
        out[m] = {"n": len(both),
                  "stated_vs_ideal": {"mean_abs_gap_points": float(np.mean(np.abs(ans_st - ideal))),
                                      "correlation": float(np.corrcoef(ans_st, ideal)[0, 1])},
                  "history_vs_references": S.history_distances(both, answers[m]["history"])}
    return out


def main() -> int:
    test_dir = config.RESULTS_DIR / "part4" / "test"
    analysis = test_dir / "analysis.json"
    scenarios = read_jsonl(config.DATA_DIR / "part4" / "test" / "scenarios.jsonl")
    answers = {m: {w: read_jsonl(test_dir / "answers" / m / f"{w}.jsonl") for w in ("stated", "history")}
               for m in MODELS}
    if not analysis.exists() or not scenarios or not all(answers[m][w] for m in MODELS for w in answers[m]):
        print("Run the pre-registered analysis first (scripts/part4_analyze.py); test inputs are missing.")
        return 1
    log = config.METRICS_DIR / "test_access.log"
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} part4_explore models={list(MODELS)} "
                f"(exploratory, after the pre-registered analysis)\n")

    res = json.loads(analysis.read_text(encoding="utf-8"))
    frames = {m: S.stated_frame(scenarios, answers[m]["stated"]) for m in MODELS}
    out = {"note": "exploratory, written after the pre-registered test analysis; no p-value here is confirmatory",
           "points": points(frames), **from_analysis(res), "framing": framing(scenarios, frames),
           "history_vs_stated": history_vs_stated(scenarios, answers)}
    (test_dir / "exploratory.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    p, f = out["points"], out["framing"]
    print(f"mean answer: base {p['base']['mean_answer']:.1f}, abliterated {p['abliterated']['mean_answer']:.1f}, "
          f"weeks up {p['base']['share_up']:.1f}%; abliterated minus base "
          f"{p['abliterated_minus_base']['mean']:.2f} points (SE {p['abliterated_minus_base']['se']:.2f})")
    print("fitted answer at ideal 50%: " + ", ".join(f"{m} {v:.1f}" for m, v in out["fitted_answer_at_ideal_50"].items()))
    print("Brier gap closed: " + ", ".join(f"{m} {100 * v:.0f}%" for m, v in out["brier_gap_closed"].items()))
    print("90% intervals: " + ", ".join(f"{h} [{v['lower']:.3f}, {v['upper']:.3f}] vs {v['smallest_effect']:.3f}"
                                        for h, v in out["intervals_90"].items()))
    for m in MODELS:
        c = f[m]["logodds_vs_no_sentence"]
        print(f"framing {m}: calm {c['calm']['estimate']:+.3f} ({c['calm']['se']:.3f}), volatile "
              f"{c['volatile']['estimate']:+.3f} ({c['volatile']['se']:.3f}); mean answers "
              + ", ".join(f"{k} {v:.1f}" for k, v in f[m]["mean_answer_points"].items()))
        h = out["history_vs_stated"][m]
        print(f"history {m}: stated gap to ideal {h['stated_vs_ideal']['mean_abs_gap_points']:.1f} points, history gap "
              + ", ".join(f"{k} {v['mean_abs_gap_points']:.1f}" for k, v in h["history_vs_references"].items() if k != "n"))
    print(f"written to {test_dir / 'exploratory.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
