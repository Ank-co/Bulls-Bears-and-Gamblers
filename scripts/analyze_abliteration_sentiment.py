"""Part 1 addendum: does abliteration change how the 35B model reads market news?

    python scripts/analyze_abliteration_sentiment.py

Pre-registered tests (docs/part1-abliteration-preregistration.md) on the test split, comparing
the abliterated model with the base model of the same quantization. Reads the prediction files
written by scripts/zero_shot_llamacpp.py and writes results/metrics/abliteration_sentiment.json
and .md. Like scripts/evaluate.py, every run is appended to results/metrics/test_access.log.
"""
from __future__ import annotations

import datetime as dt
import json
import sys

import numpy as np

from bbg import config
from bbg.data import load_split
from bbg.metrics import (bootstrap_ci, confusion, holm, macro_f1, paired_bootstrap_diff,
                         paired_rate_test)
from bbg.predictions import read_predictions

BASE = "qwen3.6-35b-i1-zeroshot"
ABL = "qwen3.6-35b-abliterated-zeroshot"
OTHER_QUANT = "qwen3.6-35b-a3b-zeroshot"          # same base model, unsloth UD-Q3_K_M
SPLIT = "test"
NAMES = [config.LABELS[i] for i in sorted(config.LABELS)]
BULLISH = NAMES.index("bullish")
BEARISH = NAMES.index("bearish")


def aligned(name: str, gold_ids: np.ndarray) -> np.ndarray:
    path = config.PREDICTIONS_DIR / SPLIT / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/zero_shot_llamacpp.py first "
                         "(docs/part1-abliteration-preregistration.md, procedure)")
    df = read_predictions(path, SPLIT).sort_values("id")
    if not np.array_equal(df["id"].to_numpy(), gold_ids):
        raise RuntimeError(f"{path.name}: ids not aligned with gold")
    return df["y_pred"].to_numpy()


def describe(y: np.ndarray, p: np.ndarray) -> dict:
    cm = confusion(y, p)
    rows = cm / cm.sum(axis=1, keepdims=True)
    return {
        "macro_f1": macro_f1(y, p),
        "macro_f1_ci95": bootstrap_ci(y, p, "macro_f1"),
        "predicted_share": {NAMES[j]: float(np.mean(p == j)) for j in range(len(NAMES))},
        "predicted_share_by_gold": {NAMES[i]: {NAMES[j]: float(rows[i, j]) for j in range(len(NAMES))}
                                    for i in range(len(NAMES))},
        "bearish_read_as_bullish": int(cm[BEARISH, BULLISH]),
        "bullish_read_as_bearish": int(cm[BULLISH, BEARISH]),
    }


def run(y: np.ndarray, base: np.ndarray, abl: np.ndarray) -> dict:
    not_bullish = y != BULLISH
    tests = {
        "H1_false_optimism": paired_rate_test(abl[not_bullish] == BULLISH, base[not_bullish] == BULLISH),
        "H2_bullish_share": paired_rate_test(abl == BULLISH, base == BULLISH),
        "H3_macro_f1": paired_bootstrap_diff(y, abl, base, "macro_f1"),
    }
    for key, p in holm({k: v["p"] for k, v in tests.items()}).items():
        tests[key]["p_holm"] = p
    for key in ("H1_false_optimism", "H2_bullish_share"):
        t = tests[key]
        t["confirmed"] = bool(t["p_holm"] < 0.05 and t["diff"] > 0)
    tests["H3_macro_f1"]["confirmed"] = bool(tests["H3_macro_f1"]["p_holm"] < 0.05)
    return tests


def render(out: dict) -> str:
    t = out["tests"]
    h1, h2, h3 = t["H1_false_optimism"], t["H2_bullish_share"], t["H3_macro_f1"]
    def verdict(r):
        if r["confirmed"]:
            return "Confirmed"
        return "Opposite direction" if r["p_holm"] < 0.05 and r["diff"] < 0 else "Not confirmed"

    def fmt_p(p):            # a bootstrap p-value cannot go below 1 / iterations
        floor = 1 / config.BOOTSTRAP_ITERS
        return f"< {floor:.0e}" if p < floor else f"{p:.3g}"

    lines = [f"## Part 1 addendum: abliterated vs base, same quantization ({SPLIT}, n = {out['n']})", "",
             "Pre-registered tests (docs/part1-abliteration-preregistration.md), Holm-corrected.", "",
             "| Hypothesis | Abliterated | Base | Difference | Only abliterated / only base | p (Holm) | Result |",
             "|---|---|---|---|---|---|---|",
             f"| H1 non-bullish tweets read as bullish (n = {h1['n']}) | {h1['rate_a']:.1%} | {h1['rate_b']:.1%} "
             f"| {100 * h1['diff']:+.1f} points | {h1['a_only']} / {h1['b_only']} | {h1['p_holm']:.3g} "
             f"| {verdict(h1)} |",
             f"| H2 all tweets read as bullish (n = {h2['n']}) | {h2['rate_a']:.1%} | {h2['rate_b']:.1%} "
             f"| {100 * h2['diff']:+.1f} points | {h2['a_only']} / {h2['b_only']} | {h2['p_holm']:.3g} "
             f"| {verdict(h2)} |",
             f"| H3 macro-F1 | {out['models'][ABL]['macro_f1']:.3f} | {out['models'][BASE]['macro_f1']:.3f} "
             f"| {h3['diff']:+.3f} [{h3['ci'][0]:+.3f}, {h3['ci'][1]:+.3f}] | | {fmt_p(h3['p_holm'])} "
             f"| {'Differs' if h3['confirmed'] else 'No difference shown'} |", "",
             "| Model | Macro-F1 [95% CI] | Read as bullish | Read as bearish | Read as neutral "
             "| Bearish read as bullish | Bullish read as bearish |", "|---|---|---|---|---|---|---|"]
    for name, m in out["models"].items():
        s, (lo, hi) = m["predicted_share"], m["macro_f1_ci95"]
        lines.append(f"| {name} | {m['macro_f1']:.3f} [{lo:.3f}, {hi:.3f}] | {s['bullish']:.1%} "
                     f"| {s['bearish']:.1%} | {s['neutral']:.1%} | {m['bearish_read_as_bullish']} "
                     f"| {m['bullish_read_as_bearish']} |")
    q = out.get("quantization")
    if q:
        lines += ["", f"Not pre-registered, descriptive: base i1-Q3_K_M minus base UD-Q3_K_M, macro-F1 "
                      f"{q['diff']:+.3f} [{q['ci'][0]:+.3f}, {q['ci'][1]:+.3f}]."]
    return "\n".join(lines) + "\n"


def main() -> int:
    gold = load_split(SPLIT).sort_values("id")
    ids, y = gold["id"].to_numpy(), gold["label"].to_numpy()
    base, abl = aligned(BASE, ids), aligned(ABL, ids)

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_DIR / "test_access.log", "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} "
                f"analyze_abliteration_sentiment models={[BASE, ABL]}\n")

    out = {"split": SPLIT, "n": int(len(y)), "base": BASE, "abliterated": ABL,
           "tests": run(y, base, abl), "models": {ABL: describe(y, abl), BASE: describe(y, base)}}
    if (config.PREDICTIONS_DIR / SPLIT / f"{OTHER_QUANT}.csv").exists():
        other = aligned(OTHER_QUANT, ids)
        out["models"][OTHER_QUANT] = describe(y, other)
        out["quantization"] = paired_bootstrap_diff(y, base, other, "macro_f1")

    (config.METRICS_DIR / "abliteration_sentiment.json").write_text(json.dumps(out, indent=2),
                                                                     encoding="utf-8")
    md = render(out)
    (config.METRICS_DIR / "abliteration_sentiment.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
