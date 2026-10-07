"""Single evaluation authority: joins every prediction file with the gold labels and scores it.

    python scripts/evaluate.py --split val
    python scripts/evaluate.py --split test --reference qwen3-1.7b-lora-s0

Without --reference, every pair of models is compared. With it, the reference is compared to
each other model. Paired tests are Holm-corrected across all comparisons of the run.
Models named <base>-s<seed> are also summarized as mean and standard deviation over seeds.
Every evaluation of the test split is appended to results/metrics/test_access.log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import re
import sys

import numpy as np

from bbg import config
from bbg.data import load_split
from bbg.metrics import (accuracy, bootstrap_ci, confusion, holm, macro_f1, mcnemar_exact,
                         paired_bootstrap_diff, per_class)
from bbg.predictions import read_predictions

BEARISH = next(i for i, n in config.LABELS.items() if n == "bearish")
BULLISH = next(i for i, n in config.LABELS.items() if n == "bullish")
SEED_NAME = re.compile(r"^(.+)-s(\d+)$")


def load_all(split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    gold = load_split(split).sort_values("id")
    y_true = gold["label"].to_numpy()
    preds = {}
    for path in sorted((config.PREDICTIONS_DIR / split).glob("*.csv")):
        df = read_predictions(path, split).sort_values("id")
        if not np.array_equal(df["id"].to_numpy(), gold["id"].to_numpy()):
            raise RuntimeError(f"{path.name}: ids not aligned with gold")
        preds[path.stem] = df["y_pred"].to_numpy()
    if not preds:
        raise SystemExit(f"No prediction files in {config.PREDICTIONS_DIR / split}")
    return y_true, preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--reference", help="model compared against all others")
    args = ap.parse_args()

    y_true, preds = load_all(args.split)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if args.split == "test":
        with open(config.METRICS_DIR / "test_access.log", "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} models={sorted(preds)}\n")

    models = {}
    for name, y_pred in preds.items():
        models[name] = {
            "macro_f1": macro_f1(y_true, y_pred),
            "macro_f1_ci95": bootstrap_ci(y_true, y_pred, "macro_f1"),
            "accuracy": accuracy(y_true, y_pred),
            "accuracy_ci95": bootstrap_ci(y_true, y_pred, "accuracy"),
            "per_class": per_class(y_true, y_pred),
            "confusion": confusion(y_true, y_pred).tolist(),
            "direction_errors": direction_errors(y_true, y_pred),
        }

    if args.reference:
        if args.reference not in preds:
            raise SystemExit(f"reference {args.reference!r} not found among {sorted(preds)}")
        pairs = [(args.reference, m) for m in sorted(preds) if m != args.reference]
    else:
        pairs = list(itertools.combinations(sorted(preds), 2))

    comparisons = {}
    for a, b in pairs:
        key = f"{a} vs {b}"
        comparisons[key] = {
            "mcnemar": mcnemar_exact(y_true, preds[a], preds[b]),
            "macro_f1_diff": paired_bootstrap_diff(y_true, preds[a], preds[b], "macro_f1"),
        }
    if comparisons:
        for test in ("mcnemar", "macro_f1_diff"):
            adj = holm({k: v[test]["p"] for k, v in comparisons.items()})
            for k, p in adj.items():
                comparisons[k][test]["p_holm"] = p

    out = {"split": args.split, "n": int(len(y_true)), "models": models,
           "seed_groups": seed_groups(models), "comparisons": comparisons}
    (config.METRICS_DIR / f"{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = render_markdown(out)
    (config.METRICS_DIR / f"{args.split}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def direction_errors(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """Bullish read as bearish or the reverse: the costly mistake for a trading signal."""
    bear, bull = BEARISH, BULLISH
    return int(np.sum((y_true == bull) & (y_pred == bear)) + np.sum((y_true == bear) & (y_pred == bull)))


def seed_groups(models: dict) -> dict:
    """Models named <base>-s<seed> with at least two seeds: mean and standard deviation."""
    groups: dict[str, list[str]] = {}
    for name in models:
        m = SEED_NAME.match(name)
        if m:
            groups.setdefault(m.group(1), []).append(name)
    out = {}
    for base, names in sorted(groups.items()):
        if len(names) < 2:
            continue
        out[base] = {"seeds": sorted(names)}
        for key in ("macro_f1", "accuracy", "direction_errors"):
            vals = np.array([models[n][key] for n in names], dtype=float)
            out[base][key] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1))}
    return out


def fmt_p(p: float) -> str:
    """Bootstrap p-values cannot go below 1/iterations: show them as an upper bound."""
    floor = 1 / config.BOOTSTRAP_ITERS
    return f"< {floor:.0e}" if p < floor else f"{p:.3g}"


def render_markdown(out: dict) -> str:
    lines = [f"## {out['split']} split (n = {out['n']})", "",
             "| Model | Macro-F1 [95% CI] | Accuracy [95% CI] | Direction errors |",
             "|---|---|---|---|"]
    ranked = sorted(out["models"].items(), key=lambda kv: kv[1]["macro_f1"], reverse=True)
    for name, m in ranked:
        lo, hi = m["macro_f1_ci95"]
        alo, ahi = m["accuracy_ci95"]
        lines.append(f"| {name} | {m['macro_f1']:.3f} [{lo:.3f}, {hi:.3f}] "
                     f"| {m['accuracy']:.3f} [{alo:.3f}, {ahi:.3f}] | {m['direction_errors']} |")
    lines += ["", "Direction errors: bullish tweets read as bearish, or the reverse."]
    if out.get("seed_groups"):
        lines += ["", "| Seed-averaged model | Macro-F1 mean ± sd | Accuracy mean ± sd | Direction errors |",
                  "|---|---|---|---|"]
        for base, g in out["seed_groups"].items():
            f, a, d = g["macro_f1"], g["accuracy"], g["direction_errors"]
            lines.append(f"| {base} ({len(g['seeds'])} seeds) | {f['mean']:.3f} ± {f['sd']:.3f} "
                         f"| {a['mean']:.3f} ± {a['sd']:.3f} | {d['mean']:.1f} ± {d['sd']:.1f} |")
    if out["comparisons"]:
        lines += ["", "| Comparison | Δ macro-F1 [95% CI] | p (Holm) | McNemar A-only / B-only | p (Holm) |",
                  "|---|---|---|---|---|"]
        for key, c in out["comparisons"].items():
            d, mc = c["macro_f1_diff"], c["mcnemar"]
            lines.append(f"| {key} | {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] "
                         f"| {fmt_p(d['p_holm'])} | {mc['a_only']} / {mc['b_only']} | {mc['p_holm']:.3g} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
