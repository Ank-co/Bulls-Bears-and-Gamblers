"""Part 3 plumbing: paths, label files with resume, and the pre-registered analysis.

Data derived from FNSPID (headline text, prices, returns) live in data/part3/ (git-ignored).
Only hashed headline ids, labels, counts and test results are written to results/part3/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .event_stats import daily_spreads, nw_mean, paired_spread_diff, ticker_date_signal
from .metrics import holm
from .returns import WINDOWS, winsorize

MAIN = "qwen3-1.7b-lora-s0"
OTHERS = ["finbert", "tfidf-logreg", "qwen3-1.7b-zeroshot"]
MODELS = [MAIN] + OTHERS
DAY_OF = {"reaction": "t1", "drift": "t1", "next": "t2"}   # day on which each window ends


def work_dir(period: str) -> Path:
    return config.DATA_DIR / "part3" / period


def results_dir(period: str) -> Path:
    return config.PART3_DIR / period


def labels_path(period: str, model: str) -> Path:
    return config.PART3_DIR / "predictions" / period / f"{model}.csv"


# --- label files, written in chunks so that a long run can resume ------------------------------
def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["id"], dtype={"id": str})["id"])


def append_labels(path: Path, ids, labels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"id": list(ids), "y_pred": [int(x) for x in labels]})
    if not set(df["y_pred"]) <= set(config.LABEL_IDS):
        raise ValueError("unknown label id")
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def read_labels(path: Path, expected_ids) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    if df["id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicated ids")
    missing = set(expected_ids) - set(df["id"])
    if missing:
        raise ValueError(f"{path.name}: {len(missing)} headlines without a label (run not finished?)")
    return df[df["id"].isin(set(expected_ids))]


# --- analysis ----------------------------------------------------------------------------------
def _bps(x: float) -> float:
    return float(1e4 * x)


def _test(r: dict) -> dict:
    return {"mean_bps": _bps(r["mean"]), "se_bps": _bps(r["se"]), "t": r["t"], "p": r["p"],
            "n_days": r["n_days"]}


def analyze(ret: pd.DataFrame, heads: pd.DataFrame, labels: dict[str, pd.DataFrame]) -> dict:
    """Pre-registered tests H1 to H3 and the descriptive results, from returns and labels."""
    ret = ret.copy()
    lo, hi = config.PART3_WINSOR
    for w in WINDOWS:
        ret[f"w_{w}"] = winsorize(ret[f"ab_{w}"], lo, hi)

    spreads, raw, signals = {}, {}, {}
    for m in MODELS:
        sig = ticker_date_signal(labels[m], heads)
        df = ret.merge(sig, on=["ticker", "date"], how="inner")
        if len(df) != len(ret):
            raise ValueError(f"{m}: {len(ret) - len(df)} ticker-dates without a signal")
        signals[m] = df["signal"].value_counts(normalize=True).reindex([1, 0, -1], fill_value=0)
        spreads[m] = {w: daily_spreads(df, f"w_{w}", DAY_OF[w]) for w in WINDOWS}
        raw[m] = {w: daily_spreads(df, f"ab_{w}", DAY_OF[w]) for w in WINDOWS}
        if m == MAIN:
            main_df = df

    tests = {"H1_reaction": _test(nw_mean(spreads[MAIN]["reaction"]["spread"]))}
    for m in OTHERS:
        tests[f"H2_vs_{m}"] = _test(paired_spread_diff(spreads[MAIN]["reaction"], spreads[m]["reaction"]))
    tests["H3_drift"] = _test(nw_mean(spreads[MAIN]["drift"]["spread"]))
    for k, p in holm({k: v["p"] for k, v in tests.items()}).items():
        tests[k]["p_holm"] = p
        tests[k]["confirmed"] = bool(p < 0.05 and tests[k]["mean_bps"] > 0)

    models = {}
    for m in MODELS:
        d = {"signal_share": {"bullish": float(signals[m][1]), "neutral": float(signals[m][0]),
                              "bearish": float(signals[m][-1])}}
        for w in WINDOWS:
            s = spreads[m][w]
            d[w] = {"spread": _test(nw_mean(s["spread"])), "long": _test(nw_mean(s["long"])),
                    "short": _test(nw_mean(s["short"])),
                    "spread_unwinsorized_bps": _bps(raw[m][w]["spread"].mean())}
        models[m] = d

    years = {}
    for w in ("reaction", "drift"):
        s = spreads[MAIN][w]["spread"]
        years[w] = {str(y): _bps(v) for y, v in s.groupby(s.index.year).mean().items()}

    drift = tests["H3_drift"]["mean_bps"]
    return {
        "n_ticker_dates": int(len(ret)), "n_headlines": int(len(heads)),
        "n_tickers": int(ret["ticker"].nunique()),
        "first_day": str(ret["t1"].min().date()), "last_day": str(ret["t1"].max().date()),
        "tests": tests, "models": models, "main_by_year_bps": years,
        "break_even_round_trip_cost_bps": drift / 2,   # each leg pays one round trip per day
        "main_signal_counts": {int(k): int(v) for k, v in main_df["signal"].value_counts().items()},
    }


def render(res: dict, period: str) -> str:
    t = res["tests"]
    names = {"H1_reaction": "H1 reaction spread, fine-tuned model",
             "H2_vs_finbert": "H2 reaction spread, fine-tuned minus FinBERT",
             "H2_vs_tfidf-logreg": "H2 reaction spread, fine-tuned minus TF-IDF",
             "H2_vs_qwen3-1.7b-zeroshot": "H2 reaction spread, fine-tuned minus zero-shot",
             "H3_drift": "H3 drift spread (open to close), fine-tuned model"}
    head = ("Pre-registered tests (docs/part3-preregistration.md), Holm-corrected."
            if period == "test" else "Dev period: pipeline check only, not a test of the hypotheses.")
    lines = [f"## Part 3, {period} ({res['first_day']} to {res['last_day']}): "
             f"{res['n_ticker_dates']:,} ticker-dates, {res['n_headlines']:,} headlines, "
             f"{res['n_tickers']:,} tickers", "", head, "",
             "| Test | Mean (bps) | t (Newey-West) | Days | p (Holm) | Result |", "|---|---|---|---|---|---|"]
    for k, label in names.items():
        r = t[k]
        p = "< 1e-16" if r["p_holm"] < 1e-16 else f"{r['p_holm']:.3g}"
        lines.append(f"| {label} | {r['mean_bps']:+.1f} | {r['t']:.2f} | {r['n_days']} | {p} "
                     f"| {'Confirmed' if r['confirmed'] else 'Not confirmed'} |")
    lines += ["", "| Model | Bullish / neutral / bearish | Reaction spread | Long | Short "
                  "| Drift spread | Next-day spread |", "|---|---|---|---|---|---|---|"]
    for m, d in res["models"].items():
        sh = d["signal_share"]
        lines.append(f"| {m} | {sh['bullish']:.0%} / {sh['neutral']:.0%} / {sh['bearish']:.0%} "
                     f"| {d['reaction']['spread']['mean_bps']:+.1f} | {d['reaction']['long']['mean_bps']:+.1f} "
                     f"| {d['reaction']['short']['mean_bps']:+.1f} | {d['drift']['spread']['mean_bps']:+.1f} "
                     f"| {d['next']['spread']['mean_bps']:+.1f} |")
    lines += ["", "Abnormal returns in basis points, stock minus SPY, winsorized at 0.5% and 99.5%. "
                  "Spread = mean over days of (bullish mean minus bearish mean).",
              f"Break-even round-trip cost of the drift spread: {res['break_even_round_trip_cost_bps']:.1f} bps per leg."]
    return "\n".join(lines) + "\n"
