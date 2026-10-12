"""Part 3 plumbing: paths, label files with resume, and the pre-registered analysis.

Data derived from FNSPID (headline text, prices, returns) live in data/part3/ (git-ignored).
Only hashed headline ids, labels, counts and test results are written to results/part3/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .event_stats import daily_ic, daily_spreads, nw_mean, paired_diff, ticker_date_signal
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


# --- control C1: does the base model remember these returns? ---------------------------------
PROBE_ANSWERS = ["higher", "lower"]


def probe_question(ticker: str, t0: pd.Timestamp, t1: pd.Timestamp) -> str:
    """Ticker and dates only, no headline: a model can only answer from memory."""
    return (f"Stock ticker: {ticker}.\n"
            f"Did this stock close higher or lower on {t1:%A %d %B %Y} than on {t0:%A %d %B %Y}?\n"
            "Answer with one word: higher or lower.")


def probe_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def probe_sample(ret: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    s = ret[["ticker", "date", "t0", "t1"]].sort_values(["date", "ticker"], ignore_index=True)
    if len(s) > size:
        s = s.iloc[np.sort(np.random.default_rng(seed).choice(len(s), size=size, replace=False))]
    return s.reset_index(drop=True)


def probe_path(period: str) -> Path:
    return results_dir(period) / "probe_scores.csv"


def probe_ic(ret: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """Daily IC between the probe score (log-prob of "higher" minus "lower") and the returns."""
    lo, hi = config.PART3_WINSOR
    df = ret.merge(scores[["ticker", "date", "score"]], on=["ticker", "date"], how="inner")
    out = {"n_ticker_dates": int(len(df))}
    for w in ("reaction", "drift"):
        df[f"w_{w}"] = winsorize(df[f"ab_{w}"], lo, hi)
        out[w] = _test(nw_mean(daily_ic(df, f"w_{w}", DAY_OF[w])))
    out["share_answering_higher"] = float((scores["score"] > 0).mean())
    return out


# --- control C2: are the ticker-dates without prices different? --------------------------------
_SHARE = {"bearish": 0, "bullish": 1, "neutral": 2}


def symbol_kind(ticker: str) -> str:
    if "." in ticker or "-" in ticker:
        return "class share or suffix (contains . or -)"
    if len(ticker) == 5 and ticker[-1] in "FY":
        return "five letters ending in F or Y (foreign, over the counter)"
    return "other"


def coverage(inc: pd.DataFrame, inc_labels: pd.DataFrame, exc: pd.DataFrame, exc_labels: pd.DataFrame) -> dict:
    """Compares the headlines kept (sample, with prices) with those dropped for lack of prices.
    Labels come from the same TF-IDF model. Descriptive only: the dropped ones have no returns."""
    def describe(h: pd.DataFrame, lab: pd.DataFrame) -> dict:
        y = h[["id"]].merge(lab, on="id", how="left")["y_pred"]
        if y.isna().any():
            raise ValueError("headlines without a TF-IDF label")
        td = h.groupby(["ticker", "date"]).size()
        kinds = pd.Series(h["ticker"].unique()).map(symbol_kind).value_counts(normalize=True)
        return {"headlines": int(len(h)), "ticker_dates": int(len(td)), "tickers": int(h["ticker"].nunique()),
                "headlines_per_ticker_date": float(td.mean()),
                "tfidf_label_share": {k: float((y == v).mean()) for k, v in _SHARE.items()},
                "top_publishers_share": {str(k): float(v) for k, v in
                                         h["publisher"].value_counts(normalize=True).head(5).items()},
                "symbol_kind_share": {str(k): float(v) for k, v in kinds.items()}}
    return {"kept_sample": describe(inc, inc_labels), "dropped_no_prices": describe(exc, exc_labels)}


# --- analysis ----------------------------------------------------------------------------------
def _bps(x: float) -> float:
    return float(1e4 * x)


def _test(r: dict, scale=float) -> dict:
    return {"mean": scale(r["mean"]), "se": scale(r["se"]), "t": r["t"], "p": r["p"], "n_days": r["n_days"]}


def analyze(ret: pd.DataFrame, heads: pd.DataFrame, labels: dict[str, pd.DataFrame],
            probe_scores: pd.DataFrame, coverage_result: dict) -> dict:
    """Pre-registered tests H1 to H3 (on the daily IC), controls C1 and C2, descriptive results."""
    ret = ret.copy()
    lo, hi = config.PART3_WINSOR
    for w in WINDOWS:
        ret[f"w_{w}"] = winsorize(ret[f"ab_{w}"], lo, hi)

    ics, spreads, raw, signals = {}, {}, {}, {}
    for m in MODELS:
        sig = ticker_date_signal(labels[m], heads)
        df = ret.merge(sig, on=["ticker", "date"], how="inner")
        if len(df) != len(ret):
            raise ValueError(f"{m}: {len(ret) - len(df)} ticker-dates without a signal")
        signals[m] = df["signal"].value_counts(normalize=True).reindex([1, 0, -1], fill_value=0)
        ics[m] = {w: daily_ic(df, f"w_{w}", DAY_OF[w]) for w in WINDOWS}
        spreads[m] = {w: daily_spreads(df, f"w_{w}", DAY_OF[w]) for w in WINDOWS}
        raw[m] = {w: daily_spreads(df, f"ab_{w}", DAY_OF[w]) for w in WINDOWS}

    tests = {"H1_reaction": _test(nw_mean(ics[MAIN]["reaction"]))}
    for m in OTHERS:
        tests[f"H2_vs_{m}"] = _test(paired_diff(ics[MAIN]["reaction"], ics[m]["reaction"]))
    tests["H3_drift"] = _test(nw_mean(ics[MAIN]["drift"]))
    for k, p in holm({k: v["p"] for k, v in tests.items()}).items():
        tests[k]["p_holm"] = p
        tests[k]["confirmed"] = bool(p < 0.05 and tests[k]["mean"] > 0)

    models = {}
    for m in MODELS:
        d = {"signal_share": {"bullish": float(signals[m][1]), "neutral": float(signals[m][0]),
                              "bearish": float(signals[m][-1])}}
        for w in WINDOWS:
            s = spreads[m][w]
            d[w] = {"ic": _test(nw_mean(ics[m][w])),
                    "spread_bps": _test(nw_mean(s["spread"]), _bps),
                    "long_bps": _test(nw_mean(s["long"]), _bps), "short_bps": _test(nw_mean(s["short"]), _bps),
                    "spread_unwinsorized_bps": _bps(raw[m][w]["spread"].mean())}
        models[m] = d

    years = {}
    for w in ("reaction", "drift"):
        s = ics[MAIN][w]
        g = s.groupby(s.index.year)
        years[w] = {str(y): float(v) for y, v in g.mean()[g.size() >= 20].items()}   # full years only

    drift_bps = models[MAIN]["drift"]["spread_bps"]["mean"]
    return {
        "n_ticker_dates": int(len(ret)), "n_headlines": int(len(heads)),
        "n_tickers": int(ret["ticker"].nunique()),
        "first_day": str(ret["t1"].min().date()), "last_day": str(ret["t1"].max().date()),
        "tests": tests, "models": models, "main_ic_by_year": years,
        "controls": {"C1_memory_probe": probe_ic(ret, probe_scores), "C2_coverage": coverage_result},
        "break_even_round_trip_cost_bps": drift_bps / 2,   # each leg pays one round trip per day
    }


def render(res: dict, period: str) -> str:
    t = res["tests"]
    names = {"H1_reaction": "H1 reaction, fine-tuned model",
             "H2_vs_finbert": "H2 reaction, fine-tuned minus FinBERT",
             "H2_vs_tfidf-logreg": "H2 reaction, fine-tuned minus TF-IDF",
             "H2_vs_qwen3-1.7b-zeroshot": "H2 reaction, fine-tuned minus zero-shot",
             "H3_drift": "H3 drift (open to close), fine-tuned model"}
    head = ("Pre-registered tests (docs/part3-preregistration.md) on the daily rank correlation (IC) "
            "between the model's score and the abnormal return, Holm-corrected."
            if period == "test" else "Dev period: pipeline check only, not a test of the hypotheses.")
    lines = [f"## Part 3, {period} ({res['first_day']} to {res['last_day']}): "
             f"{res['n_ticker_dates']:,} ticker-dates, {res['n_headlines']:,} headlines, "
             f"{res['n_tickers']:,} tickers", "", head, "",
             "| Test | Mean daily IC | t (Newey-West) | Days | p (Holm) | Result |", "|---|---|---|---|---|---|"]
    for k, label in names.items():
        r = t[k]
        p = "< 1e-16" if r["p_holm"] < 1e-16 else f"{r['p_holm']:.3g}"
        lines.append(f"| {label} | {r['mean']:+.3f} | {r['t']:.2f} | {r['n_days']} | {p} "
                     f"| {'Confirmed' if r['confirmed'] else 'Not confirmed'} |")
    lines += ["", "| Model | Bullish / neutral / bearish | Reaction IC | Reaction spread (bps) | Long | Short "
                  "| Drift IC | Drift spread (bps) | Next-day IC |", "|---|---|---|---|---|---|---|---|---|"]
    for m, d in res["models"].items():
        sh, r, dr = d["signal_share"], d["reaction"], d["drift"]
        lines.append(f"| {m} | {sh['bullish']:.0%} / {sh['neutral']:.0%} / {sh['bearish']:.0%} "
                     f"| {r['ic']['mean']:+.3f} | {r['spread_bps']['mean']:+.1f} | {r['long_bps']['mean']:+.1f} "
                     f"| {r['short_bps']['mean']:+.1f} | {dr['ic']['mean']:+.3f} | {dr['spread_bps']['mean']:+.1f} "
                     f"| {d['next']['ic']['mean']:+.3f} |")
    c1 = res["controls"]["C1_memory_probe"]
    lines += ["", f"**C1, memory probe** (base Qwen3-1.7B, ticker and dates only, {c1['n_ticker_dates']:,} "
                  f"ticker-dates): IC with the reaction {c1['reaction']['mean']:+.3f} (t {c1['reaction']['t']:.2f}, "
                  f"p {c1['reaction']['p']:.3g}), with the drift {c1['drift']['mean']:+.3f} "
                  f"(t {c1['drift']['t']:.2f}, p {c1['drift']['p']:.3g}); answers \"higher\" "
                  f"{c1['share_answering_higher']:.0%} of the time."]
    kept, drop = res["controls"]["C2_coverage"]["kept_sample"], res["controls"]["C2_coverage"]["dropped_no_prices"]
    lines += ["", "**C2, ticker-dates kept vs dropped for lack of prices** (TF-IDF labels, descriptive)", "",
              "| | Headlines | Headlines per ticker-date | Bullish | Neutral | Bearish |", "|---|---|---|---|---|---|"]
    for name, d in (("Kept (sample)", kept), ("Dropped, no prices", drop)):
        sh = d["tfidf_label_share"]
        lines.append(f"| {name} | {d['headlines']:,} | {d['headlines_per_ticker_date']:.2f} | {sh['bullish']:.0%} "
                     f"| {sh['neutral']:.0%} | {sh['bearish']:.0%} |")
    lines += ["", "Market-adjusted returns: stock minus SPY, winsorized at 0.5% and 99.5%. IC: Spearman correlation "
                  "between the score and the return over all ticker-dates of a day, averaged over days. "
                  "Spread: bullish mean minus bearish mean, averaged over days.",
              f"Break-even round-trip cost of the drift spread: {res['break_even_round_trip_cost_bps']:.1f} bps per leg."]
    return "\n".join(lines) + "\n"
