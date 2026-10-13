"""Part 3, exploratory (NOT pre-registered): what does the memory probe's link with the drift mean?

    python scripts/part3_explore.py --period test

Written after the test results, to read control C1, which came out positive on the drift window.
Nothing here changes a pre-registered conclusion. Writes results/part3/<period>/exploratory.json:
- rank correlation between the probe score and each reader's score;
- drift IC of each reader on the probe's sample, and of the fine-tuned model controlling for the
  probe (daily cross-sectional regressions on ranks, averaged over days, Newey-West);
- the probe split into a between-stock part (the stock's mean probe score over the sample) and a
  within-stock part (the deviation of a date from that mean), in one joint daily regression; the
  separate ICs of the two parts are also given, but they do not add up to the total;
- a stock's number of news days: over the whole sample (retrospective, it uses later dates) and
  before the date only (known at the time);
- data checks: unique stock-dates, complete labels, days present in each daily series.
All results are written, whatever their direction. None of the p-values here is confirmatory.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from bbg import config, part3
from bbg.event_stats import daily_ic, nw_mean, ticker_date_signal
from bbg.returns import winsorize

MAIN = part3.MAIN


def short(r: dict) -> dict:
    return {"mean": r["mean"], "t": r["t"], "p": r["p"], "n_days": r["n_days"]}


def fama_macbeth(df: pd.DataFrame, xs: list[str], y: str, day: str) -> dict:
    """Average over days of the slopes of standardized ranks of y on standardized ranks of xs."""
    rows = []
    for _, g in df.groupby(day):
        if len(g) < config.PART3_MIN_PER_DAY:
            continue
        Y = g[y].rank()
        X = pd.concat([g[x].rank() for x in xs], axis=1)
        if Y.std(ddof=0) == 0 or (X.std(ddof=0) == 0).any():
            continue
        X = (X - X.mean()) / X.std(ddof=0)
        Y = (Y - Y.mean()) / Y.std(ddof=0)
        rows.append(np.linalg.lstsq(np.c_[np.ones(len(g)), X.to_numpy()], Y.to_numpy(), rcond=None)[0][1:])
    b = pd.DataFrame(rows, columns=xs)
    return {x: short(nw_mean(b[x])) for x in xs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(config.PART3_PERIODS), required=True)
    args = ap.parse_args()

    work = part3.work_dir(args.period)
    ret = pd.read_parquet(work / "returns.parquet")
    ret["w_drift"] = winsorize(ret["ab_drift"], *config.PART3_WINSOR)
    heads = pd.read_parquet(work / "headlines.parquet", columns=["id", "ticker", "date"])
    df = ret
    for m in part3.MODELS:
        lab = part3.read_labels(part3.labels_path(args.period, m), heads["id"])
        sig = ticker_date_signal(lab, heads)[["ticker", "date", "score"]].rename(columns={"score": m})
        df = df.merge(sig, on=["ticker", "date"])
    probe = pd.read_csv(part3.probe_path(args.period), parse_dates=["date"]).rename(columns={"score": "probe"})
    sub = df.merge(probe, on=["ticker", "date"])

    out = {"note": "exploratory, written after the test results; not part of the pre-registration; "
                   "p-values are descriptive, not confirmatory",
           "probe_sample": int(len(sub))}
    calendar = sorted(ret["t1"].unique())
    out["data_checks"] = {
        "duplicate_stock_dates_returns": int(ret.duplicated(["ticker", "date"]).sum()),
        "duplicate_stock_dates_probe": int(probe.duplicated(["ticker", "date"]).sum()),
        "stock_dates_with_all_four_readers": int(len(df)), "stock_dates_in_returns": int(len(ret)),
        "trading_days_in_sample": len(calendar),
        "days_in_probe_series": int(daily_ic(sub.assign(score=sub["probe"]), "w_drift", "t1").size),
    }
    out["rank_correlation_probe_vs_reader"] = {m: float(sub["probe"].corr(sub[m], method="spearman"))
                                               for m in part3.MODELS}
    out["drift_ic_on_probe_sample"] = {c: short(nw_mean(daily_ic(sub.assign(score=sub[c]), "w_drift", "t1")))
                                       for c in part3.MODELS + ["probe"]}
    out["fine_tuned_alone"] = fama_macbeth(sub, [MAIN], "w_drift", "t1")
    out["fine_tuned_with_probe"] = fama_macbeth(sub, [MAIN, "probe"], "w_drift", "t1")
    sub["probe_between_stocks"] = sub.groupby("ticker")["probe"].transform("mean")
    sub["probe_within_stock"] = sub["probe"] - sub["probe_between_stocks"]
    parts = ["probe_between_stocks", "probe_within_stock"]
    out["probe_parts_joint_regression"] = fama_macbeth(sub, parts, "w_drift", "t1")
    out["probe_parts_separate_ic_not_additive"] = {c: short(nw_mean(daily_ic(sub.assign(score=sub[c]), "w_drift", "t1")))
                                                   for c in parts}
    df = df.sort_values(["ticker", "date"])
    df["news_days_before"] = df.groupby("ticker").cumcount()          # earlier news days of the stock
    df["news_days_whole_sample"] = df.groupby("ticker")["ticker"].transform("size")
    sub = sub.merge(df[["ticker", "date", "news_days_before", "news_days_whole_sample"]], on=["ticker", "date"])
    out["news_days"] = {}
    for c, note in (("news_days_whole_sample", "retrospective: uses dates after the observation"),
                    ("news_days_before", "known at the time: earlier news days of the stock in the sample")):
        out["news_days"][c] = {
            "note": note,
            "rank_correlation_with_probe": float(sub["probe"].corr(sub[c], method="spearman")),
            "drift_ic_on_probe_sample": short(nw_mean(daily_ic(sub.assign(score=sub[c]), "w_drift", "t1"))),
            "drift_ic_full_sample": short(nw_mean(daily_ic(df.assign(score=df[c]), "w_drift", "t1")))}

    path = part3.results_dir(args.period) / "exploratory.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
