"""Part 3, step 1: sample of ticker-dates from FNSPID, with their return windows.

    python scripts/part3_prepare.py --period dev \
        --news ~/fnspid/Stock_news/All_external.csv --prices ~/fnspid/Stock_price/full_history.zip

Applies the rules of docs/part3-design.md in a fixed order, computes abnormal returns for every
remaining ticker-date, then draws the fixed random sample among those with returns. Writes:
    data/part3/<period>/headlines.parquet, returns.parquet      (git-ignored, FNSPID-derived)
    data/part3/<period>/excluded_headlines.parquet              (no prices: for control C2)
    data/part3/<period>/filter_examples.csv                     (git-ignored, for the dev review)
    results/part3/<period>/sample_report.json                   (counts only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from bbg import config, news, part3, returns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(config.PART3_PERIODS), required=True)
    ap.add_argument("--news", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()
    for p in (args.news, args.prices):
        if not p.exists():
            print(f"{p} not found (download command in docs/part3-design.md)")
            return 1

    start, end = config.PART3_PERIODS[args.period]
    print(f"{args.period}: reading headlines dated {start} to {end}...", flush=True)
    raw, report = news.read_period(args.news.expanduser(), start, end, args.chunksize)

    moves = raw["title"].map(news.is_price_move)
    kept, counts = news.apply_rules(raw)
    report.update(counts)

    td = news.ticker_dates(kept)
    report["ticker_dates_candidates"] = int(len(td))
    print(f"  {len(kept):,} headlines kept on {len(td):,} ticker-dates; loading prices...", flush=True)
    px = returns.load_prices(args.prices.expanduser(), set(td["ticker"]) | {config.PART3_MARKET})
    ret, rcounts = returns.abnormal_returns(td, px, config.PART3_MARKET)
    report.update(rcounts)

    usable = ret[["ticker", "date"]].assign(_ok=True)
    excluded = kept.merge(usable, on=["ticker", "date"], how="left")
    excluded = excluded[excluded["_ok"].isna()].drop(columns="_ok")

    size = config.PART3_SAMPLE_SIZE[args.period]
    chosen = news.sample(ret[["ticker", "date"]], size, config.PART3_SEED)
    ret = ret.merge(chosen, on=["ticker", "date"], how="inner")
    heads = kept.merge(chosen, on=["ticker", "date"], how="inner")
    report.update({
        "sample_seed": config.PART3_SEED, "sample_size_target": size,
        "sample_ticker_dates": int(len(ret)), "sample_headlines": int(len(heads)),
        "sample_tickers": int(ret["ticker"].nunique()),
        "sample_per_year": {str(k): int(v) for k, v in ret["date"].dt.year.value_counts().sort_index().items()},
        "sample_top_publishers": heads["publisher"].value_counts().head(10).to_dict(),
    })

    work = part3.work_dir(args.period)
    work.mkdir(parents=True, exist_ok=True)
    heads[["id", "ticker", "date", "title", "publisher"]].to_parquet(work / "headlines.parquet", index=False)
    ret.to_parquet(work / "returns.parquet", index=False)
    excluded[["id", "ticker", "date", "title", "publisher"]].to_parquet(work / "excluded_headlines.parquet", index=False)
    report["excluded_headlines_no_prices"] = int(len(excluded))
    ex = pd.concat([raw[moves].sample(min(100, int(moves.sum())), random_state=0).assign(dropped=True),
                    heads.sample(min(100, len(heads)), random_state=0).assign(dropped=False)])
    ex[["dropped", "ticker", "date", "title"]].to_csv(work / "filter_examples.csv", index=False)

    out = part3.results_dir(args.period)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWritten: {work} (not in git) and {out / 'sample_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
