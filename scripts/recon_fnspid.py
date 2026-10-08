"""Part 3 reconnaissance: what is really inside the FNSPID files, before the design is frozen.

    python scripts/recon_fnspid.py --news ~/fnspid/Stock_news/All_external.csv \
                                   --prices ~/fnspid/Stock_price/full_history.zip

Reads the news file in chunks (CPU only, about 10 minutes for 6 GB) and a few price files.
Prints a summary and writes results/part3/recon.json. Only counts and a handful of example
headlines are written; the dataset itself stays outside the repository (CC BY-NC 4.0).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from bbg import config

WANTED = ["Date", "Article_title", "Stock_symbol", "Publisher", "Url"]
PROBES = ["AAPL", "MSFT", "JPM", "TSLA", "SPY", "IVV", "VOO", "QQQ", "^GSPC", "GSPC"]
FIRST_YEAR = 2015


def parse_dates(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S UTC", utc=True, errors="coerce")
    rest = d.isna() & s.notna()
    if rest.any():
        d[rest] = pd.to_datetime(s[rest], utc=True, errors="coerce", format="mixed")
    return d


def scan_news(path: Path, chunksize: int) -> dict:
    header = pd.read_csv(path, nrows=5)
    cols = [c for c in WANTED if c in header.columns]
    out = {"columns": list(header.columns), "used_columns": cols,
           "examples": header[cols].astype(str).to_dict("records")}
    rows, bad_dates, no_title = 0, 0, 0
    per_year, per_publisher, hours_et, minute_zero = Counter(), Counter(), Counter(), Counter()
    keys, tickers = [], set()
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize, dtype=str):
        rows += len(chunk)
        d = parse_dates(chunk["Date"])
        bad_dates += int(d.isna().sum())
        no_title += int(chunk["Article_title"].isna().sum())
        ok = d.notna() & chunk["Article_title"].notna() & chunk["Stock_symbol"].notna()
        chunk, d = chunk[ok], d[ok]
        per_year.update(d.dt.year.astype(int).tolist())
        if "Publisher" in chunk:
            per_publisher.update(chunk["Publisher"].fillna("(none)").tolist())
        recent = d.dt.year >= FIRST_YEAR
        dr, cr = d[recent], chunk[recent]
        et = dr.dt.tz_convert("America/New_York")
        hours_et.update(et.dt.hour.tolist())
        midnight_utc = (dr.dt.hour == 0) & (dr.dt.minute == 0) & (dr.dt.second == 0)
        minute_zero.update(midnight_utc.map({True: "00:00:00 UTC", False: "other"}).tolist())
        title_key = pd.util.hash_pandas_object(cr["Article_title"].str.lower().str.strip(), index=False)
        day = et.dt.strftime("%Y%m%d").astype(np.int64).to_numpy()
        keys.append(pd.DataFrame({"title": title_key.to_numpy(), "day": day,
                                  "ticker": cr["Stock_symbol"].str.upper().to_numpy()}))
        tickers.update(cr["Stock_symbol"].str.upper().unique().tolist())
        print(f"  {rows:,} rows read", flush=True)

    k = pd.concat(keys, ignore_index=True)
    per_headline = k.groupby(["title", "day"])["ticker"].nunique()
    dup = k.duplicated(["title", "day", "ticker"]).sum()
    out.update({
        "rows": rows, "unparsed_dates": bad_dates, "missing_titles": no_title,
        "rows_per_year": dict(sorted(per_year.items())),
        "top_publishers": per_publisher.most_common(15),
        f"since_{FIRST_YEAR}": {
            "rows": int(len(k)),
            "hour_et_histogram": dict(sorted(hours_et.items())),
            "timestamps_exactly_midnight_utc": dict(minute_zero),
            "distinct_tickers": len(tickers),
            "headline_days": int(len(per_headline)),
            "headline_days_with_one_ticker": int((per_headline == 1).sum()),
            "duplicate_rows_same_title_day_ticker": int(dup),
        },
    })
    return out, tickers


def scan_prices(path: Path, news_tickers: set[str]) -> dict:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        by_ticker = {Path(n).stem.upper(): n for n in names}
        out = {"files": len(names), "example_names": names[:5],
               "probes": {}, "news_tickers_with_prices": len(news_tickers & set(by_ticker)),
               "news_tickers": len(news_tickers)}
        for t in PROBES:
            if t not in by_ticker:
                out["probes"][t] = None
                continue
            df = pd.read_csv(io.BytesIO(z.read(by_ticker[t])))
            info = {"columns": list(df.columns), "rows": len(df), "head": df.head(3).astype(str).to_dict("records")}
            date_col = next((c for c in df.columns if c.lower() == "date"), None)
            if date_col:
                dates = pd.to_datetime(df[date_col], errors="coerce")
                info["first"], info["last"] = str(dates.min().date()), str(dates.max().date())
            lower = {c.lower().replace(" ", "_"): c for c in df.columns}
            if "adj_close" in lower and "close" in lower:
                ratio = df[lower["adj_close"]] / df[lower["close"]]
                info["adj_close_over_close"] = {"min": float(ratio.min()), "max": float(ratio.max()),
                                                "distinct_rounded": int(ratio.round(4).nunique())}
            out["probes"][t] = info
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()
    for p in (args.news, args.prices):
        if not p.exists():
            print(f"{p} not found (download command in docs/part3-design.md)")
            return 1
    print("news file...", flush=True)
    news, tickers = scan_news(args.news, args.chunksize)
    print("price archive...", flush=True)
    prices = scan_prices(args.prices, tickers)
    out_dir = config.RESULTS_DIR / "part3"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"news": news, "prices": prices}
    (out_dir / "recon.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str)[:6000])
    print(f"\nFull report: {out_dir / 'recon.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
