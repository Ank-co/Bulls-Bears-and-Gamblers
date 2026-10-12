"""Part 3: daily prices from the FNSPID archive, return windows around a news date, abnormal returns.

The publication hour is unknown for most headlines, so the windows only use the date D:

    reaction  last close before D  ->  first open after D     (contains the reaction, whatever the hour)
    drift     that open            ->  its close               (the news is public for sure)
    next      that close           ->  the next close

Abnormal return = stock return minus market return (SPY) over the same window. Open and close are
scaled by adjusted close / close, so that a dividend does not show up as a price gap.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = ("reaction", "drift", "next")


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols)
    need = {"date", "open", "close", "adj_close"}
    if not need <= set(df.columns):
        raise ValueError(f"price file without {sorted(need - set(df.columns))}")
    df = df[["date", "open", "close", "adj_close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open", "close", "adj_close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df[(df["open"] > 0) & (df["close"] > 0) & (df["adj_close"] > 0)]
    factor = df["adj_close"] / df["close"]
    out = pd.DataFrame({"date": df["date"], "open_adj": df["open"] * factor, "close_adj": df["adj_close"]})
    return out.drop_duplicates("date").sort_values("date", ignore_index=True)


def price_files(zip_path: Path) -> dict[str, str]:
    """Ticker -> member name in the archive (macOS metadata files ignored)."""
    with zipfile.ZipFile(zip_path) as z:
        return {Path(n).stem.upper(): n for n in z.namelist()
                if n.lower().endswith(".csv") and "__macosx" not in n.lower()
                and not Path(n).name.startswith("._")}


def load_prices(zip_path: Path, tickers) -> pd.DataFrame:
    """Long table: ticker, date, open_adj, close_adj, for the tickers that have a price file."""
    files = price_files(zip_path)
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        for t in sorted(set(tickers) & set(files)):
            try:
                df = _standardize(pd.read_csv(io.BytesIO(z.read(files[t]))))
            except ValueError:
                continue
            frames.append(df.assign(ticker=t))
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "open_adj", "close_adj"])
    return pd.concat(frames, ignore_index=True)[["ticker", "date", "open_adj", "close_adj"]]


def windows(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """t0 = last trading day before D, t1 = first trading day after D, t2 = the one after t1."""
    cal = np.asarray(calendar.sort_values(), dtype="datetime64[ns]")
    d = dates.to_numpy(dtype="datetime64[ns]")
    i0 = np.searchsorted(cal, d, side="left") - 1
    i1 = np.searchsorted(cal, d, side="right")
    i2 = i1 + 1

    def pick(i):
        ok = (i >= 0) & (i < len(cal))
        out = np.full(len(i), np.datetime64("NaT", "ns"))
        out[ok] = cal[i[ok]]
        return out

    return pd.DataFrame({"t0": pick(i0), "t1": pick(i1), "t2": pick(i2)}, index=dates.index)


def _window_returns(px: pd.DataFrame, keys: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Raw returns for each row of keys (columns by + t0, t1, t2) from price table px."""
    out = keys.copy()
    close = px[by + ["date", "close_adj"]]
    opn = px[by + ["date", "open_adj"]]
    out = out.merge(close.rename(columns={"date": "t0", "close_adj": "c0"}), on=by + ["t0"], how="left")
    out = out.merge(opn.rename(columns={"date": "t1", "open_adj": "o1"}), on=by + ["t1"], how="left")
    out = out.merge(close.rename(columns={"date": "t1", "close_adj": "c1"}), on=by + ["t1"], how="left")
    out = out.merge(close.rename(columns={"date": "t2", "close_adj": "c2"}), on=by + ["t2"], how="left")
    out["reaction"] = out["o1"] / out["c0"] - 1
    out["drift"] = out["c1"] / out["o1"] - 1
    out["next"] = out["c2"] / out["c1"] - 1
    return out.drop(columns=["c0", "o1", "c1", "c2"])


def abnormal_returns(td: pd.DataFrame, prices: pd.DataFrame, market: str) -> tuple[pd.DataFrame, dict]:
    """td: ticker, date. Returns one row per ticker-date with raw and abnormal returns."""
    mkt = prices[prices["ticker"] == market]
    if mkt.empty:
        raise ValueError(f"market series {market} missing from the price archive")
    cal = pd.DatetimeIndex(mkt["date"])
    keys = pd.concat([td[["ticker", "date"]].reset_index(drop=True),
                      windows(td["date"].reset_index(drop=True), cal)], axis=1)
    counts = {"ticker_dates": int(len(keys))}
    stock = _window_returns(prices, keys, ["ticker"])
    m = _window_returns(mkt.drop(columns="ticker"), keys[["t0", "t1", "t2"]].drop_duplicates(), [])
    m = m.rename(columns={w: f"mkt_{w}" for w in WINDOWS})
    out = stock.merge(m, on=["t0", "t1", "t2"], how="left")
    for w in WINDOWS:
        out[f"ab_{w}"] = out[w] - out[f"mkt_{w}"]
    has_price_file = td["ticker"].isin(set(prices["ticker"])).to_numpy()
    counts["dropped_no_price_file"] = int((~has_price_file).sum())
    usable = out["ab_reaction"].notna() & out["ab_drift"].notna()
    counts["dropped_missing_prices_around_date"] = int((~usable & has_price_file).sum())
    out = out[usable].reset_index(drop=True)
    counts["ticker_dates_with_returns"] = int(len(out))
    counts["next_day_missing"] = int(out["ab_next"].isna().sum())
    return out, counts


def winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)
