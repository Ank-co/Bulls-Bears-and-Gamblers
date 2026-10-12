"""Part 3: FNSPID headlines turned into a sample of ticker-dates, with rules fixed in advance.

Every rule here is deterministic and counted (see `build`). The sample never contains text that
goes into the repository: only hashed headline ids, tickers and dates are written to results/.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

NEW_YORK = "America/New_York"
COLUMNS = ["Date", "Article_title", "Stock_symbol", "Publisher"]

# --- price-move headlines --------------------------------------------------------------------
# They describe the return instead of reading the news ("71 Biggest Movers From Friday",
# "Why Tesla Shares Are Trading Higher Today", "Apple Jumps 5% After ..."). Kept out because they
# would create a mechanical link between the label and the return.
_VERBS = ("jump|jumps|jumped|jumping|soar|soars|soared|soaring|surge|surges|surged|surging|"
          "spike|spikes|spiked|climb|climbs|climbed|climbing|gain|gains|gained|gaining|"
          "rise|rises|rose|risen|rising|fall|falls|fell|falling|drop|drops|dropped|dropping|"
          "slide|slides|slid|sliding|sink|sinks|sank|sinking|tumble|tumbles|tumbled|tumbling|"
          "plunge|plunges|plunged|plunging|slump|slumps|slumped|dip|dips|dipped|"
          "decline|declines|declined|declining|skyrocket|skyrockets|skyrocketed|rocket|rockets|"
          "rocketed|shed|sheds|tank|tanks|tanked|crater|craters|cratered|slip|slips|slipped|"
          "plummet|plummets|plummeted|rally|rallies|rallied|rallying|pop|pops|popped")
_ROUNDUP = re.compile(
    r"\bmovers?\b|\b(?:gainers|losers)\b|\b52[- ]week\s+(?:highs?|lows?)\b"
    r"|\bstocks?\s+(?:moving|that\s+hit|hitting)\b"
    r"|\b(?:pre-?market|after-?hours|mid-?day)\b.{0,30}\b(?:session|movers|trading|update)\b",
    re.IGNORECASE)
_DIRECTION = re.compile(
    r"\b(?:trading|moving|trades|traded)\s+(?:sharply\s+|slightly\s+)?(?:higher|lower)\b"
    r"|\b(?:shares|stocks?)\b(?:\s+\S+){0,4}?\s+(?:are|is|were|was)\s+(?:sharply\s+|slightly\s+)?(?:higher|lower|up|down)\b"
    r"|\b(?:up|down)\s+\d+(?:\.\d+)?\s?%",
    re.IGNORECASE)
_VERB_PCT = re.compile(r"\b(?:" + _VERBS + r")\b.{0,40}?\d+(?:\.\d+)?\s?%", re.IGNORECASE)
_FUNDAMENTAL = re.compile(
    r"\b(?:revenue|revenues|sales|earnings|eps|profit|profits|income|margin|margins|dividend|"
    r"deliveries|production|subscribers|users|traffic|bookings|orders|shipments|guidance)\b",
    re.IGNORECASE)


def is_price_move(title: str) -> bool:
    """True when the headline reports a price move rather than news."""
    if _ROUNDUP.search(title) or _DIRECTION.search(title):
        return True
    m = _VERB_PCT.search(title)
    # "Revenue rises 12%" is news about the company, not about its share price.
    return bool(m) and not _FUNDAMENTAL.search(title[:m.start()])


# --- dates -----------------------------------------------------------------------------------
def parse_stamp(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S UTC", utc=True, errors="coerce")
    rest = out.isna() & s.notna()
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], utc=True, errors="coerce", format="mixed")
    return out


def event_date(stamp: pd.Series) -> pd.Series:
    """Calendar date D of the news.

    Most FNSPID rows only have a date, stored as 00:00:00 UTC: D is that date. Rows with a real
    time get their New York date. Returns naive datetime64 values at midnight.
    """
    utc_day = stamp.dt.tz_localize(None).dt.normalize()
    ny_day = stamp.dt.tz_convert(NEW_YORK).dt.tz_localize(None).dt.normalize()
    midnight = (stamp.dt.hour == 0) & (stamp.dt.minute == 0) & (stamp.dt.second == 0)
    return utc_day.where(midnight, ny_day)


# --- text ------------------------------------------------------------------------------------
_SPACES = re.compile(r"\s+")


def normalize_title(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(_SPACES, " ", regex=True).str.strip()


def headline_ids(ticker: pd.Series, day: pd.Series, title_norm: pd.Series) -> pd.Series:
    keys = ticker + "|" + day.dt.strftime("%Y-%m-%d") + "|" + title_norm
    return keys.map(lambda k: hashlib.sha1(k.encode("utf-8")).hexdigest()[:16])


# --- reading ---------------------------------------------------------------------------------
def read_period(path: Path, start: str, end: str, chunksize: int = 500_000) -> tuple[pd.DataFrame, dict]:
    """Rows whose date D falls in [start, end], with ticker, D, title and publisher."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    frames, counts = [], {"rows_read": 0, "unparsed_or_empty": 0}
    for chunk in pd.read_csv(path, usecols=lambda c: c in COLUMNS, chunksize=chunksize, dtype=str):
        counts["rows_read"] += len(chunk)
        stamp = parse_stamp(chunk["Date"])
        ok = stamp.notna() & chunk["Article_title"].notna() & chunk["Stock_symbol"].notna()
        counts["unparsed_or_empty"] += int((~ok).sum())
        chunk, stamp = chunk[ok], stamp[ok]
        day = event_date(stamp)
        keep = (day >= lo) & (day <= hi)
        if keep.any():
            frames.append(pd.DataFrame({
                "ticker": chunk.loc[keep, "Stock_symbol"].str.upper().str.strip(),
                "date": day[keep],
                "title": chunk.loc[keep, "Article_title"].str.strip(),
                "publisher": chunk.loc[keep, "Publisher"].fillna("") if "Publisher" in chunk else "",
            }))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ticker", "date", "title", "publisher"])
    counts["rows_in_period"] = int(len(df))
    return df, counts


# --- rules -----------------------------------------------------------------------------------
def apply_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rules 1 to 3 of the design: one ticker per headline, no duplicates, no price moves."""
    df = df.copy()
    df["title_norm"] = normalize_title(df["title"])
    counts = {}

    n_tickers = df.groupby(["title_norm", "date"])["ticker"].transform("nunique")
    counts["dropped_several_tickers"] = int((n_tickers > 1).sum())
    df = df[n_tickers == 1]

    dup = df.duplicated(["ticker", "date", "title_norm"])
    counts["dropped_duplicates"] = int(dup.sum())
    df = df[~dup]

    move = df["title"].map(is_price_move)
    counts["dropped_price_moves"] = int(move.sum())
    df = df[~move]

    df = df.assign(id=headline_ids(df["ticker"], df["date"], df["title_norm"]))
    if df["id"].duplicated().any():
        raise RuntimeError("headline id collision")
    counts["headlines_kept"] = int(len(df))
    return df.reset_index(drop=True), counts


def ticker_dates(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["ticker", "date"]).size().rename("n_headlines").reset_index()
              .sort_values(["date", "ticker"], ignore_index=True))


def sample(td: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Fixed random sample of ticker-dates (all of them if there are fewer)."""
    td = td.sort_values(["date", "ticker"], ignore_index=True)
    if len(td) <= size:
        return td
    idx = np.random.default_rng(seed).choice(len(td), size=size, replace=False)
    return td.iloc[np.sort(idx)].reset_index(drop=True)
