"""Part 3 statistics: from headline labels to daily measures, tested with Newey-West.

A model's labels are turned into one score per ticker-date (mean of +1 bullish, 0 neutral,
-1 bearish) and a signal (its sign). Two daily measures:

- information coefficient (IC): rank correlation between the score and the abnormal return over
  all the ticker-dates of the day, neutral ones included. It rewards a model that is right often,
  not one that only labels a few obvious headlines. The tests use it.
- long-short spread: mean abnormal return of the ticker-dates read as bullish minus the mean of
  those read as bearish. It gives the size of the effect in basis points.

Each test is on the series of daily values, so that stocks reacting to the same market day are
not counted as independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config

SCORE = {i: {"bearish": -1, "bullish": 1, "neutral": 0}[n] for i, n in config.LABELS.items()}


def ticker_date_signal(labels: pd.DataFrame, headlines: pd.DataFrame) -> pd.DataFrame:
    """labels: id, y_pred. headlines: id, ticker, date. One row per ticker-date: score and signal."""
    df = headlines[["id", "ticker", "date"]].merge(labels[["id", "y_pred"]], on="id", how="left")
    if df["y_pred"].isna().any():
        raise ValueError(f"{int(df['y_pred'].isna().sum())} headlines have no label")
    df["s"] = df["y_pred"].astype(int).map(SCORE)
    out = df.groupby(["ticker", "date"])["s"].mean().rename("score").reset_index()
    out["signal"] = np.sign(out["score"]).astype(int)
    return out


def daily_spreads(df: pd.DataFrame, ret: str, day: str) -> pd.DataFrame:
    """Per day: long (signal +1) mean, short (signal -1) mean, spread; days with both legs only."""
    g = df[df["signal"] != 0].groupby([day, "signal"])[ret].agg(["mean", "size"]).unstack("signal")
    g = g.dropna(subset=[("mean", 1), ("mean", -1)])
    return pd.DataFrame({"long": g[("mean", 1)], "short": g[("mean", -1)],
                         "spread": g[("mean", 1)] - g[("mean", -1)],
                         "n_long": g[("size", 1)].astype(int), "n_short": g[("size", -1)].astype(int)})


def daily_ic(df: pd.DataFrame, ret: str, day: str, min_n: int = config.PART3_MIN_PER_DAY) -> pd.Series:
    """Per day: Spearman correlation between score and return, over all ticker-dates of the day.
    Days with fewer than min_n ticker-dates, or without variation in the score, are skipped."""
    def ic(g: pd.DataFrame) -> float:
        if len(g) < min_n or g["score"].nunique() < 2 or g[ret].nunique() < 2:
            return np.nan
        return float(g["score"].rank().corr(g[ret].rank()))
    return df.groupby(day)[["score", ret]].apply(ic).dropna().rename("ic")


def paired_diff(a: pd.Series, b: pd.Series, lags: int = config.PART3_NW_LAGS) -> dict:
    """Mean daily difference a - b on the days where both are defined."""
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    return nw_mean(both["a"] - both["b"], lags)


def nw_mean(series: pd.Series, lags: int = config.PART3_NW_LAGS) -> dict:
    """Mean of a daily series with Newey-West (Bartlett, `lags` lags) standard error, two-sided p."""
    y = series.dropna().to_numpy(dtype=float)
    if len(y) < lags + 2:
        raise ValueError("series too short for the Newey-West test")
    if np.allclose(y, y[0]):                 # no variation at all (identical series): no test
        return {"mean": float(y[0]), "se": 0.0, "t": float("nan"), "p": 1.0, "n_days": int(len(y))}
    fit = sm.OLS(y, np.ones_like(y)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean": float(fit.params[0]), "se": float(fit.bse[0]), "t": float(fit.tvalues[0]),
            "p": float(fit.pvalues[0]), "n_days": int(len(y))}


def paired_spread_diff(a: pd.DataFrame, b: pd.DataFrame, lags: int = config.PART3_NW_LAGS) -> dict:
    """Mean daily difference of two models' spreads, on the days where both are defined."""
    both = a[["spread"]].join(b[["spread"]], lsuffix="_a", rsuffix="_b", how="inner")
    return nw_mean(both["spread_a"] - both["spread_b"], lags)
