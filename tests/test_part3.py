"""Part 3 checks: sample rules, date windows, abnormal returns, daily spreads and their test."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from bbg import event_stats, news, returns


# --- headlines ---------------------------------------------------------------------------------
@pytest.mark.parametrize("title,expected", [
    ("71 Biggest Movers From Friday", True),
    ("Stocks That Hit 52-Week Highs On Friday", True),
    ("46 Stocks Moving In Friday's Mid-Day Session", True),
    ("Why Tesla Shares Are Trading Higher Today", True),
    ("Shares of XYZ are lower after downgrade", True),
    ("Apple Jumps 5% After Strong iPhone Sales", True),
    ("Why Is Nvidia (NVDA) Up 5.3% Since Last Earnings Report?", True),
    ("Revenue Rises 12% at Agilent", False),
    ("Q3 Earnings Fall 8% on Weak Demand", False),
    ("B of A Securities Maintains Neutral on Agilent, Raises Price Target to $88", False),
    ("Demand is higher than expected, says CEO", False),
    ("FDA approves Pfizer drug", False),
    # cases found on dev
    ("Deere Sees FY 2017 Sales up 11% YoY", False),
    ("Online Spend Up 18.4% YoY, According to Adobe Analytics Data", False),
    ("Movers And Shakers In Diabetes Management", False),
    ("Benzinga's Volume Movers", True),
    ("40 Biggest Movers From Yesterday", True),
    ("Cyclacel Pharmaceuticals (CYCC) Tumbles on Failed Tests", True),
    ("Revenue Rises On Strong Demand", False),
    ("Stocks With Rising Relative Strength: Capital One Financial", True),
    ("Ares Capital cruises past estimates", False),
    ("Comerica Increases Prime Lending Rate From 3.75% To 4%", False),
    ("Seagate Technology Plunges 10% Following Q3 Guidance", True),
])
def test_price_move_filter(title, expected):
    assert news.is_price_move(title) is expected


def test_event_date_uses_utc_date_when_no_time_and_new_york_date_otherwise():
    s = pd.Series(["2018-03-05 00:00:00 UTC",      # date only: D = 5 March
                   "2018-03-06 01:30:00 UTC",      # 20:30 in New York on 5 March
                   "2018-03-06 14:00:00 UTC"])     # 09:00 in New York on 6 March
    d = news.event_date(news.parse_stamp(s))
    assert list(d.dt.strftime("%Y-%m-%d")) == ["2018-03-05", "2018-03-05", "2018-03-06"]


def test_rules_drop_multi_ticker_duplicates_and_moves():
    day = pd.Timestamp("2018-03-05")
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB", "AAA", "AAA", "CCC", "DDD"],
        "date": [day] * 6,
        "title": ["Market roundup", "Market roundup", "AAA wins contract", "AAA  wins contract",
                  "CCC Jumps 9% on takeover talk", "DDD names new CEO"],
        "publisher": [""] * 6,
    })
    kept, counts = news.apply_rules(df)
    assert counts == {"dropped_several_tickers": 2, "dropped_duplicates": 1,
                      "dropped_price_moves": 1, "headlines_kept": 2}
    assert sorted(kept["ticker"]) == ["AAA", "DDD"] and kept["id"].is_unique


def test_sample_is_deterministic_and_capped():
    td = pd.DataFrame({"ticker": [f"T{i}" for i in range(100)],
                       "date": pd.date_range("2018-01-01", periods=100)})
    a, b = news.sample(td, 30, seed=1), news.sample(td, 30, seed=1)
    assert len(a) == 30 and a.equals(b) and not a.equals(news.sample(td, 30, seed=2))
    assert len(news.sample(td, 500, seed=1)) == 100


def test_read_period_keeps_only_the_period(tmp_path):
    path = tmp_path / "news.csv"
    pd.DataFrame({"Unnamed: 0": [0, 1, 2], "Date": ["2017-12-31 00:00:00 UTC", "2018-01-02 00:00:00 UTC",
                                                     "garbage"],
                  "Article_title": ["a", "b", "c"], "Stock_symbol": ["aaa", "bbb", "ccc"],
                  "Url": "u", "Publisher": "p", "Article": None}).to_csv(path, index=False)
    df, counts = news.read_period(path, "2018-01-01", "2018-12-31", chunksize=2)
    assert list(df["ticker"]) == ["BBB"] and counts["no_valid_date"] == 1 and counts["rows_read"] == 3


# --- windows and returns -----------------------------------------------------------------------
# Trading days: Thu 1, Fri 2, Mon 5, Tue 6 (Wed 7 is a holiday), Thu 8, Fri 9 March 2018.
CAL = pd.DatetimeIndex(["2018-03-01", "2018-03-02", "2018-03-05", "2018-03-06", "2018-03-08", "2018-03-09"])


def test_windows_around_weekends_and_holidays():
    d = pd.Series(pd.to_datetime(["2018-03-02", "2018-03-03", "2018-03-07"]))
    w = returns.windows(d, CAL)
    fmt = lambda c: list(w[c].dt.strftime("%m-%d"))  # noqa: E731
    assert fmt("t0") == ["03-01", "03-02", "03-06"]   # last close strictly before D
    assert fmt("t1") == ["03-05", "03-05", "03-08"]   # first open strictly after D
    assert fmt("t2") == ["03-06", "03-06", "03-09"]


def prices_frame(ticker, opens, closes, adj=None):
    adj = closes if adj is None else adj
    return pd.DataFrame({"date": CAL.strftime("%Y-%m-%d"), "open": opens, "high": 0, "low": 0,
                         "close": closes, "adj close": adj, "volume": 1})


def make_zip(tmp_path, frames: dict) -> str:
    path = tmp_path / "prices.zip"
    with zipfile.ZipFile(path, "w") as z:
        for t, df in frames.items():
            z.writestr(f"full_history/{t}.csv", df.to_csv(index=False))
            z.writestr(f"__MACOSX/full_history/._{t}.csv", "junk")
    return path


def test_abnormal_returns_and_dividend_adjustment(tmp_path):
    spy = prices_frame("SPY", [100, 100, 101, 102, 102, 103], [100, 100, 102, 102, 103, 103])
    # AAA: +10% gap at the open after the news, then flat; ex-dividend (2) on 9 March
    aaa_close = [50, 50, 55, 55, 55, 53]
    aaa_adj = [c * 53 / 55 for c in aaa_close[:5]] + [53]   # dividend of 2 paid on 9 March
    aaa = prices_frame("AAA", [50, 50, 55, 55, 55, 53], aaa_close, aaa_adj)
    path = make_zip(tmp_path, {"SPY": spy, "AAA": aaa})
    assert set(returns.price_files(path)) == {"SPY", "AAA"}
    px = returns.load_prices(path, ["SPY", "AAA", "ZZZ"])
    td = pd.DataFrame({"ticker": ["AAA", "ZZZ", "AAA"],
                       "date": pd.to_datetime(["2018-03-03", "2018-03-03", "2018-03-08"])})
    out, counts = returns.abnormal_returns(td, px, "SPY")
    assert counts["dropped_no_price_file"] == 1 and counts["ticker_dates_with_returns"] == 2
    r = out.iloc[0]                                 # news dated Saturday 3 March
    assert r["reaction"] == pytest.approx(55 / 50 - 1)
    assert r["mkt_reaction"] == pytest.approx(101 / 100 - 1)
    assert r["ab_reaction"] == pytest.approx(0.10 - 0.01)
    assert r["ab_drift"] == pytest.approx(0 - (102 / 101 - 1))
    last = out.iloc[1]                              # news on 8 March: t1 = 9 March, no t2
    assert np.isnan(last["ab_next"])
    assert last["reaction"] == pytest.approx(0.0)   # the ex-dividend gap is not a loss
    assert last["drift"] == pytest.approx(0.0)


def test_missing_market_series_is_an_error(tmp_path):
    path = make_zip(tmp_path, {"AAA": prices_frame("AAA", [1] * 6, [1] * 6)})
    with pytest.raises(ValueError):
        returns.abnormal_returns(pd.DataFrame({"ticker": ["AAA"], "date": [CAL[2]]}),
                                 returns.load_prices(path, ["AAA"]), "SPY")


# --- signals and statistics --------------------------------------------------------------------
def test_ticker_date_signal_is_the_sign_of_the_mean_score():
    heads = pd.DataFrame({"id": list("abcde"), "ticker": ["X", "X", "X", "Y", "Z"],
                          "date": pd.to_datetime(["2018-01-02"] * 5)})
    # labels: 0 bearish, 1 bullish, 2 neutral
    labels = pd.DataFrame({"id": list("abcde"), "y_pred": [1, 1, 0, 2, 0]})
    s = event_stats.ticker_date_signal(labels, heads).set_index("ticker")
    assert s.loc["X", "score"] == pytest.approx(1 / 3) and s.loc["X", "signal"] == 1
    assert s.loc["Y", "signal"] == 0 and s.loc["Z", "signal"] == -1
    with pytest.raises(ValueError):
        event_stats.ticker_date_signal(labels.iloc[:4], heads)


def synthetic_panel(effect: float, n_days=250, per_day=40, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-01", periods=n_days)
    rows = []
    for d in days:
        common = rng.normal(0, 0.01)              # same-day shock shared by all stocks
        sig = rng.choice([-1, 0, 1], size=per_day)
        ret = common + effect * sig + rng.normal(0, 0.02, per_day)
        rows.append(pd.DataFrame({"t1": d, "signal": sig, "ab_reaction": ret}))
    return pd.concat(rows, ignore_index=True)


def test_spread_test_detects_an_effect_and_not_a_null():
    strong = event_stats.daily_spreads(synthetic_panel(0.004), "ab_reaction", "t1")
    r = event_stats.nw_mean(strong["spread"])
    assert r["mean"] == pytest.approx(0.008, abs=0.002) and r["p"] < 1e-6
    null = event_stats.daily_spreads(synthetic_panel(0.0, seed=1), "ab_reaction", "t1")
    assert event_stats.nw_mean(null["spread"])["p"] > 0.01
    assert (strong["n_long"] > 0).all() and (strong["n_short"] > 0).all()


def test_paired_difference_uses_common_days_only():
    a = event_stats.daily_spreads(synthetic_panel(0.004, seed=2), "ab_reaction", "t1")
    b = event_stats.daily_spreads(synthetic_panel(0.001, seed=3), "ab_reaction", "t1").iloc[:200]
    r = event_stats.paired_spread_diff(a, b)
    assert r["n_days"] == 200 and r["mean"] == pytest.approx(0.006, abs=0.002) and r["p"] < 1e-4


def test_newey_west_matches_the_textbook_formula():
    rng = np.random.default_rng(5)
    y = pd.Series(np.convolve(rng.normal(size=400), [1, 0.6, 0.3], mode="same"))
    r = event_stats.nw_mean(y, lags=5)
    e = y.to_numpy() - y.mean()
    n = len(e)
    s = e @ e / n + 2 * sum((1 - k / 6) * (e[k:] @ e[:-k]) / n for k in range(1, 6))
    assert r["se"] == pytest.approx(np.sqrt(s / n), rel=1e-6)


def test_daily_ic_rewards_being_right_often_not_being_selective():
    rng = np.random.default_rng(7)
    rows = []
    for d in pd.bdate_range("2018-01-01", periods=120):
        r = rng.normal(0, 0.02, 60)
        often = np.sign(r + rng.normal(0, 0.02, 60))            # right two times out of three
        rare = np.where(np.abs(r) > 0.04, np.sign(r), 0)          # only the extreme moves
        rows.append(pd.DataFrame({"t1": d, "ret": r, "often": often, "rare": rare}))
    df = pd.concat(rows, ignore_index=True)
    ic = {k: event_stats.daily_ic(df.assign(score=df[k], signal=df[k]), "ret", "t1") for k in ("often", "rare")}
    sp = {k: event_stats.daily_spreads(df.assign(signal=df[k]), "ret", "t1")["spread"].mean() for k in ("often", "rare")}
    assert sp["rare"] > sp["often"]                  # the spread prefers the selective model
    assert ic["often"].mean() > ic["rare"].mean()    # the IC prefers the one that is right more often
    assert event_stats.paired_diff(ic["often"], ic["rare"])["p"] < 0.01


def test_daily_ic_skips_thin_or_constant_days():
    df = pd.DataFrame({"t1": ["a"] * 5 + ["b"] * 12 + ["c"] * 12,
                       "score": [1, -1, 0, 1, -1] + [0] * 12 + list(np.linspace(-1, 1, 12)),
                       "ret": list(range(5)) + list(range(12)) + list(range(12))})
    ic = event_stats.daily_ic(df, "ret", "t1")
    assert list(ic.index) == ["c"] and ic["c"] == pytest.approx(1.0)


# --- controls ----------------------------------------------------------------------------------
from bbg import part3  # noqa: E402


def test_probe_question_contains_no_headline_and_both_dates():
    q = part3.probe_question("AAPL", pd.Timestamp("2018-03-02"), pd.Timestamp("2018-03-05"))
    assert "AAPL" in q and "Friday 02 March 2018" in q and "Monday 05 March 2018" in q
    assert q.endswith("higher or lower.") and part3.PROBE_ANSWERS == ["higher", "lower"]


def test_probe_ic_finds_memory_when_there_is_some():
    rng = np.random.default_rng(11)
    days = pd.bdate_range("2018-01-02", periods=150)
    ret = pd.DataFrame({"ticker": [f"T{i}" for i in range(150 * 40)],
                        "date": np.repeat(days - pd.Timedelta(days=1), 40), "t1": np.repeat(days, 40),
                        "ab_reaction": rng.normal(0, 0.02, 6000), "ab_drift": rng.normal(0, 0.01, 6000)})
    ret["t0"] = ret["date"] - pd.Timedelta(days=1)
    remembers = ret[["ticker", "date"]].assign(score=np.sign(ret["ab_drift"]) + rng.normal(0, 1, 6000))
    forgets = ret[["ticker", "date"]].assign(score=rng.normal(size=6000))
    assert part3.probe_ic(ret, remembers)["drift"]["p"] < 1e-6
    r = part3.probe_ic(ret, forgets)
    assert r["drift"]["p"] > 0.01 and r["n_ticker_dates"] == 6000
    assert len(part3.probe_sample(ret, 1000, seed=1)) == 1000


def test_coverage_compares_kept_and_dropped():
    inc = pd.DataFrame({"id": ["a", "b", "c"], "ticker": ["AAA", "AAA", "BBB"],
                        "date": pd.to_datetime(["2018-01-02"] * 3), "publisher": ["p", "p", "q"]})
    exc = pd.DataFrame({"id": ["x", "y"], "ticker": ["BRK.B", "TOYOF"],
                        "date": pd.to_datetime(["2018-01-02"] * 2), "publisher": ["p", "p"]})
    res = part3.coverage(inc, pd.DataFrame({"id": ["a", "b", "c"], "y_pred": [1, 0, 2]}),
                         exc, pd.DataFrame({"id": ["x", "y"], "y_pred": [0, 0]}))
    assert res["kept_sample"]["headlines_per_ticker_date"] == pytest.approx(1.5)
    assert res["dropped_no_prices"]["tfidf_label_share"]["bearish"] == 1.0
    assert set(res["dropped_no_prices"]["symbol_kind_share"]) == {
        "class share or suffix (contains . or -)", "five letters ending in F or Y (foreign, over the counter)"}
    with pytest.raises(ValueError):
        part3.coverage(inc, pd.DataFrame({"id": ["a"], "y_pred": [1]}), exc,
                       pd.DataFrame({"id": ["x", "y"], "y_pred": [0, 0]}))
