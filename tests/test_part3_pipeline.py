"""Part 3 end to end on synthetic FNSPID-like files: prepare, classify (stub models), analyze.

The synthetic market has a planted reaction to the news at the next open and no drift after it,
so H1 must be detected, H3 must not, and a model that labels at random must lose H2.
"""
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bbg import config, part3

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(f"t_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_fnspid(tmp: Path, seed=0):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2015-12-01", "2018-01-31")
    tickers = [f"T{i:02d}" for i in range(40)]
    rows, gaps = [], {t: np.zeros(len(cal)) for t in tickers}
    words = {1: "wins a large contract", -1: "faces a lawsuit", 0: "holds annual meeting"}
    for _ in range(12000):
        t = tickers[rng.integers(len(tickers))]
        d = pd.Timestamp("2016-01-01") + pd.Timedelta(days=int(rng.integers(0, 730)))
        s = int(rng.choice([1, -1, 0]))
        rows.append({"Date": f"{d:%Y-%m-%d} 00:00:00 UTC", "Article_title": f"{t} {words[s]} #{rng.integers(1e9)}",
                     "Stock_symbol": t.lower(), "Url": "u", "Publisher": "p"})
        i1 = np.searchsorted(cal, d, side="right")               # first open after D
        if i1 < len(cal):
            gaps[t][i1] += 0.02 * s
    rows.append({"Date": "2016-05-02 00:00:00 UTC", "Article_title": "T01 Shares Are Trading Higher",
                 "Stock_symbol": "T01", "Url": "u", "Publisher": "p"})
    rows.append({"Date": "2016-05-02 00:00:00 UTC", "Article_title": "Roundup of the day",
                 "Stock_symbol": "T02", "Url": "u", "Publisher": "p"})
    rows.append({"Date": "2016-05-02 00:00:00 UTC", "Article_title": "Roundup of the day",
                 "Stock_symbol": "T03", "Url": "u", "Publisher": "p"})
    news_path = tmp / "All_external.csv"
    pd.DataFrame(rows).to_csv(news_path, index=True)

    zpath = tmp / "full_history.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for t in tickers + ["SPY"]:
            g = gaps.get(t, np.zeros(len(cal)))
            close, opens, c = [], [], 100.0
            for k in range(len(cal)):
                o = c * (1 + g[k] + rng.normal(0, 0.002))
                c = o * (1 + rng.normal(0, 0.01))                    # no drift after the open
                opens.append(o)
                close.append(c)
            df = pd.DataFrame({"date": cal.strftime("%Y-%m-%d"), "volume": 1, "open": opens, "high": 0,
                               "low": 0, "close": close, "adj close": close})
            z.writestr(f"full_history/{t}.csv", df.iloc[::-1].to_csv(index=False))
            z.writestr(f"__MACOSX/full_history/._{t}.csv", "junk")
    return news_path, zpath


def stub(name):
    rng = np.random.default_rng(len(name))

    def by_words(texts):
        return [1 if "wins" in s else 0 if "lawsuit" in s else 2 for s in texts]

    if name == "finbert":                                 # right two times out of three
        return lambda texts: [x if rng.random() < 0.67 else int(rng.integers(3)) for x in by_words(texts)]
    if name == "tfidf-logreg":                            # random labels
        return lambda texts: [int(rng.integers(3)) for _ in texts]
    return by_words


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "PART3_DIR", tmp_path / "results" / "part3")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "results" / "metrics")
    return tmp_path


def run(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"] + argv)
    return mod.main()


def test_pipeline_end_to_end(sandbox, monkeypatch):
    news_path, zpath = synthetic_fnspid(sandbox)
    assert run(load("part3_prepare"), ["--period", "dev", "--news", str(news_path),
                                       "--prices", str(zpath), "--chunksize", "1000"], monkeypatch) == 0
    report = json.loads((sandbox / "results/part3/dev/sample_report.json").read_text())
    assert report["dropped_price_moves"] == 1 and report["dropped_several_tickers"] == 2
    assert report["dropped_no_price_file"] == 0 and report["sample_ticker_dates"] > 8000
    heads = pd.read_parquet(sandbox / "data/part3/dev/headlines.parquet")
    assert heads["date"].min() >= pd.Timestamp("2016-01-01") and heads["date"].max() <= pd.Timestamp("2017-12-31")

    classify = load("part3_classify")
    monkeypatch.setattr(classify, "make_classifier", lambda name, device, bs: stub(name))
    monkeypatch.setattr(classify, "check_against_part1", lambda name, fn: 1.0)
    monkeypatch.setattr(classify, "write_runtime", lambda *a, **k: None)
    monkeypatch.setattr(classify, "CHUNK", 1000)
    assert run(classify, ["--period", "dev"], monkeypatch) == 0
    lab = pd.read_csv(part3.labels_path("dev", "finbert"))
    assert len(lab) == len(heads) and lab["id"].is_unique

    # resume: delete the end of one file, rerun, nothing is labelled twice
    path = part3.labels_path("dev", "qwen3-1.7b-zeroshot")
    pd.read_csv(path).iloc[:1500].to_csv(path, index=False)
    assert run(classify, ["--period", "dev", "--model", "qwen3-1.7b-zeroshot"], monkeypatch) == 0
    again = pd.read_csv(path)
    assert len(again) == len(heads) and again["id"].is_unique

    assert run(load("part3_analyze"), ["--period", "dev"], monkeypatch) == 0
    res = json.loads((sandbox / "results/part3/dev/analysis.json").read_text())
    t = res["tests"]
    assert t["H1_reaction"]["confirmed"] and t["H1_reaction"]["mean"] > 0.3
    spread = res["models"]["qwen3-1.7b-lora-s0"]["reaction"]["spread_bps"]["mean"]
    assert spread == pytest.approx(400, abs=60)          # planted: +2% bullish, -2% bearish
    assert t["H2_vs_tfidf-logreg"]["confirmed"]          # random labels carry no information
    assert not t["H2_vs_qwen3-1.7b-zeroshot"]["confirmed"]   # identical labels: no difference
    assert not t["H3_drift"]["confirmed"]                # no drift was planted
    assert "Dev period" in (sandbox / "results/part3/dev/analysis.md").read_text()
    assert not (sandbox / "results/metrics/test_access.log").exists()


def test_classify_refuses_a_model_that_is_not_the_part1_model(sandbox, monkeypatch):
    work = part3.work_dir("dev")
    work.mkdir(parents=True)
    pd.DataFrame({"id": ["a"], "title": ["x"]}).to_parquet(work / "headlines.parquet")
    classify = load("part3_classify")
    monkeypatch.setattr(classify, "make_classifier", lambda name, device, bs: stub(name))
    monkeypatch.setattr(classify, "check_against_part1", lambda name, fn: 0.95)
    assert run(classify, ["--period", "dev", "--model", "finbert"], monkeypatch) == 1
    assert not part3.labels_path("dev", "finbert").exists()


def test_read_labels_refuses_incomplete_files(tmp_path):
    p = tmp_path / "m.csv"
    part3.append_labels(p, ["a", "b"], [0, 1])
    with pytest.raises(ValueError):
        part3.read_labels(p, ["a", "b", "c"])
    part3.append_labels(p, ["b"], [1])
    with pytest.raises(ValueError):
        part3.read_labels(p, ["a", "b"])
    with pytest.raises(ValueError):
        part3.append_labels(p, ["z"], [7])
