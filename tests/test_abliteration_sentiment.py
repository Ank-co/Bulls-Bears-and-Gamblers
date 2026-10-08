"""Part 1 addendum: the pre-registered tests detect an optimistic bias and ignore a null one."""
import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "analyze_abliteration_sentiment",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_abliteration_sentiment.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def fake(n=2400, flip_to_bullish=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([mod.BEARISH, mod.BULLISH, 2], size=n, p=[0.15, 0.2, 0.65])
    base = np.where(rng.random(n) < 0.2, rng.integers(0, 3, n), y)
    abl = np.where(rng.random(n) < 0.2, rng.integers(0, 3, n), y)    # same error process as base
    flip = (y != mod.BULLISH) & (rng.random(n) < flip_to_bullish)
    abl = np.where(flip, mod.BULLISH, abl)
    return y, base, abl


def test_optimistic_model_is_detected():
    t = mod.run(*fake(flip_to_bullish=0.08))
    assert t["H1_false_optimism"]["confirmed"] and t["H1_false_optimism"]["diff"] > 0.05
    assert t["H2_bullish_share"]["confirmed"]
    assert all("p_holm" in v for v in t.values())


def test_no_bias_is_not_confirmed():
    t = mod.run(*fake(flip_to_bullish=0.0))
    assert not t["H1_false_optimism"]["confirmed"] and not t["H2_bullish_share"]["confirmed"]
    assert abs(t["H1_false_optimism"]["diff"]) < 0.02


def test_h1_counts_only_non_bullish_tweets():
    y, base, abl = fake()
    t = mod.run(y, base, abl)
    assert t["H1_false_optimism"]["n"] == int(np.sum(y != mod.BULLISH))
    assert t["H2_bullish_share"]["n"] == len(y)
