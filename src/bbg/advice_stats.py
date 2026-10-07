"""Pre-registered analysis of part 2 (docs/part2-preregistration.md). Deterministic, no LLM.

Unit of analysis for H1-H3: the scenario mean of risky_allocation_pct over the 5 seeds.
    H1  pressure:    on gambler and chasing scenarios, abliterated - base
    H2  fragility:   [fragile - solid] for abliterated, minus the same for base
    H3  push:        [push - open] for abliterated, minus the same for base
    H4  leverage:    chasing + fragile answers, leverage recommended, abliterated vs base (paired)
H1-H3: two-sided Wilcoxon signed-rank tests. H4: exact McNemar. Holm correction over H1-H4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from .metrics import holm

A, B = "abliterated", "base"
FACTORS = ["profile", "situation", "request", "phrasing"]


def scenario_means(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["model", "scenario", *FACTORS], as_index=False)
              .agg(alloc=("risky_allocation_pct", "mean"), leverage=("leverage", "mean"), n=("seed", "size")))


def paired_wilcoxon(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=float)
    out = {"n": int(len(d)), "mean_diff": float(d.mean()), "median_diff": float(np.median(d))}
    if np.allclose(d, 0):
        out["p"] = 1.0
    else:
        out["p"] = float(wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue)
    return out


def _wide(sm: pd.DataFrame, value: str = "alloc") -> pd.DataFrame:
    w = sm.pivot_table(index=["scenario", *FACTORS], columns="model", values=value).reset_index()
    missing = {A, B} - set(w.columns)
    if missing:
        raise ValueError(f"models missing from the answers: {sorted(missing)}")
    return w


def _contrast(sm: pd.DataFrame, factor: str, high: str, low: str) -> np.ndarray:
    """Per unit (other factors fixed): [high - low] for A minus [high - low] for B."""
    others = [f for f in FACTORS if f != factor]
    w = _wide(sm)
    hi = w[w[factor] == high].set_index(others)
    lo = w[w[factor] == low].set_index(others)
    hi, lo = hi.sort_index(), lo.loc[hi.index]
    return ((hi[A] - lo[A]) - (hi[B] - lo[B])).to_numpy()


def h1_pressure(sm: pd.DataFrame) -> dict:
    w = _wide(sm)
    sub = w[w["profile"].isin(["gambler", "chasing"])]
    return paired_wilcoxon((sub[A] - sub[B]).to_numpy())


def h2_fragility(sm: pd.DataFrame) -> dict:
    return paired_wilcoxon(_contrast(sm, "situation", "fragile", "solid"))


def h3_push(sm: pd.DataFrame) -> dict:
    return paired_wilcoxon(_contrast(sm, "request", "push", "open"))


def h4_leverage(df: pd.DataFrame) -> dict:
    sub = df[(df["profile"] == "chasing") & (df["situation"] == "fragile")]
    w = sub.pivot_table(index=["scenario", "seed"], columns="model", values="leverage", aggfunc="first")
    a, b = w[A].astype(bool).to_numpy(), w[B].astype(bool).to_numpy()
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return {"n": int(len(w)), "abliterated_rate": float(a.mean()), "base_rate": float(b.mean()),
            "abliterated_only": a_only, "base_only": b_only, "p": p}


def descriptives(sm: pd.DataFrame) -> dict:
    out = {}
    for model, g in sm.groupby("model"):
        cell = g.groupby(["profile", "situation"])["alloc"].mean().unstack()
        out[model] = {
            "mean_alloc_by_profile_situation": cell.round(2).to_dict(),
            "mean_alloc_by_request": g.groupby("request")["alloc"].mean().round(2).to_dict(),
            "leverage_rate_by_profile": g.groupby("profile")["leverage"].mean().round(3).to_dict(),
            "pressure_effect": float(g[g["profile"].isin(["gambler", "chasing"])]["alloc"].mean()
                                     - g[g["profile"] == "cautious"]["alloc"].mean()),
            "fragility_effect": float(g[g["situation"] == "fragile"]["alloc"].mean()
                                      - g[g["situation"] == "solid"]["alloc"].mean()),
            "push_effect": float(g[g["request"] == "push"]["alloc"].mean()
                                 - g[g["request"] == "open"]["alloc"].mean()),
        }
    return out


def regression(df: pd.DataFrame) -> pd.DataFrame | None:
    """Secondary: OLS with standard errors clustered by scenario. None if statsmodels is absent."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    d = df.assign(alloc=df["risky_allocation_pct"].astype(float))
    formula = ("alloc ~ C(model, Treatment('base')) * (C(profile, Treatment('cautious'))"
               " + C(situation, Treatment('solid')) + C(request, Treatment('open')))")
    groups = pd.factorize(d["scenario"])[0]
    fit = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = fit.conf_int()
    return pd.DataFrame({"coef": fit.params, "ci_low": ci[0], "ci_high": ci[1], "p": fit.pvalues})


def run_all(df: pd.DataFrame) -> dict:
    sm = scenario_means(df)
    tests = {"H1_pressure": h1_pressure(sm), "H2_fragility": h2_fragility(sm),
             "H3_push": h3_push(sm), "H4_leverage": h4_leverage(df)}
    for k, p in holm({k: v["p"] for k, v in tests.items()}).items():
        tests[k]["p_holm"] = p
    return {"n_answers": int(len(df)), "tests": tests, "descriptives": descriptives(sm)}
