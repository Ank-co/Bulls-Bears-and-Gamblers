"""Part 4: measures on the models' answers (see docs/part4-design.md, "Measures").

Everything here is a deterministic computation on the answers and on the scenario numbers; tests
check it against answers with known, planted behaviour.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from bbg import fictional

CLIP = (1, 99)                      # answers are clipped to 1..99 before taking log-odds
TERMS = ["base_rate", "e_congruent", "e_reversed", "distractor", "dramatic_tone"]
# smallest effects of interest, used to size the test (log-odds weights and differences)
SMALLEST_EFFECTS = {"H1": 0.10, "H2": 0.05, "H3": 0.10, "H4": 0.10, "H5": 0.05}


def answer_logodds(p: float) -> float:
    q = min(max(p, CLIP[0]), CLIP[1]) / 100
    return math.log(q / (1 - q))


def regressors(sc: dict, reading: dict | None = None) -> dict:
    """The regressors of one stated scenario, from what its text shows.

    An item whose track record is 50% / 50% is a distractor. An evidence item is congruent when its
    track record points the way of its reading, reversed otherwise. The reading is the conventional
    one, or a model's own (reading[(kind, version)] in +1 good, -1 bad, 0 neither); evidence a model
    reads as neither goes to e_unread. The tone term always uses the wording (euphoric +1, alarmist -1).
    """
    out = {"base_rate": fictional.logit(sc["base_rate"] / 100), "e_congruent": 0.0, "e_reversed": 0.0,
           "e_unread": 0.0, "distractor": 0.0, "dramatic_tone": 0.0}
    for it in sc["items"]:
        up, down = fictional.shown_version_pct(it)
        contrib = math.log(up / down)
        t = it["textbook"] if reading is None else reading.get((it["family"], it["textbook"]), 0)
        if contrib == 0:
            out["distractor"] += t
        elif t == 0:
            out["e_unread"] += contrib
        elif (contrib > 0) == (t > 0):
            out["e_congruent"] += contrib
        else:
            out["e_reversed"] += contrib
        if it["dramatic"]:
            out["dramatic_tone"] += it["textbook"]
    out["ideal"] = out["base_rate"] + out["e_congruent"] + out["e_reversed"] + out["e_unread"]
    return out


def stated_frame(scenarios: list[dict], answers: list[dict], reading: dict | None = None) -> pd.DataFrame:
    """One row per scenario answered: outcome, answer (NaN on failure), regressors."""
    by_id = {a["id"]: a for a in answers}
    rows = []
    for sc in scenarios:
        a = by_id.get(sc["id"])
        if a is None:
            continue
        r = regressors(sc, reading)
        rows.append({"id": sc["id"], "y": sc["outcome"], "answer": a["value"] if a["error"] is None else np.nan,
                     "error": a["error"], **r})
    df = pd.DataFrame(rows)
    df["answer_logodds"] = [answer_logodds(v) if v == v else np.nan for v in df["answer"]]
    df["ideal_p"] = 1 / (1 + np.exp(-df["ideal"]))
    return df


def _fit(y: pd.Series, X: pd.DataFrame, groups=None):
    X = sm.add_constant(X, has_constant="add")
    if groups is None:
        return sm.OLS(y, X).fit(cov_type="HC1")
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def _coefs(fit) -> dict:
    return {k: {"estimate": float(fit.params[k]), "se": float(fit.bse[k]), "t": float(fit.tvalues[k])}
            for k in fit.params.index}


def grether(df: pd.DataFrame) -> dict:
    """Belief-updating regression on the answers that are not failures. Ideal: 0, 1, 1, 1, 0, 0."""
    d = df.dropna(subset=["answer_logodds"])
    terms = TERMS + (["e_unread"] if (d["e_unread"] != 0).any() else [])
    fit = _fit(d["answer_logodds"], d[terms])
    return {"n": int(len(d)), "weights": _coefs(fit)}


def shift_amplification_noise(df: pd.DataFrame) -> dict:
    """Regression of the answer's log-odds on the ideal log-odds: intercept, slope, residual sd."""
    d = df.dropna(subset=["answer_logodds"])
    fit = _fit(d["answer_logodds"], d[["ideal"]])
    c = _coefs(fit)
    return {"n": int(len(d)), "shift": c["const"], "amplification": c["ideal"],
            "noise_sd": float(np.sqrt(fit.mse_resid))}


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), CLIP[0] / 100, CLIP[1] / 100)
    y = np.asarray(y, float)
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}


def accuracy(df: pd.DataFrame, extra: dict[str, np.ndarray] | None = None) -> dict:
    """Brier and log loss of the model and of the references, on the same answered scenarios."""
    d = df.dropna(subset=["answer"])
    out = {"model": scores(d["answer"] / 100, d["y"]), "ideal": scores(d["ideal_p"], d["y"]),
           "base_rate_only": scores(1 / (1 + np.exp(-d["base_rate"])), d["y"])}
    for name, p in (extra or {}).items():
        out[name] = scores(pd.Series(p, index=df.index).loc[d.index], d["y"])
    return out


def failures(answers: list[dict]) -> dict:
    errs = Counter(a["error"] for a in answers if a["error"] is not None)
    return {"n": len(answers), "failures": int(sum(errs.values())),
            "rate": float(sum(errs.values()) / max(len(answers), 1)), "by_reason": dict(errs)}


def runtime(answers: list[dict]) -> dict:
    sec = [a["seconds"] for a in answers]
    tok = [a["prompt_tokens"] for a in answers if a.get("prompt_tokens") is not None]
    return {"mean_seconds": float(np.mean(sec)) if sec else None,
            "mean_prompt_tokens": float(np.mean(tok)) if tok else None}


def paired(df_a: pd.DataFrame, df_b: pd.DataFrame, shift: float = 0.1, sizes=(1000, 2000)) -> dict:
    """Base (a) against abliterated (b) on the same scenarios: answer differences, correlation of the
    two models' errors around their own trend, and the power to detect a shift of `shift` log-odds."""
    m = df_a[["id", "answer_logodds", "ideal"]].merge(df_b[["id", "answer_logodds"]], on="id",
                                                      suffixes=("_a", "_b")).dropna()
    d = m["answer_logodds_b"] - m["answer_logodds_a"]
    res = {}
    for s in ("a", "b"):
        fit = sm.OLS(m[f"answer_logodds_{s}"], sm.add_constant(m[["ideal"]])).fit()
        res[s] = fit.resid
    # H4 tests the shift once a difference of slope is allowed for: its noise is the spread of the
    # difference around a line in the ideal log-odds, which is what the power uses
    trend = sm.OLS(d, sm.add_constant(m[["ideal"]])).fit()
    sd = float(np.sqrt(trend.mse_resid))
    out = {"n": int(len(m)), "mean_difference": float(d.mean()), "sd_difference": float(d.std(ddof=1)),
           "sd_difference_around_trend": sd,
           "error_correlation": float(np.corrcoef(res["a"], res["b"])[0, 1]), "power": []}
    for n in sizes:
        et = shift * math.sqrt(n) / sd if sd > 0 else float("inf")
        out["power"].append({"scenarios": n, "shift_logit": shift, "expected_t": et,
                             "power_5pct": float(stats.norm.cdf(et - stats.norm.ppf(0.95))),
                             "power_1pct": float(stats.norm.cdf(et - stats.norm.ppf(0.99)))})
    return out


def stacked_shift_slope(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """H4 and H5: difference of intercept and of slope (b minus a) on the ideal log-odds, with both
    models' answers to the same scenarios stacked and standard errors clustered by scenario."""
    a = df_a.dropna(subset=["answer_logodds"]).assign(model_b=0.0)
    b = df_b.dropna(subset=["answer_logodds"]).assign(model_b=1.0)
    s = pd.concat([a, b], ignore_index=True)
    s["ideal_x_b"] = s["ideal"] * s["model_b"]
    fit = _fit(s["answer_logodds"], s[["ideal", "model_b", "ideal_x_b"]], groups=pd.factorize(s["id"])[0])
    c = _coefs(fit)
    return {"n_rows": int(len(s)), "shift_difference": c["model_b"], "slope_difference": c["ideal_x_b"]}


def history_distances(scenarios: list[dict], answers: list[dict]) -> dict:
    """History version: how far the model's answers are from each reference, in points."""
    by_id = {a["id"]: a for a in answers if a["error"] is None}
    rows = [(by_id[s["id"]]["value"], s["history_references"]) for s in scenarios if s["id"] in by_id]
    if not rows:
        return {"n": 0}
    ans = np.array([min(max(v, CLIP[0]), CLIP[1]) for v, _ in rows], float)
    out = {"n": len(rows)}
    for name in rows[0][1]:
        ref = 100 * np.array([r[name] for _, r in rows])
        out[name] = {"mean_abs_gap_points": float(np.mean(np.abs(ans - ref))),
                     "correlation": float(np.corrcoef(ans, ref)[0, 1]) if np.std(ans) > 0 else None}
    return out


def reading_agreement(items: list[dict], answers: list[dict]) -> dict:
    """Agreement of a model's reading with the conventional one, per kind of news, and the model's own
    reading of each version (majority over the variants; a tie or 'neither' gives 0)."""
    by_id = {a["id"]: a for a in answers}
    per_kind = defaultdict(Counter)
    votes = defaultdict(list)
    for it in items:
        a = by_id.get(it["id"])
        if a is None or a["error"] is not None:
            per_kind[it["family"]]["failure"] += 1
            continue
        v = {"good": 1, "bad": -1, "neither": 0}[a["value"]]
        per_kind[it["family"]]["agrees" if v == it["version"] else "opposite" if v == -it["version"] else "neither"] += 1
        votes[(it["family"], it["version"])].append(v)
    own = {}
    for key, vs in votes.items():
        c = Counter(vs).most_common()
        own[key] = c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else 0
    total = Counter()
    for c in per_kind.values():
        total.update(c)
    return {"overall": dict(total), "per_kind": {k: dict(v) for k, v in per_kind.items()},
            "own_reading": {f"{k}|{'good' if t == 1 else 'bad'}": v for (k, t), v in own.items()}}


def own_reading_map(agreement: dict) -> dict:
    """The model's own reading as a mapping (kind, version) -> direction, for the sensitivity analysis."""
    return {(k.split("|")[0], 1 if k.split("|")[1] == "good" else -1): v
            for k, v in agreement["own_reading"].items()}


def planning(df_base: pd.DataFrame, df_abl: pd.DataFrame, sizes=(2000, 3000, 4000)) -> dict:
    """Power of each hypothesis at its smallest effect of interest, for several test sizes, from the
    standard errors of the dev run scaled by the square root of the number of scenarios."""
    d = df_base.dropna(subset=["answer_logodds"])
    fit = _fit(d["answer_logodds"], d[TERMS])
    cov = fit.cov_params()
    se_h3 = math.sqrt(cov.loc["e_congruent", "e_congruent"] + cov.loc["e_reversed", "e_reversed"]
                      - 2 * cov.loc["e_congruent", "e_reversed"])
    st = stacked_shift_slope(df_base, df_abl)
    se = {"H1": float(fit.bse["base_rate"]), "H2": float(fit.bse["distractor"]), "H3": se_h3,
          "H4": st["shift_difference"]["se"], "H5": st["slope_difference"]["se"]}
    n0 = len(d)
    out = {"dev_scenarios": n0, "dev_se": se, "smallest_effects": SMALLEST_EFFECTS, "power": {}}
    for h, e in SMALLEST_EFFECTS.items():
        out["power"][h] = []
        for n in sizes:
            z = e / (se[h] * math.sqrt(n0 / n))
            out["power"][h].append({"scenarios": n, "power_5pct": float(stats.norm.cdf(z - stats.norm.ppf(0.95))),
                                    "power_1pct": float(stats.norm.cdf(z - stats.norm.ppf(0.99)))})
    return out
