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


# ---------------------------------------------------------------- confirmatory tests (pre-registered)

DIRECTIONS = {"H1": "less", "H2": "greater", "H3": "less", "H4": "greater", "H5": "two-sided"}
ALPHA = 0.05
POINTS = 2.5          # smallest effect of interest: 2.5 points of probability on a typical answer


def _p_value(z: float, direction: str) -> float:
    if direction == "less":
        return float(stats.norm.cdf(z))
    if direction == "greater":
        return float(stats.norm.sf(z))
    return float(2 * stats.norm.sf(abs(z)))


def holm(p: dict, alpha: float = ALPHA) -> dict:
    """Holm's step-down procedure over the whole family: True where the hypothesis is rejected."""
    order = sorted(p, key=p.get)
    out, still = {}, True
    for i, h in enumerate(order):
        still = still and p[h] <= alpha / (len(order) - i)
        out[h] = still
    return out


def confirmatory(df_base: pd.DataFrame, df_abl: pd.DataFrame, alpha: float = ALPHA,
                 bias: dict | None = None) -> dict:
    """The five pre-registered tests, with their directions and Holm's correction over the five.

    H1 w_base < 1, H2 w_dist > 0, H3 w_rev < w_cong (base model, robust standard errors); H4 higher
    intercept and H5 different slope for the abliterated model on the ideal log-odds (both models
    stacked, standard errors clustered by scenario). Normal reference distribution. 'bias' moves each
    null value by the measurement bias that rounding answers to whole percentages and clipping them
    to 1..99 create when the model behaves exactly as the null says (estimated by simulation)."""
    d = df_base.dropna(subset=["answer_logodds"])
    fit = _fit(d["answer_logodds"], d[TERMS])
    cov = fit.cov_params()
    st = stacked_shift_slope(df_base, df_abl)
    est = {"H1": (fit.params["base_rate"] - 1, fit.bse["base_rate"]),
           "H2": (fit.params["distractor"], fit.bse["distractor"]),
           "H3": (fit.params["e_reversed"] - fit.params["e_congruent"],
                  math.sqrt(cov.loc["e_reversed", "e_reversed"] + cov.loc["e_congruent", "e_congruent"]
                            - 2 * cov.loc["e_reversed", "e_congruent"])),
           "H4": (st["shift_difference"]["estimate"], st["shift_difference"]["se"]),
           "H5": (st["slope_difference"]["estimate"], st["slope_difference"]["se"])}
    out = {}
    for h, (e, se) in est.items():
        null = (bias or {}).get(h, 0.0)
        z = float((e - null) / se)
        out[h] = {"estimate": float(e), "se": float(se), "null_value": float(null), "z": z,
                  "direction": DIRECTIONS[h], "p": _p_value(z, DIRECTIONS[h])}
    rejected = holm({h: r["p"] for h, r in out.items()}, alpha)
    for h in out:
        out[h]["rejected_holm"] = bool(rejected[h])
    z_check = float(fit.params["e_congruent"] / fit.bse["e_congruent"])
    out["manipulation_check"] = {"w_congruent": float(fit.params["e_congruent"]), "z": z_check,
                                 "p": _p_value(z_check, "greater"), "passed": _p_value(z_check, "greater") < alpha}
    return out


def smallest_effects(points: float = POINTS, n: int = 100_000, seed: int = 5) -> dict:
    """Each hypothesis's smallest effect of interest, from one criterion: an effect that moves a typical
    answer by `points` points of probability around 50%, where a change of x in log-odds moves the
    probability by about x / 4. Typical sizes of the regressors come from the generator, not from any
    model's answers."""
    rng = np.random.default_rng(seed)
    regs = [regressors(fictional.draw(rng)) for _ in range(n)]
    base = np.mean([abs(r["base_rate"]) for r in regs if r["base_rate"] != 0])
    dist = np.mean([abs(r["distractor"]) for r in regs if r["distractor"] != 0])
    contrib = []
    rng2 = np.random.default_rng(seed + 1)
    for _ in range(n // 4):
        for it in fictional.draw(rng2)["items"]:
            up, down = fictional.shown_version_pct(it)
            if up != down:
                contrib.append(abs(math.log(up / down)))
    ideal = np.mean([abs(r["ideal"]) for r in regs])
    k = 4 * points / 100
    typical = {"base_rate_logodds": float(base), "distractor_sum": float(dist),
               "evidence_logodds": float(np.mean(contrib)), "ideal_logodds": float(ideal)}
    effects = {"H1": k / typical["base_rate_logodds"], "H2": k / typical["distractor_sum"],
               "H3": k / typical["evidence_logodds"], "H4": k, "H5": k / typical["ideal_logodds"]}
    return {"points": points, "typical_magnitudes": typical, "effects": {h: float(v) for h, v in effects.items()}}


def simulate_power(dev_base: pd.DataFrame, dev_abl: pd.DataFrame, effects: dict,
                   sizes=(2000, 3000, 4000), reps: int = 300, failure_rates=(0.0, 0.05), seed: int = 9) -> dict:
    """Power of the exact confirmatory analysis, by simulation from the dev run.

    Each simulated test run draws its scenarios from the dev scenarios with replacement, each scenario
    keeping its own regressors and the residuals of both models on it (case bootstrap): the errors keep
    their shape, their link with the regressors (heteroskedasticity) and their dependence between the
    two models. The base model's answers follow its belief-updating regression fitted on dev, with the
    tested quantities set by the world; the abliterated model's answers add a shift and a slope on the
    ideal log-odds. Answers are rounded to whole percentages, a share of them is dropped as failures,
    and confirmatory() runs with Holm's correction over the five tests. 'at_smallest_effects' sets each
    tested quantity at its smallest effect of interest; 'null' sets every one at its null value, which
    gives the family-wise error. A first stage measures the bias that rounding and clipping create
    under the null; the tests then move their null values by it."""
    m = dev_base.merge(dev_abl[["id", "answer_logodds"]], on="id", suffixes=("", "_abl")).dropna(
        subset=["answer_logodds", "answer_logodds_abl"]).reset_index(drop=True)
    g = _fit(m["answer_logodds"], m[TERMS]).params
    mu_dev = g["const"] + m[TERMS] @ g[TERMS]
    e_b = (m["answer_logodds"] - mu_dev).to_numpy()
    e_a = sm.OLS(m["answer_logodds_abl"] - mu_dev, sm.add_constant(m[["ideal"]])).fit().resid.to_numpy()
    rng = np.random.default_rng(seed)
    worlds = {
        "at_smallest_effects": {"base_rate": 1 - effects["H1"], "distractor": effects["H2"],
                                "e_reversed": g["e_congruent"] - effects["H3"],
                                "shift": effects["H4"], "slope": -effects["H5"]},
        "null": {"base_rate": 1.0, "distractor": 0.0, "e_reversed": g["e_congruent"], "shift": 0.0, "slope": 0.0},
    }
    out = {"method": "case bootstrap of the dev scenarios with both models' residuals",
           "dev_scenarios": int(len(m)), "dev_coefficients": {k: float(v) for k, v in g.items()},
           "dev_residual_sd": {"base": float(e_b.std()), "abliterated": float(e_a.std())},
           "dev_residual_correlation": float(np.corrcoef(e_b, e_a)[0, 1]), "reps": reps, "runs": []}
    def one_run(coef, w, n, f, bias):
        j = rng.integers(0, len(m), n)
        s_ = m.iloc[j].reset_index(drop=True)
        mu = coef["const"] + s_[TERMS] @ coef[TERMS]
        frames = []
        for x in (mu + e_b[j], mu + w["shift"] + w["slope"] * s_["ideal"] + e_a[j]):
            ans = np.rint(100 / (1 + np.exp(-x))).astype(float)
            ans[rng.random(n) < f] = np.nan
            fr = s_[TERMS + ["e_unread", "ideal"]].assign(id=np.arange(n), answer=ans)
            fr["answer_logodds"] = [answer_logodds(v) if v == v else np.nan for v in ans]
            frames.append(fr)
        return confirmatory(frames[0], frames[1], bias=bias)

    def coefficients(w):
        coef = g.copy()
        coef["base_rate"], coef["distractor"], coef["e_reversed"] = w["base_rate"], w["distractor"], w["e_reversed"]
        return coef

    # stage 1: the measurement bias, i.e. the mean estimate when every null is exactly true
    first = [one_run(coefficients(worlds["null"]), worlds["null"], max(sizes), 0.0, None) for _ in range(reps)]
    bias = {h: float(np.mean([r[h]["estimate"] for r in first])) for h in DIRECTIONS}
    out["measurement_bias"] = bias
    out["measurement_bias_se"] = {h: float(np.std([r[h]["estimate"] for r in first]) / np.sqrt(reps))
                                  for h in DIRECTIONS}
    # for the record: the family-wise error without that correction (fresh runs, largest size)
    raw = [one_run(coefficients(worlds["null"]), worlds["null"], max(sizes), 0.0, None) for _ in range(reps)]
    out["null_without_correction"] = {
        "scenarios": max(sizes), "any_rejected": float(np.mean([any(r[h]["rejected_holm"] for h in DIRECTIONS) for r in raw])),
        "rejection_rate": {h: float(np.mean([r[h]["rejected_holm"] for r in raw])) for h in DIRECTIONS}}
    # stage 2: power and family-wise error of the tests with the null values moved by that bias
    for world, w in worlds.items():
        coef = coefficients(w)
        for n in sizes:
            for f in failure_rates:
                hits = {h: 0 for h in DIRECTIONS}
                any_hit = all_hit = 0
                estimates = {h: [] for h in DIRECTIONS}
                for _ in range(reps):
                    res = one_run(coef, w, n, f, bias)
                    rej = [res[h]["rejected_holm"] for h in DIRECTIONS]
                    for h, r in zip(DIRECTIONS, rej):
                        hits[h] += r
                        estimates[h].append(res[h]["estimate"])
                    any_hit += any(rej)
                    all_hit += all(rej)
                out["runs"].append({"world": world, "scenarios": n, "failure_rate": f,
                                    "rejection_rate": {h: hits[h] / reps for h in hits},
                                    "mean_estimate": {h: float(np.mean(v)) for h, v in estimates.items()},
                                    "any_rejected": any_hit / reps, "all_rejected": all_hit / reps})
    return out


# ---------------------------------------------------------------- reading the confirmatory results


def effect(h: str, r: dict) -> float:
    """The tested quantity measured in the hypothesis's direction (positive = in the predicted way);
    H5 is two-sided, its effect is the absolute difference."""
    d = r["estimate"] - r["null_value"]
    return {"H1": -d, "H2": d, "H3": -d, "H4": d, "H5": abs(d)}[h]


def interpret(conf: dict, smallest: dict) -> dict:
    """Pre-registered reading of each test against its smallest effect of interest.

    rejected, effect >= smallest: 'confirmed, at least the smallest effect of interest'
    rejected, effect <  smallest: 'detected, below the smallest effect of interest'
    not rejected, upper end of the 90% interval of the effect below the smallest: 'no effect of interest'
    not rejected otherwise: 'inconclusive'"""
    out = {}
    for h in DIRECTIONS:
        r = conf[h]
        e = effect(h, r)
        upper = (abs(r["estimate"] - r["null_value"]) if h == "H5" else e) + stats.norm.ppf(0.95) * r["se"]
        if r["rejected_holm"]:
            verdict = ("confirmed, at least the smallest effect of interest" if e >= smallest[h]
                       else "detected, below the smallest effect of interest")
        else:
            verdict = "no effect of interest" if upper < smallest[h] else "inconclusive"
        out[h] = {"effect": float(e), "smallest_effect": float(smallest[h]), "upper_90": float(upper),
                  "verdict": verdict}
    return out


def holm_thresholds(conf: dict, alpha: float = ALPHA) -> dict:
    """The alpha each hypothesis was compared with in Holm's procedure (by its rank)."""
    order = sorted(DIRECTIONS, key=lambda h: conf[h]["p"])
    return {h: alpha / (len(order) - i) for i, h in enumerate(order)}


def clear_kinds_reading() -> dict:
    """A reading that keeps the conventional one for the 12 clear kinds and reads the 8 debatable
    kinds as 'neither', so that they leave the congruent / reversed / distractor terms."""
    return {(f.key, t): (0 if f.debatable else t) for f in fictional.FAMILIES for t in (1, -1)}


def calibration_table(df: pd.DataFrame, bins: int = 10) -> list[dict]:
    d = df.dropna(subset=["answer"])
    b = np.clip((d["answer"] / 100 * bins).astype(int), 0, bins - 1)
    return [{"bin": f"{100 * k // bins}-{100 * (k + 1) // bins}", "n": int((b == k).sum()),
             "mean_answer": float(d["answer"][b == k].mean() / 100), "frequency_up": float(d["y"][b == k].mean())}
            for k in range(bins) if (b == k).sum() > 0]
