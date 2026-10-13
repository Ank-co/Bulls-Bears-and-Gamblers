"""Part 4, design check: the numbers quoted in docs/part4-design.md, from a simulation of the fictional world.

    python scripts/part4_design_check.py

No language model is involved. The world is generated in this order: the outcome from the base rate,
then each news item from the outcome and its reliability, then the distractors independently of the
outcome. The script checks that:
1. the ideal observer of the stated version is calibrated against the outcomes and beats the base rate;
2. the belief-updating regression, fed the ideal observer itself, returns the ideal weights;
3. the chosen reliabilities leave most ideal answers away from 0 and 1;
4. with the planned number of scenarios, the weights of a noisy, biased model are estimated precisely,
   and a shift between two models is detected (expected t and power, with independent or correlated
   noise);
5. in the history version (simulated with the real generator, bbg.fictional), where the track records
   are samples of past episodes, three reference answers differ: the observed frequencies taken as true
   (calculation), a Bayesian with uniform priors (inference), and a Bayesian who knows the design.
Writes results/part4/design_check.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import statsmodels.api as sm
from scipy import stats

from bbg import config, fictional

RELIABILITIES = (0.55, 0.65, 0.75, 0.85)
BASE_RATES = (0.3, 0.5, 0.7)
MAX_EVIDENCE, MAX_DISTRACTORS, P_REVERSED = 3, 2, 1 / 3
TERMS = ["const", "base_rate", "congruent", "reversed", "distractor", "dramatic_tone"]


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def world(n: int, rng: np.random.Generator) -> dict:
    """Stated version. Item arrays have one column per slot; 'on' marks the slots in use."""
    pi = rng.choice(BASE_RATES, n)
    y = (rng.random(n) < pi).astype(int)                         # 1. outcome from the base rate
    sign_y = 2 * y - 1
    k = rng.integers(1, MAX_EVIDENCE + 1, n)
    ev_on = np.arange(MAX_EVIDENCE)[None, :] < k[:, None]
    r = rng.choice(RELIABILITIES, (n, MAX_EVIDENCE))
    rev = rng.random((n, MAX_EVIDENCE)) < P_REVERSED
    right = rng.random((n, MAX_EVIDENCE)) < r                    # 2. item from outcome and reliability
    inworld = np.where(right, sign_y[:, None], -sign_y[:, None])    # direction it points to in this world
    textbook = np.where(rev, -inworld, inworld)                   # how it reads in a textbook
    d = rng.integers(0, MAX_DISTRACTORS + 1, n)
    di_on = np.arange(MAX_DISTRACTORS)[None, :] < d[:, None]
    di_text = np.where(rng.random((n, MAX_DISTRACTORS)) < 0.5, 1, -1)   # 3. independent of the outcome
    ev_dramatic = rng.random((n, MAX_EVIDENCE)) < 0.5                   # tone, whatever the role
    di_dramatic = rng.random((n, MAX_DISTRACTORS)) < 0.5
    e = inworld * logit(r) * ev_on
    out = {"y": y, "base_rate": logit(pi),
           "congruent": np.where(rev, 0, e).sum(1), "reversed": np.where(rev, e, 0).sum(1),
           "distractor": (di_text * di_on).sum(1),
           "dramatic_tone": (textbook * ev_on * ev_dramatic).sum(1) + (di_text * di_on * di_dramatic).sum(1),
           "items": {"r": r, "reversed": rev, "on": ev_on, "textbook": textbook, "inworld": inworld,
                     "distractor_on": di_on, "distractor_textbook": di_text}}
    out["ideal"] = out["base_rate"] + out["congruent"] + out["reversed"]
    return out


def design(w: dict) -> np.ndarray:
    return sm.add_constant(np.c_[[w[t] for t in TERMS[1:]]].T, has_constant="add")


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: mean |answer - frequency| over equal-width bins, weighted by size."""
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    gap = np.abs(np.bincount(b, p, bins) - np.bincount(b, y, bins))
    return float(gap.sum() / len(p))


def calibration_gap(p: np.ndarray, y: np.ndarray) -> float:
    deciles = np.clip((p * 10).astype(int), 0, 9)
    return float(max(abs(p[deciles == b].mean() - y[deciles == b].mean())
                     for b in range(10) if (deciles == b).sum() > 1000))


def history_from_generator(n: int, seed: int) -> dict:
    """History version, from the real scenario generator (bbg.fictional): the base rate shown as 20
    past weeks, and each kind of news as the versions that came before the last 10 rises and the last
    10 falls. Returns the outcomes and the three reference answers."""
    rng = np.random.default_rng(seed)
    draws = [fictional.draw(rng) for _ in range(n)]
    refs = [fictional.history_references(d) for d in draws]
    stated = np.array([fictional.ideal_stated(d) for d in draws])
    return {"y": np.array([d["outcome"] for d in draws]),
            "refs": {"true_parameters": stated, **{k: np.array([r[k] for r in refs]) for k in refs[0]}}}


def main() -> int:
    res = {"reliabilities": RELIABILITIES, "base_rates": BASE_RATES, "max_evidence": MAX_EVIDENCE,
           "max_distractors": MAX_DISTRACTORS, "p_reversed": P_REVERSED,
           "history": {"past_weeks": fictional.PAST_WEEKS, "past_rises_and_falls_each": fictional.PAST_EACH}}

    w = world(400_000, np.random.default_rng(0))
    p = sigmoid(w["ideal"])
    res["1_calibration_max_gap_over_deciles"] = calibration_gap(p, w["y"])
    res["1_brier"] = {"ideal": float(np.mean((p - w["y"]) ** 2)),
                      "base_rate_only": float(np.mean((sigmoid(w["base_rate"]) - w["y"]) ** 2)),
                      "coin": 0.25}

    b = sm.OLS(w["ideal"], design(w)).fit().params
    res["2_ideal_weights"] = dict(zip(TERMS, np.round(b, 6).tolist()))

    res["3_share_of_ideal_p"] = {"outside_5_95": float(np.mean((p < .05) | (p > .95))),
                                 "outside_1_99": float(np.mean((p < .01) | (p > .99))),
                                 "deciles": np.round(np.quantile(p, np.arange(.1, 1, .1)), 3).tolist()}

    # a model that underweights the base rate, discounts reversed evidence and follows distractors
    true = {"const": 0.0, "base_rate": 0.6, "congruent": 0.8, "reversed": 0.4, "distractor": 0.15,
            "dramatic_tone": 0.1}
    res["4_precision"] = {"assumed_model_weights": true, "runs": []}
    for n in (1000, 2000):
        for noise in (0.5, 1.0):
            g = np.random.default_rng(1)
            w = world(n, g)
            lp = sum(true[t] * (w[t] if t != "const" else 1) for t in TERMS) + g.normal(0, noise, n)
            lp = logit(np.clip(sigmoid(lp), 0.01, 0.99))              # answers between 1 and 99
            fit = sm.OLS(lp, design(w)).fit(cov_type="HC1")
            res["4_precision"]["runs"].append({"scenarios": n, "noise_sd_logit": noise,
                                               "estimate": dict(zip(TERMS, np.round(fit.params, 3).tolist())),
                                               "se": dict(zip(TERMS, np.round(fit.bse, 3).tolist()))})

    # two models answering the same scenarios, the second shifted up by 0.1 log-odds, each with noise of
    # one log-odds unit; their noises have correlation rho (two versions of one model are likely to err
    # together: rho > 0). The paired t-statistic has expectation shift * sqrt(n) / sqrt(2 (1 - rho));
    # the power is given one-sided at 5% and at 1% (the strictest Holm threshold over five tests).
    # A Monte Carlo average over 500 replications checks the formula.
    res["4_paired_shift"] = []
    for n in (1000, 2000):
        for rho in (0.0, 0.5):
            expected_t = 0.1 * np.sqrt(n) / np.sqrt(2 * (1 - rho))
            g = np.random.default_rng(2)
            e1 = g.normal(0, 1.0, (500, n))
            e2 = rho * e1 + np.sqrt(1 - rho ** 2) * g.normal(0, 1.0, (500, n))
            d = 0.1 + e2 - e1
            t = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(n))
            res["4_paired_shift"].append({
                "scenarios": n, "shift_logit": 0.1, "noise_sd_logit": 1.0, "noise_correlation": rho,
                "expected_t": round(float(expected_t), 2), "mean_t_500_runs": round(float(t.mean()), 2),
                "power_5pct": round(float(stats.norm.cdf(expected_t - stats.norm.ppf(0.95))), 2),
                "power_1pct": round(float(stats.norm.cdf(expected_t - stats.norm.ppf(0.99))), 2)})

    # the four references on the same 100,000 scenarios, with the same metrics and conventions
    h = history_from_generator(100_000, seed=3)
    # the true probabilities are calibrated by construction: their ECE is the noise floor of the metric
    res["5_history_references"] = {"scenarios": int(len(h["y"])), "clip": 1e-6, "ece_bins": 10}
    for name, q in h["refs"].items():
        q = np.clip(q, 1e-6, 1 - 1e-6)
        fit = sm.GLM(h["y"], sm.add_constant(logit(q)), family=sm.families.Binomial()).fit()
        res["5_history_references"][name] = {
            "brier": round(float(np.mean((q - h["y"]) ** 2)), 4),
            "ece_points": round(100 * ece(q, h["y"]), 2),
            "calibration_slope": round(float(fit.params[1]), 3), "slope_se": round(float(fit.bse[1]), 3),
            "calibration_intercept": round(float(fit.params[0]), 3), "intercept_se": round(float(fit.bse[0]), 3)}
    ref = h["refs"]
    res["5_history_mean_abs_gap_points"] = {
        "frequencies_vs_uniform_prior": round(float(100 * np.mean(abs(ref["frequencies_taken_as_true"]
                                                                       - ref["uniform_prior_bayes"]))), 2),
        "frequencies_vs_design_aware": round(float(100 * np.mean(abs(ref["frequencies_taken_as_true"]
                                                                      - ref["design_aware_bayes"]))), 2),
        "uniform_prior_vs_design_aware": round(float(100 * np.mean(abs(ref["uniform_prior_bayes"]
                                                                        - ref["design_aware_bayes"]))), 2)}

    out = config.RESULTS_DIR / "part4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "design_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
