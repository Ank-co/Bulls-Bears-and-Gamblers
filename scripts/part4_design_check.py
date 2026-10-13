"""Part 4, design check: the numbers quoted in docs/part4-design.md, from a simulation of the fictional world.

    python scripts/part4_design_check.py

No language model is involved. The world is generated in this order: the outcome from the base rate,
then each news item from the outcome and its reliability, then the distractors independently of the
outcome. The script checks that:
1. the ideal observer of the stated version is calibrated against the outcomes and beats the base rate;
2. the belief-updating regression, fed the ideal observer itself, returns the ideal weights;
3. the chosen reliabilities leave most ideal answers away from 0 and 1;
4. with the planned number of scenarios, the weights of a noisy, biased model are estimated precisely,
   and a shift between two models is detected (with independent or correlated noise);
5. in the history version, where the track records are samples of 20 past episodes, three reference
   answers differ: the observed frequencies taken as true (calculation), a Bayesian with uniform priors
   (inference), and a Bayesian who knows the design (the calibrated one).
Writes results/part4/design_check.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import statsmodels.api as sm

from bbg import config

RELIABILITIES = (0.55, 0.65, 0.75, 0.85)
BASE_RATES = (0.3, 0.5, 0.7)
MAX_EVIDENCE, MAX_DISTRACTORS, P_REVERSED = 3, 2, 1 / 3
EPISODES = 20
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
    dramatic = rng.random((n, MAX_DISTRACTORS)) < 0.5
    e = inworld * logit(r) * ev_on
    out = {"y": y, "base_rate": logit(pi),
           "congruent": np.where(rev, 0, e).sum(1), "reversed": np.where(rev, e, 0).sum(1),
           "distractor": (di_text * di_on).sum(1), "dramatic_tone": (di_text * di_on * dramatic).sum(1),
           "items": {"r": r, "reversed": rev, "on": ev_on, "textbook": textbook, "inworld": inworld,
                     "distractor_on": di_on, "distractor_textbook": di_text}}
    out["ideal"] = out["base_rate"] + out["congruent"] + out["reversed"]
    return out


def design(w: dict) -> np.ndarray:
    return sm.add_constant(np.c_[[w[t] for t in TERMS[1:]]].T, has_constant="add")


def calibration_gap(p: np.ndarray, y: np.ndarray) -> float:
    deciles = np.clip((p * 10).astype(int), 0, 9)
    return float(max(abs(p[deciles == b].mean() - y[deciles == b].mean())
                     for b in range(10) if (deciles == b).sum() > 1000))


def oracle_rate(counts: np.ndarray, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Posterior mean of a rate given counts out of EPISODES, under a discrete prior."""
    like = values[None, :] ** counts[:, None] * (1 - values[None, :]) ** (EPISODES - counts[:, None])
    post = like * weights[None, :]
    return (post * values[None, :]).sum(1) / post.sum(1)


def history(n: int, rng: np.random.Generator) -> dict:
    """History version: the base rate and each kind of news are shown as 20 past episodes.

    For each kind of news, q is how often the asset moved in the item's textbook direction: r for a
    congruent kind, 1 - r for a reversed one, 0.5 for a distractor. The items are shown in random order,
    so the model is not told which ones are distractors.
    """
    w = world(n, rng)
    it = w["items"]
    pi = sigmoid(w["base_rate"])
    q = np.where(it["reversed"], 1 - it["r"], it["r"])
    q = np.c_[np.where(it["on"], q, np.nan), np.where(it["distractor_on"], 0.5, np.nan)]
    t = np.c_[it["textbook"], it["distractor_textbook"]]
    on = ~np.isnan(q)
    weeks_up = rng.binomial(EPISODES, pi)
    counts = rng.binomial(EPISODES, np.nan_to_num(q, nan=0.5))

    # the design's prior on q for a news item: distractor 1/3, congruent 4/9, reversed 2/9
    vals = np.array([0.5] + list(RELIABILITIES) + [1 - r for r in RELIABILITIES])
    wts = np.array([1 / 3] + [2 / 3 * (1 - P_REVERSED) / 4] * 4 + [2 / 3 * P_REVERSED / 4] * 4)
    base_vals, base_wts = np.array(BASE_RATES), np.ones(len(BASE_RATES)) / len(BASE_RATES)

    def answer(base_hat, q_hat):
        contrib = np.where(on, t * logit(np.where(on, q_hat, 0.5)), 0)
        return sigmoid(logit(base_hat) + contrib.sum(1))

    clip = lambda x: np.clip(x, 0.5 / EPISODES, 1 - 0.5 / EPISODES)
    refs = {
        "true_parameters": answer(pi, np.where(on, q, 0.5)),
        "frequencies_taken_as_true": answer(clip(weeks_up / EPISODES), clip(counts / EPISODES)),
        "uniform_prior_bayes": answer((weeks_up + 1) / (EPISODES + 2), (counts + 1) / (EPISODES + 2)),
        "design_aware_bayes": answer(oracle_rate(weeks_up, base_vals, base_wts),
                                     oracle_rate(counts.ravel(), vals, wts).reshape(counts.shape)),
    }
    return {"y": w["y"], "refs": refs}


def main() -> int:
    res = {"reliabilities": RELIABILITIES, "base_rates": BASE_RATES, "max_evidence": MAX_EVIDENCE,
           "max_distractors": MAX_DISTRACTORS, "p_reversed": P_REVERSED, "episodes": EPISODES}

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

    # two models answering the same scenarios, the second shifted up by 0.1 log-odds; their noises
    # have correlation rho (two versions of one model are likely to err together: rho > 0)
    res["4_paired_shift"] = []
    for n in (1000, 2000):
        for rho in (0.0, 0.5):
            g = np.random.default_rng(2)
            w = world(n, g)
            e1 = g.normal(0, 1.0, n)
            e2 = rho * e1 + np.sqrt(1 - rho ** 2) * g.normal(0, 1.0, n)
            d = (0.1 + 0.7 * w["ideal"] + e2) - (0.7 * w["ideal"] + e1)
            res["4_paired_shift"].append({"scenarios": n, "shift_logit": 0.1, "noise_sd_logit": 1.0,
                                          "noise_correlation": rho,
                                          "t": round(float(d.mean() / (d.std(ddof=1) / np.sqrt(n))), 2)})

    h = history(400_000, np.random.default_rng(3))
    res["5_history_references"] = {}
    for name, q in h["refs"].items():
        res["5_history_references"][name] = {
            "calibration_max_gap_over_deciles": round(calibration_gap(q, h["y"]), 4),
            "brier": round(float(np.mean((q - h["y"]) ** 2)), 4)}
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
