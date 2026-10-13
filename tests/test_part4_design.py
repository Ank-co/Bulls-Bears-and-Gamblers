"""Part 4 design check: the simulated world and its ideal observer behave as the design says.

The same tests will be applied to the real scenario generator when it is written.
"""
import importlib.util
from pathlib import Path

import numpy as np
import statsmodels.api as sm

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "part4_design_check.py"
spec = importlib.util.spec_from_file_location("part4_design_check", SCRIPT)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)


def test_world_follows_the_generation_order():
    w = dc.world(200_000, np.random.default_rng(4))
    it, y = w["items"], w["y"]
    sign_y = (2 * y - 1)[:, None]
    for r in dc.RELIABILITIES:
        m = it["on"] & (it["r"] == r)
        assert abs((it["inworld"] == sign_y)[m].mean() - r) < 0.01          # item points to the outcome at rate r
        cong, rev = m & ~it["reversed"], m & it["reversed"]
        assert abs((it["textbook"] == sign_y)[cong].mean() - r) < 0.01
        assert abs((it["textbook"] == sign_y)[rev].mean() - (1 - r)) < 0.015  # reversed: textbook reads wrong
    dm = it["distractor_on"]
    assert abs((it["distractor_textbook"] == sign_y)[dm].mean() - 0.5) < 0.01  # distractors ignore the outcome


def test_ideal_observer_is_calibrated_and_beats_the_base_rate():
    w = dc.world(100_000, np.random.default_rng(5))
    p = dc.sigmoid(w["ideal"])
    assert dc.calibration_gap(p, w["y"]) < 0.02
    base = dc.sigmoid(w["base_rate"])
    assert np.mean((p - w["y"]) ** 2) < np.mean((base - w["y"]) ** 2) < 0.25


def test_regression_returns_the_ideal_weights():
    w = dc.world(5_000, np.random.default_rng(6))
    b = sm.OLS(w["ideal"], dc.design(w)).fit().params
    assert np.allclose(b, [0, 1, 1, 1, 0, 0], atol=1e-9)


def test_history_references_differ_and_only_the_design_aware_one_is_calibrated():
    h = dc.history(100_000, np.random.default_rng(7))
    ref = h["refs"]
    gaps = {k: dc.calibration_gap(v, h["y"]) for k, v in ref.items()}
    assert gaps["design_aware_bayes"] < 0.02
    assert gaps["frequencies_taken_as_true"] > 0.04
    assert np.mean(abs(ref["frequencies_taken_as_true"] - ref["design_aware_bayes"])) > 0.03


def test_worked_example_of_the_design():
    # base rate 70%, congruent item 75% pointing up, reversed item 65% pointing up, one distractor
    p = dc.sigmoid(dc.logit(0.7) + dc.logit(0.75) + dc.logit(0.65))
    assert round(100 * p) == 93
