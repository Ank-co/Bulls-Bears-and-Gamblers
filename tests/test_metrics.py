import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from bbg.metrics import (accuracy, bootstrap_ci, holm, macro_f1, mcnemar_exact,
                         paired_bootstrap_diff, paired_rate_test, _macro_f1_rows)


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    noisy = np.where(rng.random(600) < 0.3, rng.integers(0, 3, 600), y)
    return y, noisy


def test_point_metrics_match_sklearn(sample):
    y, p = sample
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro", labels=[0, 1, 2]))
    assert accuracy(y, p) == pytest.approx(accuracy_score(y, p))


def test_vectorized_macro_f1_matches_scalar(sample):
    y, p = sample
    assert _macro_f1_rows(y[None, :], p[None, :])[0] == pytest.approx(macro_f1(y, p))


def test_macro_f1_handles_absent_class():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 1])        # class 2 never appears: its F1 counts as 0
    assert macro_f1(y, p) == pytest.approx(2 / 3)


def test_bootstrap_ci_contains_point_and_is_reproducible(sample):
    y, p = sample
    lo, hi = bootstrap_ci(y, p, iters=2000)
    assert lo <= macro_f1(y, p) <= hi
    assert (lo, hi) == bootstrap_ci(y, p, iters=2000)


def test_mcnemar_counts_and_identity(sample):
    y, p = sample
    r = mcnemar_exact(y, y, p)
    assert r["b_only"] == 0 and r["a_only"] == int(np.sum(p != y))
    assert mcnemar_exact(y, p, p) == {"a_only": 0, "b_only": 0, "p": 1.0}


def test_paired_diff_sign_and_identity(sample):
    y, p = sample
    better = paired_bootstrap_diff(y, y, p, iters=2000)
    assert better["diff"] > 0 and better["ci"][0] > 0 and better["p"] < 0.01
    same = paired_bootstrap_diff(y, p, p, iters=500)
    assert same["diff"] == 0 and same["ci"] == (0.0, 0.0)


def test_holm_known_values():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)   # monotone: never below the previous step


def test_paired_rate_test_counts_discordant_pairs():
    a = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    b = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 1], dtype=bool)
    r = paired_rate_test(a, b)
    assert (r["a_only"], r["b_only"], r["n"]) == (5, 1, 10)
    assert r["diff"] == pytest.approx(0.7 - 0.3)
    assert r["p"] == pytest.approx(0.21875)          # binomial(5 of 6, 0.5), two-sided
    assert paired_rate_test(a, a)["p"] == 1.0
    with pytest.raises(ValueError):
        paired_rate_test(a, b[:5])
