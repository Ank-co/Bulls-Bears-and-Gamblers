"""Deterministic metrics and paired statistical tests.

- macro-F1 (primary) and accuracy (secondary), per-class precision / recall / F1
- percentile bootstrap confidence intervals
- paired comparisons on the same test items:
    * exact McNemar test on per-item correctness
    * paired bootstrap of the macro-F1 difference (same resampled items for both models)
- Holm-Bonferroni correction for multiple comparisons
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import binomtest

from . import config

LABELS = np.array(config.LABEL_IDS)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    k = len(LABELS)
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return m


def per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = confusion(y_true, y_pred)
    out = {}
    for c in LABELS:
        tp = m[c, c]
        fp = m[:, c].sum() - tp
        fn = m[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        out[config.LABELS[int(c)]] = {"precision": p, "recall": r, "f1": f, "support": int(m[c].sum())}
    return out


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean([v["f1"] for v in per_class(y_true, y_pred).values()]))


# --- vectorized versions over bootstrap resamples: inputs have shape (B, n) ---------------------

def _accuracy_rows(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (t == p).mean(axis=1)


def _macro_f1_rows(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    f1s = []
    for c in LABELS:
        tp = ((t == c) & (p == c)).sum(axis=1)
        fp = ((t != c) & (p == c)).sum(axis=1)
        fn = ((t == c) & (p != c)).sum(axis=1)
        denom = 2 * tp + fp + fn
        f1s.append(np.divide(2 * tp, denom, out=np.zeros(len(tp), dtype=float), where=denom > 0))
    return np.mean(f1s, axis=0)


ROW_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "macro_f1": _macro_f1_rows,
    "accuracy": _accuracy_rows,
}


def _resample_indices(n: int, iters: int, seed: int, chunk: int = 1000):
    rng = np.random.default_rng(seed)
    done = 0
    while done < iters:
        b = min(chunk, iters - done)
        yield rng.integers(0, n, size=(b, n))
        done += b


def bootstrap_ci(y_true, y_pred, metric: str = "macro_f1", iters: int = config.BOOTSTRAP_ITERS,
                 seed: int = config.BOOTSTRAP_SEED, alpha: float = 0.05) -> tuple[float, float]:
    t, p = np.asarray(y_true), np.asarray(y_pred)
    fn = ROW_METRICS[metric]
    stats = np.concatenate([fn(t[idx], p[idx]) for idx in _resample_indices(len(t), iters, seed)])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric: str = "macro_f1",
                          iters: int = config.BOOTSTRAP_ITERS, seed: int = config.BOOTSTRAP_SEED,
                          alpha: float = 0.05) -> dict:
    """metric(A) - metric(B) on the same resampled items. p is the two-sided bootstrap p-value."""
    t, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    fn = ROW_METRICS[metric]
    diffs = np.concatenate([fn(t[idx], a[idx]) - fn(t[idx], b[idx])
                            for idx in _resample_indices(len(t), iters, seed)])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    point = float(fn(t[None, :], a[None, :])[0] - fn(t[None, :], b[None, :])[0])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {"diff": point, "ci": (float(lo), float(hi)), "p": float(min(1.0, p))}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on per-item correctness.

    a_only: items A gets right and B gets wrong; b_only: the reverse.
    """
    t = np.asarray(y_true)
    ca, cb = np.asarray(pred_a) == t, np.asarray(pred_b) == t
    a_only, b_only = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "p": p}


def paired_rate_test(flag_a, flag_b) -> dict:
    """Exact McNemar test on a paired yes/no property (for example "predicted bullish").

    a_only: items where only A has the property; b_only: the reverse. diff = rate(A) - rate(B).
    """
    fa, fb = np.asarray(flag_a, dtype=bool), np.asarray(flag_b, dtype=bool)
    if fa.shape != fb.shape or fa.size == 0:
        raise ValueError("flags must be non-empty and paired")
    a_only, b_only = int(np.sum(fa & ~fb)), int(np.sum(~fa & fb))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return {"n": int(fa.size), "rate_a": float(fa.mean()), "rate_b": float(fb.mean()),
            "diff": float(fa.mean() - fb.mean()), "a_only": a_only, "b_only": b_only, "p": p}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted
