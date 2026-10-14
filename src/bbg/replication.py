"""Replication of Part 2, the Part 1 addendum and Part 4 on two more model families.

Pre-registration: docs/replication-preregistration.md. Each pair is a base model and its abliterated
copy, both quantized by the same quantizer with the same method. The tests are the original ones,
run per pair through the original code (bbg.advice_stats, scripts/analyze_abliteration_sentiment.py,
bbg.part4_stats and scripts/part4_analyze.py). This module adds only the bookkeeping: which files
belong to which pair, the checks that both models of a pair answered the same items under the same
conditions, the pre-registered replication criteria, and the power of the tests that have no dev run.
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config

ALPHA = 0.05
PREREGISTRATION = Path("docs") / "replication-preregistration.md"
OUT = config.RESULTS_DIR / "replication"
ADVICE_RAW = OUT / "advice" / "raw"
PREDICTIONS = OUT / "predictions"

# Short names are the folders of the answers; server names are the sections of the llama.cpp preset.
PAIRS = {
    "qwen": {"family": "Qwen3.6-35B-A3B (original pair)",
             "short": {"base": "base", "abliterated": "abliterated"},
             "server": {"base": "qwen3.6-35b-base-i1", "abliterated": "qwen3.6-35b-abliterated"},
             "addendum": {"base": "qwen3.6-35b-i1-zeroshot", "abliterated": "qwen3.6-35b-abliterated-zeroshot"}},
    "gemma": {"family": "Gemma 4 26B-A4B",
              "short": {"base": "gemma-base", "abliterated": "gemma-abliterated"},
              "server": {"base": "gemma-4-26b-a4b-base-i1", "abliterated": "gemma-4-26b-a4b-abliterated"},
              "addendum": {"base": "gemma-4-26b-a4b-base-i1-zeroshot",
                           "abliterated": "gemma-4-26b-a4b-abliterated-zeroshot"}},
    "llama": {"family": "Llama 3.1 8B",
              "short": {"base": "llama-base", "abliterated": "llama-abliterated"},
              "server": {"base": "llama-3.1-8b-base-i1", "abliterated": "llama-3.1-8b-abliterated"},
              "addendum": {"base": "llama-3.1-8b-base-i1-zeroshot", "abliterated": "llama-3.1-8b-abliterated-zeroshot"}},
}
REPLICATIONS = ("gemma", "llama")
ROLES = ("base", "abliterated")

# Archive of the exact files (SHA-256 from the Hugging Face listings, checked locally on 2026-10-14).
FILES = {
    "gemma-base": {"repo": "mradermacher/gemma-4-26B-A4B-it-i1-GGUF", "file": "gemma-4-26B-A4B-it.i1-Q4_K_M.gguf",
                   "repo_commit": "2758390afaf21eff610fa762736f72edab4c7df6",
                   "sha256": "01f2546777b7c3f4ab5f5c0c84e1cb6b7a213739d156edcb25e342afada22a59",
                   "source_model": "google/gemma-4-26B-A4B-it"},
    "gemma-abliterated": {"repo": "mradermacher/Huihui-gemma-4-26B-A4B-it-abliterated-i1-GGUF",
                          "file": "Huihui-gemma-4-26B-A4B-it-abliterated.i1-Q4_K_M.gguf",
                          "repo_commit": "00f632730d7c1b6999c86d4d6035fa0ce39483d8",
                          "sha256": "0f9077c9c7b40f1b056b89a48422017717190618e3136d2d3c20c758e3603676",
                          "source_model": "huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated"},
    "llama-base": {"repo": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF",
                   "file": "Meta-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf",
                   "repo_commit": "15a6b2be4570af9886c5cde164ddc8f41b4d7520",
                   "sha256": "85465a58bf902b8fe36e1e719fd1ddc23305110a6b19233f10560dfd1180f77a",
                   "source_model": "meta-llama/Llama-3.1-8B-Instruct"},
    "llama-abliterated": {"repo": "mradermacher/Meta-Llama-3.1-8B-Instruct-abliterated-i1-GGUF",
                          "file": "Meta-Llama-3.1-8B-Instruct-abliterated.i1-Q4_K_M.gguf",
                          "repo_commit": "b659b6c902f09585ea669991c0a52e67043e89a9",
                          "sha256": "74f27fbf0877b4d76d9708efab5fc6dceb6a8bc1ed2394397a020f6c71d57c9a",
                          "source_model": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"},
}
SERVER = {"llama.cpp": "b10786 (commit de8656bd9, released 2026-09-03)", "ctx-size": 8192, "parallel": 1,
          "flash-attn": "on", "cache-type-k": "q8_0", "cache-type-v": "q8_0", "jinja": True, "no-mmproj": True}

# What the original pair found, for the sign that a replication must share. Directional tests
# carry their direction in their p-value; the two-sided ones must also match the original's sign.
ORIGINAL = {
    "part2": {"H1_pressure": "confirmed", "H2_fragility": "not confirmed", "H3_push": "confirmed",
              "H4_leverage": "confirmed"},
    "addendum": {"H1_false_optimism": "confirmed", "H2_bullish_share": "confirmed", "H3_macro_f1": "confirmed"},
    "part4": {h: "confirmed" for h in ("H1", "H2", "H3", "H4", "H5")},
}
SIGN = {"part2": {"H1_pressure": 1, "H2_fragility": 1, "H3_push": 1, "H4_leverage": 1},
        "addendum": {"H1_false_optimism": 1, "H2_bullish_share": 1, "H3_macro_f1": -1},
        "part4": {"H1": -1, "H2": 1, "H3": -1, "H4": 1, "H5": -1}}


# ---------------------------------------------------------------- the pre-registration must be public


def git(*args: str, cwd: Path = config.ROOT) -> subprocess.CompletedProcess:
    """git without writing any lock and without trusting the repository globally."""
    return subprocess.run(["git", "-c", f"safe.directory={cwd}", "--no-optional-locks", *args],
                          cwd=cwd, capture_output=True, text=True)


def preregistration_status(root: Path = config.ROOT, path: Path = PREREGISTRATION,
                           remote: str = "origin/main") -> tuple[bool, str]:
    """True only if the pre-registration is committed, unchanged since, and contained in the remote
    branch as last fetched or pushed: the push is the public time stamp."""
    if not (root / path).exists():
        return False, f"{path} does not exist"
    r = git("log", "-1", "--format=%H", "--", str(path), cwd=root)
    if r.returncode != 0:
        return False, f"{path} is not committed (git: {r.stderr.strip()})"
    commit = r.stdout.strip()
    if not commit:
        return False, f"{path} is not committed"
    if git("diff", "--quiet", "HEAD", "--", str(path), cwd=root).returncode != 0:
        return False, f"{path} was modified after its last commit"
    if git("merge-base", "--is-ancestor", commit, remote, cwd=root).returncode != 0:
        return False, f"the commit of {path} ({commit[:7]}) is not in {remote}: push it first"
    return True, commit


def log_test_access(what: str) -> None:
    import datetime as dt
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_DIR / "test_access.log", "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {what}\n")


def load_script(name: str):
    """A frozen analysis script, imported as is so that the replication runs exactly its code."""
    spec = importlib.util.spec_from_file_location(f"frozen_{name}", config.ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- same items, same conditions


def check_same_items(items_a: list, items_b: list, what: str) -> None:
    """Both models of a pair must have answered exactly the same items, in the same order."""
    if list(items_a) != list(items_b):
        missing = sorted(set(items_a) ^ set(items_b))[:5]
        raise ValueError(f"{what}: the two models did not answer the same items (e.g. {missing})")


def check_fingerprint(meta_path: Path, expected: str, what: str) -> None:
    """The cache's fingerprint covers the model, prompts, schema and sampling: it must be the one
    the pre-registered code computes for this model."""
    if not meta_path.exists():
        raise FileNotFoundError(f"{what}: {meta_path} is missing")
    found = json.loads(meta_path.read_text(encoding="utf-8"))["fingerprint"]
    if found != expected:
        raise ValueError(f"{what}: produced with other prompts, schema, sampling or model ({found} != {expected})")


# ---------------------------------------------------------------- replication criteria


def criterion_a(rejected: bool, estimate_sign: float, expected_sign: int) -> bool:
    """Statistical replication: rejected after Holm, in the pre-registered direction (for a
    two-sided test, with the sign the original pair found)."""
    return bool(rejected and np.sign(estimate_sign) == expected_sign)


def part4_criteria(conf: dict, smallest: dict) -> dict:
    """Criteria A and B for the five Part 4 tests of one pair (B: A and the effect reaches the
    smallest effect of interest, which comes from the generator and is the same for every pair)."""
    from . import part4_stats as S
    out = {}
    for h in S.DIRECTIONS:
        r = conf[h]
        a = criterion_a(r["rejected_holm"], r["estimate"] - r["null_value"], SIGN["part4"][h])
        e = S.effect(h, r)
        out[h] = {"A": a, "B": bool(a and e >= smallest[h]), "effect": float(e), "smallest_effect": float(smallest[h])}
    return out


def part2_criteria(tests: dict) -> dict:
    out = {}
    for h, r in tests.items():
        d = r["abliterated_rate"] - r["base_rate"] if h == "H4_leverage" else r["mean_diff"]
        out[h] = {"A": criterion_a(r["p_holm"] < ALPHA, d, SIGN["part2"][h])}
    return out


def addendum_criteria(tests: dict) -> dict:
    return {h: {"A": criterion_a(r["p_holm"] < ALPHA, r["diff"], SIGN["addendum"][h])} for h, r in tests.items()}


def classify(successes: int, total: int = 2, originally: str = "confirmed") -> str:
    """Pre-registered reading of one hypothesis over the replication pairs."""
    if originally != "confirmed":
        return f"supported in {successes} of {total} new families (not confirmed in the original pair)"
    if successes == total:
        return f"met the criterion in {total} of {total} replications: holds in the three families tested"
    if successes == 0:
        return (f"met the criterion in 0 of {total} replications: neither replication met the fixed criterion, "
                "which does not show that the effect is specific to Qwen")
    return f"met the criterion in {successes} of {total} replications: depends on the family"


# ---------------------------------------------------------------- data of each experiment


def advice_frame(pair: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Part 2 answers of one pair, with the original model labels ('base', 'abliterated') so that
    bbg.advice_stats runs unchanged."""
    from .advice import build_scenarios
    raw_dir = raw_dir or ADVICE_RAW
    meta = pd.DataFrame(build_scenarios()).drop(columns="text")
    frames = {}
    for role in ROLES:
        path = raw_dir / f"{PAIRS[pair]['short'][role]}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run scripts/replication_run.py --pair {pair}")
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        df = pd.DataFrame(recs).merge(meta, on="scenario")
        df.insert(0, "model", role)
        frames[role] = df.sort_values(["scenario", "seed"]).reset_index(drop=True)
    check_same_items(list(zip(frames["base"]["scenario"], frames["base"]["seed"])),
                     list(zip(frames["abliterated"]["scenario"], frames["abliterated"]["seed"])), "Part 2")
    return pd.concat(frames.values(), ignore_index=True)


# ---------------------------------------------------------------- power of the tests without a dev run


def wilcoxon_power(d: np.ndarray, scale: float, alpha: float, reps: int = 4000, seed: int = 0) -> float:
    """Power of the two-sided Wilcoxon test in the original direction, resampling the original units
    with their mean moved to `scale` times the original mean (their spread kept)."""
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    centred = d - d.mean() + scale * d.mean()
    sign = np.sign(d.mean())
    hits = 0
    for _ in range(reps):
        s = rng.choice(centred, size=len(d), replace=True)
        if np.allclose(s, 0):
            continue
        p = wilcoxon(s, zero_method="pratt", alternative="two-sided").pvalue
        hits += (p < alpha) and (np.sign(s.mean()) == sign)
    return hits / reps


def mcnemar_power(n_pairs: int, a_only: int, b_only: int, scale: float, alpha: float) -> float:
    """Exact power of the McNemar test in the original direction when the difference of rates is
    `scale` times the original one (b_only kept, part of a_only becoming concordant)."""
    rate_a = (b_only + scale * (a_only - b_only)) / n_pairs
    rate_b = b_only / n_pairs
    p_disc = rate_a + rate_b
    if p_disc <= 0:
        return 0.0
    pi = rate_a / p_disc
    nd = np.arange(1, n_pairs + 1)
    weight = stats.binom.pmf(nd, n_pairs, p_disc)
    # the exact two-sided p of x discordant pairs in A's favour (x > nd / 2) is 2 P(X >= x | 0.5):
    # it is at most alpha from x = k + 1 on, k being the (1 - alpha / 2) quantile
    k = stats.binom.ppf(1 - alpha / 2, nd, 0.5)
    exact = 2 * stats.binom.sf(k, nd, 0.5) < alpha          # strict, as the analysis uses p < alpha
    k = np.where(exact, k, k + 1)
    return float(min(1.0, np.sum(weight * stats.binom.sf(k, nd, pi))))


def normal_power(diff: float, se: float, scale: float, alpha: float) -> float:
    """Two-sided z test at level alpha, counted only in the original direction."""
    return float(stats.norm.cdf(abs(scale * diff) / se - stats.norm.ppf(1 - alpha / 2)))


def power_without_dev(advice_answers: pd.DataFrame, addendum_results: dict, scales=(1.0, 0.5)) -> dict:
    """Part 2 and the addendum have no dev run: their power is computed for an effect as large as the
    original pair's, and half as large, with the original pair's spread. Each test is taken at the
    Bonferroni level alpha / k of its family, a lower bound of its power under Holm."""
    from . import advice_stats as A
    sm = A.scenario_means(advice_answers)
    w = A._wide(sm)
    h1 = (w[w["profile"].isin(["gambler", "chasing"])][A.A] - w[w["profile"].isin(["gambler", "chasing"])][A.B]).to_numpy()
    h3 = A._contrast(sm, "request", "push", "open")
    lev = A.h4_leverage(advice_answers)
    out = {"method": "original pair's units resampled (Part 2) or exact / normal power from the original "
                     "counts and interval (addendum); each test at alpha / k, a lower bound under Holm",
           "part2": {}, "addendum": {}}
    k2, k3 = 4, 3
    for s in scales:
        key = f"x{s:g}"
        out["part2"][key] = {
            "H1_pressure": wilcoxon_power(h1, s, ALPHA / k2),
            "H2_fragility": None,
            "H3_push": wilcoxon_power(h3, s, ALPHA / k2, seed=1),
            "H4_leverage": mcnemar_power(lev["n"], lev["abliterated_only"], lev["base_only"], s, ALPHA / k2)}
        t = addendum_results["tests"]
        h3a = t["H3_macro_f1"]
        se = (h3a["ci"][1] - h3a["ci"][0]) / (2 * stats.norm.ppf(0.975))
        out["addendum"][key] = {
            "H1_false_optimism": mcnemar_power(t["H1_false_optimism"]["n"], t["H1_false_optimism"]["a_only"],
                                               t["H1_false_optimism"]["b_only"], s, ALPHA / k3),
            "H2_bullish_share": mcnemar_power(t["H2_bullish_share"]["n"], t["H2_bullish_share"]["a_only"],
                                              t["H2_bullish_share"]["b_only"], s, ALPHA / k3),
            "H3_macro_f1": normal_power(h3a["diff"], se, s, ALPHA / k3)}
    out["original_effects"] = {"part2": {"H1_pressure_mean": float(h1.mean()), "H3_push_mean": float(h3.mean()),
                                         "H4_leverage": lev},
                               "addendum": {h: {k: v for k, v in r.items() if k in ("diff", "a_only", "b_only", "n")}
                                            for h, r in addendum_results["tests"].items()}}
    return out


def fmt_pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"
