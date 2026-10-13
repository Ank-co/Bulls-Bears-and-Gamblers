"""Part 4 measures: they find back planted behaviours, and recode correctly with a model's own reading."""
import math

import numpy as np
import pytest
from scipy import stats

from bbg import fictional as F
from bbg import part4_stats as S


@pytest.fixture(scope="module")
def scs():
    banned = F.real_place_names()
    return [F.scenario("dev", i, banned, seed=8) for i in range(4000)]


def answers_from(scs, f, seed=0):
    """Integer answers from a log-odds function of the regressors, with noise; as a model would give."""
    rng = np.random.default_rng(seed)
    out = []
    for sc in scs:
        x = f(S.regressors(sc)) + rng.normal(0, 0.3)
        out.append({"id": sc["id"], "value": int(round(100 / (1 + math.exp(-x)))), "error": None,
                    "seconds": 1.0, "prompt_tokens": 250})
    return out


def test_belief_updating_weights_are_found_back(scs):
    planted = {"const": 0.2, "base_rate": 0.6, "e_congruent": 0.8, "e_reversed": 0.4, "distractor": 0.3,
               "dramatic_tone": 0.2}
    ans = answers_from(scs, lambda r: planted["const"] + sum(planted[k] * r[k] for k in S.TERMS))
    w = S.grether(S.stated_frame(scs, ans))["weights"]
    for k, v in planted.items():
        assert abs(w[k]["estimate"] - v) < 4 * w[k]["se"] + 0.03, (k, w[k])


def test_shift_amplification_and_noise_are_found_back(scs):
    ans = answers_from(scs, lambda r: -0.25 + 0.7 * r["ideal"])
    s = S.shift_amplification_noise(S.stated_frame(scs, ans))
    assert abs(s["shift"]["estimate"] + 0.25) < 0.05 and abs(s["amplification"]["estimate"] - 0.7) < 0.03
    assert 0.25 < s["noise_sd"] < 0.4


def test_paired_comparison_and_power(scs):
    a = S.stated_frame(scs, answers_from(scs, lambda r: 0.8 * r["ideal"], seed=1))
    b = S.stated_frame(scs, answers_from(scs, lambda r: 0.2 + 0.8 * r["ideal"], seed=2))
    p = S.paired(a, b)
    assert abs(p["mean_difference"] - 0.2) < 0.05
    q = p["power"][1]
    assert q["expected_t"] == pytest.approx(0.1 * math.sqrt(q["scenarios"]) / p["sd_difference_around_trend"])
    d = S.stacked_shift_slope(a, b)
    assert abs(d["shift_difference"]["estimate"] - 0.2) < 0.05 and abs(d["slope_difference"]["estimate"]) < 0.03


def test_own_reading_moves_items_between_congruent_and_reversed():
    sc = F.scenarios("dev", 50)
    target = next(s for s in sc if any(it["role"] == "evidence" for it in s["items"]))
    it = next(i for i in target["items"] if i["role"] == "evidence")
    full = {(f.key, t): t for f in F.FAMILIES for t in (1, -1)}             # a model reading as we do
    conventional = S.regressors(target)
    assert S.regressors(target, full) == conventional
    flipped = S.regressors(target, {**full, (it["family"], it["textbook"]): -it["textbook"]})
    unread = S.regressors(target, {**full, (it["family"], it["textbook"]): 0})
    assert flipped["ideal"] == pytest.approx(conventional["ideal"])          # the answer key never moves
    moved = math.log(F.shown_version_pct(it)[0] / F.shown_version_pct(it)[1])
    if it["reversed"]:
        assert flipped["e_congruent"] == pytest.approx(conventional["e_congruent"] + moved)
    else:
        assert flipped["e_reversed"] == pytest.approx(conventional["e_reversed"] + moved)
    assert unread["e_unread"] == pytest.approx(moved)


def test_failures_are_counted_and_left_out_of_the_weights(scs):
    ans = answers_from(scs[:500], lambda r: r["ideal"])
    for a in ans[:40]:
        a.update(value=None, error="not JSON")
    df = S.stated_frame(scs[:500], ans)
    assert S.failures(ans)["failures"] == 40 and S.failures(ans)["by_reason"] == {"not JSON": 40}
    assert S.grether(df)["n"] == 460


def test_reading_agreement_and_own_reading():
    from bbg.part4_llm import reading_items
    items = reading_items()
    ans = []
    for it in items:
        v = "good" if it["version"] == 1 else "bad"
        if it["family"] == "rates":
            v = "bad" if it["version"] == 1 else "good"                     # reads rate cuts as bad news
        if it["family"] == "prices" and it["variant"] == 0:
            v = "neither"
        ans.append({"id": it["id"], "value": v, "error": None})
    r = S.reading_agreement(items, ans)
    assert r["per_kind"]["rates"] == {"opposite": 6} and r["per_kind"]["prices"] == {"neither": 2, "agrees": 4}
    own = S.own_reading_map(r)
    assert own[("rates", 1)] == -1 and own[("prices", 1)] == 1 and own[("orders", -1)] == -1


def test_planning_scales_the_dev_noise_with_the_size(scs):
    a = S.stated_frame(scs[:300], answers_from(scs[:300], lambda r: 0.8 * r["ideal"], seed=3))
    b = S.stated_frame(scs[:300], answers_from(scs[:300], lambda r: 0.1 + 0.8 * r["ideal"], seed=4))
    pl = S.planning(a, b, sizes=(300, 1200))
    for h, rows in pl["power"].items():
        assert 0 <= rows[0]["power_1pct"] <= rows[1]["power_1pct"] <= 1          # more scenarios, more power
        assert rows[0]["power_1pct"] <= rows[0]["power_5pct"]
    se, e = pl["dev_se"]["H1"], pl["smallest_effects"]["H1"]
    assert pl["power"]["H1"][0]["power_5pct"] == pytest.approx(stats.norm.cdf(e / se - stats.norm.ppf(0.95)))
