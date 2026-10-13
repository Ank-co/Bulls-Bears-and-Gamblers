"""Part 4 scenario generator: the numbers follow the design, the text carries exactly them, and nothing
else in a scenario says anything about the outcome."""
import re

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from bbg import fictional as F


@pytest.fixture(scope="module")
def draws():
    rng = np.random.default_rng(11)
    return [F.draw(rng) for _ in range(40_000)]


@pytest.fixture(scope="module")
def dev():
    return F.scenarios("dev", 400)


@pytest.fixture(scope="module")
def many():
    """Full scenarios (text included) on a seed of their own, for the leak test."""
    banned = F.real_place_names()
    return [F.scenario("dev", i, banned, seed=7) for i in range(30_000)]


def close(freq, expected, n, k=4):
    """Within k standard errors of sampling."""
    return abs(freq - expected) < k * np.sqrt(expected * (1 - expected) / n)


# ---------------------------------------------------------------- the numbers


def test_generation_order_on_the_generator(draws):
    rows = [(d["outcome"], it) for d in draws for it in d["items"]]
    for r in F.RELIABILITIES:
        ev = [(y, it) for y, it in rows if it["role"] == "evidence" and it["reliability"] == r]
        points_right = [it["textbook"] * (-1 if it["reversed"] else 1) == (1 if y else -1) for y, it in ev]
        assert close(np.mean(points_right), r / 100, len(ev))
    dis = [it["textbook"] == (1 if y else -1) for y, it in rows if it["role"] == "distractor"]
    assert close(np.mean(dis), 0.5, len(dis))                            # distractors ignore the outcome


def test_displayed_percentages_are_the_frequencies_given_the_outcome(draws):
    """The test that the first wording would have failed: among weeks that rose (or fell), the version
    shown came out as often as the track record says, at every base rate."""
    for base in F.BASE_RATES:
        for y in (1, 0):
            groups = {}
            for d in draws:
                if d["base_rate"] != base or d["outcome"] != y:
                    continue
                for it in d["items"]:
                    groups.setdefault((it["role"], it["reliability"], it["reversed"]), []).append(it)
            for key, its in groups.items():
                good_freq = np.mean([it["textbook"] == 1 for it in its])
                up, down = F.good_version_pct(its[0])
                assert close(good_freq, (up if y else down) / 100, len(its)), (base, y, key)


def test_news_items_are_independent_given_the_outcome(draws):
    """The ideal answer multiplies one ratio per news item: that requires independence given the outcome."""
    for y in (1, 0):
        pairs = [(d["items"][0]["textbook"], d["items"][1]["textbook"]) for d in draws
                 if d["outcome"] == y and len(d["items"]) >= 2]
        a = np.array(pairs, dtype=float)
        assert abs(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) < 4 / np.sqrt(len(a))


def test_tone_is_independent_of_the_role(draws):
    for role in ("evidence", "distractor"):
        tone = [it["dramatic"] for d in draws for it in d["items"] if it["role"] == role]
        assert close(np.mean(tone), F.P_DRAMATIC, len(tone))


def test_history_counts_follow_the_rates_and_ignore_this_week(draws):
    for y in (1, 0):
        for base in F.BASE_RATES:
            weeks = [d["weeks_up"] for d in draws if d["outcome"] == y and d["base_rate"] == base]
            assert close(np.mean(weeks) / F.PAST_WEEKS, base / 100, len(weeks) * F.PAST_WEEKS)
        groups = {}
        for d in draws:
            if d["outcome"] == y:
                for it in d["items"]:
                    groups.setdefault(F.good_version_pct(it), []).append(it)
        for (up, down), its in groups.items():
            n = len(its) * F.PAST_EACH
            assert close(np.mean([it["good_before_rises"] for it in its]) / F.PAST_EACH, up / 100, n)
            assert close(np.mean([it["good_before_falls"] for it in its]) / F.PAST_EACH, down / 100, n)


def test_ideal_answer_is_calibrated(draws):
    p = np.array([F.ideal_stated(d) for d in draws])
    y = np.array([d["outcome"] for d in draws])
    for b in range(10):
        m = (p >= b / 10) & (p < (b + 1) / 10)
        if m.sum() > 1500:
            assert close(y[m].mean(), p[m].mean(), m.sum())


def _features(sc: dict) -> list[float]:
    """Everything a reader of the stated text can see, apart from what the ideal answer already uses."""
    its, n = sc["items"], sc["names"]
    f = [sc["framing"] == "calm", sc["framing"] == "volatile", len(its),
         sum(it["role"] == "distractor" for it in its), sum(it["dramatic"] for it in its),
         sum(it["dramatic"] * it["textbook"] for it in its), sum(it["textbook"] for it in its),
         sum(it["textbook"] for it in its if it["role"] == "distractor"),
         its[0]["role"] == "distractor", its[0]["textbook"], its[-1]["textbook"],
         int(re.search(r"(\d+)%", " ".join(it["sentence"] for it in its) + " 0%").group(1)),
         len(n["country"]), len(n["company"]), n["company"].endswith("Group")]
    f += [sum(it["topic"] == t for it in its) for t in F.TOPICS[1:]]
    f += [n["sector"] == s for s in F.SECTORS[1:]]
    f += [n["month"] == m for m in F.MONTHS[1:]]
    return [float(x) for x in f]


def test_nothing_else_in_the_text_predicts_the_outcome(many):
    """Once the ideal answer is known, framing, tone, topics, order of the news, names, sector, month and
    numbers in the sentences must add nothing: a likelihood-ratio test of all of them together."""
    y = np.array([sc["outcome"] for sc in many])
    offset = np.array([F.logit(sc["ideal_stated"]) for sc in many])
    q = 1 / (1 + np.exp(-offset))                                        # the ideal answer alone
    null_llf = float(np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)))

    def lr_test(X):
        full = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit()
        return stats.chi2.sf(2 * (full.llf - null_llf), X.shape[1]), np.max(np.abs(full.tvalues))

    X = sm.add_constant(np.array([_features(sc) for sc in many]))
    p, max_t = lr_test(X)
    assert p > 0.001 and max_t < 4.5, (p, max_t)
    # the test can see a leak: a feature equal to the outcome 55% of the time is caught
    planted = np.where(np.random.default_rng(0).random(len(y)) < 0.55, y, 1 - y)
    assert lr_test(np.c_[X, planted])[0] < 1e-6


# ---------------------------------------------------------------- the words


def test_text_carries_exactly_the_generator_numbers(dev):
    for sc in dev:
        p = F.parse_stated(sc["stated"])
        assert p["base_rate"] == sc["base_rate"]
        assert p["items"] == [(it["family"], it["textbook"], *F.shown_version_pct(it)) for it in sc["items"]]
        assert p["ideal"] == pytest.approx(sc["ideal_stated"], abs=1e-12)
        h = F.parse_history(sc["history"])
        assert h["weeks_up"] == sc["weeks_up"]
        assert h["items"] == [(it["family"], it["textbook"], *F.shown_counts(it)) for it in sc["items"]]


def test_one_convention_for_percentages(dev):
    for sc in dev:
        assert sc["base_rate"] in F.BASE_RATES
        for it in sc["items"]:
            up, down = F.shown_version_pct(it)
            assert isinstance(up, int) and isinstance(down, int) and up + down == 100
        assert all(s == str(int(s)) for s in re.findall(r"(\d+)%", sc["stated"]))   # whole percentages
    # with 20 past rises and falls, every stated percentage is a reachable count
    assert all((r * F.PAST_EACH) % 100 == 0 for r in F.RELIABILITIES + (50,))


def test_every_sentence_and_label_matches_exactly_one_news():
    rng = np.random.default_rng(3)
    names = F.invent_names(rng, set())
    slots = {**names, "pct": 7, "years": 9}
    for fam in F.FAMILIES:
        for t in (1, -1):
            s = (fam.good if t == 1 else fam.bad).format(**slots)
            assert F.read_sentence(s) == (fam.key, t)
            for wrap in F.DRAMATIC[t]:
                assert F.read_sentence(wrap.format(country=names["country"], s=s)) == (fam.key, t)
    wrong = F.DRAMATIC[-1][0].format(country=names["country"], s=F.FAMILIES[0].good.format(**slots))
    with pytest.raises(ValueError):
        F.read_sentence(wrong)                                           # alarmist wording on good news
    labels = [lab for f in F.FAMILIES for lab in (f.good_label, f.bad_label)]
    assert len(labels) == len(set(labels))


def test_a_track_record_about_another_news_is_rejected(dev):
    sc = dev[0]
    it = sc["items"][0]
    shown, other = F.label(it["family"], it["textbook"]), F.label(it["family"], -it["textbook"])
    with pytest.raises(ValueError):
        F.parse_stated(sc["stated"].replace(f"news of {shown} came", f"news of {other} came", 1))
    with pytest.raises(ValueError):
        F.parse_history(sc["history"].replace(f"the news was {shown} ", f"the news was {other} ", 1))


def test_kinds_of_news_are_distinct_within_a_scenario(dev):
    for sc in dev:
        fams = [it["family"] for it in sc["items"]]
        assert len(fams) == len(set(fams))


FORBIDDEN = re.compile(r"\bwill\b|\bsoar|\bplunge|\brally|\btumble|\bcrash|expected to (rise|fall)|"
                       r"\bnext week\b|\btomorrow\b|\b(19|20)\d\d\b|suggest", re.I)


def test_no_word_gives_the_outcome_a_real_date_or_an_interpretation(dev):
    for sc in dev:
        for text in (sc["stated"], sc["history"]):
            assert not FORBIDDEN.search(text), FORBIDDEN.search(text).group(0)


def test_names_are_invented():
    banned = F.real_place_names()
    assert len(banned) > 5000 and "france" in banned and "bayern" in banned
    for sc in F.scenarios("dev", 200):
        n = sc["names"]
        words = [n["country"], n["company"].split()[0], *n["person"].split(), n["bank"].split()[0]]
        assert not any(w.casefold() in banned for w in words)
    first = F.invent_names(np.random.default_rng(0), set())
    again = F.invent_names(np.random.default_rng(0), {first["country"].casefold()})
    assert again["country"] != first["country"]                          # a real name forces a new draw


def test_history_references_on_the_same_scenarios(draws):
    """Same scenarios, same metric: the design-aware reference is calibrated (calibration slope near 1),
    the frequencies taken as true are overconfident (slope below 1)."""
    sub = draws[:20_000]
    refs = [F.history_references(d) for d in sub]
    y = np.array([d["outcome"] for d in sub])
    slopes = {}
    for name in ("design_aware_bayes", "frequencies_taken_as_true"):
        x = np.array([F.logit(min(max(r[name], 1e-6), 1 - 1e-6)) for r in refs])
        fit = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()
        slopes[name] = (fit.params[1], fit.bse[1])
    b, se = slopes["design_aware_bayes"]
    assert abs(b - 1) < 4 * se
    b, se = slopes["frequencies_taken_as_true"]
    assert b < 1 - 4 * se


def test_generation_is_deterministic():
    a, b = F.scenarios("dev", 5), F.scenarios("dev", 5)
    assert [s["stated"] for s in a] == [s["stated"] for s in b]
    assert [s["history"] for s in a] == [s["history"] for s in b]
    assert F.scenarios("test", 1)[0]["stated"] != a[0]["stated"]


def test_history_subset_and_sizes():
    assert F.SPLITS["dev"]["history"] <= F.SPLITS["dev"]["scenarios"]
    sc = F.scenarios("dev", 120)
    assert [s["in_history_subset"] for s in sc] == [i < F.SPLITS["dev"]["history"] for i in range(120)]


def test_debatable_kinds_follow_the_written_criterion():
    """Debatable = the effect runs through the economy or the central bank: macro data, central bank,
    world raw-material prices. Fixed before any model is run."""
    expected = {f.key for f in F.FAMILIES if f.topic in ("macro", "central_bank")} | {"commodities"}
    assert {f.key for f in F.FAMILIES if f.debatable} == expected
    assert len(F.FAMILIES) == 20 and len(expected) == 8
