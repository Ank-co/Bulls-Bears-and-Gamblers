"""Part 4: scenarios of an invented market, with an answer key computed by code.

See docs/part4-design.md. A scenario is drawn in this order and no other:
1. the outcome (the share price is up or down in one week) from the base rate;
2. each evidence item: which version of its kind of news came out (the good or the bad one),
   given the outcome: the version that conventionally points to the outcome with probability r for a
   congruent kind, the opposite one with probability r for a reversed kind;
3. each distractor's version, at random, independently of the outcome.
The past track records of the history version are drawn from the same probabilities, independently of
this week's outcome. Only then are names, sentences and tone chosen: they carry no information beyond
the numbers (tests/test_fictional.py checks that nothing else in a scenario predicts the outcome once
the ideal answer is known).

Track records give the probability of the news given the outcome, for the version that came out:
"news of a rate cut came before 75% of the share's past weekly rises and 25% of its past weekly falls".
They never say what the news "suggests", so the answer key does not depend on how a reader interprets
the news. All probabilities are whole percentages: the generator, the text and the answer key use the
same integers.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bbg import config

RELIABILITIES = (55, 65, 75, 85)            # percent
BASE_RATES = (30, 50, 70)                   # percent
MAX_EVIDENCE, MAX_DISTRACTORS, P_REVERSED = 3, 2, 1 / 3
PAST_WEEKS = 20                             # history version: past weekly moves counted for the base rate
PAST_EACH = 20                              # history version: past rises and past falls counted per item
P_DRAMATIC = 0.5                            # any news item, whatever its role
FRAMINGS = ("calm", "volatile", "none")
SPLITS = {"dev": {"seed": 41, "scenarios": 300, "history": 100},
          "test": {"seed": 42, "scenarios": 4000, "history": 500}}     # size set after the dev run

PLACE_NAMES = config.DATA_DIR / "part4" / "real_place_names.txt"


# ---------------------------------------------------------------- the numbers


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def good_version_pct(item: dict) -> tuple[int, int]:
    """Percent of past rises, and of past falls, that came after the good version of this kind of news."""
    if item["role"] == "distractor":
        return 50, 50
    r = item["reliability"]
    return (100 - r, r) if item["reversed"] else (r, 100 - r)


def shown_version_pct(item: dict) -> tuple[int, int]:
    """The same for the version that came out this week: what the stated track record displays."""
    up, down = good_version_pct(item)
    return (up, down) if item["textbook"] == 1 else (100 - up, 100 - down)


def draw(rng: np.random.Generator) -> dict:
    """The numbers of one scenario, in generation order."""
    base_rate = int(rng.choice(BASE_RATES))
    outcome = int(rng.random() < base_rate / 100)                 # 1. the outcome
    sign = 1 if outcome else -1
    items = []
    for _ in range(int(rng.integers(1, MAX_EVIDENCE + 1))):       # 2. evidence, given the outcome
        r = int(rng.choice(RELIABILITIES))
        rev = bool(rng.random() < P_REVERSED)
        points_to = sign if rng.random() < r / 100 else -sign     # direction it points to in this world
        items.append({"role": "evidence", "reliability": r, "reversed": rev,
                      "textbook": -points_to if rev else points_to})
    for _ in range(int(rng.integers(0, MAX_DISTRACTORS + 1))):    # 3. distractors, independent
        items.append({"role": "distractor", "reliability": None, "reversed": False,
                      "textbook": int(rng.choice([1, -1]))})
    # past episodes (history version): counts drawn from the same probabilities, not from this week
    weeks_up = int(rng.binomial(PAST_WEEKS, base_rate / 100))
    for it in items:
        up, down = good_version_pct(it)
        it["good_before_rises"] = int(rng.binomial(PAST_EACH, up / 100))
        it["good_before_falls"] = int(rng.binomial(PAST_EACH, down / 100))
        it["dramatic"] = bool(rng.random() < P_DRAMATIC)          # tone, whatever the role
    items = [items[i] for i in rng.permutation(len(items))]
    return {"base_rate": base_rate, "outcome": outcome, "items": items, "weeks_up": weeks_up,
            "framing": str(rng.choice(FRAMINGS))}


def ideal_logodds(base_rate: float, items: list[tuple[float, float]]) -> float:
    """base_rate as a probability; items: (P(news | rise), P(news | fall)) for the version shown."""
    return logit(base_rate) + sum(math.log(up / down) for up, down in items)


def ideal_stated(sc: dict) -> float:
    return sigmoid(ideal_logodds(sc["base_rate"] / 100,
                                 [(u / 100, d / 100) for u, d in map(shown_version_pct, sc["items"])]))


# design-aware prior for the history version: the base rates, and for any news item a distractor
# (1/3), a congruent kind (4/9) or a reversed kind (2/9), as (P(good | rise), P(good | fall), weight)
_TYPES = ([(0.5, 0.5, 1 / 3)] + [(r / 100, 1 - r / 100, 2 / 3 * (1 - P_REVERSED) / 4) for r in RELIABILITIES]
          + [(1 - r / 100, r / 100, 2 / 3 * P_REVERSED / 4) for r in RELIABILITIES])


def shown_counts(item: dict) -> tuple[int, int]:
    """How often the version shown this week came before the past rises, and before the past falls."""
    if item["textbook"] == 1:
        return item["good_before_rises"], item["good_before_falls"]
    return PAST_EACH - item["good_before_rises"], PAST_EACH - item["good_before_falls"]


def history_counts(sc: dict) -> tuple[int, list[tuple[int, int]]]:
    """Past rises among the weeks counted, and the shown counts of every item."""
    return sc["weeks_up"], [shown_counts(it) for it in sc["items"]]


def history_references(sc: dict) -> dict:
    """Three reference answers for the history version (none of them is the answer key)."""
    up, counts = history_counts(sc)
    n, m = PAST_WEEKS, PAST_EACH
    clip = lambda p, k: min(max(p, 0.5 / k), 1 - 0.5 / k)
    freq = ideal_logodds(clip(up / n, n), [(clip(a / m, m), clip(b / m, m)) for a, b in counts])
    unif = ideal_logodds((up + 1) / (n + 2), [((a + 1) / (m + 2), (b + 1) / (m + 2)) for a, b in counts])

    base_like = np.array([(b / 100) ** up * (1 - b / 100) ** (n - up) for b in BASE_RATES])
    base_post = float((base_like * np.array(BASE_RATES) / 100).sum() / base_like.sum())
    design_items = []
    for it, (a, b) in zip(sc["items"], counts):
        # a and b count the version shown; express each type's probabilities for that version
        types = [(pu, pd, w) if it["textbook"] == 1 else (1 - pu, 1 - pd, w) for pu, pd, w in _TYPES]
        like = np.array([w * pu ** a * (1 - pu) ** (m - a) * pd ** b * (1 - pd) ** (m - b) for pu, pd, w in types])
        post = like / like.sum()
        design_items.append((float(sum(p * t[0] for p, t in zip(post, types))),
                             float(sum(p * t[1] for p, t in zip(post, types)))))
    design = ideal_logodds(base_post, design_items)
    return {"frequencies_taken_as_true": sigmoid(freq), "uniform_prior_bayes": sigmoid(unif),
            "design_aware_bayes": sigmoid(design)}


# ---------------------------------------------------------------- the words


@dataclass(frozen=True)
class Family:
    """A kind of news, with its good and bad versions (the conventional reading, fixed in advance).

    The labels name each version in the track records. 'debatable' gives, for the kinds whose
    conventional reading can be argued either way, the argument for the other reading. The criterion is
    mechanical: a kind is debatable when its effect on a share runs through the economy as a whole or the
    central bank's response (macro data, central bank, world raw-material prices), because then a
    standard argument exists for both signs. It never changes the answer key, only the reading of H2
    and H3.
    """
    key: str
    topic: str
    good: str
    bad: str
    good_label: str
    bad_label: str
    debatable: str = ""


FAMILIES = (
    Family("orders", "company",
           "{company} reports that its order book grew {pct}% over the last quarter.",
           "{company} reports that its order book shrank {pct}% over the last quarter.",
           "order-book growth", "an order-book decline"),
    Family("results", "company",
           "{company} reported quarterly profits {pct}% above analysts' forecasts.",
           "{company} reported quarterly profits {pct}% below analysts' forecasts.",
           "profits above forecasts", "profits below forecasts"),
    Family("contract", "company",
           "{company} won a multi-year supply contract with a state-owned utility.",
           "{company} lost its largest supply contract to a competitor.",
           "a contract win", "a contract loss"),
    Family("leadership", "company",
           "{company} named {person}, a widely respected industry veteran, as chief executive.",
           "The chief executive of {company}, {person}, resigned without explanation.",
           "a respected new chief executive", "a chief executive's resignation"),
    Family("analysts", "company",
           "An analyst at {bank} raised {company} to \"buy\".",
           "An analyst at {bank} cut {company} to \"sell\".",
           "an upgrade to buy", "a downgrade to sell"),
    Family("industry", "macro",
           "The national statistics office reports that industrial orders rose {pct}% in {month}.",
           "The national statistics office reports that industrial orders fell {pct}% in {month}.",
           "rising industrial orders", "falling industrial orders",
           "strong activity can raise fears of rate rises"),
    Family("jobs", "macro",
           "Unemployment in {country} fell to its lowest level in {years} years.",
           "Unemployment in {country} rose to its highest level in {years} years.",
           "falling unemployment", "rising unemployment",
           "low unemployment can raise fears of rate rises"),
    Family("prices", "macro",
           "Inflation in {country} came in below expectations in {month}.",
           "Inflation in {country} came in above expectations in {month}.",
           "inflation below expectations", "inflation above expectations",
           "low inflation can signal weak demand"),
    Family("retail", "macro",
           "Retail sales in {country} rose {pct}% in {month}.",
           "Retail sales in {country} fell {pct}% in {month}.",
           "rising retail sales", "falling retail sales",
           "strong spending can raise fears of rate rises"),
    Family("rates", "central_bank",
           "The central bank of {country} cut its policy rate by half a point.",
           "The central bank of {country} raised its policy rate by half a point.",
           "a rate cut", "a rate rise",
           "a rate cut can signal that the economy is weakening"),
    Family("guidance", "central_bank",
           "The central bank governor, {person}, signalled that rates should stay low for longer.",
           "The central bank governor, {person}, signalled that further rate rises are likely.",
           "a signal of low rates for longer", "a signal of further rate rises",
           "low rates for longer can signal weak growth"),
    Family("lending", "central_bank",
           "The central bank eased its lending rules for the banks of {country}.",
           "The central bank tightened its lending rules for the banks of {country}.",
           "eased lending rules", "tightened lending rules",
           "eased lending rules can signal that the banks are under strain"),
    Family("tax", "politics",
           "The government of {country} announced a cut in corporate taxes.",
           "The government of {country} announced a rise in corporate taxes.",
           "a corporate tax cut", "a corporate tax rise"),
    Family("budget", "politics",
           "Parliament passed the annual budget after a short debate.",
           "Parliament rejected the annual budget, raising the risk of a government shutdown.",
           "a budget passed", "a budget rejected"),
    Family("coalition", "politics",
           "The ruling coalition in {country} won a confidence vote by a wide margin.",
           "The ruling coalition in {country} collapsed, and early elections were called.",
           "a confidence vote won", "a coalition collapse"),
    Family("regulation", "politics",
           "The government announced a plan to cut red tape for businesses.",
           "The government announced new rules that raise compliance costs for listed companies.",
           "less red tape", "new compliance costs"),
    Family("trade", "international",
           "{country} signed a free-trade agreement with its largest trading partner.",
           "{country}'s largest trading partner imposed new tariffs on its goods.",
           "a free-trade agreement", "new tariffs"),
    Family("sanctions", "international",
           "Neighbouring countries lifted their sanctions on exports from {country}.",
           "Neighbouring countries imposed sanctions on exports from {country}.",
           "sanctions lifted", "sanctions imposed"),
    Family("commodities", "international",
           "World prices of the raw materials {country} imports fell {pct}% this month.",
           "World prices of the raw materials {country} imports rose {pct}% this month.",
           "cheaper raw materials", "dearer raw materials",
           "cheaper raw materials can signal weak world demand"),
    Family("investors", "international",
           "A large foreign pension fund announced new investments in {country}.",
           "A large foreign pension fund announced that it is withdrawing its investments from {country}.",
           "new foreign investment", "a foreign withdrawal"),
)
TOPICS = tuple(dict.fromkeys(f.topic for f in FAMILIES))
_BY_KEY = {f.key: f for f in FAMILIES}
_LABELS = {lab: (f.key, t) for f in FAMILIES for t, lab in ((1, f.good_label), (-1, f.bad_label))}

DRAMATIC = {1: ("Euphoria in {country}: {s} Commentators speak of a turning point.",
                "Jubilation in {country}: {s} Some call it the best news in years."),
            -1: ("Alarm in {country}: {s} Commentators fear the worst.",
                 "Shock in {country}: {s} Some call it a disaster.")}

FRAMING_TEXT = {"calm": "It has been a calm year on this exchange.",
                "volatile": "It has been a volatile year on this exchange, with large swings in both directions.",
                "none": ""}
HEADER = "Market: the {country} Stock Exchange. Asset: the shares of {company}, {sector_with_article} company."
BASE_STATED = "In this market, the share price of {company} rises over a week (five trading days) with probability {p}%."
BASE_HISTORY = "Over the past {n} weeks, the share price of {company} rose {up} times and fell {down} times."
NEWS_INTRO = "News this week, each followed by its track record in this market:"
TRACK_STATED = ("Track record: news of {label} came before {up}% of the share's past weekly rises and before "
                "{down}% of its past weekly falls.")
TRACK_HISTORY = ("Track record: in the weeks before the share's last {m} weekly rises, the news was {shown} "
                 "{a} times and {other} {m_a} times; before its last {m} weekly falls, {shown} {b} times and "
                 "{other} {m_b} times.")
QUESTION = ("What is the probability, from 0 to 100, that the share price of {company} is higher in one week "
            "(five trading days) than at today's close? Answer in JSON with one field, \"probability_up\", "
            "an integer from 0 to 100.")
SYSTEM = "You are an equity analyst."

SECTORS = ("logistics", "chemicals", "retail", "telecoms", "construction", "food processing",
           "machinery", "software", "shipping", "pharmaceuticals", "steel", "insurance")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")
_SYL = ("bra", "dor", "fen", "gal", "hel", "kar", "lun", "mor", "nes", "ost", "pra", "ril", "sev",
        "tal", "ul", "var", "wen", "zor", "bel", "cas", "dre", "ev", "lis", "quen", "tov", "vol")
_COUNTRY_END = ("ia", "a", "and", "esh", "ora")
_COMPANY_END = ("ex", "on", "ica", "is", "um", "ar")
_COMPANY_KIND = ("Group", "Holdings", "Industries", "Systems", "Works")
_FIRST_END = ("ra", "na", "lo", "rik", "mir")
_LAST_END = ("ek", "ov", "ard", "ens")


def real_place_names(path: Path = PLACE_NAMES) -> set[str]:
    return {line.strip().casefold() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def _word(rng, n_syl, endings) -> str:
    return ("".join(rng.choice(_SYL, n_syl)) + rng.choice(endings)).capitalize()


def invent_names(rng: np.random.Generator, banned: set[str]) -> dict:
    """Invented names, none equal to a real country or region, all different from each other."""
    while True:
        names = {"country": _word(rng, int(rng.integers(2, 4)), _COUNTRY_END),
                 "company_base": _word(rng, 2, _COMPANY_END),
                 "first": _word(rng, 1, _FIRST_END), "last": _word(rng, 2, _LAST_END),
                 "bank_base": _word(rng, 2, _COMPANY_END)}
        words = list(names.values())
        if len(set(words)) == len(words) and not any(w.casefold() in banned for w in words):
            break
    return {"country": names["country"],
            "company": f"{names['company_base']} {rng.choice(_COMPANY_KIND)}",
            "person": f"{names['first']} {names['last']}",
            "bank": f"{names['bank_base']} Securities",
            "sector": str(rng.choice(SECTORS)),
            "month": str(rng.choice(MONTHS))}


def label(family: str, version: int) -> str:
    f = _BY_KEY[family]
    return f.good_label if version == 1 else f.bad_label


def write(sc: dict, rng: np.random.Generator, banned: set[str]) -> dict:
    """Chooses names, kinds of news and sentences, then renders both versions."""
    names = invent_names(rng, banned)
    slots = {**names, "pct": int(rng.integers(2, 10)), "years": int(rng.integers(3, 12))}
    unused = list(FAMILIES)
    for it in sc["items"]:
        topic = str(rng.choice(sorted({f.topic for f in unused})))
        candidates = [f for f in unused if f.topic == topic]
        fam = candidates[int(rng.integers(0, len(candidates)))]
        unused.remove(fam)
        sentence = (fam.good if it["textbook"] == 1 else fam.bad).format(**slots)
        if it["dramatic"]:
            sentence = DRAMATIC[it["textbook"]][int(rng.integers(0, 2))].format(country=names["country"], s=sentence)
        it.update({"family": fam.key, "topic": fam.topic, "sentence": sentence})
    sc["names"] = names
    sc["system"] = SYSTEM
    sc["stated"] = render(sc, "stated")
    sc["history"] = render(sc, "history")
    return sc


def _track(it: dict, version: str) -> str:
    shown, other = label(it["family"], it["textbook"]), label(it["family"], -it["textbook"])
    if version == "history":
        a, b = shown_counts(it)
        return TRACK_HISTORY.format(m=PAST_EACH, shown=shown, other=other, a=a, m_a=PAST_EACH - a,
                                    b=b, m_b=PAST_EACH - b)
    up, down = shown_version_pct(it)
    return TRACK_STATED.format(label=shown, up=up, down=down)


def render(sc: dict, version: str) -> str:
    n = sc["names"]
    article = "an" if n["sector"][0] in "aeiou" else "a"
    parts = [HEADER.format(country=n["country"], company=n["company"], sector_with_article=f"{article} {n['sector']}"),
             FRAMING_TEXT[sc["framing"]]]
    if version == "stated":
        parts.append(BASE_STATED.format(company=n["company"], p=sc["base_rate"]))
    else:
        parts.append(BASE_HISTORY.format(n=PAST_WEEKS, company=n["company"], up=sc["weeks_up"],
                                         down=PAST_WEEKS - sc["weeks_up"]))
    head = " ".join(p for p in parts if p)
    news = [NEWS_INTRO] + [f"{i}. {it['sentence']} {_track(it, version)}" for i, it in enumerate(sc["items"], 1)]
    blocks = [head, "\n".join(news), QUESTION.format(company=n["company"])]
    return "\n\n".join(blocks)


def scenario(split: str, index: int, banned: set[str], seed: int | None = None) -> dict:
    """One scenario of a split. 'seed' replaces the split's seed (for tests only)."""
    rng = np.random.default_rng([SPLITS[split]["seed"] if seed is None else seed, index])
    sc = {"id": f"{split}-{index:05d}", "split": split, "index": index,
          "in_history_subset": index < SPLITS[split]["history"], **draw(rng)}
    sc = write(sc, rng, banned)
    sc["ideal_stated"] = ideal_stated(sc)
    sc["history_references"] = history_references(sc)
    return sc


def scenarios(split: str, n: int | None = None) -> list[dict]:
    banned = real_place_names()
    return [scenario(split, i, banned) for i in range(SPLITS[split]["scenarios"] if n is None else n)]


# ---------------------------------------------------------------- reading the text back


def _pattern(template: str) -> str:
    out, pos = "", 0
    for m in re.finditer(r"\{(\w+)\}", template):
        out += re.escape(template[pos:m.start()]) + ".+?"
        pos = m.end()
    return out + re.escape(template[pos:])


_SENTENCES = [(f.key, t, re.compile(_pattern(f.good if t == 1 else f.bad))) for f in FAMILIES for t in (1, -1)]
_WRAPS = [(t, re.compile(re.escape(w).replace(r"\{country\}", r"[^:]+?").replace(r"\{s\}", r"(.+)")))
          for t, ws in DRAMATIC.items() for w in ws]
_BASE = re.compile(r"rises over a week \(five trading days\) with probability (\d+)%\.")
_LINE = re.compile(r"^(\d+)\. (.*?) (Track record: .*)$", re.M)
_T_STATED = re.compile(r"Track record: news of (.+?) came before (\d+)% of the share's past weekly rises and "
                       r"before (\d+)% of its past weekly falls\.$")
_T_HISTORY = re.compile(r"Track record: in the weeks before the share's last (\d+) weekly rises, the news was (.+?) "
                        r"(\d+) times and (.+?) (\d+) times; before its last (\d+) weekly falls, (.+?) (\d+) times "
                        r"and (.+?) (\d+) times\.$")
_WEEKS = re.compile(r"Over the past (\d+) weeks, the share price of .+? rose (\d+) times and fell (\d+) times\.")


def read_sentence(sentence: str) -> tuple[str, int]:
    """The (kind of news, conventional direction) of a rendered sentence. Exactly one must match."""
    inner, wrap_dir = sentence, None
    for t, rx in _WRAPS:
        m = rx.fullmatch(sentence)
        if m:
            inner, wrap_dir = m.group(1), t
            break
    hits = [(key, t) for key, t, rx in _SENTENCES if rx.fullmatch(inner)]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} templates match: {sentence!r}")
    if wrap_dir is not None and wrap_dir != hits[0][1]:
        raise ValueError(f"dramatic wording contradicts the news: {sentence!r}")
    return hits[0]


def parse_stated(text: str) -> dict:
    """Reads the stated text alone: base rate, and per item (kind, version, % of rises, % of falls)."""
    base = _BASE.search(text)
    if not base:
        raise ValueError("base rate not found")
    items = []
    for _, sentence, track in _LINE.findall(text):
        family, t = read_sentence(sentence)
        m = _T_STATED.fullmatch(track)
        if not m:
            raise ValueError(f"track record not understood: {track!r}")
        if _LABELS.get(m.group(1)) != (family, t):
            raise ValueError(f"track record describes another news: {track!r}")
        items.append((family, t, int(m.group(2)), int(m.group(3))))
    if not items:
        raise ValueError("no news found")
    base_rate = int(base.group(1))
    ideal = sigmoid(ideal_logodds(base_rate / 100, [(u / 100, d / 100) for _, _, u, d in items]))
    return {"base_rate": base_rate, "items": items, "ideal": ideal}


def parse_history(text: str) -> dict:
    """Reads the history text alone: past rises among the weeks counted, and per item (kind, version,
    times the version shown came before the past rises, and before the past falls)."""
    w = _WEEKS.search(text)
    if not w or int(w.group(2)) + int(w.group(3)) != int(w.group(1)):
        raise ValueError("past weeks not understood")
    items = []
    for _, sentence, track in _LINE.findall(text):
        family, t = read_sentence(sentence)
        m = _T_HISTORY.fullmatch(track)
        if not m:
            raise ValueError(f"track record not understood: {track!r}")
        n1, s1, a, o1, a2, n2, s2, b, o2, b2 = m.groups()
        shown, other = label(family, t), label(family, -t)
        if (s1, s2, o1, o2) != (shown, shown, other, other):
            raise ValueError(f"track record describes another news: {track!r}")
        if int(a) + int(a2) != int(n1) or int(b) + int(b2) != int(n2):
            raise ValueError(f"counts do not add up: {track!r}")
        items.append((family, t, int(a), int(b)))
    return {"weeks_up": int(w.group(2)), "items": items}
