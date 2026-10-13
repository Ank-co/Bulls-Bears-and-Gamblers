# Part 4 design: reading the news in a fictional market

Status: **draft for discussion, second version** (after a first review). Nothing is frozen and no
language model has been run. As in Part 3, the pre-registration will be written after a dev run and
committed before the test scenarios are run. Every number below comes from
`scripts/part4_design_check.py` (a simulation, no language model) and is written to
`results/part4/design_check.json`.

## Question

Part 3 found that the models' reading of real headlines lines up with prices, but could not separate
reading from remembering: the models were trained on text covering the test years. Part 4 removes memory
by construction. Every scenario takes place in an invented market, and the right answer is computed by
code from rules the scenario states. The question becomes:

> Do the models use financial news in proportion to what it is worth?

Five things are measured: whether the models take the base rate into account, whether they weigh evidence
by its strength, whether news worth nothing still moves them, whether evidence given in the scenario beats
their textbook priors when the two disagree, and whether abliteration changes any of this.

## Why the answer key is computed, not written

In an invented market, the outcome is whatever the author decides. If the author writes that a rate rise
sends the market down, a model that answers "down" only shows that it shares the author's textbook: the
test would measure agreement with the author, not the use of information. If a language model wrote the
scenarios and their outcomes, a language model would set the answer key.

So the market follows an explicit random process, and the best possible answer given what the scenario
says is a probability computed by code (the ideal observer). The models are compared with that probability
and with the drawn outcomes.

## The world

One scenario is one asset (a company's share, or the main index of the country) on an invented exchange,
in an invented country, with invented companies, people and dates. The model is asked for the probability,
from 0 to 100, that the asset closes higher in five trading days than today.

| Element | Values | Meaning |
|---|---|---|
| Base rate | 30%, 50%, 70% | probability that the asset rises over five trading days in this market |
| Evidence | 1 to 3 news items, each of its own kind | each points to the true outcome with probability r |
| Reliability r | 55%, 65%, 75%, 85% | drawn per kind of news, so that no real-world idea of which source is reliable can help |
| Reversed kinds | one kind in three | the track record shows that the market moved against the news's textbook meaning |
| Distractors | 0 to 2 news items | track record shows no relation with the asset (followed by a rise half of the time) |
| Tone of distractors | neutral or dramatic | same content, alarmist or euphoric wording on half of them |
| Topic | company, macro data, central bank, politics, international | drawn independently of the item's value: a political item can be evidence and an earnings item a distractor |
| Framing | calm market, volatile market, none | one opening sentence with no effect on the outcome |

Each news item is written with a textbook meaning ("orders rose 6%" reads as good news). For a congruent
kind of news, the item's textbook meaning is the direction it points to. For a reversed kind, the
scenario's track record says that the market moved the other way in the past, and the ideal observer
follows the track record.

**Generation order.** The generator works in this order and no other, so that the ideal observer's
formula is the true conditional probability:

1. draw the outcome (up or down) from the base rate;
2. for each evidence item, draw whether it points to the true outcome (probability r), then write it with
   the textbook meaning that matches that direction (its opposite for a reversed kind);
3. draw each distractor's textbook direction at random, independently of the outcome.

A test checks this on the generator itself, not only on its text: over many generated scenarios, items
point to the outcome at rate r, reversed items read wrong at rate r, and distractors agree with the
outcome half of the time (`tests/test_part4_design.py` does it for the simulation; the same test will run
on the real generator).

## Two versions of each scenario, which test two different skills

A first draft said the two versions carried the same information. They do not.

- **Stated (confirmatory).** The base rate and each reliability are given as probabilities of this market
  ("such reports point the right way with probability 75%"). The ideal answer is a calculation with known
  numbers, with no ambiguity.
- **History (exploratory).** The model sees raw past episodes instead: the asset's moves over the past
  20 weeks, and for each kind of news its last 20 occurrences with what the asset did next. These are
  real samples, drawn from the true rates, so 15 out of 20 does not mean the rate is 75%. The model has
  to count and then infer, and the right answer depends on what it assumes about rates before seeing the
  counts.

For the history version, three reference answers are computed:

| Reference | What it assumes | Calibration gap (max over deciles) | Brier |
|---|---|---|---|
| True rates (not available to the model) | the hidden parameters | 0.6 points | 0.150 |
| Frequencies taken as true (a calculation) | 15 out of 20 is 75% | 7.7 points | 0.174 |
| Bayesian with uniform priors (an inference) | 15 out of 20 gives 16/22 = 73% | 5.1 points | 0.172 |
| Bayesian who knows the design | the list of possible rates and their shares | 1.1 points | 0.168 |

On average, the three observable references differ by 2 to 6 points. Only the design-aware one is close
to calibrated (its remaining gap comes from an approximation that ignores how many items a scenario
holds), and it uses knowledge the model does not have. So the history version does not get a single
answer key: it is analysed descriptively, by asking which reference the model's answers resemble, and how
much accuracy it loses compared with the stated version of the same world.

## Example (stated version)

> Market: the [Country] stock exchange. Asset: shares of [Company], a logistics firm. In this market, a
> share like this one rises over five trading days with probability 70%.
>
> News this week, with the track record of each kind of news in this market:
>
> 1. The national statistics office reports that industrial orders rose 6% in [month]. Track record: such
>    reports point the right way with probability 75%.
> 2. [Country]'s opposition leader called for early elections. Track record: news of this kind has no
>    relation with the share (followed by a rise with probability 50%).
> 3. An analyst at [Bank] cut the share to "sell". Track record: after such notes, the share moves in the
>    opposite direction to what the note suggests, with probability 65%.
>
> What is the probability, from 0 to 100, that [Company]'s share closes higher in five trading days than
> today? Answer in JSON: {"probability_up": integer}.

The ideal answer is about 93: the base rate (70%), item 1 (reliability 75%, up) and item 3 (reversed,
65%, so up as well) all point up, and item 2 carries nothing. A reader that only follows the tone (one
good item, one bad, one political) would stay near the middle.

## The ideal observer (stated version)

The news items are independent given the outcome, so the best answer has a closed form:

    log-odds(up) = log-odds(base rate) + sum over evidence items of ( +/- log-odds(r) )

with the sign given by the direction the item points to in this world, and distractors adding nothing.
It is computed by code, and two tests guard it:

- **text to numbers:** the base rate, reliabilities and directions read back from each rendered prompt
  give the same ideal answer as the generator, so the text carries exactly the information the benchmark
  uses;
- **numbers to outcomes:** the generator's outcomes follow the distribution the formula assumes (see
  generation order).

In the simulation (400,000 scenarios), the ideal observer is calibrated within 0.5 points on every decile
of its answers. Its Brier score is 0.150, against 0.224 for the base rate alone and 0.250 for a coin. With
these reliabilities, 13% of ideal answers fall outside 5 to 95% and 1.4% outside 1 to 99%, so answers are
rarely forced against the ends of the scale.

## Models and baselines

- **Qwen3.6-35B-A3B base**, quantization i1-Q3_K_M, the base model of the Part 1 addendum.
- **The same model abliterated**, same quantization.
- **Qwen3-1.7B zero-shot**, as a floor, descriptive only. The fine-tuned 1.7B model labels one tweet and
  was not trained to forecast, so it does not enter as a contestant.
- **Tone reader:** FinBERT labels each news sentence, without its track record, and the sum of tones is
  turned into a probability by a logistic regression fitted on the dev scenarios. It reads tone and
  cannot read track records, which shows what a model gains by reasoning beyond tone.
- **Base rate only** and the **ideal observer**.

Settings as in Parts 1 and 2: thinking disabled, greedy decoding, one JSON answer per prompt. Answers are
clipped to 1 to 99 before taking log-odds.

## Measures

**Belief-updating regression** (stated version). This is the standard tool of experimental economics for
Bayesian updating (Grether, 1980, *Quarterly Journal of Economics*; reviewed by
[Benjamin, 2019](https://www.nber.org/papers/w25200)). For each model:

    log-odds(answer) = c + w_base * log-odds(base rate)
                         + w_cong * E_congruent + w_rev * E_reversed
                         + w_dist * D + w_tone * T + error

where E_congruent and E_reversed are the ideal log-odds contributions of congruent and reversed evidence,
D is the sum of the textbook directions of the distractors (+1 good news, -1 bad news) and T the same sum
over dramatic distractors only. The ideal observer has c = 0, w_base = w_cong = w_rev = 1 and
w_dist = w_tone = 0. A weight below 1 describes the fitted model, not a mechanism: it can come from a
cognitive shortcut, a calculation error or a misreading of the scenario, and is reported as such.

**Shift, amplification and noise.** Extreme or optimistic answers can have three different sources, and
each model gets all three, from the regression of its answer's log-odds on the ideal log-odds:

- **shift:** the intercept, answers higher (or lower) than the ideal everywhere;
- **amplification:** the slope, answers pushed away from 50% in both directions (above 1) or pulled
  toward it (below 1);
- **noise:** the residual standard deviation, answers scattered around their own trend.

A model can be extreme because it amplifies or because it is noisy; only the slope is called
amplification. Calibration curves by decile are shown for every model.

**Accuracy.** Brier score and log loss against the drawn outcomes, compared with the ideal observer, the
base rate alone and the tone reader.

## Draft hypotheses

To be frozen in the pre-registration after the dev run, with these directions, all one-sided, alpha 0.05,
Holm correction over the five tests. All on the stated version.

- **Manipulation check (not a hypothesis):** the base model's w_cong is positive. If it is not, the model
  does not read the evidence at all, and the rest is reported as uninterpretable.
- **H1, base-rate underweighting:** w_base < 1 for the base model.
- **H2, distraction:** w_dist > 0 for the base model: news worth nothing moves the answer in the direction
  of its textbook meaning.
- **H3, priors over evidence:** w_rev < w_cong for the base model: a track record that contradicts the
  textbook is followed less than one that agrees with it.
- **H4, abliteration and shift:** the abliterated model's intercept is higher than the base model's, in
  the regression on the ideal log-odds. The Part 1 addendum found it reads neutral news as bullish more
  often, and Fafuła (2026) found abliterated models more optimistic in their own stock calls.
- **H5, abliteration and amplification:** the abliterated model's slope on the ideal log-odds is higher
  than the base model's.

Standard errors: H1 to H3 use one answer per scenario and robust standard errors (HC1). H4 and H5 stack
the two models' answers to the same scenarios, with standard errors clustered by scenario. Any analysis
that pools the two versions of a scenario is clustered by scenario too.

**Parse failures.** Counted per model and per version. Before the test, the prompt is adjusted on dev
until failures are below 2%. In the test, failures are excluded from the main analysis, and a logistic
regression describes whether they depend on the scenario (number of items, extremity of the ideal answer,
version). A sensitivity analysis replaces them with 50 and with the base rate; if a conclusion changes,
it is reported. Above 5% for a model, failures are reported as a limit.

Exploratory, labelled as such: the history version and its three references, the topic (are politics
and international news discounted?), the framing (calm or volatile), the 1.7B floor, the tone reader,
and a run with thinking enabled if time allows.

## Size and precision

Since the confirmatory tests use the stated version only, the stated prompts get the size: **2,000
scenarios in the stated version** (short prompts), and **500 of them also in the history version** (long
prompts, exploratory). This costs less compute than the first draft (1,000 scenarios in both versions)
and gives more precision where it counts.

In the simulation, a model with biases (weights 0.6, 0.8, 0.4, 0.15, 0.1) and noise of one log-odds unit
gets standard errors of 0.015 to 0.043 on its weights with 2,000 scenarios (0.022 to 0.065 with 1,000).
For a shift of 0.1 log-odds between the base and abliterated models (about 2.5 points around 50%), the
t-statistic depends on how much the two models' errors move together:

| Scenarios | Independent errors | Correlation 0.5 |
|---|---|---|
| 1,000 | 1.6 | 2.9 |
| 2,000 | 2.3 | 3.4 |

Two versions of one model will most likely err together (positive correlation), which helps the paired
comparison; only a negative correlation would hurt. The final size will be set from the dev run: the
models' real noise, and a power simulation built on the observed base minus abliterated differences.
The test results are never used to change it.

## Procedure

1. **Code:** generator, the two versions, ideal observer and history references, round-trip and
   distribution tests, runner (the llama.cpp client of Parts 1 and 2), analysis.
2. **Dev:** 300 scenarios (seed 41) in the stated version, 100 of them in history, both 35B models.
   Checks the JSON parsing, runtime, noise level, the base minus abliterated differences and the prompt
   wording, and fits the tone reader's mapping. Dev results do not count.
3. **Pre-registration commit:** hypotheses with directions, measures, size, failure rules and the test
   seed (42).
4. **Test:** scenarios generated and run after that commit, analysed once and logged in
   `results/metrics/test_access.log`.
5. **README and figure.**

## Validity checks, by code

- **Names:** invented by a syllable generator and checked against the tickers of the FNSPID price archive
  and a list of country names; no match allowed.
- **Text to numbers and numbers to outcomes:** the two tests of the ideal observer above.
- **No outcome in the text:** templates only; a unit test rejects forbidden words such as "will rise" or
  "fell after".
- **Tone manipulation:** FinBERT scores dramatic distractors as more polarized than neutral ones. This one
  uses a model to check a manipulation, and is reported as such.

## Cost, to confirm on dev

- Prompts: about 300 to 400 tokens in the stated version, 1,500 to 2,000 in history.
- The 35B model took 0.6 s per tweet in Part 1. Stated prompts should take a few seconds each and history
  prompts several times more: about one night per 35B model for 2,000 stated and 500 history prompts.
  Measured on dev before the pre-registration.
- Work: one to two weeks part time, most of it the generator and its tests.

## Limits known in advance

- **Synthetic by design:** Part 4 measures how models use information under known rules, not whether they
  predict real markets. Part 3 (real data, possibly contaminated by memory) and Part 4 (clean, artificial)
  bracket the question; neither answers it alone.
- Templates are less varied than real news, and the models may treat a scenario as an arithmetic puzzle.
  That is partly the point, but it limits what the results say about free reading.
- News items are independent given the outcome; real news is correlated.
- In the stated version, track records are given: the test is the use of a stated track record, not
  judging a source's credibility in the wild. The history version gets closer to that, without a single
  answer key.
- One model family, one quantization, English only, thinking disabled, JSON answers, as in Part 2.

## Related work

- Grether (1980, *Quarterly Journal of Economics*) and [Benjamin (2019)](https://www.nber.org/papers/w25200):
  the regression above and what it has found in people (base rates and evidence weighed below what Bayes'
  rule asks).
- [Gupta et al. (2025)](https://arxiv.org/abs/2503.04722) test language models on biased coin flips: they
  broadly update like Bayesians, start from miscalibrated priors, and in-context evidence can outweigh an
  explicit instruction about the coin's bias.
- [Ilanyan and Jain (2026)](https://iclr.cc/virtual/2026/10018243) find base-rate neglect and updates
  weakly linked to the evidence in a language model playing poker.
- [Yee and Sharma (2026)](https://arxiv.org/abs/2602.01022) use synthetic scenarios to measure behavioural
  biases of language models, with correct answers set by the authors rather than computed.
- [Fafuła (2026)](https://arxiv.org/abs/2607.17427) find abliterated models more optimistic in their own
  stock calls.
- [Gao, Jiang and Yan (2026)](https://arxiv.org/abs/2512.23847v2) show that look-ahead from memory inflates
  forecasts on real headlines before a model's training cutoff, which motivates a world with nothing to
  remember.

After a short search, we did not find a study that combines financial news, a computed benchmark,
worthless news, reversed track records and abliteration. This needs a more thorough search before any
claim of novelty.

## Decisions

Settled after the first review:

1. Horizon of five trading days, answer as a probability from 0 to 100.
2. Confirmatory tests on the stated version; the history version is exploratory, with sampled episodes,
   one block per kind of news, and three reference answers.
3. Qwen3-1.7B kept as a descriptive floor.
4. Thinking disabled for the confirmatory run, enabled only as an exploratory run.
5. Size: 2,000 stated and 500 history scenarios, provisional, set from the dev run's noise and paired
   differences, never from the test.
