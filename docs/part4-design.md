# Part 4 design: reading the news in a fictional market

Status: **fourth version** (after three reviews), followed by the dev run and by
`docs/part4-preregistration.md`, which wins where the two differ (test size, H5 two-sided, null values
moved by the measurement bias, smallest effects of interest, no Qwen3-1.7B). The scenario generator is written and tested
(`src/bbg/fictional.py`, `tests/test_fictional.py`) and the 300 dev scenarios are generated
(`data/part4/dev/scenarios.jsonl`); no language model has been run. As in Part 3, the pre-registration
will be written after a dev run and committed before the test scenarios are generated, which the
generation script refuses to do earlier. The numbers of the simulation come from
`scripts/part4_design_check.py` (no language model) and are written to `results/part4/design_check.json`.

## Question

Part 3 found that the models' reading of real headlines lines up with prices, but could not separate
reading from remembering: the models were trained on text covering the test years. Part 4 removes memory
by construction. Every scenario takes place in an invented market, and the right answer is computed by
code from rules the scenario states. The question becomes:

> Do the models use financial news in proportion to what it is worth?

Five things are measured: whether the models take the base rate into account, whether they weigh evidence
by its strength, whether news worth nothing still moves them, whether evidence given in the scenario beats
their conventional reading of the news when the two disagree, and whether abliteration changes any of this.

## Why the answer key is computed, not written

In an invented market, the outcome is whatever the author decides. If the author writes that a rate rise
sends the market down, a model that answers "down" only shows that it shares the author's textbook: the
test would measure agreement with the author, not the use of information. If a language model wrote the
scenarios and their outcomes, a language model would set the answer key.

So the market follows an explicit random process, and the best possible answer given what the scenario
says is a probability computed by code (the ideal observer). The models are compared with that probability
and with the drawn outcomes.

## The world

One scenario is one company's share on an invented exchange, in an invented country, with invented
companies, people and months without a year. The model is asked for the probability, from 0 to 100, that
the share price is higher in one week (five trading days) than at today's close. There are 20 kinds of
news across five topics, each with a good and a bad version (40 sentences).

| Element | Values | Meaning |
|---|---|---|
| Base rate | 30%, 50%, 70% | probability that the share price rises over a week (five trading days) in this market |
| Evidence | 1 to 3 news items, each of its own kind | the version that came out depends on the outcome |
| Reliability r | 55%, 65%, 75%, 85% | drawn per kind of news, so that no real-world idea of which source is reliable can help |
| Reversed kinds | one kind in three | the track record runs against the conventional reading of the news |
| Distractors | 0 to 2 news items | the version that came out does not depend on the outcome |
| Tone | neutral or dramatic | same content, alarmist or euphoric wording on half of the news items, whatever their role |
| Topic | company, macro data, central bank, politics, international | drawn independently of the item's value: a political item can be evidence and an earnings item a distractor |
| Framing | calm market, volatile market, none | one opening sentence with no effect on the outcome |

**Generation order.** The generator works in this order and no other, so that the ideal observer's
formula is the true conditional probability:

1. draw the outcome (up or down) from the base rate;
2. for each evidence item, draw which version came out given the outcome: the version that
   conventionally points to the outcome with probability r for a congruent kind, the other one with
   probability r for a reversed kind;
3. draw each distractor's version at random, independently of the outcome.

The counts of the history version are drawn from the same probabilities, independently of this week's
outcome. Tone, names, sentences and the order of the news are drawn last, independently of everything
else.

## Track records state the news given the outcome, not what the news "suggests"

The second draft wrote track records as "after such news, the share price moves in the direction the news
suggests with probability 75%". Two problems, raised in review and confirmed by calculation:

- **It depended on interpretation.** "The direction the news suggests" requires the model to read the
  news as we do. A rate cut can be read as good news or as a sign of a weakening economy; a model that
  reads it the other way would answer "wrong" without having weighed anything wrongly.
- **It was false whenever the base rate was not 50%.** The generator's 75% is the probability of the news
  given the outcome. The sentence, read naturally, gives the probability of the outcome given the news.
  With a source at 75% and a base rate of 70%, the share rises after 87.5% of its good news and falls
  after only 56% of its bad news; with a base rate of 30%, the reverse. The sentence was true only on
  average over good and bad news.

Track records now give, for the version that came out, how often it preceded the share's past rises and
past falls:

> Track record: news of a rate cut came before 25% of the share's past weekly rises and before 75% of its
> past weekly falls.

This is the form of the classic urn experiments of belief updating. It has a single right reading,
combined with the base rate by Bayes' rule, and it never says whether the news is good or bad. A test
(`test_displayed_percentages_are_the_frequencies_given_the_outcome`) checks on the generator that, at
every base rate, the version shown came out before the stated share of rises and of falls; the second
draft's wording would have failed it.

**The conventional reading is fixed in advance** (table at the end) and used only to code which kinds
are reversed (H3) and which way distractors and dramatic wording lean (H2). It never enters the answer
key.

**Debatable kinds, by a written criterion.** A kind of news is marked debatable when its effect on a
share runs through the economy as a whole or through the central bank's response: macro data, central
bank decisions and world raw-material prices. For these, a standard argument exists for both signs (good
news is bad news when it raises the expected rates; a rate cut can signal a weakening economy). The
criterion gives 8 debatable kinds and 12 clear ones (company news, politics, trade, sanctions, foreign
investors). It was written before any model was run, does not depend on any model's answer, and a test
checks that the code follows it.

## Two versions of each scenario, which test two different skills

- **Stated (confirmatory).** The base rate and each track record are given as whole percentages. The
  ideal answer is a calculation with known numbers, with no ambiguity.
- **History (exploratory).** The model gets counts of past episodes instead: "over the past 20 weeks, the
  share price rose 13 times and fell 7 times", and for each news item "in the weeks before the share's
  last 20 weekly rises, the news was a rate cut 14 times and a rate rise 6 times; before its last 20
  weekly falls, a rate cut 7 times and a rate rise 13 times". These counts are samples, drawn from the
  same probabilities, so they equal the stated percentages only on average: the model has to infer, and
  the right answer depends on what it assumes before seeing the counts.

Counts out of 20 move in steps of 5%, so every stated percentage (55, 65, 75, 85, 50) is a possible
count. An earlier draft showed lists of 10 episodes, where 55% or 75% could never be observed and the
model had to count long lists, a known weakness of language models that has nothing to do with belief
updating; counts remove both problems, and any cue from the position of an episode in a list.

For the history version, three reference answers are computed. All are evaluated on the same 100,000
scenarios from the generator, with the same metrics and conventions (probabilities clipped to 1e-6, ten
equal-width bins). The true probabilities, calibrated by construction, give the noise floor of each
metric.

| Reference | What it assumes | Brier | Calibration error (ECE) | Calibration slope (SE) |
|---|---|---|---|---|
| True probabilities (not available to the model) | the hidden parameters | 0.1496 | 0.20 points (noise floor) | 1.006 (0.006) |
| Frequencies taken as true (a calculation) | 14 out of 20 is 70% | 0.1668 | 4.22 points | 0.742 (0.005) |
| Bayesian with uniform priors (an inference) | 14 out of 20 gives 15/22 = 68% | 0.1648 | 2.06 points | 0.865 (0.006) |
| Bayesian who knows the design | the list of possible probabilities and their shares | 0.1610 | 0.29 points | 1.010 (0.007) |

A slope below 1 means answers too far from 50%: taking the observed frequencies as true is overconfident.
On average, the three observable references differ by 2.2 to 5.7 points. Only the design-aware one is
calibrated (slope 1 within its standard error), and it uses knowledge the model does not have. So the
history version does not get a single answer key: it is analysed descriptively, by asking which reference
the model's answers resemble, and how much accuracy it loses compared with the stated version of the same
world.

## Nothing else in a scenario says anything about the outcome

Checking each cue separately (positions, frequencies) is not enough: a combination could still leak. The
generator is tested for this directly. On 30,000 scenarios with their text, a logistic regression of the
outcome on the ideal answer (as a fixed offset) plus 41 other features a reader can see (framing, tone and
its direction, number of news and of distractors, topics, which news comes first and last, sector, month,
the numbers in the sentences, length and form of the names) must find nothing: a likelihood-ratio test of
all of them together, and no single feature beyond 4.5 standard errors. With the test's seed, p = 0.28.
The same test catches a planted feature equal to the outcome 55% of the time (p below 1e-6 is required).

Writing this test found a real cue in the previous draft: dramatic wording was applied to distractors
only, so an alarmist or euphoric sentence was always worthless. Tone is now drawn for every news item,
whatever its role, and a test checks that both roles get it half of the time. Other tests check that the
news items are independent given the outcome (the ideal formula multiplies one ratio per item) and that
the history counts do not depend on this week's outcome.

## Example (stated version, in the generator's wording)

> Market: the [Country] Stock Exchange. Asset: the shares of [Company], a logistics company. In this
> market, the share price of [Company] rises over a week (five trading days) with probability 70%.
>
> News this week, each followed by its track record in this market:
>
> 1. The national statistics office reports that industrial orders rose 6% in [month]. Track record: news
>    of rising industrial orders came before 75% of the share's past weekly rises and before 25% of its
>    past weekly falls.
> 2. Alarm in [Country]: The ruling coalition in [Country] collapsed, and early elections were called.
>    Commentators fear the worst. Track record: news of a coalition collapse came before 50% of the share's
>    past weekly rises and before 50% of its past weekly falls.
> 3. An analyst at [Bank] cut [Company] to "sell". Track record: news of a downgrade to sell came before
>    65% of the share's past weekly rises and before 35% of its past weekly falls.
>
> What is the probability, from 0 to 100, that the share price of [Company] is higher in one week (five
> trading days) than at today's close? Answer in JSON with one field, "probability_up", an integer from 0
> to 100.

The ideal answer is about 93: the base rate (70%), item 1 (75% against 25%) and item 3 (65% against 35%,
a reversed kind) all point up, and item 2, alarmist as it is, carries nothing. A reader that only follows
the tone (one good item, two bad ones, one of them alarmist) would lean toward a fall.

## The ideal observer (stated version)

The news items are independent given the outcome, so the best answer has a closed form:

    log-odds(up) = log-odds(base rate) + sum over news items of log( P(news | rise) / P(news | fall) )

with the two percentages of each track record, and distractors (50% and 50%) adding nothing. It is
computed by code from the same whole percentages the text displays, and tests guard it:

- **text to numbers:** the base rate, each news item, its version and both percentages read back from
  each rendered prompt equal the generator's, and give the same ideal answer to 1e-12; a track record that
  names another version than the news shown is rejected;
- **numbers to outcomes:** at every base rate, the version shown came out before the stated share of past
  rises and of past falls, and the ideal answer is calibrated against the drawn outcomes;
- **one convention:** every probability is a whole percentage, the two percentages of a track record sum
  to 100, and the generator, the text and the answer key use the same integers.

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

Three things are measured separately, so that a wrong answer can be traced to its source:

1. **Reading the news.** Outside the scenarios, each model is asked whether news of each of the 40
   versions is good or bad for the share price of a company in a given sector. Its agreement with the
   conventional reading is published per kind of news. It selects nothing.
2. **Using the track records:** the weights on congruent and reversed evidence below.
3. **Using the base rate:** the weight on the base rate below.

**Belief-updating regression** (stated version). This is the standard tool of experimental economics for
Bayesian updating (Grether, 1980, *Quarterly Journal of Economics*; reviewed by
[Benjamin, 2019](https://www.nber.org/papers/w25200)). For each model:

    log-odds(answer) = c + w_base * log-odds(base rate)
                         + w_cong * E_congruent + w_rev * E_reversed
                         + w_dist * D + w_tone * T + error

where E_congruent and E_reversed are the ideal log-odds contributions of evidence whose track record
agrees or disagrees with the conventional reading, D is the sum of the conventional directions of the
distractors (+1 good news, -1 bad news) and T the same sum over every dramatic news item, evidence or
distractor. The ideal observer has c = 0, w_base = w_cong = w_rev = 1 and w_dist = w_tone = 0. A weight
below 1 describes the fitted model, not a mechanism: it can come from a cognitive shortcut, a calculation
error or a misreading of the scenario, and is reported as such.

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
  of its conventional reading.
- **H3, conventional reading over evidence:** w_rev < w_cong for the base model: a track record that runs
  against the conventional reading is followed less than one that agrees with it.
- **H4, abliteration and shift:** the abliterated model's intercept is higher than the base model's, in
  the regression on the ideal log-odds. The Part 1 addendum found it reads neutral news as bullish more
  often, and Fafuła (2026) found abliterated models more optimistic in their own stock calls.
- **H5, abliteration and amplification:** the abliterated model's slope on the ideal log-odds is higher
  than the base model's.

**Pre-registered sensitivity of H2 and H3.** Both rely on the conventional reading. They are re-estimated
twice: with each model's own reading of each version (measure 1) in place of the conventional one, and on
the 12 kinds not marked as debatable. If a conclusion changes, it is reported with the main result.

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
scenarios in the stated version** (short prompts), and **500 of them also in the history version**
(exploratory).

In the simulation, a model with biases (weights 0.6, 0.8, 0.4, 0.15, 0.1) and noise of one log-odds unit
gets standard errors of 0.018 to 0.033 on its weights with 2,000 scenarios (0.026 to 0.047 with 1,000).
For a shift of 0.1 log-odds between the base and abliterated models (about 2.5 points around 50%), each
with noise of one log-odds unit, the expected t-statistic and the power depend on how much the two
models' errors move together (formula, checked by 500 simulated runs):

| Scenarios | Errors | Expected t | Power at 5% | Power at 1% |
|---|---|---|---|---|
| 1,000 | independent | 2.2 | 72% | 46% |
| 1,000 | correlation 0.5 | 3.2 | 94% | 80% |
| 2,000 | independent | 3.2 | 94% | 80% |
| 2,000 | correlation 0.5 | 4.5 | 100% | 98% |

The 1% column is the strictest Holm threshold over five tests. An earlier draft gave single simulated
t-values (2.3 and 3.4), which were noisy draws of these expectations. Two versions of one model will most
likely err together (positive correlation), which helps the paired comparison; only a negative correlation
would hurt. The final size will be set from the dev run: the models' real noise, and a power simulation
built on the observed base minus abliterated differences. The test results are never used to change it.

## Procedure

1. **Code:** generator, the two versions, ideal observer and history references, tests (done); runner
   (the llama.cpp client of Parts 1 and 2), reading measure, analysis.
2. **Dev:** 300 scenarios (seed 41) in the stated version, 100 of them in history, both 35B models, and
   the reading measure. Checks the JSON parsing, runtime, noise level, the base minus abliterated
   differences and the prompt wording, and fits the tone reader's mapping. Dev results do not count.
3. **Pre-registration commit:** hypotheses with directions, measures, sensitivity analyses, size, failure
   rules and the test seed (42).
4. **Test:** scenarios generated and run after that commit, analysed once and logged in
   `results/metrics/test_access.log`.
5. **README and figure.**

## Validity checks, by code

- **Names:** invented by a syllable generator and checked against 5,298 real country and region names
  (ISO 3166, from pycountry, in `data/part4/real_place_names.txt`); a match forces a new draw.
- **Text to numbers, numbers to outcomes, one convention:** the tests of the ideal observer above.
- **No exploitable signal:** the joint test above, independence of the news given the outcome, tone drawn
  whatever the role, history counts independent of this week's outcome, and every position in the list of
  news a distractor with the same probability.
- **No outcome and no interpretation in the text:** templates only; a unit test rejects words such as
  "will", "soar" or "plunge", real years, and the word "suggest".
- **Tone manipulation:** FinBERT scores dramatic news as more polarized than neutral news. This one uses a
  model to check a manipulation, and is reported as such.
- **No selection by the models:** the models' answers never decide which news, scenarios or sentences
  enter the test. Their reading of the news is measured and published separately.

## Cost, to confirm on dev

- Prompts, measured on the 300 dev scenarios: 214 words on average in the stated version (323 at most),
  272 in history (428 at most). The token counts will come from the server on dev.
- The 35B model took 0.6 s per tweet in Part 1. Stated prompts should take a few seconds each: about one
  night per 35B model for 2,000 stated and 500 history prompts. Measured on dev before the
  pre-registration.

## Limits known in advance

- **Synthetic by design:** Part 4 measures how models use information under known rules, not whether they
  predict real markets. Part 3 (real data, possibly contaminated by memory) and Part 4 (clean, artificial)
  bracket the question; neither answers it alone.
- Templates are less varied than real news, and the models may treat a scenario as an arithmetic puzzle.
  That is partly the point, but it limits what the results say about free reading.
- News items are independent given the outcome; real news is correlated. Every kind of news is assumed
  to come out every week, in one version or the other.
- In the stated version, track records are given: the test is the use of a stated track record, not
  judging a source's credibility in the wild. The history version gets closer to that, without a single
  answer key.
- H2 and H3 depend on a conventional reading of the news, fixed by us; the sensitivity analyses show how
  much.
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

Settled after three reviews:

1. Horizon of five trading days, answer as a probability from 0 to 100.
2. Confirmatory tests on the stated version; the history version is exploratory, with counts of 20 past
   rises and 20 past falls, and three reference answers.
3. Track records state the probability of the news given the outcome, in whole percentages, and never
   what the news suggests.
4. The conventional reading of every kind of news, and the criterion for debatable kinds, are fixed
   below, before any model is run; no answer of a model selects what enters the test.
5. Tone is drawn for every news item, whatever its role.
6. Qwen3-1.7B kept as a descriptive floor.
7. Thinking disabled for the confirmatory run, enabled only as an exploratory run.
8. Size: 2,000 stated and 500 history scenarios, provisional, set from the dev run's noise and paired
   differences, never from the test.

## Conventional reading of the news, fixed before any model is run

From `FAMILIES` in `src/bbg/fictional.py`, where it is defined. Debatable kinds follow the criterion
above (effect through the economy or the central bank); a test checks that the code follows it.

| Topic | Kind | Good version | Bad version | Debatable because |
|---|---|---|---|---|
| Company | orders | order-book growth | an order-book decline |  |
| Company | results | profits above forecasts | profits below forecasts |  |
| Company | contract | a contract win | a contract loss |  |
| Company | leadership | a respected new chief executive | a chief executive's resignation |  |
| Company | analysts | an upgrade to buy | a downgrade to sell |  |
| Macro data | industry | rising industrial orders | falling industrial orders | strong activity can raise fears of rate rises |
| Macro data | jobs | falling unemployment | rising unemployment | low unemployment can raise fears of rate rises |
| Macro data | prices | inflation below expectations | inflation above expectations | low inflation can signal weak demand |
| Macro data | retail | rising retail sales | falling retail sales | strong spending can raise fears of rate rises |
| Central bank | rates | a rate cut | a rate rise | a rate cut can signal that the economy is weakening |
| Central bank | guidance | a signal of low rates for longer | a signal of further rate rises | low rates for longer can signal weak growth |
| Central bank | lending | eased lending rules | tightened lending rules | eased lending rules can signal that the banks are under strain |
| Politics | tax | a corporate tax cut | a corporate tax rise |  |
| Politics | budget | a budget passed | a budget rejected |  |
| Politics | coalition | a confidence vote won | a coalition collapse |  |
| Politics | regulation | less red tape | new compliance costs |  |
| International | trade | a free-trade agreement | new tariffs |  |
| International | sanctions | sanctions lifted | sanctions imposed |  |
| International | commodities | cheaper raw materials | dearer raw materials | cheaper raw materials can signal weak world demand |
| International | investors | new foreign investment | a foreign withdrawal |  |
