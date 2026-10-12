# Part 3 pre-registration: does the sentiment a model reads move with returns?

Written and committed **before** any headline or price of the test period (2018 to 2019) is
sampled, labelled or analysed. Everything below was built and checked on the dev period
(2016 to 2017) only. Any later change is listed in the *Deviations* section at the end.

## Question

Part 1 measures how well each model reads market sentiment against human labels. Part 3 asks
whether that reading carries information about prices, and whether the better reader of Part 1
carries more of it.

1. **Association.** Is the model's reading of a headline correlated with the market-adjusted
   return of the stock around the news?
2. **Better reader, stronger association.** Is that correlation stronger for the fine-tuned model
   than for the other readers?
3. **Prediction.** Once the news is certainly public, does the reading still predict the return
   of the next session? For large US stocks the honest expectation is little or nothing.

Only the third question is about prediction. The first two measure how well a reading lines up
with the price move around the news, which may have started before the headline was published.
No trading strategy is claimed anywhere in Part 3.

## Data

- **Headlines:** FNSPID ([Dong et al., 2024](https://arxiv.org/abs/2402.06698)),
  `Stock_news/All_external.csv`, license CC BY-NC 4.0. 13.1 million rows, of which 9.8 million
  have no ticker and are not usable here. Coverage ends in June 2020. 98% of the rows since 2015
  have a date but no publication time.
- **Prices:** FNSPID `Stock_price/full_history.zip`, daily open, close and adjusted close up to
  December 2023. SPY is the market. Only 63% of the news tickers have a price file, most likely
  because delisted companies are missing: the sample is tilted towards survivors.
- Headline text, prices and returns stay in `data/part3/` (git-ignored). The repository only
  receives hashed headline ids, labels, counts and test results.

## Sample (code: `src/bbg/news.py`)

Rules applied in this order, each one counted in `results/part3/test/sample_report.json`:

1. Headlines dated 1 January 2018 to 31 December 2019. Date D: the UTC date when the row has no
   time (stored as 00:00:00 UTC), the New York date otherwise.
2. A headline listed under several tickers on the same date is dropped.
3. Duplicates (same ticker, same normalized title, same date) are kept once.
4. Price-move headlines are dropped with the fixed rules of `is_price_move`: lists of movers,
   gainers and losers, 52-week highs and lows, pre-market, mid-day and after-hours updates,
   relative-strength ratings, "shares trading higher", "shares are lower", and a move verb
   ("jumps", "tumbles"...) followed by a percentage or by "on", "after", "following", "as",
   "amid" or "post". A company figure is not a price move: the verb and percentage rules do not
   apply after "revenue", "sales", "earnings", "EPS" and similar words, nor to year-over-year
   comparisons.
5. Ticker-dates without a price file, or without the prices needed around D, are dropped.
6. A fixed random sample of **150,000 ticker-dates** is drawn (seed 31, `src/bbg/config.py`).

## Returns (code: `src/bbg/returns.py`)

The hour of publication is unknown, so the windows only depend on D. With t0 the last trading
day before D, t1 the first trading day after D and t2 the next one:

| Window | From | To |
|---|---|---|
| Reaction | close of t0 | open of t1 |
| Drift | open of t1 | close of t1 |
| Next day | close of t1 | close of t2 |

The reaction window contains the reaction whatever the hour, but also the whole session of D, so
H1 is an association with the move around the news, not a causal effect. The drift windows start
after any possible publication time.

Return = stock minus SPY over the same window: a market-adjusted return, not an abnormal return
from a factor model (no beta, no size or value factors). Open and close are scaled by adjusted
close / close, so dividends do not create gaps. Returns are winsorized at 0.5% and 99.5% over the
sample. The tests below compare stocks with each other on the same day, so subtracting the same
market return from all of them does not change their ranking; a beta adjustment would change it
only slightly over a single session. The choice matters for the spreads in basis points, not for
the tests.

## Signals

Each headline is labelled by four Part 1 models, unchanged, with the Part 1 prompt:
the fine-tuned Qwen3-1.7B (LoRA, seed 0, the main model), Qwen3-1.7B zero-shot, FinBERT and
TF-IDF with logistic regression. Before labelling, each model re-labels the Part 1 dev tweets and
must agree with its stored Part 1 predictions on at least 99% of them, otherwise the run stops.
The score of a ticker-date is the mean of +1 (bullish), 0 (neutral) and -1 (bearish) over its
headlines; the signal is the sign of the score.

## Measure and hypotheses (code: `src/bbg/event_stats.py`, `src/bbg/part3.py`)

**Measure.** For each trading day, the information coefficient (IC) is the Spearman correlation
between the score and the market-adjusted return over all the ticker-dates whose window ends that day,
neutral ones included. Days with fewer than 10 ticker-dates, or where the score does not vary,
are skipped. Each test is on the series of daily values: mean with Newey-West standard errors
(Bartlett kernel, 5 lags), so that stocks reacting to the same market day are not counted as
independent.

- **H1, association.** The mean daily IC of the fine-tuned model on the reaction window is
  positive.
- **H2, better reader.** On the reaction window, the mean daily IC of the fine-tuned model minus
  that of each other reader is positive: FinBERT, TF-IDF, zero-shot (three tests, days where both
  are defined).
- **H3, prediction.** The mean daily IC of the fine-tuned model on the drift window is positive.
  Expected: no effect, or a small one.

The five p-values are two-sided and Holm-corrected together. A hypothesis is confirmed if its
corrected p is below 0.05 and the mean is positive.

**Reported without a test:** long-short spreads in basis points (mean return of the bullish
ticker-dates minus that of the bearish ones, per day), long and short legs separately, the
next-day window, results by year, unwinsorized spreads, the break-even transaction cost of the
drift spread, and the share of bullish, neutral and bearish signals of each model. The spreads
are equally weighted, rebalanced daily and gross of costs, with no execution, liquidity or risk
model: they give the size of an association, not the profit of a strategy.

## Controls

**C1, memory probe (code: `scripts/part3_probe.py`).** The Qwen models were pretrained on text
that covers 2018 and 2019. To check whether they remember what these stocks did, the base
Qwen3-1.7B is asked, for a fixed random sample of 20,000 test ticker-dates (seed 32), with the
ticker and the two dates only and no headline: "Did this stock close higher or lower on t1 than
on t0?". The score is the log-probability of "higher" minus that of "lower", read from the
model's scores of the two answers. Its daily IC with the reaction and with the drift returns is
measured as above (Newey-West, two-sided, not part of the Holm family of H1 to H3).

**C2, coverage (code: `scripts/part3_coverage.py`).** Ticker-dates without prices cannot be
analysed, so no sensitivity analysis on them is possible. What both groups have is compared:
number of headlines, publishers, kind of ticker symbol (class shares and suffixes, five-letter
over-the-counter symbols ending in F or Y), and the tone of the headlines read by the TF-IDF model.
Descriptive only.

**Interpretation rules, fixed now.**
- If C1 shows a positive IC with p < 0.05 on either window, every result of the Qwen models is
  reported as possibly affected by memory of the prices.
- If H3 is confirmed, it is reported as a prediction only if C1 shows no memory and at least one of
  FinBERT and TF-IDF, which carry no knowledge of these prices, also has a positive mean drift IC.
  Otherwise it is reported as not separable from look-ahead.
- H1 and H2 are reported as associations, never as predictions.

## What the dev period changed, and why

The pipeline was first run on dev with the design of `docs/part3-design.md`. Two choices were
changed after looking at it; both are reported here so that the reader can judge them.

1. **The test measure: long-short spread replaced by the daily IC.** The spread only looks at
   the headlines a model labels bullish or bearish, so it rewards a model that labels few of
   them. On dev, the zero-shot model reads 91% of ticker-dates as neutral and has by far the
   largest spread, although it is the weakest reader of Part 1. The IC uses every ticker-date
   and rewards being right often. First dev run, both measures, reaction window:

   | Model | Signals bullish / neutral / bearish | Spread (bps) | Mean daily IC |
   |---|---|---|---|
   | Fine-tuned | 28% / 56% / 16% | +323 | +0.213 |
   | Zero-shot | 6% / 91% / 3% | +567 | +0.151 |
   | TF-IDF | 28% / 56% / 16% | +220 | +0.144 |
   | FinBERT | 26% / 52% / 22% | +192 | +0.129 |

   With the spread, H2 against the zero-shot model would have tested selectivity, not reading.
   The spreads stay in the report as the size of the effect.

2. **The price-move filter, refined after reading 100 dropped and 100 kept dev headlines.**
   Company figures such as "Sales up 11% YoY" were wrongly dropped; headlines such as
   "Cyclacel Tumbles on Failed Tests" and relative-strength ratings were wrongly kept. The rules
   above are the corrected ones: they flag 1.2% of the dev headlines the first filter had kept,
   and keep the company figures it had dropped. The cases are unit tests in `tests/test_part3.py`.

The dev period was then run again with the final code (`results/part3/dev/`). After a reading of
this document, the two controls above and the interpretation rules were added, and the dev period
was run once more to check them; the hypotheses, the sample and the measure did not change.

| Dev, final code (30,000 ticker-dates, 45,941 headlines, 504 days) | Mean daily IC | t (Newey-West) |
|---|---|---|
| H1 reaction, fine-tuned model | +0.204 | 27.1 |
| H2 reaction, fine-tuned minus FinBERT | +0.075 | 12.8 |
| H2 reaction, fine-tuned minus TF-IDF | +0.062 | 11.1 |
| H2 reaction, fine-tuned minus zero-shot | +0.068 | 7.9 |
| H3 drift, fine-tuned model | +0.002 | 0.4 |

Dev is not a test: these numbers only show that the pipeline runs end to end and what size of
effect to expect. The fine-tuned model's reaction spread is +306 bps and its drift spread +5.6 bps,
which would break even at 2.8 bps of round-trip cost per leg. All four models re-labelled the
Part 1 dev tweets as in Part 1 (agreement 100%, 99.9% for the zero-shot model).

Controls on dev (third run, same sample):

- **C1.** The base model answers "higher" for almost every ticker-date, but its confidence varies
  (log-probability gap from -0.8 to 12, standard deviation 1.35), and that variation is unrelated
  to the returns: daily IC -0.006 with the reaction (t -0.7) and +0.009 with the drift (t 1.3),
  over 503 days. With 20,000 ticker-dates the probe would detect an IC of about 0.016 (two
  standard errors); the fine-tuned model's reaction IC is 0.20.
- **C2.** Kept and dropped ticker-dates look alike: 1.53 and 1.58 headlines per ticker-date, the
  same main publishers, and a TF-IDF tone of 25% / 60% / 15% (bullish / neutral / bearish) for the
  kept ones against 22% / 63% / 15% for the dropped ones. 99% of the dropped tickers have ordinary
  symbols, so the missing price files are not mainly a symbol format problem.

## Procedure

1. Commit this file.
2. In WSL, with llama-server stopped:
   ```bash
   python scripts/part3_prepare.py --period test --news ~/fnspid/Stock_news/All_external.csv \
       --prices ~/fnspid/Stock_price/full_history.zip
   python scripts/part3_classify.py --period test --model all
   python scripts/part3_probe.py --period test
   python scripts/part3_coverage.py --period test
   python scripts/part3_analyze.py --period test      # once; logged in results/metrics/test_access.log
   ```
3. Commit the sample report, the labels, the probe scores, the coverage report and the analysis.

## Known limits

- **Look-ahead.** The Qwen models were pretrained on text that covers 2018 and 2019, so they
  could in principle carry knowledge of later price moves
  ([Glasserman and Lin, 2023](https://arxiv.org/abs/2309.17322)). FNSPID ends before their
  training cutoff, so no later period is left to test on. Control C1 checks whether the base model
  remembers the returns from the ticker and the dates. It cannot check whether a headline reminds
  the model of a famous event and its outcome: that would need the headlines with company names
  removed ([Glasserman and Lin](https://arxiv.org/abs/2309.17322) replace them with a placeholder),
  which needs a ticker-to-name table that FNSPID does not provide. Other controls: FinBERT was
  trained on text that largely predates the test period (BERT's 2018 corpus, Reuters news from
  2008 to 2010), TF-IDF only learnt word weights from the Part 1 tweets, and the zero-shot and
  fine-tuned models share the same pretraining, so the difference between them is not
  look-ahead.
- **Survivorship.** 37% of the news tickers have no price file. Their returns are unknown, so the
  results may not hold for them; C2 only shows whether their news look different.
- **Timing.** Without the hour, the reaction window also contains the session of D.
- **Sources.** Mostly Benzinga and Seeking Alpha headlines, English only, US stocks.
- **Out of scope.** The abliterated model of Part 2 is not part of Part 3: served by llama.cpp at
  about 0.6 s per headline, it would need about 38 hours on this sample. Whether its extra bullish
  calls carry price information could be a later addendum with its own pre-registration.

## Deviations

None so far.
