# Part 3 design draft: does the sentiment a model reads move with returns?

**Status: draft, not frozen.** Some choices below depend on what the data files really contain
(timestamps, price adjustment, ticker coverage). They are checked first with
`scripts/recon_fnspid.py`; then this draft becomes `docs/part3-preregistration.md`, committed
before any return is computed on the test period.

## Question

Part 1 measures how well each model reads market sentiment against human labels. A trader
cares about something else: does that reading carry information about prices? Two questions:

1. **Reaction.** When a model reads a headline as bullish, does the stock move up when the
   market opens, compared with headlines it reads as bearish? And does the better reader of
   Part 1 separate the moves better?
2. **Drift.** Once the news is public at the open, is there anything left to earn on that day?
   For large US stocks in 2020 to 2023, the honest prior is little or nothing; a null result is
   a result.

## Data

- **Headlines:** FNSPID ([Dong et al., 2024](https://arxiv.org/abs/2402.06698)),
  `Stock_news/All_external.csv` (5.7 GB): publication time in UTC, title, ticker, publisher.
  License CC BY-NC 4.0: research use, never redistributed in this repository.
- **Prices:** FNSPID `Stock_price/full_history.zip` (0.6 GB), daily prices per ticker.
- **Periods:** dev = 2018 to 2019, used to build and debug the whole pipeline, returns included.
  Test = 2020 to 2023, opened once, after the pre-registration is committed.

Download (WSL, Linux filesystem, outside the repository):

```bash
hf download Zihan1004/FNSPID Stock_news/All_external.csv Stock_price/full_history.zip \
   --repo-type dataset --local-dir ~/fnspid
```

## Sample, rules fixed in advance

1. One ticker per headline: a headline listed under several tickers on the same day is dropped
   (the sentiment cannot be attributed).
2. Price-move headlines are dropped ("shares are trading higher", "52-week high", lists of
   gainers and losers): they describe the return instead of predicting it and would create a
   mechanical link. The filter is a fixed regular expression, written and tested on dev; the
   number of dropped headlines is reported.
3. Duplicates (same ticker, same title, same day) keep the earliest copy.
4. **Timing.** Times converted to New York time. Only overnight news is kept: published after
   16:00 or before 09:00 (weekends and holidays included). News during the session is dropped,
   since daily prices cannot tell what happened after it (its count is reported). The event day
   is the first trading day whose open follows the publication.
5. Several headlines for the same ticker and event day are combined: score = mean of
   +1 (bullish), 0 (neutral), -1 (bearish); its sign is the signal of that ticker-day.
6. A fixed random sample of ticker-days keeps the compute within one night on the laptop:
   about 150,000 on test and 30,000 on dev (seed and exact sizes fixed in the pre-registration).

## Returns

All returns are abnormal: the stock minus the market over the same window. The market is SPY
if the price archive has it, otherwise the equal-weighted mean of all stocks that day.

| Name | Window | Meaning |
|---|---|---|
| Reaction | last close before the news → first open after it | how the market priced the news |
| Drift, day 0 | open → close of the event day | what a trader entering at the open would earn |
| Drift, day 1 | close of the event day → next close | slower reaction |

Prices are split-adjusted (from the adjusted close if open prices are not adjusted). Abnormal
returns are winsorized at 0.5% and 99.5% to limit data errors; the unwinsorized results are
reported too.

## Signals

Each headline is classified by four models of Part 1, unchanged:

| Model | Why it is here |
|---|---|
| Qwen3-1.7B + LoRA (seed 0) | the best reader of Part 1 |
| Qwen3-1.7B zero-shot | same backbone before fine-tuning: isolates what fine-tuning adds |
| FinBERT | pretrained before 2020: cannot know what happened in the test period |
| TF-IDF + logistic regression | trained only on the Part 1 tweets: cannot know either |

The 35B model is left out: 150,000 headlines at 0.64 s each would take 27 hours.

## Hypotheses (draft)

- **H1, reaction.** For the fine-tuned model, ticker-days read as bullish have a higher abnormal
  reaction than those read as bearish (long-short spread > 0).
- **H2, better reader, more information.** The reaction spread of the fine-tuned model is
  larger than that of FinBERT, of TF-IDF and of the zero-shot model (three paired comparisons
  on the same ticker-days).
- **H3, drift.** The long-short abnormal return from the open to the close of the event day
  is positive for the fine-tuned model.

Descriptive, not tested: day-1 drift, long and short legs separately (published work finds the
negative news stronger), results by year, break-even transaction cost.

## Statistics

For each event day, long-short spread = mean abnormal return of the bullish ticker-days minus
mean of the bearish ones. The test is on the series of daily spreads, mean with Newey-West
standard errors (5 lags), so that stocks reacting to the same market day are not counted as
independent. H2 uses the daily difference of spreads between two models, on the same days.
Holm correction over the five tests (H1, the three H2 comparisons, H3).

## Look-ahead bias

The Qwen models were pretrained on text that covers 2020 to 2023, so in principle they could
carry knowledge of later price moves ([Glasserman and Lin, 2023](https://arxiv.org/abs/2309.17322)).
FNSPID ends in 2023, so there is no period after their training to test on. Two controls
instead: FinBERT and TF-IDF cannot know the test period, and the zero-shot and fine-tuned
Qwen models share the same pretraining, so the difference between them is not look-ahead.
This limit is stated in the README.

## Compute

About 150,000 + 30,000 headlines: fine-tuned model about 3 hours, zero-shot model about
2 hours, FinBERT and TF-IDF a few minutes. One night on the laptop GPU, with llama-server
stopped (8 GB of GPU memory).

## Closest work

[Lopez-Lira and Tang (2023)](https://arxiv.org/abs/2304.07619) score headlines with GPT models
and find returns that keep drifting after the news is public (October 2021 to May 2024), stronger
for small firms and for bad news, and shrinking over time. Part 3 asks a narrower question with a laptop: does a better sentiment reader, as
measured in Part 1, translate into more price information, with look-ahead-free baselines?
