# Part 3 design draft: does the sentiment a model reads move with returns?

**Status: draft, not frozen.** Updated after the reconnaissance of the data files
(`scripts/recon_fnspid.py`, summary below). Once reviewed, it becomes
`docs/part3-preregistration.md`, committed before any return is computed on the test period.

## Question

Part 1 measures how well each model reads market sentiment against human labels. A trader
cares about something else: does that reading carry information about prices? Two questions:

1. **Reaction.** When a model reads a headline as bullish, does the stock move up around the
   news, compared with headlines it reads as bearish? And does the better reader of Part 1
   separate the moves better?
2. **Drift.** Once the news is certainly public, is there anything left to earn? For large US
   stocks, the honest prior is little or nothing; a null result is a result.

## What the data contain (reconnaissance)

- **Headlines:** FNSPID ([Dong et al., 2024](https://arxiv.org/abs/2402.06698)),
  `Stock_news/All_external.csv`: 13.1 million rows (title, ticker, publisher, date). License
  CC BY-NC 4.0: research use, never redistributed in this repository.
  - **Coverage ends in June 2020.** 2016: 375,000 rows, 2017: 286,000, 2018: 480,000,
    2019: 519,000.
  - **98% of the rows since 2015 have a date but no time** (stored as 00:00:00 UTC). The
    publication hour is unknown, which rules out an "overnight news only" design.
  - Publishers: Seeking Alpha, Zacks, Benzinga (listed by author name), GuruFocus, Investor's
    Business Daily. Many Benzinga rows are lists of movers ("71 Biggest Movers From Friday",
    "Stocks That Hit 52-Week Highs").
  - 74% of headline-days are attached to a single ticker.
- **Prices:** FNSPID `Stock_price/full_history.zip`: daily open, high, low, close, adjusted
  close and volume, up to December 2023. Open and close are split-adjusted; the adjusted close
  also accounts for dividends. SPY is present. The archive also contains macOS metadata files,
  ignored.
  - **Only 63% of the news tickers have a price file** (4,167 of 6,576), most likely because
    delisted companies are missing. The sample is therefore tilted towards survivors; this is
    stated as a limit.

## Periods

- **Dev = 2016 to 2017**: build and debug the whole pipeline, returns included.
- **Test = 2018 to 2019**: opened once, after the pre-registration is committed.

Download (WSL, Linux filesystem, outside the repository):

```bash
hf download Zihan1004/FNSPID Stock_news/All_external.csv Stock_price/full_history.zip \
   --repo-type dataset --local-dir ~/fnspid
```

## Sample, rules fixed in advance

1. One ticker per headline: a headline listed under several tickers on the same day is dropped
   (the sentiment cannot be attributed).
2. Price-move headlines are dropped (lists of movers, "52-week high", "shares are trading
   higher"): they describe the return instead of reading the news, and would create a
   mechanical link. The filter is a fixed regular expression, written and checked on dev only;
   the number of dropped headlines is reported.
3. Duplicates (same ticker, same title, same date) are kept once.
4. Tickers without a price file, or without prices around the date, are dropped (count
   reported).
5. **Unit: a ticker-date.** Several headlines for the same ticker and date are combined:
   score = mean of +1 (bullish), 0 (neutral), -1 (bearish); its sign is the signal.
6. A fixed random sample of ticker-dates keeps the compute within one night on the laptop:
   about 150,000 on test and 30,000 on dev (seed and exact sizes fixed in the pre-registration).

## Timing and returns

The date D of a headline is known, its hour is not, and the date itself may be a UTC date
(a headline published at 21:00 New York time is dated the next day). The windows are built so
that the answer does not depend on the hour:

| Name | Window | Meaning |
|---|---|---|
| Reaction | last close **before** D → first open **after** D | contains the reaction, whatever the hour of publication |
| Drift | that open → its close | the news is public for sure: what a trader entering at the open would earn |
| Drift, next day | that close → next close | slower reaction |

The reaction window contains the whole session of day D, so it also contains moves that
happened before the news. This is why the price-move filter matters, and why H1 is read as
an association with the move around the news, not as a causal effect. The drift windows start
after any possible publication time, so they are free of that problem.

All returns are abnormal: the stock minus SPY over the same window. Open and close are scaled
by the ratio adjusted close / close, so dividends do not create false gaps. Abnormal returns
are winsorized at 0.5% and 99.5% to limit data errors; unwinsorized results are reported too.

## Signals

Each headline is classified by four models of Part 1, unchanged:

| Model | Why it is here |
|---|---|
| Qwen3-1.7B + LoRA (seed 0) | the best reader of Part 1 |
| Qwen3-1.7B zero-shot | same backbone before fine-tuning: isolates what fine-tuning adds |
| FinBERT | trained on text that predates the test period |
| TF-IDF + logistic regression | trained only on the Part 1 tweets |

The 35B model is left out: 150,000 headlines at 0.64 s each would take 27 hours.

## Hypotheses (draft)

- **H1, reaction.** For the fine-tuned model, ticker-dates read as bullish have a higher
  abnormal reaction than those read as bearish (long-short spread > 0).
- **H2, better reader, more information.** The reaction spread of the fine-tuned model is
  larger than that of FinBERT, of TF-IDF and of the zero-shot model (three paired comparisons
  on the same ticker-dates).
- **H3, drift.** The long-short abnormal return from the open to the close after the news is
  positive for the fine-tuned model.

Descriptive, not tested: next-day drift, long and short legs separately (published work finds
the negative news stronger), results by year and by publisher, break-even transaction cost.

## Statistics

For each trading day, long-short spread = mean abnormal return of the bullish ticker-dates
minus mean of the bearish ones, grouped by the day the window ends. The test is on the series of
daily spreads, mean with Newey-West standard errors (5 lags), so that stocks reacting to the
same market day are not counted as independent. H2 uses the daily difference of spreads between
two models, on the same days. Holm correction over the five tests (H1, the three H2
comparisons, H3).

## Look-ahead bias

The Qwen models were pretrained on text that covers 2018 to 2019, so in principle they could
carry knowledge of later price moves ([Glasserman and Lin, 2023](https://arxiv.org/abs/2309.17322)).
There is no period after their training in these data. Two controls instead: FinBERT and TF-IDF
were trained on text that predates the test period or contains no prices, and the zero-shot and
fine-tuned Qwen models share the same pretraining, so the difference between them is not
look-ahead. This limit is stated in the README.

## Compute

About 150,000 + 30,000 ticker-dates (somewhat more headlines): fine-tuned model about 3 hours,
zero-shot model about 2 hours, FinBERT and TF-IDF a few minutes. One night on the laptop GPU,
with llama-server stopped (8 GB of GPU memory).

## Closest work

[Lopez-Lira and Tang (2023)](https://arxiv.org/abs/2304.07619) score headlines with GPT models
and find returns that keep drifting after the news is public (October 2021 to May 2024),
stronger for small firms and for bad news, and shrinking over time. Part 3 asks a narrower
question on a laptop: does a better sentiment reader, as measured in Part 1, translate into
more price information, with baselines that carry no knowledge of the test period's prices?
