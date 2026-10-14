# Part 3, does the reading move with prices?

*Short version in the [README](../README.md).*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/part3-dark.svg">
  <img alt="Daily rank correlation between each model's reading of the headlines and the stock returns, around the news and after it" src="figures/part3-light.svg" width="100%">
</picture>

Headlines from FNSPID ([Dong et al., 2024](https://arxiv.org/abs/2402.06698)): Benzinga, Seeking Alpha, Zacks and others. Test period 2018 and 2019: **150,000 stock-days** (one stock on one news date), 246,764 headlines, 3,701 US stocks, each headline labelled by the four Part 1 readers, unchanged. Return: stock minus SPY (market-adjusted, no factor model).

- **Around the news**: from the last close before the news date to the first open after it. Most headlines have a date but no hour, so this window contains the reaction whatever the hour, and also moves that may come before the news. It measures an association, not a prediction.
- **After the news**: from that open to the close. The news is public for sure: this is the prediction test.
- **Measure**: each day, the rank correlation (IC) between a reader's score and the returns of all the stocks with news that day, averaged over 504 days, with Newey-West errors.

The design was built and run on 2016 and 2017, then frozen in a [pre-registration](part3-preregistration.md) before 2018 was touched. The dev runs changed two choices, both reported there: the test measure (a bullish minus bearish spread rewards a model that rarely takes a side) and the filter of headlines that only describe a price move ("Stock X jumps 5%").

| Pre-registered test (2018 and 2019) | Mean daily IC | t | Result |
|---|---|---|---|
| H1. Fine-tuned reading vs move around the news | +0.198 | 40.5 | Confirmed |
| H2. Fine-tuned minus FinBERT | +0.064 | 21.3 | Confirmed |
| H2. Fine-tuned minus TF-IDF | +0.043 | 15.1 | Confirmed |
| H2. Fine-tuned minus zero-shot | +0.065 | 17.6 | Confirmed |
| H3. Fine-tuned reading vs open to close after the news | +0.018 | 6.0 | Significant, not separable from look-ahead (see C1) |

- **The ranking of Part 1 holds against prices.** The fine-tuned model lines up best with the move around the news, then TF-IDF (0.155), FinBERT (0.134) and the zero-shot model (0.133). In basis points, stocks it reads as bullish beat those it reads as bearish by 306 bps over that window.
- **After the news is public, little is left, and only on bad news.** Stocks read as bearish keep falling (-15 bps from open to close); stocks read as bullish do not move (-0.2 bps). The 15 bps daily spread would break even at 7.6 bps of trading cost per leg. No strategy is claimed: equal weights, daily rebalancing, no execution, liquidity or risk model.
- **Memory control (C1).** The base Qwen model, asked with the ticker and the two dates only, no headline, gives answers unrelated to the move around the news (IC -0.006) but related to the move after it (IC +0.022, t 3.1). Under the rule fixed in advance, H3 is therefore reported as not separable from look-ahead, and every Qwen result as possibly affected by memory of the prices.
- **What the control does not settle** (exploratory, written after the results; every output is in `results/part3/test/exploratory.json` and its p-values are descriptive only). The probe's score is unrelated to every reader's score (rank correlations below 0.01). Which stock it is explains part of that link, perhaps not all of it: in a joint regression, the stock's average probe score is associated with the drift (t 2.8), and the date-to-date deviation from it points the same way, more weakly (t 1.9, p 0.06). A stock's news coverage known at the time gives mixed results (t 2.0 on the probe's stock-days, 1.1 on the full sample). On the probe's 20,000 stock-days, the readers' own drift associations are too imprecise to compare with it (t 0.3 to 1.6). The positive control therefore stays unexplained. FinBERT and TF-IDF, which cannot remember these prices, show a drift association on the full sample (IC 0.012 and 0.015, reported without a test in the pre-registration): this suggests, without establishing it, that a small post-news drift exists apart from any memory.
- **Coverage control (C2).** 14% of the candidate stock-days have no prices in FNSPID, most likely companies that later disappeared. Their headlines look like the others (TF-IDF tone 22% bullish, 66% neutral, 12% bearish, against 27%, 62%, 11%), but their returns are unknown.

## Limits

The Qwen models were pretrained on text covering the test years, and the memory control cannot check whether a headline reminds a model of a famous event and its outcome (that would need headlines with company names removed). Publication hours are unknown, returns are market-adjusted only, 14% of stock-days lack prices, and the headlines come from a few English-language sources. The abliterated model is not part of Part 3: at 0.6 s per headline it would need about 38 hours.
