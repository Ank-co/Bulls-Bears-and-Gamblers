## Part 3, test (2018-01-02 to 2020-01-02): 150,000 ticker-dates, 246,764 headlines, 3,701 tickers

Pre-registered tests (docs/part3-preregistration.md) on the daily rank correlation (IC) between the model's score and the abnormal return, Holm-corrected.

| Test | Mean daily IC | t (Newey-West) | Days | p (Holm) | Result |
|---|---|---|---|---|---|
| H1 reaction, fine-tuned model | +0.198 | 40.49 | 504 | < 1e-16 | Confirmed |
| H2 reaction, fine-tuned minus FinBERT | +0.064 | 21.32 | 504 | < 1e-16 | Confirmed |
| H2 reaction, fine-tuned minus TF-IDF | +0.043 | 15.13 | 504 | < 1e-16 | Confirmed |
| H2 reaction, fine-tuned minus zero-shot | +0.065 | 17.60 | 504 | < 1e-16 | Confirmed |
| H3 drift (open to close), fine-tuned model | +0.018 | 6.02 | 504 | 1.76e-09 | Confirmed |

| Model | Bullish / neutral / bearish | Reaction IC | Reaction spread (bps) | Long | Short | Drift IC | Drift spread (bps) | Next-day IC |
|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b-lora-s0 | 27% / 60% / 13% | +0.198 | +305.8 | +111.3 | -194.5 | +0.018 | +15.3 | +0.007 |
| finbert | 29% / 55% / 15% | +0.134 | +197.8 | +81.0 | -116.8 | +0.012 | +11.4 | +0.000 |
| tfidf-logreg | 27% / 61% / 11% | +0.155 | +229.6 | +89.2 | -140.4 | +0.015 | +10.5 | +0.007 |
| qwen3-1.7b-zeroshot | 8% / 89% / 3% | +0.133 | +437.1 | +174.5 | -262.6 | +0.005 | +7.3 | -0.001 |

**C1, memory probe** (base Qwen3-1.7B, ticker and dates only, 20,000 ticker-dates): IC with the reaction -0.006 (t -0.69, p 0.489), with the drift +0.022 (t 3.07, p 0.00217); answers "higher" 100% of the time.

**C2, ticker-dates kept vs dropped for lack of prices** (TF-IDF labels, descriptive)

| | Headlines | Headlines per ticker-date | Bullish | Neutral | Bearish |
|---|---|---|---|---|---|
| Kept (sample) | 246,764 | 1.65 | 27% | 62% | 11% |
| Dropped, no prices | 44,226 | 1.62 | 22% | 66% | 12% |

Market-adjusted returns: stock minus SPY, winsorized at 0.5% and 99.5%. IC: Spearman correlation between the score and the return over all ticker-dates of a day, averaged over days. Spread: bullish mean minus bearish mean, averaged over days.
Break-even round-trip cost of the drift spread: 7.6 bps per leg.
