## Part 3, dev (2016-01-04 to 2018-01-02): 30,000 ticker-dates, 45,941 headlines, 3,048 tickers

Dev period: pipeline check only, not a test of the hypotheses.

| Test | Mean daily IC | t (Newey-West) | Days | p (Holm) | Result |
|---|---|---|---|---|---|
| H1 reaction, fine-tuned model | +0.204 | 27.06 | 504 | < 1e-16 | Confirmed |
| H2 reaction, fine-tuned minus FinBERT | +0.075 | 12.82 | 504 | < 1e-16 | Confirmed |
| H2 reaction, fine-tuned minus TF-IDF | +0.062 | 11.11 | 504 | < 1e-16 | Confirmed |
| H2 reaction, fine-tuned minus zero-shot | +0.068 | 7.86 | 498 | 7.77e-15 | Confirmed |
| H3 drift (open to close), fine-tuned model | +0.002 | 0.40 | 504 | 0.688 | Not confirmed |

| Model | Bullish / neutral / bearish | Reaction IC | Reaction spread (bps) | Long | Short | Drift IC | Drift spread (bps) | Next-day IC |
|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b-lora-s0 | 28% / 56% / 17% | +0.204 | +306.0 | +143.3 | -162.7 | +0.002 | +5.6 | +0.008 |
| finbert | 26% / 52% / 22% | +0.130 | +191.8 | +109.8 | -82.0 | +0.006 | +5.4 | -0.001 |
| tfidf-logreg | 27% / 57% / 16% | +0.142 | +206.6 | +107.5 | -99.2 | -0.006 | -6.4 | -0.004 |
| qwen3-1.7b-zeroshot | 6% / 91% / 3% | +0.136 | +550.3 | +258.3 | -292.0 | +0.004 | +8.2 | +0.001 |

**C1, memory probe** (base Qwen3-1.7B, ticker and dates only, 20,000 ticker-dates): IC with the reaction -0.006 (t -0.69, p 0.492), with the drift +0.009 (t 1.25, p 0.21); answers "higher" 100% of the time.

**C2, ticker-dates kept vs dropped for lack of prices** (TF-IDF labels, descriptive)

| | Headlines | Headlines per ticker-date | Bullish | Neutral | Bearish |
|---|---|---|---|---|---|
| Kept (sample) | 45,941 | 1.53 | 25% | 60% | 15% |
| Dropped, no prices | 70,699 | 1.58 | 22% | 63% | 15% |

Market-adjusted returns: stock minus SPY, winsorized at 0.5% and 99.5%. IC: Spearman correlation between the score and the return over all ticker-dates of a day, averaged over days. Spread: bullish mean minus bearish mean, averaged over days.
Break-even round-trip cost of the drift spread: 2.8 bps per leg.
