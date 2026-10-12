## Part 3, dev (2016-01-04 to 2018-01-02): 30,000 ticker-dates, 46,156 headlines, 3,023 tickers

Dev period: pipeline check only, not a test of the hypotheses.

| Test | Mean (bps) | t (Newey-West) | Days | p (Holm) | Result |
|---|---|---|---|---|---|
| H1 reaction spread, fine-tuned model | +322.9 | 24.33 | 498 | < 1e-16 | Confirmed |
| H2 reaction spread, fine-tuned minus FinBERT | +129.6 | 12.67 | 498 | < 1e-16 | Confirmed |
| H2 reaction spread, fine-tuned minus TF-IDF | +101.4 | 9.75 | 498 | < 1e-16 | Confirmed |
| H2 reaction spread, fine-tuned minus zero-shot | -229.2 | -8.04 | 366 | 1.82e-15 | Not confirmed |
| H3 drift spread (open to close), fine-tuned model | +9.2 | 1.43 | 498 | 0.153 | Not confirmed |

| Model | Bullish / neutral / bearish | Reaction spread | Long | Short | Drift spread | Next-day spread |
|---|---|---|---|---|---|---|
| qwen3-1.7b-lora-s0 | 28% / 56% / 16% | +322.9 | +150.0 | -172.9 | +9.2 | +5.6 |
| finbert | 26% / 52% / 22% | +192.4 | +105.7 | -86.8 | +2.2 | +4.0 |
| tfidf-logreg | 28% / 56% / 16% | +220.0 | +113.1 | -107.0 | -3.8 | +2.7 |
| qwen3-1.7b-zeroshot | 6% / 91% / 3% | +566.6 | +276.9 | -289.7 | -15.6 | +15.0 |

Abnormal returns in basis points, stock minus SPY, winsorized at 0.5% and 99.5%. Spread = mean over days of (bullish mean minus bearish mean).
Break-even round-trip cost of the drift spread: 4.6 bps per leg.
