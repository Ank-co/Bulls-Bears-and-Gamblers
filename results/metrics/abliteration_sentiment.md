## Part 1 addendum: abliterated vs base, same quantization (test, n = 2388)

Pre-registered tests (docs/part1-abliteration-preregistration.md), Holm-corrected.

| Hypothesis | Abliterated | Base | Difference | Only abliterated / only base | p (Holm) | Result |
|---|---|---|---|---|---|---|
| H1 non-bullish tweets read as bullish (n = 1913) | 22.8% | 10.9% | +11.9 points | 229 / 1 | 2.68e-67 | Confirmed |
| H2 all tweets read as bullish (n = 2388) | 36.9% | 26.5% | +10.4 points | 250 / 2 | 1.76e-71 | Confirmed |
| H3 macro-F1 | 0.720 | 0.788 | -0.069 [-0.081, -0.057] | | < 1e-04 | Differs |

| Model | Macro-F1 [95% CI] | Read as bullish | Read as bearish | Read as neutral | Bearish read as bullish | Bullish read as bearish |
|---|---|---|---|---|---|---|
| qwen3.6-35b-abliterated-zeroshot | 0.720 [0.701, 0.739] | 36.9% | 21.5% | 41.6% | 8 | 7 |
| qwen3.6-35b-i1-zeroshot | 0.788 [0.771, 0.806] | 26.5% | 21.5% | 52.0% | 1 | 9 |
| qwen3.6-35b-a3b-zeroshot | 0.757 [0.738, 0.776] | 29.4% | 22.5% | 48.1% | 4 | 13 |

Not pre-registered, descriptive: base i1-Q3_K_M minus base UD-Q3_K_M, macro-F1 +0.031 [+0.021, +0.042].
