# Part 1 addendum, pre-registration: does abliteration change how the model reads the market?

Written and committed **before** the two models below classified any tweet. Any later change to
the design or the analysis is listed in the *Deviations* section at the end, with its reason.

## Question

Part 2 shows that the abliterated model follows a gambling client instead of protecting them.
A recent study reports that abliterations of Qwen3-30B-A3B and Gemma make more optimistic
weekly calls on stocks than their base models ([Fafuła, 2026](https://arxiv.org/abs/2607.17427)).
This addendum asks two questions on the Part 1 task, where the right answer is known:

1. Does the abliterated model read market news with an optimistic bias, labelling tweets as
   bullish when they are not?
2. What does abliteration cost in classification quality?

## Status of the test set

The test set was already opened once, for the Part 1 models (git history: `Test set evaluated
once`). The two models below are added afterwards, so this is not a blind evaluation of the
test set. Nothing about them is chosen with the test set: they are used zero-shot, with exactly
the prompt, grammar and decoding of the existing 35B baseline (`src/bbg/prompts.py`,
`src/bbg/llamacpp.py`), and no other configuration is tried. Both dev and test are classified;
the tests below use the test set only.

## Models

| Prediction name | Server preset | File |
|---|---|---|
| `qwen3.6-35b-i1-zeroshot` | `qwen3.6-35b-base-i1` | `mradermacher/Qwen3.6-35B-A3B-i1-GGUF`, `i1-Q3_K_M` |
| `qwen3.6-35b-abliterated-zeroshot` | `qwen3.6-35b-abliterated` | `mradermacher/Huihui-Qwen3.6-35B-A3B-abliterated-i1-GGUF`, `i1-Q3_K_M` |

These are the two files of Part 2 (SHA-256 in [part2-preregistration.md](part2-preregistration.md)),
from the same quantizer with the same method, so a difference between them is not a
quantization artefact. The existing baseline `qwen3.6-35b-a3b-zeroshot` is the same base model
in another quantization (unsloth `UD-Q3_K_M`); its comparison with the i1 base is reported as
a description of the quantization effect, not as a test.

## Hypotheses

Unit: a tweet of the test set (n = 2,388), paired between the two models. Expected direction:
the abliterated model is more optimistic.

- **H1, false optimism (primary).** Among tweets whose gold label is not bullish (bearish or
  neutral), the abliterated model labels a larger share as bullish than the base model.
  Exact McNemar test on the paired indicator "predicted bullish".
- **H2, overall optimism.** Over all tweets, the abliterated model labels a larger share as
  bullish. Exact McNemar test on the same indicator.
- **H3, capability.** The macro-F1 of the two models differs. Paired bootstrap of the difference
  (10,000 resamples, seed 2026). No direction is predicted; a small loss for the abliterated
  model would be consistent with published measures of the cost of abliteration.

The three p-values are two-sided and Holm-corrected together. H1 and H2 are confirmed if the
corrected p is below 0.05 **and** the effect goes in the predicted direction; a significant
effect in the other direction is reported as such. H3 is confirmed if the corrected p is below
0.05, whatever the sign.

## Reported without a test

- Macro-F1 with bootstrap interval, confusion matrix and share of each predicted label, overall
  and by gold label, for both models.
- Bearish tweets read as bullish, and bullish tweets read as bearish, for both models.
- Base i1 minus base `UD-Q3_K_M`, macro-F1 (quantization).

## Procedure

1. Commit this file.
2. Classify dev and test with each model (WSL, llama.cpp router running on Windows):
   ```bash
   LLAMA_MODEL=qwen3.6-35b-base-i1 python scripts/zero_shot_llamacpp.py --name qwen3.6-35b-i1-zeroshot
   LLAMA_MODEL=qwen3.6-35b-abliterated python scripts/zero_shot_llamacpp.py --name qwen3.6-35b-abliterated-zeroshot
   ```
3. Run the pre-registered tests: `python scripts/analyze_abliteration_sentiment.py`
   (writes `results/metrics/abliteration_sentiment.{json,md}`, logged in `test_access.log`).
4. Commit the predictions and the analysis.

## Deviations

None so far.
