# Part 4 pre-registration: reading the news in a fictional market

Status: **frozen when committed**, before the test scenarios exist (the generation script refuses to
create them until this file exists). The design and its reasons are in `docs/part4-design.md`; where the
two differ, this document wins. Any change after this commit goes to "Deviations" at the end.

## Question

Do language models use financial news in proportion to what it is worth? In an invented market, with
nothing to remember, the best possible answer to every scenario is computed by code from what the scenario
states (the ideal observer). The models' answers are compared with it.

## What is frozen

- **Code:** the scenario generator (`src/bbg/fictional.py`), the requests to the models
  (`src/bbg/part4_llm.py`, `scripts/part4_run.py`), the measures (`src/bbg/part4_stats.py`), the tone
  reader (`scripts/part4_tone.py`) and the test analysis (`scripts/part4_analyze.py`), as in the commit
  that adds this file. The analysis code was tested on planted answers before the test exists
  (`tests/test_part4_analyze.py`).
- **Test scenarios:** seed 42, **4,000 scenarios** in the stated version, the first 500 of them also in
  the history version.
- **Models:** `qwen3.6-35b-base-i1` (base) and `qwen3.6-35b-abliterated`, the two quantized Qwen3.6-35B-A3B
  models of the Part 1 addendum, served by llama.cpp. Thinking disabled, temperature 0, top_k 1, seed 0,
  answers forced into a JSON schema, at most 32 tokens. The Qwen3-1.7B floor of the design is dropped.
- **Reading measure:** the same 120 prompts as on dev (20 kinds of news, 2 versions, 3 filled variants,
  seed 43), asked again to each model in the test run.
- **Frozen on dev:** the tone reader's mapping (`results/part4/tone_reader.json`), and the measurement
  biases and smallest effects of interest (`results/part4/dev/power.json`, below).

## Confirmatory tests

All on the stated version, answers clipped to 1..99 and taken in log-odds (`docs/part4-design.md`,
"Measures"). Alpha 0.05, **Holm's correction over exactly these five tests**, normal reference
distribution.

| Test | Hypothesis | Quantity | Direction | Standard errors |
|---|---|---|---|---|
| H1 | The base model underweights the base rate | w_base - 1 | below its null value | robust (HC1) |
| H2 | News worth nothing moves the base model's answer in the direction of its conventional reading | w_dist | above | robust (HC1) |
| H3 | The base model follows a track record that runs against the conventional reading less than one that agrees with it | w_rev - w_cong | below | robust (HC1), from the covariance of the two weights |
| H4 | The abliterated model's answers are higher, on the same scenarios | intercept difference, on the ideal log-odds | above | clustered by scenario, both models stacked |
| H5 | The abliterated model's slope on the ideal log-odds differs from the base model's | slope difference | two-sided | clustered by scenario, both models stacked |

**Manipulation check (not a test of the family):** the base model's w_cong is positive (one-sided, 5%).
If it is not, the base model does not use the evidence, and H1 to H3 are reported as uninterpretable.

**Null values moved by the measurement bias.** Answers are whole percentages clipped to 1..99. Simulating
the analysis with a model that behaves exactly as every null says, this rounding and clipping alone bias
the estimates (H1 -0.0135, H2 -0.0009, H3 -0.0078, H4 +0.0049, H5 +0.0014) and the five tests reject at
least one null in 13% of simulated test runs instead of 5%. Each test's null value is therefore moved by
its bias, read from `results/part4/dev/power.json`; with that correction, the simulated family-wise error
is 3% to 6%. The bias is estimated from the dev residuals and frozen; it is small next to the smallest
effects of interest below.

## Smallest effects of interest

One criterion, written before any power was computed from it: **an effect matters if it moves a typical
answer by at least 2.5 points of probability**, half of the 5-point step in which probabilities are usually
given. Near 50%, a change x in log-odds moves the probability by about x / 4, so each effect is
0.1 divided by the typical size of its regressor in the generator (not in any model's answers):

| Test | Typical size of the regressor | Smallest effect of interest |
|---|---|---|
| H1 | base-rate log-odds, 0.847 (when the base rate is not 50%) | 0.118 |
| H2 | sum of distractor directions, 1.336 (when not zero) | 0.075 |
| H3 | evidence log-odds of one news item, 0.912 | 0.110 |
| H4 | the shift itself | 0.100 |
| H5 | ideal log-odds, 1.597 | 0.063 |

## How each result will be read

- Rejected by Holm, effect at least the smallest of interest: **confirmed, at least the smallest effect
  of interest**.
- Rejected by Holm, effect below it: **detected, below the smallest effect of interest**.
- Not rejected, and the upper end of the effect's 90% interval below the smallest effect: **no effect of
  interest**.
- Not rejected otherwise: **inconclusive**.

Weights below 1 describe the fitted answers, not a mechanism: a shortcut, a calculation error or a
misreading of the scenario give the same pattern. H1 to H3 say nothing about the abliterated model, whose
weights are descriptive.

## Sensitivity analyses

- **H2 and H3 and the reading of the news.** Both rely on the conventional reading fixed in
  `docs/part4-design.md`. They are re-estimated (1) with the base model's own reading from the reading
  measure (majority over the 3 variants; a tie or "neither" moves the item out of the congruent,
  reversed and distractor terms), and (2) with the 8 debatable kinds moved out of those terms, keeping
  the 12 clear ones. Each is compared with the threshold its test had in Holm's procedure. If a
  conclusion changes, it is reported with the main result.
- **Failures.** Counted per model and version and excluded from the main analysis. If there are any, a
  logistic regression describes whether they depend on the scenario (number of news, extremity of the
  ideal answer, framing), and the five tests are rerun with failures replaced by 50, then by the base
  rate; any change of decision is reported. Above 5% failures for a model, the failures are reported as
  a limit.

## Size and power

`scripts/part4_power.py` simulates the analysis above, with its tests, its clustering, Holm's correction,
the moved null values, rounding, clipping and failures (0% and 5%). Each simulated test run resamples the
300 dev scenarios with replacement, each with its own regressors and both models' dev residuals, so the
errors keep their shape, their link with the regressors and their dependence between the two models;
300 runs per cell. With every tested quantity at its smallest effect of interest (`results/part4/dev/power.md`):

| Scenarios | H1 | H2 | H3 | H4 | H5 | All five rejected |
|---|---|---|---|---|---|---|
| 2,000 | 96% to 97% | 99% to 100% | 83% to 85% | 100% | 100% | 81% to 83% |
| 4,000 | 100% | 100% | 97% to 99% | 100% | 100% | 97% to 99% |

(ranges over 0% and 5% failures). With every null true, at least one rejection in 4% to 5% of runs at
4,000 scenarios. These numbers hold if the test answers are as noisy as the dev ones.

## What the dev run changed

The dev run (300 scenarios, `results/part4/dev/`) set up the test; its answers are never pooled with the
test answers.

1. **Size:** 2,000 to 4,000 stated scenarios. At 2,000, H3 had 83% to 85% power at its smallest effect of
   interest.
2. **H5 became two-sided.** The design predicted that the abliterated model's answers would be more
   extreme (a higher slope). On dev its slope was lower (difference -0.14, t -5.7). The prior for that
   direction was weak; flipping it after seeing dev would fit the hypothesis to the data, so the test is
   two-sided. H4 keeps its direction (the Part 1 addendum and Fafuła 2026 predict optimism; dev agrees).
3. **Null values moved by the measurement bias**, found by simulating the exact analysis.
4. **Smallest effects of interest** from one written criterion, replacing values first chosen one by one.
5. **The Qwen3-1.7B floor is dropped:** it was descriptive and would need another client.

Unchanged, on purpose: the prompts and the 40 sentences. On dev, both models read "Parliament passed the
annual budget" as neither good nor bad in all 3 variants, and the abliterated model gave the opposite of
the conventional reading in 6 of its 120 answers, all on bad news; changing the sentences after that
would let the models choose the stimuli. The sensitivity analyses cover it. No failures occurred on dev (1,040 answers), so the failure
rule stays as designed.

## Descriptive and exploratory

Descriptive, for both models: belief-updating weights, shift, amplification and noise, Brier score and
log loss against the ideal observer, the base rate alone and the tone reader, calibration by decile,
reading of the news, paired differences between the two models.

Exploratory, labelled as such: the history version against its three reference answers; the framing
(calm, volatile, none); the weight of worthless political and international news against other worthless
news; anything written after the results.

## Procedure

1. Commit this file.
2. `python scripts/part4_generate.py --split test` (4,000 scenarios, seed 42).
3. `python scripts/part4_run.py --split test` (reading, stated and history, both models).
4. `python scripts/part4_tone.py --split test` (mapping frozen on dev).
5. `python scripts/part4_analyze.py`, run once; every run is logged in `results/metrics/test_access.log`.
6. README section, figure, and the commit of every file above.

## Known limits

- Synthetic by design: the results describe how the models use information under stated rules, not
  whether they predict real markets.
- Templates are less varied than real news; news items are independent given the outcome; every kind of
  news comes out every week in one version or the other.
- H2 and H3 depend on a conventional reading of the news; the sensitivity analyses show how much.
- The measurement-bias correction and the power assume the test answers are noisy like the dev ones.
- The tone reader is weak: on dev, FinBERT read 52% of neutral sentences in their conventional
  direction, and its Brier score (0.241) was close to a coin's (0.25).
- One model family at one quantization, English only, thinking disabled, JSON answers.

## Deviations

1. **This file was not committed before the test run.** Step 1 of the procedure was skipped. This file
   and the analysis code were placed in the repository before the test scenarios were generated (file
   times on the author's machine: 2026-10-13 20:11 UTC for this file and `scripts/part4_analyze.py`,
   21:42 UTC for `results/part4/dev/power.json`, 21:45 UTC for `data/part4/test/scenarios.jsonl`) and
   were not modified afterwards: everything above this section is byte for byte the text placed at
   20:11. But they were committed only with the results, on 2026-10-14, so the public record does not
   prove that they came first, and file times on a personal machine cannot be checked by others. The
   analysis was run once (`results/metrics/test_access.log`, 2026-10-14 02:52 local time), with no
   failures in the 9,240 answers of the test run.
