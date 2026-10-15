# Replication on two more model families

*Short version in the [README](../README.md).*

Part 2, the addendum and Part 4 were run on one pair of models: Qwen3.6-35B-A3B and its abliterated copy, from the same quantizer. This replication asks whether their results hold in two more families, each a base model and its abliterated copy quantized by the same quantizer with the same method (mradermacher, `i1-Q4_K_M`):

- **Gemma 4 26B-A4B**, abliterated by the author of the original pair's abliteration (Huihui), and also a mixture of experts with about 4B active parameters: only the family changes.
- **Llama 3.1 8B**, abliterated by mlabonne: the family and the abliteration recipe change together.

Same inputs (the 48 advice scenarios, the 2,388 test tweets, the 4,000 test scenarios of the invented market), same requests, same tests, run through the original code. The [pre-registration](replication-preregistration.md) was committed and pushed before any test run: commit `7ba9bd2` at 23:39 on 2026-10-14, the runs started at 23:40 (`results/metrics/test_access.log`), the answers were committed before the analysis, and the analysis ran once. The scripts of the runs and of the analysis refuse to start until the pre-registration is committed, unchanged and contained in `origin/main`. Every number below comes from `results/replication/analysis.json`, `power.json` and the original analyses.

## How a result counts

- **Criterion A, statistical replication:** the hypothesis is rejected after Holm's correction, in the pre-registered direction, and for the two-sided tests (Part 4 H5, addendum H3) with the sign the original pair found.
- **Criterion B, substantive replication (Part 4 only):** criterion A, and the effect reaches the smallest effect of interest fixed by the generator (the same for every model). Part 2 and the addendum have no such threshold.
- Over the two new pairs, 2 of 2 means the result holds in the three families tested, 1 of 2 that it depends on the family, 0 of 2 that neither replication met the criterion, which does not show that the effect is specific to Qwen. **A test that is not rejected is not evidence that the effect is absent.**
- **Power, before the test:** in Part 4, 100% for every test and both pairs at the smallest effects of interest (simulation on each pair's dev answers), with a family-wise error of 1.7% to 4.3% when every null is true; in Part 2 and the addendum, 98% to 100% at the original pair's effect. At half the original effect, Part 2 H3 (43%) and H4 (32%) would probably miss it, a limit written before the test.
- No failed answer in any experiment. The format was checked before the test on items outside the test (`results/replication/checks.json`).

## Part 2, advising the client

| Test | Qwen (original) | Gemma 4 26B-A4B | Llama 3.1 8B |
|---|---|---|---|
| H1. Riskier advice to gamblers and loss-chasers (abliterated minus base) | +42.8 points (p Holm 7 × 10⁻⁵) | +30.5 points (p Holm 7 × 10⁻⁵, A: yes) | +8.2 points (p Holm 2 × 10⁻⁴, A: yes) |
| H2. A fragile situation lowers the advice less for the abliterated model | −2.3 points (p Holm 0.65) | −10.8 points (p Holm 1 × 10⁻⁴, A: no) | −5.4 points (p Holm 8 × 10⁻⁴, A: no) |
| H3. A client pushing for confirmation moves the abliterated model more | +23.6 points (p Holm 1 × 10⁻⁴) | +11.9 points (p Holm 7 × 10⁻⁵, A: yes) | +2.6 points (p Holm 0.14, A: no) |
| H4. Leverage to fragile clients chasing losses (abliterated vs base) | 43% vs 0% (p Holm 5 × 10⁻⁴) | 0% vs 0% (p Holm 1, A: no) | 0% vs 0% (p Holm 1, A: no) |

*Mean difference over 24 scenario means (H1 to H3) or rates over 30 paired answers (H4); Holm over the four tests of each pair.*

- **The abliterated model gives gamblers riskier advice in all three families**, with very different sizes: +42.8, +30.5 and +8.2 points of the savings put in high-risk products.
- **The push effect replicates in Gemma** (+11.9 points) **but is not detected in Llama** (+2.6 points, p after Holm 0.14), at about a ninth of the original effect, the size the power analysis had flagged as hard to detect.
- **No model recommends leverage to fragile clients chasing losses** (0% against 0% in both pairs), so H4 has nothing to compare.
- **H2, not confirmed in the original pair, goes the other way in both families.** Read after the results, not tested: the base models put almost nothing at risk (on average 1.5% of the savings for Gemma and 2.7% for Llama, over all scenarios), so only the abliterated model has room to lower its advice for a fragile client.

## Addendum, reading the market

| Test | Qwen (original) | Gemma 4 26B-A4B | Llama 3.1 8B |
|---|---|---|---|
| H1. Tweets that are not bullish read as bullish (abliterated vs base) | 22.8% vs 10.9% (p Holm 3 × 10⁻⁶⁷) | 17.1% vs 21.2% (p Holm 1 × 10⁻¹¹, A: no) | 2.9% vs 5.1% (p Holm 2 × 10⁻⁶, A: no) |
| H2. All tweets read as bullish (abliterated vs base) | 36.9% vs 26.5% (p Holm 2 × 10⁻⁷¹) | 32.2% vs 35.6% (p Holm 1 × 10⁻¹¹, A: no) | 13.2% vs 16.9% (p Holm 2 × 10⁻¹¹, A: no) |
| H3. Macro-F1 (abliterated vs base) | 0.720 vs 0.788 (p Holm < 10⁻¹⁰⁰) | 0.765 vs 0.717 (p Holm < 10⁻¹⁰⁰, A: no) | 0.698 vs 0.742 (p Holm < 10⁻¹⁰⁰, A: yes) |

- **The optimism of the abliterated Qwen does not replicate; on the tweets it reverses.** Both new abliterated models read fewer tweets as bullish than their base models, significantly.
- **Accuracy goes both ways:** macro-F1 falls for Llama (−0.044), as for Qwen, and rises for Gemma (+0.048).

## Part 4, weighing the news in an invented market

| Test | Qwen (original) | Gemma 4 26B-A4B | Llama 3.1 8B |
|---|---|---|---|
| H1. Base rate underweighted (weight minus 1) | −0.161 (rejected; effect 0.147 / 0.118) | −0.585 (rejected; effect 0.582 / 0.118; A: yes, B: yes) | −0.156 (rejected; effect 0.155 / 0.118; A: yes, B: yes) |
| H2. Worthless news moves the base model (weight) | +0.113 (rejected; effect 0.114 / 0.075) | +0.022 (not rejected; effect 0.020 / 0.075; A: no, B: no) | +0.022 (rejected; effect 0.023 / 0.075; A: yes, B: no) |
| H3. Reversed minus usual track record (weights) | −0.379 (rejected; effect 0.371 / 0.110) | −0.109 (rejected; effect 0.113 / 0.110; A: yes, B: yes) | −0.073 (rejected; effect 0.071 / 0.110; A: yes, B: no) |
| H4. Abliterated answers higher (shift, log-odds) | +0.332 (rejected; effect 0.327 / 0.100) | −0.142 (not rejected; effect −0.141 / 0.100; A: no, B: no) | −0.005 (not rejected; effect −0.010 / 0.100; A: no, B: no) |
| H5. Abliterated slope differs (log-odds) | −0.170 (rejected; effect 0.171 / 0.063) | −0.054 (rejected; effect 0.057 / 0.063; A: yes, B: no) | −0.046 (rejected; effect 0.046 / 0.063; A: yes, B: no) |

*Estimate, Holm decision, then the effect in the predicted direction against the smallest effect of interest. Each pair's null values are moved by its own measurement bias (`power.json`).*

- **All three base models underweight the base rate**, substantively: weights 0.84 (Qwen), 0.41 (Gemma) and 0.84 (Llama), against 1 for the ideal observer.
- **All three follow a reversed track record less than a usual one.** The effect reaches the smallest effect of interest for Gemma, by a narrow margin (0.113 against 0.110, 90% interval 0.081 to 0.144), and not for Llama (0.071).
- **Worthless news:** 0.11 for Qwen; 0.02 for Gemma, where the upper end of the 90% interval (0.038) lies below the smallest effect of interest, the one place where the data show that a meaningful effect is absent; 0.02 for Llama, detected but below the smallest effect.
- **The higher answers of the abliterated model do not replicate.** Gemma's abliterated model answers lower than its base model (49.4% against 52.2% on average), Llama's the same (58.8% against 58.9%). The share of weeks up is 49.3% in every case.
- **The flatter slope replicates statistically, not substantively:** −0.054 (Gemma) and −0.046 (Llama), about a third of Qwen's −0.170, below the smallest effect of interest (0.063).
- The sensitivity analyses (each base model's own reading of the news, the 12 clear kinds) leave every H2 and H3 decision unchanged.

**Descriptive, not tested.**

- **Llama 3.1 8B's base model barely uses the news:** weight 0.095 on news with the usual direction, against 0.51 for Qwen and 0.63 for Gemma. Its manipulation check passes, but its Brier score (0.224) is no better than the base rate alone (0.222): it answers mostly from the base rate, which is why its other effects are small.
- **Gemma's base model weighs the news more than the base rate** (0.63 against 0.41), the pattern usually called base-rate neglect, where Qwen showed the opposite (0.51 against 0.84). It has the best Brier score of the three (0.171, ideal observer 0.147).

## Over the two new families

| Original finding (Qwen pair) | Criterion A met | Criterion B met (Part 4) |
|---|---|---|
| Part 2 H1. The abliterated model gives gamblers and loss-chasers riskier advice | 2 of 2 | no threshold fixed |
| Part 2 H3. It moves more when the client pushes for confirmation | 1 of 2 | no threshold fixed |
| Part 2 H4. It recommends leverage to fragile clients chasing losses | 0 of 2 | no threshold fixed |
| Addendum H1. It reads tweets that are not bullish as bullish more often | 0 of 2 | no threshold fixed |
| Addendum H2. It reads all tweets as bullish more often | 0 of 2 | no threshold fixed |
| Addendum H3. Its macro-F1 is lower | 1 of 2 | no threshold fixed |
| Part 4 H1. The base model underweights the base rate | 2 of 2 | 2 of 2 |
| Part 4 H2. Worthless news moves the base model | 1 of 2 | 0 of 2 |
| Part 4 H3. A reversed track record is followed less | 2 of 2 | 1 of 2 |
| Part 4 H4. The abliterated model answers higher | 0 of 2 | 0 of 2 |
| Part 4 H5. The abliterated model's slope is flatter | 2 of 2 | 0 of 2 |

Part 2 H2 is left out: it was not confirmed in the original pair, so there was nothing to replicate (it is supported in 0 of 2 new families).

## What it changes

- **Abliteration's effect on advice generalizes.** In three families, the abliterated model gives clients who sound like gamblers riskier advice than its base model, by 8 to 43 points.
- **Its effect on how the model reads the market does not.** The optimism found with Qwen does not replicate: on labelled tweets it reverses with both new families, and in the invented market Gemma's abliterated model answers lower and Llama's the same. The earlier synthesis, one edit with three effects, held for one file; across families, the compliance with the client is what remains.
- **Base-rate underweighting generalizes; the use of the news does not.** How much a base model listens to the news differs a lot between families, from Gemma, which follows it more than the base rate, to Llama 3.1 8B, which hardly uses it.

## Notes

- In `results/replication/analysis.md`, the p column of the Part 4 rows is labelled "p (Holm)" but shows the raw p, with the Holm decision next to it. Example: Gemma H2 has p = 0.031 and is not rejected because its threshold in Holm's procedure was 0.025. The decisions are those of the pre-registered procedure; the file is left as the analysis wrote it.
- Answers: `results/replication/advice/raw/` (Part 2), `results/replication/predictions/` (addendum), `results/part4/test/answers/<model>/` (Part 4).

## Limits

- Three families in all, one quantization each, English only, thinking disabled, JSON answers, single questions.
- For Llama, the family and the abliteration recipe change together; for Gemma, only the family.
- Part 2's sampling is the one Qwen recommends, kept for every family so that only the model changes.
- Gemma's chat template, embedded in its files, predates Google's July 2026 update (tool calls and turn closures, which the requests here do not use).
- The replication reuses the test inputs of the original pair; nothing was chosen with them, and the new models answered them once.

## Reproduce

```bash
python scripts/part4_run.py --split dev --model gemma-base=gemma-4-26b-a4b-base-i1 --model gemma-abliterated=gemma-4-26b-a4b-abliterated --model llama-base=llama-3.1-8b-base-i1 --model llama-abliterated=llama-3.1-8b-abliterated
python scripts/replication_power.py                 # power and measurement bias, dev only
python scripts/replication_checks.py                # format checks on items outside the test
python scripts/replication_run.py --pair llama      # refused until the pre-registration is pushed
python scripts/replication_run.py --pair gemma
python scripts/replication_analyze.py               # run once
```
