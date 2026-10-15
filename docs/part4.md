# Part 4, weighing the news in an invented market

*Short version in the [README](../README.md).*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/part4-dark.svg">
  <img alt="Weights each model gives to the base rate, to news, to worthless news and to dramatic wording, against the ideal observer, and calibration of both models" src="figures/part4-light.svg" width="100%">
</picture>

Part 3 could not separate reading from remembering. Part 4 removes memory by construction: invented countries, exchanges and companies, and an outcome drawn by code. Each scenario states a base rate (the share rises over a week with probability 30%, 50% or 70%), then one to five news items, each followed by its track record, and asks for the probability, from 0 to 100, that the share is higher in a week. Since the process is known, the best possible answer (the ideal observer) is Bayes' rule applied to the stated numbers, computed by code. One news item from a test scenario:

> Sevdreica Systems reported quarterly profits 2% below analysts' forecasts. Track record: news of profits below forecasts came before 85% of the share's past weekly rises and before 15% of its past weekly falls.

- **The world** ([design](part4-design.md)). 20 kinds of news over five topics, each with a good and a bad version. Each kind gets a reliability from 55% to 85%, and one kind in three is **reversed**: its track record runs against the usual reading, as above, where news usually read as bad came before rises. **Worthless** items have a 50/50 record. Half of all items, whatever their role, get **dramatic wording** ("Alarm in ...: ... Commentators fear the worst.").
- **Nothing else leaks.** On 30,000 generated scenarios, a logistic regression of the outcome on 41 features a reader can see (topics, positions, tone, numbers, names), with the ideal answer as an offset, finds nothing (p = 0.28) and catches a planted leak. Writing it found a real leak in a draft (dramatic wording only on worthless news), fixed before any model was run.
- **Measure.** The belief-updating regression of experimental economics (Grether, 1980; reviewed by [Benjamin, 2019](https://www.nber.org/papers/w25200)): the log-odds of the answer on the log-odds of the base rate, of news whose record has the usual direction, of reversed news, and on the direction of worthless news and of dramatic wording. The ideal observer's weights are 1, 1, 1, 0 and 0.
- **The base model reads the news as intended.** Asked separately whether each of the 40 sentences is good or bad for a company (120 prompts), the base model gives the usual reading 109 times, "neither" 11 times, and never the opposite. The abliterated model gives the opposite reading 6 times. These answers select nothing.
- **Models.** The two Qwen3.6-35B-A3B files of the addendum, greedy decoding, thinking disabled, JSON answers. A 300-scenario dev run set the test size, the measurement bias and the power; its answers are never pooled with the test.

**Pre-registered tests** ([hypotheses and analysis plan](part4-preregistration.md), 4,000 test scenarios, no failed answer)

| Hypothesis | Estimate | Smallest effect of interest | p (Holm) | Result |
|---|---|---|---|---|
| H1. The base model underweights the base rate (weight minus 1) | −0.16 | 0.12 | 9 × 10⁻¹⁵ | Confirmed |
| H2. Worthless news moves the base model (weight) | +0.11 | 0.075 | 3 × 10⁻¹⁸ | Confirmed |
| H3. A reversed track record is followed less (reversed minus usual weight) | −0.38 | 0.11 | 7 × 10⁻⁴⁴ | Confirmed |
| H4. The abliterated model answers higher (shift, log-odds) | +0.33 | 0.10 | < 10⁻¹⁰⁰ | Confirmed |
| H5. The abliterated model's slope on the ideal answer differs (two-sided) | −0.17 | 0.063 | < 10⁻¹⁰⁰ | Confirmed |

*Each smallest effect of interest is the effect that moves a typical answer by 2.5 points of probability, written down before any power was computed from it. Answers are whole percentages, clipped to 1 to 99; the bias this causes in the estimates was simulated on dev, and each null value is moved by it (for H1, −0.014 instead of 0).*

- **The base model uses the news, at half its worth.** News whose record has the usual direction gets a weight of 0.51 (ideal 1). The model closes 40% of the gap in Brier score between the base rate alone (0.222) and the ideal observer (0.147): it scores 0.193.
- **The base rate is underweighted too, but less.** H1 holds as written: the base rate gets a weight of 0.84 (ideal 1). Since the news gets even less, the answers stay closer to the base rate than they should: the pattern is conservatism, too little updating, rather than dropping the base rate for the story. H1 is also the closest call: the lower end of its 90% interval (0.116) is just below the smallest effect of interest (0.118). These weights describe the answers, not a mechanism.
- **Worthless news moves it.** A 50/50 record carries nothing, yet each worthless item moves the answer by 0.11 log-odds in its usual direction, about 3 points near 50%. Dramatic wording adds little (0.05, not tested).
- **The textbook beats the stated record.** H3 is the largest effect: a reversed record is followed with a weight of 0.13, against 0.51 when the record agrees with the usual reading. The model reads these sentences the usual way, so this is not a misreading: when the usual meaning and the stated record disagree, the answers follow the record much less. The conclusions do not change with the model's own reading of each sentence (H2 0.11, H3 −0.36) or on the 12 kinds whose usual reading is not debatable (H2 0.09, H3 −0.40), both pre-registered.
- **Abliteration makes the model more optimistic and less responsive.** On the same scenarios, the abliterated model answers 7.9 points higher on average (60.9% against 53.0%, while the share rose in 49.3% of the weeks): higher in 2,420 of the 4,000 scenarios, equal in 823. Its slope on the ideal answer is 0.30 against 0.47, and all its weights are lower. After its most frequent answers, from 60% to 69% (1,365 of 4,000), the share rose in 50% of the weeks, and its Brier score (0.214) barely beats the base rate alone. This is the optimism of the addendum and of [Fafuła (2026)](https://arxiv.org/abs/2607.17427), measured against a computed answer. The design expected the opposite slope (more extreme answers); the dev run showed a lower one, so H5 was made two-sided before the test rather than flipped.
- **Tone alone does worse than a coin.** A reader of tone only (FinBERT, with a mapping fitted on dev) scores 0.255, against 0.25 for always answering 50%. Here tone carries little information by construction (a third of the kinds are reversed, worthless items are random), and FinBERT reads only 53% of the neutral-worded sentences in their usual direction.

**Exploratory** (written after the results; `results/part4/test/exploratory.json` and `analysis.json`, no confirmatory p-value)

- **Counts instead of percentages.** In the history version of 500 scenarios, the track records are counts ("in the weeks before the share's last 20 weekly rises, the news was a rate cut 14 times and a rate rise 6 times"). The base model's answers end 20 to 22 points from each of three reasonable reference answers (correlation about 0.5), against 15 points from the ideal for the stated version of the same scenarios (correlation 0.73). The abliterated model: 26 to 28 points, against 20.
- **Framing.** An opening sentence about a calm or a volatile market changes the mean answer by less than one point for either model. With the ideal answer held fixed, the largest of the four differences (volatile, abliterated model) is 0.06 log-odds, nominal p 0.013, uncorrected.
- **Topics.** Worthless political and international news moves the base model less than other worthless news (0.08 against 0.14, standard errors about 0.02).

**Deviation from the protocol.** The pre-registration and the analysis code were placed in the repository before the test scenarios were generated, but committed only with the results: the public record does not prove that they came first. The analysis was run once (`results/metrics/test_access.log`). Details in the [Deviations](part4-preregistration.md#deviations) section.

## Limits

Synthetic by design: it measures how the models use stated information, not whether they predict real markets. The templates are less varied than real news, and news items are independent given the outcome. H2 and H3 rest on a usual reading of the news fixed by us; the sensitivity analyses show that the conclusions hold without it. One model family at one quantization, English only, thinking disabled, JSON answers. The pre-registration was committed after the test run.

## Replication

With Gemma 4 26B-A4B and Llama 3.1 8B, the base model also underweights the base rate (weights 0.41 and 0.84) and follows a reversed track record less (substantively for Gemma only), and worthless news shows no effect of interest with Gemma. The higher answers of the abliterated model do not replicate (49.4% against 52.2% with Gemma, 58.8% against 58.9% with Llama), and the flatter slope replicates at about a third of the size found here, below the smallest effect of interest. Details in [replication.md](replication.md).
