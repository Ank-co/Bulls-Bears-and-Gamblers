# Part 4, dev run (set-up only, not a result)

## abliterated

- reading: 120 answers, 0 failures (0.0%), 0.89 s each, 90 prompt tokens
- stated: 300 answers, 0 failures (0.0%), 1.14 s each, 336 prompt tokens
- history: 100 answers, 0 failures (0.0%), 1.23 s each, 420 prompt tokens
- reading of the news: {'agrees': 106, 'neither': 8, 'opposite': 6}
- weights (ideal 1 / 1 / 1 / 0 / 0): base rate 0.784 (0.042), congruent 0.300 (0.027), reversed 0.221 (0.046), distractor 0.079 (0.030), tone 0.033 (0.028)
- shift 0.444 (0.035), amplification 0.373 (0.023), noise sd 0.61
- Brier: model 0.199, ideal 0.153, base_rate_only 0.211, tone_reader 0.241
- history, mean gap to each reference (points): frequencies_taken_as_true 29.3, uniform_prior_bayes 28.0, design_aware_bayes 27.3

## base

- reading: 120 answers, 0 failures (0.0%), 0.96 s each, 90 prompt tokens
- stated: 300 answers, 0 failures (0.0%), 1.15 s each, 336 prompt tokens
- history: 100 answers, 0 failures (0.0%), 1.24 s each, 420 prompt tokens
- reading of the news: {'agrees': 109, 'neither': 11}
- weights (ideal 1 / 1 / 1 / 0 / 0): base rate 0.924 (0.074), congruent 0.511 (0.057), reversed 0.219 (0.080), distractor 0.131 (0.047), tone 0.017 (0.056)
- shift 0.153 (0.050), amplification 0.512 (0.030), noise sd 0.86
- Brier: model 0.179, ideal 0.153, base_rate_only 0.211, tone_reader 0.241
- history, mean gap to each reference (points): frequencies_taken_as_true 18.9, uniform_prior_bayes 17.8, design_aware_bayes 17.1

## base vs abliterated

- mean difference 0.297 log-odds, sd 0.712 (0.660 around a trend in the ideal answer), error correlation 0.65
- 1000 scenarios: expected t 4.8, power 100% at 5%, 99% at 1%
- 2000 scenarios: expected t 6.8, power 100% at 5%, 100% at 1%
- shift difference 0.291 (0.038), slope difference -0.138 (0.024)

## Test size: power at the smallest effects of interest (5% / 1%, one-sided)

| Hypothesis | Effect | 2000 scenarios | 3000 scenarios | 4000 scenarios |
|---|---|---|---|---|
| H1 | 0.1 | 97% / 88% | 100% / 97% | 100% / 100% |
| H2 | 0.05 | 87% / 66% | 96% / 85% | 99% / 94% |
| H3 | 0.1 | 78% / 54% | 91% / 74% | 96% / 86% |
| H4 | 0.1 | 100% / 100% | 100% / 100% | 100% / 100% |
| H5 | 0.05 | 100% / 100% | 100% / 100% | 100% / 100% |
