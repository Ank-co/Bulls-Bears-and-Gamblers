# Part 4, test analysis (pre-registered, run once)

Manipulation check (base model uses the evidence): weight 0.512, passed.

| Hypothesis | Estimate (SE) | Null | p | Holm | Effect / smallest | Verdict |
|---|---|---|---|---|---|---|
| H1. Base rate underweighted (base model) | -0.161 (0.019) | -0.014 | 8.7e-15 | rejected | 0.147 / 0.118 | confirmed, at least the smallest effect of interest |
| H2. Worthless news moves the answer (base model) | 0.113 (0.013) | -0.001 | 1.5e-18 | rejected | 0.114 / 0.075 | confirmed, at least the smallest effect of interest |
| H3. Reversed track records followed less (base model) | -0.379 (0.027) | -0.008 | 2.4e-44 | rejected | 0.371 / 0.110 | confirmed, at least the smallest effect of interest |
| H4. Abliterated model answers higher | 0.332 (0.011) | 0.005 | 2.4e-189 | rejected | 0.327 / 0.100 | confirmed, at least the smallest effect of interest |
| H5. Abliterated model's slope differs | -0.170 (0.007) | 0.001 | 2e-126 | rejected | 0.171 / 0.063 | confirmed, at least the smallest effect of interest |

Sensitivity of H2 and H3:
- own reading: H2 0.112 (p 1.3e-17, same conclusion), H3 -0.365 (p 1.2e-41, same conclusion)
- clear kinds: H2 0.086 (p 1.4e-07, same conclusion), H3 -0.397 (p 2.4e-30, same conclusion)

Failures: base 0 stated, 0 history; abliterated 0 stated, 0 history

## base (descriptive)

- weights: const 0.173 (0.012), base_rate 0.839 (0.019), e_congruent 0.512 (0.013), e_reversed 0.133 (0.020), distractor 0.113 (0.013), dramatic_tone 0.049 (0.013)
- shift 0.171, amplification 0.471, noise sd 0.85
- Brier: model 0.1926, ideal 0.1474, base_rate_only 0.2224, tone_reader 0.2548
- reading of the news: {'agrees': 109, 'neither': 11}

## abliterated (descriptive)

- weights: const 0.506 (0.007), base_rate 0.719 (0.011), e_congruent 0.272 (0.008), e_reversed 0.073 (0.011), distractor 0.069 (0.008), dramatic_tone 0.036 (0.008)
- shift 0.503, amplification 0.301, noise sd 0.58
- Brier: model 0.2144, ideal 0.1474, base_rate_only 0.2224, tone_reader 0.2548
- reading of the news: {'agrees': 106, 'neither': 8, 'opposite': 6}
