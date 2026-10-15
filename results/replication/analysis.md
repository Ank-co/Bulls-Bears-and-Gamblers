# Replication: pre-registered analysis of the test runs (run once)

Criterion A: rejected after Holm in the pre-registered direction (original sign for two-sided tests). Criterion B (Part 4): A and the effect reaches the smallest effect of interest. A test that is not rejected does not show that the effect is absent.

## Gemma 4 26B-A4B

| Experiment | Test | Estimate | p (Holm) | A | B |
|---|---|---|---|---|---|
| Part 2 | H1_pressure | +30.5 points | 7.19e-05 | yes | |
| Part 2 | H2_fragility | -10.8 points | 9.76e-05 | no | |
| Part 2 | H3_push | +11.9 points | 7.19e-05 | yes | |
| Part 2 | H4_leverage | 0% vs 0% | 1 | no | |
| Addendum | H1_false_optimism | -4.0 points | 1.17e-11 | no | |
| Addendum | H2_bullish_share | -3.4 points | 1.17e-11 | no | |
| Addendum | H3_macro_f1 | +0.048 | 0 | no | |
| Part 4 | H1 | -0.585 (null -0.004) | 0 (rejected) | yes | yes (effect 0.582 / 0.118) |
| Part 4 | H2 | +0.022 (null +0.002) | 0.0307 (not rejected) | no | no (effect 0.020 / 0.075) |
| Part 4 | H3 | -0.109 (null +0.004) | 2e-09 (rejected) | yes | yes (effect 0.113 / 0.110) |
| Part 4 | H4 | -0.142 (null -0.002) | 1 (not rejected) | no | no (effect -0.141 / 0.100) |
| Part 4 | H5 | -0.054 (null +0.002) | 5.08e-42 (rejected) | yes | no (effect 0.057 / 0.063) |

Part 4 manipulation check (base model uses the evidence): weight 0.632, passed.
Part 4 sensitivity: own reading: H2 same, H3 same; clear kinds: H2 same, H3 same

## Llama 3.1 8B

| Experiment | Test | Estimate | p (Holm) | A | B |
|---|---|---|---|---|---|
| Part 2 | H1_pressure | +8.2 points | 0.000211 | yes | |
| Part 2 | H2_fragility | -5.4 points | 0.000824 | no | |
| Part 2 | H3_push | +2.6 points | 0.141 | no | |
| Part 2 | H4_leverage | 0% vs 0% | 1 | no | |
| Addendum | H1_false_optimism | -2.2 points | 1.97e-06 | no | |
| Addendum | H2_bullish_share | -3.7 points | 2.04e-11 | no | |
| Addendum | H3_macro_f1 | -0.044 | 0 | yes | |
| Part 4 | H1 | -0.156 (null -0.001) | 1.97e-70 (rejected) | yes | yes (effect 0.155 / 0.118) |
| Part 4 | H2 | +0.022 (null -0.001) | 1.71e-05 (rejected) | yes | no (effect 0.023 / 0.075) |
| Part 4 | H3 | -0.073 (null -0.002) | 6.8e-18 (rejected) | yes | no (effect 0.071 / 0.110) |
| Part 4 | H4 | -0.005 (null +0.005) | 0.851 (not rejected) | no | no (effect -0.010 / 0.100) |
| Part 4 | H5 | -0.046 (null +0.000) | 1.44e-23 (rejected) | yes | no (effect 0.046 / 0.063) |

Part 4 manipulation check (base model uses the evidence): weight 0.095, passed.
Part 4 sensitivity: own reading: H2 same, H3 same; clear kinds: H2 same, H3 same

## Over the two new families

| Experiment | Test | Original pair | Criterion A | Criterion B |
|---|---|---|---|---|
| part2 | H1_pressure | confirmed | met the criterion in 2 of 2 replications: holds in the three families tested |  |
| part2 | H2_fragility | not confirmed | supported in 0 of 2 new families (not confirmed in the original pair) |  |
| part2 | H3_push | confirmed | met the criterion in 1 of 2 replications: depends on the family |  |
| part2 | H4_leverage | confirmed | met the criterion in 0 of 2 replications: neither replication met the fixed criterion, which does not show that the effect is specific to Qwen |  |
| addendum | H1_false_optimism | confirmed | met the criterion in 0 of 2 replications: neither replication met the fixed criterion, which does not show that the effect is specific to Qwen |  |
| addendum | H2_bullish_share | confirmed | met the criterion in 0 of 2 replications: neither replication met the fixed criterion, which does not show that the effect is specific to Qwen |  |
| addendum | H3_macro_f1 | confirmed | met the criterion in 1 of 2 replications: depends on the family |  |
| part4 | H1 | confirmed | met the criterion in 2 of 2 replications: holds in the three families tested | met the criterion in 2 of 2 replications: holds in the three families tested |
| part4 | H2 | confirmed | met the criterion in 1 of 2 replications: depends on the family | met the criterion in 0 of 2 replications: neither replication met the fixed criterion, which does not show that the effect is specific to Qwen |
| part4 | H3 | confirmed | met the criterion in 2 of 2 replications: holds in the three families tested | met the criterion in 1 of 2 replications: depends on the family |
| part4 | H4 | confirmed | met the criterion in 0 of 2 replications: neither replication met the fixed criterion, which does not show that the effect is specific to Qwen | met the criterion in 0 of 2 replications: neither replication met the fixed criterion, which does not show that the effect is specific to Qwen |
| part4 | H5 | confirmed | met the criterion in 2 of 2 replications: holds in the three families tested | met the criterion in 0 of 2 replications: neither replication met the fixed criterion, which does not show that the effect is specific to Qwen |
