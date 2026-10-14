# Replication: power of the pre-registered tests (before the test)

Estimates conditional on the data used, not guarantees of detecting the true effect.

## Part 4 (4,000 test scenarios, 300 simulated runs per cell, dev noise of each pair)

Smallest effects of interest (frozen, from the generator): H1 0.118, H2 0.075, H3 0.110, H4 0.100, H5 0.063.

| Pair | World | Failures | H1 | H2 | H3 | H4 | H5 | Any rejected | All rejected |
|---|---|---|---|---|---|---|---|---|---|
| gemma | at_smallest_effects | 0% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| gemma | at_smallest_effects | 5% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| gemma | null | 0% | 1% | 1% | 1% | 0% | 1% | 4% | 0% |
| gemma | null | 5% | 1% | 0% | 0% | 0% | 1% | 3% | 0% |
| llama | at_smallest_effects | 0% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| llama | at_smallest_effects | 5% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| llama | null | 0% | 2% | 1% | 1% | 0% | 1% | 4% | 0% |
| llama | null | 5% | 0% | 0% | 0% | 0% | 1% | 2% | 0% |

Measurement bias of gemma (null values of its tests): H1 -0.0039, H2 +0.0017, H3 +0.0039, H4 -0.0018, H5 +0.0025.
Measurement bias of llama (null values of its tests): H1 -0.0012, H2 -0.0006, H3 -0.0018, H4 +0.0047, H5 +0.0001.

## Part 2 and addendum (no dev run: effect of the original pair, and half of it)

Each test at alpha / k of its family, a lower bound of its power under Holm.

| Experiment | Test | Original effect | Half the effect |
|---|---|---|---|
| part2 | H1_pressure | 100% | 100% |
| part2 | H2_fragility | n/a | n/a |
| part2 | H3_push | 100% | 43% |
| part2 | H4_leverage | 98% | 32% |
| addendum | H1_false_optimism | 100% | 100% |
| addendum | H2_bullish_share | 100% | 100% |
| addendum | H3_macro_f1 | 100% | 100% |

Part 2 H2 has no power figure: it was not confirmed in the original pair, so there is no original effect to size it on.

Tests below 80% power (they stay unchanged; the limit is reported before the test): none.
