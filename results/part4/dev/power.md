# Part 4, test size: power of the exact analysis (simulation, dev noise)

Smallest effects of interest (criterion: 2.5 points of probability on a typical answer): H1 0.118, H2 0.075, H3 0.110, H4 0.100, H5 0.063.

300 simulated test runs per cell, resampling the 300 dev scenarios with both models' residuals. Holm's correction over the five tests, alpha 0.05.

| World | Scenarios | Failures | H1 | H2 | H3 | H4 | H5 | Any rejected | All rejected |
|---|---|---|---|---|---|---|---|---|---|
| at_smallest_effects | 2000 | 0% | 96% | 100% | 85% | 100% | 100% | 100% | 83% |
| at_smallest_effects | 2000 | 5% | 97% | 99% | 83% | 100% | 100% | 100% | 81% |
| at_smallest_effects | 3000 | 0% | 100% | 100% | 96% | 100% | 100% | 100% | 95% |
| at_smallest_effects | 3000 | 5% | 98% | 100% | 93% | 100% | 100% | 100% | 92% |
| at_smallest_effects | 4000 | 0% | 100% | 100% | 97% | 100% | 100% | 100% | 97% |
| at_smallest_effects | 4000 | 5% | 100% | 100% | 99% | 100% | 100% | 100% | 99% |
| null | 2000 | 0% | 1% | 0% | 1% | 2% | 1% | 5% | 0% |
| null | 2000 | 5% | 1% | 1% | 1% | 0% | 1% | 3% | 0% |
| null | 3000 | 0% | 2% | 1% | 1% | 1% | 2% | 6% | 0% |
| null | 3000 | 5% | 2% | 0% | 1% | 1% | 1% | 5% | 0% |
| null | 4000 | 0% | 0% | 1% | 1% | 2% | 1% | 4% | 0% |
| null | 4000 | 5% | 1% | 1% | 0% | 3% | 1% | 5% | 0% |

Measurement bias under the null (rounding to whole percentages, clipping to 1..99), used as null values: H1 -0.0135, H2 -0.0009, H3 -0.0078, H4 +0.0049, H5 +0.0014.
Without that correction, 4000 scenarios, no failures: any rejected 13% (H1 7%, H2 1%, H3 0%, H4 4%, H5 2%).
