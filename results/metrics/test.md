## test split (n = 2388)

| Model | Macro-F1 [95% CI] | Accuracy [95% CI] | Direction errors |
|---|---|---|---|
| qwen3-1.7b-lora-s0 | 0.887 [0.872, 0.902] | 0.911 [0.899, 0.922] | 11 |
| qwen3-1.7b-lora-s1 | 0.883 [0.868, 0.897] | 0.907 [0.895, 0.918] | 10 |
| qwen3-1.7b-lora-s2 | 0.882 [0.867, 0.897] | 0.907 [0.895, 0.918] | 8 |
| tfidf-logreg | 0.764 [0.743, 0.784] | 0.826 [0.810, 0.841] | 69 |
| qwen3.6-35b-a3b-zeroshot | 0.757 [0.738, 0.776] | 0.775 [0.758, 0.792] | 17 |
| finbert | 0.668 [0.647, 0.689] | 0.725 [0.707, 0.743] | 70 |
| qwen3-1.7b-zeroshot | 0.561 [0.534, 0.586] | 0.735 [0.717, 0.753] | 6 |
| majority | 0.264 [0.259, 0.268] | 0.656 [0.637, 0.674] | 0 |

Direction errors: bullish tweets read as bearish, or the reverse.

| Seed-averaged model | Macro-F1 mean ± sd | Accuracy mean ± sd | Direction errors |
|---|---|---|---|
| qwen3-1.7b-lora (3 seeds) | 0.884 ± 0.003 | 0.908 ± 0.002 | 9.7 ± 1.5 |

| Comparison | Δ macro-F1 [95% CI] | p (Holm) | McNemar A-only / B-only | p (Holm) |
|---|---|---|---|---|
| finbert vs majority | +0.404 [+0.381, +0.427] | < 1e-04 | 548 / 382 | 4.65e-07 |
| finbert vs qwen3-1.7b-lora-s0 | -0.219 [-0.242, -0.197] | < 1e-04 | 89 / 532 | 2.12e-76 |
| finbert vs qwen3-1.7b-lora-s1 | -0.215 [-0.238, -0.192] | < 1e-04 | 89 / 522 | 4.22e-74 |
| finbert vs qwen3-1.7b-lora-s2 | -0.214 [-0.236, -0.192] | < 1e-04 | 84 / 517 | 1.26e-75 |
| finbert vs qwen3-1.7b-zeroshot | +0.108 [+0.073, +0.142] | < 1e-04 | 414 / 438 | 0.861 |
| finbert vs qwen3.6-35b-a3b-zeroshot | -0.089 [-0.113, -0.066] | < 1e-04 | 308 / 427 | 7.75e-05 |
| finbert vs tfidf-logreg | -0.095 [-0.121, -0.069] | < 1e-04 | 216 / 456 | 1.28e-19 |
| majority vs qwen3-1.7b-lora-s0 | -0.623 [-0.639, -0.607] | < 1e-04 | 102 / 711 | 1.12e-111 |
| majority vs qwen3-1.7b-lora-s1 | -0.619 [-0.635, -0.602] | < 1e-04 | 111 / 710 | 3.2e-106 |
| majority vs qwen3-1.7b-lora-s2 | -0.618 [-0.634, -0.602] | < 1e-04 | 103 / 702 | 6.47e-109 |
| majority vs qwen3-1.7b-zeroshot | -0.297 [-0.322, -0.270] | < 1e-04 | 40 / 230 | 2.01e-32 |
| majority vs qwen3.6-35b-a3b-zeroshot | -0.493 [-0.514, -0.472] | < 1e-04 | 469 / 754 | 3.08e-15 |
| majority vs tfidf-logreg | -0.500 [-0.521, -0.477] | < 1e-04 | 180 / 586 | 1.16e-49 |
| qwen3-1.7b-lora-s0 vs qwen3-1.7b-lora-s1 | +0.005 [-0.003, +0.012] | 0.646 | 30 / 20 | 0.811 |
| qwen3-1.7b-lora-s0 vs qwen3-1.7b-lora-s2 | +0.005 [-0.002, +0.012] | 0.63 | 30 / 20 | 0.811 |
| qwen3-1.7b-lora-s0 vs qwen3-1.7b-zeroshot | +0.327 [+0.299, +0.355] | < 1e-04 | 503 / 84 | 2.25e-72 |
| qwen3-1.7b-lora-s0 vs qwen3.6-35b-a3b-zeroshot | +0.130 [+0.110, +0.149] | < 1e-04 | 421 / 97 | 7.43e-48 |
| qwen3-1.7b-lora-s0 vs tfidf-logreg | +0.124 [+0.103, +0.145] | < 1e-04 | 283 / 80 | 1.45e-26 |
| qwen3-1.7b-lora-s1 vs qwen3-1.7b-lora-s2 | +0.000 [-0.007, +0.008] | 1 | 24 / 24 | 1 |
| qwen3-1.7b-lora-s1 vs qwen3-1.7b-zeroshot | +0.322 [+0.294, +0.350] | < 1e-04 | 501 / 92 | 9.3e-68 |
| qwen3-1.7b-lora-s1 vs qwen3.6-35b-a3b-zeroshot | +0.125 [+0.106, +0.145] | < 1e-04 | 414 / 100 | 3.68e-45 |
| qwen3-1.7b-lora-s1 vs tfidf-logreg | +0.119 [+0.099, +0.140] | < 1e-04 | 280 / 87 | 1.1e-23 |
| qwen3-1.7b-lora-s2 vs qwen3-1.7b-zeroshot | +0.322 [+0.294, +0.350] | < 1e-04 | 496 / 87 | 3.63e-69 |
| qwen3-1.7b-lora-s2 vs qwen3.6-35b-a3b-zeroshot | +0.125 [+0.106, +0.144] | < 1e-04 | 418 / 104 | 2.15e-44 |
| qwen3-1.7b-lora-s2 vs tfidf-logreg | +0.119 [+0.098, +0.140] | < 1e-04 | 276 / 83 | 3.17e-24 |
| qwen3-1.7b-zeroshot vs qwen3.6-35b-a3b-zeroshot | -0.197 [-0.229, -0.166] | < 1e-04 | 436 / 531 | 0.0124 |
| qwen3-1.7b-zeroshot vs tfidf-logreg | -0.203 [-0.234, -0.172] | < 1e-04 | 204 / 420 | 3.48e-17 |
| qwen3.6-35b-a3b-zeroshot vs tfidf-logreg | -0.006 [-0.030, +0.018] | 1 | 277 / 398 | 2.55e-05 |
