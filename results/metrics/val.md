## val split (n = 941)

| Model | Macro-F1 [95% CI] | Accuracy [95% CI] | Direction errors |
|---|---|---|---|
| qwen3-1.7b-lora-s0 | 0.886 [0.861, 0.908] | 0.910 [0.892, 0.928] | 6 |
| qwen3-1.7b-lora-s2 | 0.883 [0.859, 0.906] | 0.905 [0.886, 0.923] | 3 |
| qwen3-1.7b-lora-s1 | 0.880 [0.855, 0.903] | 0.905 [0.886, 0.923] | 7 |
| tfidf-logreg | 0.758 [0.724, 0.790] | 0.819 [0.795, 0.844] | 27 |
| qwen3.6-35b-a3b-zeroshot | 0.750 [0.719, 0.779] | 0.772 [0.744, 0.798] | 8 |
| finbert | 0.648 [0.613, 0.680] | 0.699 [0.670, 0.728] | 34 |
| qwen3-1.7b-zeroshot | 0.543 [0.501, 0.583] | 0.722 [0.693, 0.750] | 3 |
| majority | 0.262 [0.254, 0.269] | 0.646 [0.616, 0.677] | 0 |

Direction errors: bullish tweets read as bearish, or the reverse.

| Seed-averaged model | Macro-F1 mean ± sd | Accuracy mean ± sd | Direction errors |
|---|---|---|---|
| qwen3-1.7b-lora (3 seeds) | 0.883 ± 0.003 | 0.907 ± 0.002 | 5.3 ± 2.1 |

| Comparison | Δ macro-F1 [95% CI] | p (Holm) | McNemar A-only / B-only | p (Holm) |
|---|---|---|---|---|
| finbert vs majority | +0.386 [+0.348, +0.421] | < 1e-04 | 220 / 170 | 0.078 |
| finbert vs qwen3-1.7b-lora-s0 | -0.238 [-0.275, -0.203] | < 1e-04 | 30 / 228 | 1.81e-37 |
| finbert vs qwen3-1.7b-lora-s1 | -0.232 [-0.268, -0.196] | < 1e-04 | 29 / 223 | 7.05e-37 |
| finbert vs qwen3-1.7b-lora-s2 | -0.236 [-0.271, -0.201] | < 1e-04 | 30 / 224 | 1.62e-36 |
| finbert vs qwen3-1.7b-zeroshot | +0.105 [+0.051, +0.158] | 0.001 | 169 / 190 | 1 |
| finbert vs qwen3.6-35b-a3b-zeroshot | -0.102 [-0.141, -0.064] | < 1e-04 | 119 / 187 | 0.000966 |
| finbert vs tfidf-logreg | -0.111 [-0.151, -0.071] | < 1e-04 | 79 / 192 | 6.54e-11 |
| majority vs qwen3-1.7b-lora-s0 | -0.624 [-0.648, -0.597] | < 1e-04 | 41 / 289 | 1.21e-45 |
| majority vs qwen3-1.7b-lora-s1 | -0.618 [-0.643, -0.591] | < 1e-04 | 39 / 283 | 2.09e-45 |
| majority vs qwen3-1.7b-lora-s2 | -0.622 [-0.646, -0.595] | < 1e-04 | 41 / 285 | 1.06e-44 |
| majority vs qwen3-1.7b-zeroshot | -0.281 [-0.321, -0.239] | < 1e-04 | 16 / 87 | 7.55e-12 |
| majority vs qwen3.6-35b-a3b-zeroshot | -0.488 [-0.521, -0.453] | < 1e-04 | 170 / 288 | 3.49e-07 |
| majority vs tfidf-logreg | -0.496 [-0.529, -0.461] | < 1e-04 | 66 / 229 | 5.61e-21 |
| qwen3-1.7b-lora-s0 vs qwen3-1.7b-lora-s1 | +0.006 [-0.006, +0.018] | 1 | 13 / 9 | 1 |
| qwen3-1.7b-lora-s0 vs qwen3-1.7b-lora-s2 | +0.003 [-0.010, +0.015] | 1 | 14 / 10 | 1 |
| qwen3-1.7b-lora-s0 vs qwen3-1.7b-zeroshot | +0.343 [+0.299, +0.388] | < 1e-04 | 211 / 34 | 4.84e-31 |
| qwen3-1.7b-lora-s0 vs qwen3.6-35b-a3b-zeroshot | +0.136 [+0.107, +0.166] | < 1e-04 | 160 / 30 | 2.3e-21 |
| qwen3-1.7b-lora-s0 vs tfidf-logreg | +0.128 [+0.096, +0.161] | < 1e-04 | 114 / 29 | 6.14e-12 |
| qwen3-1.7b-lora-s1 vs qwen3-1.7b-lora-s2 | -0.004 [-0.017, +0.009] | 1 | 13 / 13 | 1 |
| qwen3-1.7b-lora-s1 vs qwen3-1.7b-zeroshot | +0.337 [+0.292, +0.382] | < 1e-04 | 207 / 34 | 4.06e-30 |
| qwen3-1.7b-lora-s1 vs qwen3.6-35b-a3b-zeroshot | +0.130 [+0.100, +0.160] | < 1e-04 | 157 / 31 | 3.01e-20 |
| qwen3-1.7b-lora-s1 vs tfidf-logreg | +0.122 [+0.089, +0.155] | < 1e-04 | 114 / 33 | 1.47e-10 |
| qwen3-1.7b-lora-s2 vs qwen3-1.7b-zeroshot | +0.341 [+0.296, +0.385] | < 1e-04 | 207 / 34 | 4.06e-30 |
| qwen3-1.7b-lora-s2 vs qwen3.6-35b-a3b-zeroshot | +0.133 [+0.104, +0.163] | < 1e-04 | 157 / 31 | 3.01e-20 |
| qwen3-1.7b-lora-s2 vs tfidf-logreg | +0.125 [+0.094, +0.159] | < 1e-04 | 115 / 34 | 1.91e-10 |
| qwen3-1.7b-zeroshot vs qwen3.6-35b-a3b-zeroshot | -0.207 [-0.256, -0.158] | < 1e-04 | 157 / 204 | 0.078 |
| qwen3-1.7b-zeroshot vs tfidf-logreg | -0.215 [-0.265, -0.164] | < 1e-04 | 78 / 170 | 5.08e-08 |
| qwen3.6-35b-a3b-zeroshot vs tfidf-logreg | -0.008 [-0.048, +0.032] | 1 | 111 / 156 | 0.0488 |
