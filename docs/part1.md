# Part 1, reading the market

*Short version in the [README](../README.md).*

This part builds the tools of the later ones: the readers whose reading of real headlines Part 3 links to prices, and the labelled benchmark the addendum uses. A fine-tuned model beating models used zero-shot is expected; what this part measures is how large the gap is and which errors each reader makes.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/part1-dark.svg">
  <img alt="Macro-F1 and direction errors of each model on the test set" src="figures/part1-light.svg" width="100%">
</picture>

Classify financial news tweets as **bullish**, **bearish** or **neutral** ([Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment), MIT license). Test set: the official validation file, 2,388 tweets, scored once.

| Model | Macro-F1 [95% CI] | Accuracy | Direction errors | Time per tweet |
|---|---|---|---|---|
| **Qwen3-1.7B + LoRA (3 seeds)** | **0.884 ± 0.003** | **0.908** | **9.7** | 58 ms (GPU) |
| TF-IDF + logistic regression | 0.764 [0.743, 0.784] | 0.826 | 69 | 0.1 ms (CPU) |
| Qwen3.6-35B-A3B, zero-shot | 0.757 [0.738, 0.776] | 0.775 | 17 | 638 ms (GPU + CPU) |
| FinBERT, off the shelf | 0.668 [0.647, 0.689] | 0.725 | 70 | 1.4 ms (GPU) |
| Qwen3-1.7B, zero-shot | 0.561 [0.534, 0.586] | 0.735 | 6 | 39 ms (GPU) |
| Majority class | 0.264 [0.259, 0.268] | 0.656 | 0 | |

*Direction errors: bullish tweets read as bearish, or the reverse. For a trading signal, this is the costly mistake.*

- **Every seed beats every baseline.** Macro-F1 gap of +0.119 to +0.124 over TF-IDF and +0.125 to +0.130 over the 35B model, p (Holm) < 10⁻⁴, paired bootstrap and exact McNemar tests on the same tweets.
- **No sign of overfitting the model selection.** Dev 0.883, test 0.884.
- **The 35B model is only 3B active parameters per token** (MoE). It is 20 times larger in memory and stored knowledge, not in computation.
- **The baselines fail in opposite ways.** The small zero-shot model hides in "neutral" (it finds 23% of bullish tweets), while the 35B model and FinBERT see sentiment where there is none (they label 30% and 24% of neutral tweets as bullish or bearish).

**Protocol**

- **Frozen splits, leakage removed.** 69 training tweets also appear in the test set once case, links and spacing are ignored (2.9% of the test set). They are removed from training, along with 72 duplicates. The test set is never modified. Every split is fingerprinted (`data/splits/manifest.json`).
- **Facts are computed by code.** Models only write predictions. A single script joins them with the gold labels, refuses any incomplete file, and computes every metric. The language models score the three possible answers instead of generating free text; the 35B model is constrained by a grammar.
- **Fine-tuning.** LoRA (r = 16) on all linear layers, trained by minimizing minus the log-probability of the gold answer, computed by the same function that scores answers at evaluation. Learning rate and class weighting are chosen on the dev set by code (four runs), then the chosen configuration is trained with three seeds, which also control the adapter initialization.
- **The test set was opened once, after the protocol was committed** (see the git history: `Protocol frozen before test evaluation`, then `Test set evaluated once`).

  

## Limits

One dataset of tweets without timestamps or prices; Part 3 links the same models to returns on another dataset. The fine-tuned model was still improving after two epochs; the protocol was kept as committed rather than tuned further.
