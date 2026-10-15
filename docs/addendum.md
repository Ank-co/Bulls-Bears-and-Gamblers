# Addendum, does abliteration change how the model reads the market?

*Short version in the [README](../README.md).*

Part 2 shows that the abliterated model follows the client. Does it also read the news differently? Both versions of the 35B model, with the same quantization, classify the Part 1 tweets zero-shot, with the prompt and grammar of the Part 1 baseline. The hypotheses were committed before the first tweet was classified ([pre-registration](part1-abliteration-preregistration.md)). The test set had already been opened for Part 1; nothing was chosen with it.

| Pre-registered test (2,388 test tweets) | Abliterated | Base | p (Holm) | Result |
|---|---|---|---|---|
| H1. Tweets that are not bullish, read as bullish | 22.8% | 10.9% | 3 × 10⁻⁶⁷ | Confirmed |
| H2. All tweets read as bullish | 36.9% | 26.5% | 2 × 10⁻⁷¹ | Confirmed |
| H3. Macro-F1 | 0.720 | 0.788 | < 10⁻⁴ | Lower for the abliterated model (−0.069) |

- **The bias goes one way.** 229 tweets are read as bullish by the abliterated model only, 1 by the base model only.
- **It comes from neutral news.** The abliterated model reads 27% of neutral tweets as bullish, against 13% for the base model. Bearish tweets are almost never flipped (2.3% against 0.3%), and bullish tweets are found slightly more often (94% against 89%).
- **This is the optimism reported without ground truth.** [Fafuła (2026)](https://arxiv.org/abs/2607.17427) finds more optimistic stock calls after abliteration; here the right answer is known, so the optimism shows up as an error. Together with Part 2, the same edit makes the model both more optimistic about the news and more compliant with the client.
- **Quantization matters too.** With this quantization (mradermacher `i1-Q3_K_M`), the base model scores 0.788, above the 0.757 of the file used in Part 1 (unsloth `UD-Q3_K_M`). This comparison was not pre-registered and is reported as a description. The fine-tuned 1.7B model (0.884) stays well ahead of both.

## Replication

**The optimism does not replicate.** With Gemma 4 26B-A4B and Llama 3.1 8B, the abliterated model reads fewer tweets that are not bullish as bullish than its base model (17.1% against 21.2%, and 2.9% against 5.1%), significantly. Macro-F1 falls for Llama (−0.044), as here, but rises for Gemma (+0.048). The reading above, the same edit making the model more optimistic, holds for this pair of files only. Details in [replication.md](replication.md).
