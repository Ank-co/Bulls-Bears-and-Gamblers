# Related work

*Short version in the [README](../README.md).*

- [FinLoRA (2025)](https://arxiv.org/html/2505.19819v1) benchmarks LoRA on the same dataset and test file. Their best fine-tuned model, Llama 3.1 8B, reports 88.0% accuracy; this 1.7B model reaches 90.8%. The comparison is indicative only: training leakage was removed here, and F1 is not computed the same way.
- [Ross and Lo (2026)](https://arxiv.org/pdf/2604.23837) generate synthetic client profiles and collect allocations in JSON from GPT models, and find that advice collapses onto stated risk tolerance. Part 2 uses a similar design and adds client pressure and abliteration.
- [Cho et al. (2026)](https://arxiv.org/abs/2603.09303) measure the risk profiles of language models with personas.
- [Zhao et al. (2026)](https://arxiv.org/pdf/2604.24668) measure sycophancy in professional financial tasks, not retail advice.
- [Fafuła (2026), "Abliteration Is Not a Scalpel"](https://arxiv.org/abs/2607.17427) is the closest work: Huihui abliterations of Qwen3-30B-A3B and Gemma make more optimistic weekly calls on stocks than their base models (+7.4 and +12.2 points of upside calls), with instruction-following preserved. It studies the model's own market calls, not advice to a client; the addendum measures the same optimism on labelled news, and Part 4 against a computed answer.
- [Lopez-Lira and Tang (2023)](https://arxiv.org/abs/2304.07619) score headlines with GPT models and find returns that keep drifting after the news is public, stronger for bad news and for small firms. Part 3 finds the same asymmetry on bad news, with small models on a laptop, and a memory control.
- [Glasserman and Lin (2023)](https://arxiv.org/abs/2309.17322) study how look-ahead in pretrained models could inflate return predictions, and test for it by removing company names from the headlines.
- [Gao, Jiang and Yan (2026)](https://arxiv.org/abs/2512.23847v2), found after our results, use the same kind of query as C1: the company, its ticker and a date, with no news. With Llama-3.3-70B they measure how much probability the model puts on "up" or "down" rather than "unknown", find it substantial before the training cutoff and close to zero after, and show that headline forecasts predict better where it is high, before the cutoff only. C1 differs in three ways: a 1.7B model, no "unknown" option, and the probe's direction tested on its own against the returns.
- Grether (1980, *Quarterly Journal of Economics*) introduced the belief-updating regression of Part 4; [Benjamin (2019)](https://www.nber.org/papers/w25200) reviews what it has found in people, who also weigh base rates and evidence below what Bayes' rule asks.
- [Gupta et al. (2025)](https://arxiv.org/abs/2503.04722) test language models on biased coin flips: they broadly update like Bayesians from miscalibrated priors, and evidence in the prompt can outweigh an explicit instruction about the bias.
- [Ilanyan and Jain (2026)](https://iclr.cc/virtual/2026/10018243) find base-rate neglect and updates weakly linked to the evidence in a language model playing poker.
- [Yee and Koh (2026)](https://arxiv.org/abs/2602.01022) measure behavioural biases of language models on synthetic financial scenarios, scored against answers set by the authors rather than computed from a stated process.
- [Sharma et al. (2024)](https://arxiv.org/abs/2310.13548) document sycophancy in assistants trained from human feedback.
- [What does abliteration actually cost](https://www.greaterwrong.com/posts/ipAXsLjkyqC6s7Cin/what-does-abliteration-actually-cost) measures the cost of a Huihui abliterated Qwen on MMLU and TruthfulQA.

To our knowledge, no published study combines investment suitability, client pressure and abliteration.
