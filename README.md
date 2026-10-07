# Bulls, Bears and Gamblers

**Can a small language model, fine-tuned on a laptop GPU, read market sentiment better than a model twenty times its size? And when a client sounds like a gambler, which models push them further?**

> Work in progress. Part 1 is being built, results will be published here once the test set has been scored.

## Part 1, reading the market

Classify financial news tweets as **bullish**, **bearish** or **neutral** ([Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment), MIT license).

| # | Model | Question it answers |
|---|---|---|
| 1 | Majority class | What is the floor? |
| 2 | TF-IDF + logistic regression | Is a language model needed at all? |
| 3 | FinBERT, off the shelf | How does an existing finance specialist do? |
| 4 | Qwen3-1.7B, zero-shot | What does fine-tuning add? |
| 5 | Qwen3.6-35B-A3B, zero-shot | Does a much larger generalist win? (35B parameters in total, about 3B active per token) |
| 6 | **Qwen3-1.7B + LoRA, 3 seeds** | The contribution |

### Protocol

- **Frozen splits.** The official validation file (2,388 tweets) is the test set, scored once at the end. A stratified 10% of the training file is the dev set, used for every choice. Every split is fingerprinted (`data/splits/manifest.json`).
- **Leakage removed.** 69 training tweets also appear in the test set once case, links and spacing are ignored (2.9% of the test set). They are removed from training, along with 72 duplicates inside the training file. The test set itself is never modified.
- **Constrained outputs.** Every model answers with one of the three labels. The language models score the three possible answers instead of generating free text, and the 35B model is restricted by a grammar. No free text is parsed.
- **Facts are computed by code.** All models write predictions in one format. A single script reads the test labels, refuses any file produced on other data or missing a single example, and computes the metrics.
- **Statistics.** Macro-F1 (the classes are imbalanced) with bootstrap 95% confidence intervals, exact McNemar tests with Holm correction, and paired bootstrap intervals on the macro-F1 gap between models.
- **Direction errors.** For a trading signal, reading a bullish tweet as bearish (or the reverse) costs more than missing a signal. Every model is also scored on the number of such swaps.
- **Fine-tuning protocol.** LoRA on Qwen3-1.7B, trained by minimizing minus the log-probability of the gold answer, computed by the same function that scores answers at evaluation. Learning rate and class weighting are chosen on the dev set by code, then the chosen configuration is trained with three seeds (which also control the adapter initialization).
- **Variance.** The fine-tuned model is reported as mean and standard deviation over the three seeds.
- **Cost.** Latency per tweet and peak VRAM, measured on the same laptop (RTX 5070 Laptop, 8 GB).

## Part 2, advising the client (pre-registered, in progress)

Does a model adapt its advice to the client's risk profile, as suitability rules require from a human adviser? Both models receive the same 48 scenarios (4 risk profiles, 2 financial situations, an open question or a client pushing for confirmation, 3 phrasings), 5 times each, and must answer in a JSON schema: share of savings recommended for high-risk products, and whether to use leverage. The comparison is a base model against an *abliterated* version of the same model (refusal behaviour removed), quantized identically. Hypotheses, tests and commitments were written before any answer was collected: [docs/part2-preregistration.md](docs/part2-preregistration.md).

## Reproduce

Windows 11 + WSL2 Ubuntu, see [docs/setup-wsl.md](docs/setup-wsl.md).

```bash
bash setup/setup_wsl.sh               # venv, PyTorch for CUDA 12.8, tests, GPU check
source ~/.venvs/bbg/bin/activate
python scripts/prepare_data.py        # download and freeze the splits
python scripts/baselines_classic.py   # baselines 1 and 2, selected on the dev set
python scripts/finbert.py             # baseline 3
python scripts/zero_shot_qwen.py      # baseline 4
python scripts/zero_shot_llamacpp.py  # baseline 5, needs a llama.cpp server (docs/setup-wsl.md)
python scripts/run_lora.py            # model 6: selection on dev, 3 seeds, predictions
python scripts/evaluate.py --split val

python scripts/run_advice.py          # part 2: 480 answers from two llama.cpp models
python scripts/analyze_advice.py      # part 2: pre-registered tests
```

Every model writes `results/predictions/<split>/<model>.csv`; `scripts/evaluate.py` is the only
place where scores are computed. Each evaluation of the test split is logged in
`results/metrics/test_access.log`.

## License

Code under the MIT license. The dataset belongs to its authors (MIT license).
