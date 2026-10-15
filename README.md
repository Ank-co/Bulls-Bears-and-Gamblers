# Bulls, Bears and Gamblers

**Four questions about language models in markets, answered with experiments on a single laptop GPU (RTX 5070 Laptop, 8 GB).**

Each experiment fixes its protocol before the test data are opened (in Part 4 it was committed only after the run, a deviation reported on its page), opens the test data once, and computes every score with code from what the models answer. No model grades another model's text.

**Full write-up, one page per part:** [Part 1](docs/part1.md) · [Part 2](docs/part2.md) · [Addendum](docs/addendum.md) · [Part 3](docs/part3.md) · [Part 4](docs/part4.md) · [Replication](docs/replication.md) · [Related work](docs/related-work.md). The pre-registrations and design notes are in [docs](docs/).

## At a glance

| | Question | Main result | Tests fixed in advance |
|---|---|---|---|
| [Part 1](docs/part1.md) | Can a small model fine-tuned on the laptop read market sentiment better than a model twenty times its size? | Yes: macro-F1 **0.884** for Qwen3-1.7B with LoRA, against 0.757 for the 35B model zero-shot | Protocol committed before the test set was opened |
| [Part 2](docs/part2.md) | When a client sounds like a gambler, does the model protect them, and what changes once its refusals are removed? | The base model protects; the abliterated model follows the client (**+43 points** of savings in high-risk products for gamblers and loss-chasers), also with Gemma 4 and Llama 3.1 | 3 of 4 confirmed |
| [Addendum](docs/addendum.md) | Does the same edit change how the model reads the news? | Tweets that are not bullish read as bullish: **23%** against 11%, reversed with Gemma 4 and Llama 3.1 | 3 of 3 confirmed |
| [Part 3](docs/part3.md) | Does a model's reading of real headlines move with stock prices? | The fine-tuned model lines up best with the move around the news; what is left after it cannot be separated from memory | 2 of 3 confirmed, the third not separable from look-ahead |
| [Part 4](docs/part4.md) | In an invented market where the best answer is computed, do models weigh each piece of news by what it is worth? | News used at **half** its worth, and at a quarter when its track record contradicts the usual reading | 5 of 5 confirmed, pre-registration committed after the run |
| [Replication](docs/replication.md) | Do the results of Part 2, the addendum and Part 4 hold in two more model families? | Compliance with the client and base-rate underweighting hold in all three families; the optimism of the abliterated model does not replicate | Pre-registered and pushed before the runs: 4 of 11 findings replicate in both new families, 3 in one, 4 in neither |

## Part 1, reading the market

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/part1-dark.svg">
  <img alt="Macro-F1 and direction errors of each model on the test set" src="docs/figures/part1-light.svg" width="100%">
</picture>

Classify 2,388 held-out financial tweets as bullish, bearish or neutral. Qwen3-1.7B, fine-tuned with LoRA in 22 minutes, reaches a macro-F1 of **0.884 ± 0.003** over three seeds, against 0.764 for TF-IDF and 0.757 for a 35B model used zero-shot (p < 10⁻⁴ for every comparison). It confuses bullish and bearish tweets 7 times less often than TF-IDF or FinBERT, and runs in 58 ms per tweet against 638 ms for the 35B model. [Protocol, table and details](docs/part1.md)

## Part 2, advising the client

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/part2-dark.svg">
  <img alt="Mean share of savings recommended in high-risk products, by client profile, for the base and abliterated models" src="docs/figures/part2-light.svg" width="100%">
</picture>

Two versions of Qwen3.6-35B-A3B advise the same 48 client scenarios on €20,000 of savings: the base model, and a copy whose refusal behaviour was removed (abliteration). The base model is a strict adviser: 0% in high-risk products in 76% of its answers. The abliterated model recommends **43 points more** at risk to gamblers and loss-chasers, moves **24 points** when the client pushes for confirmation (the base model does not move), and recommends leverage to 43% of fragile clients chasing losses. In the example fixed before the campaign, its advice reads as measured while putting 60% of a fragile client's savings in leveraged products, which is why the study scores the numbers a model commits to, never its text. In the [replication](docs/replication.md), the abliterated Gemma 4 and Llama 3.1 also give gamblers riskier advice (+30 and +8 points), but no model of those families recommends leverage to fragile clients. [Protocol, tests and examples](docs/part2.md)

**Addendum.** The same two files classify the Part 1 tweets. The abliterated model reads 23% of the tweets that are not bullish as bullish, against 11%: 229 tweets flip to bullish, 1 the other way. **This does not replicate:** with Gemma 4 and Llama 3.1, the abliterated model reads fewer tweets as bullish than its base model (17% against 21%, 2.9% against 5.1%). [Details](docs/addendum.md)

## Part 3, does the reading move with prices?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/part3-dark.svg">
  <img alt="Daily rank correlation between each model's reading of the headlines and the stock returns, around the news and after it" src="docs/figures/part3-light.svg" width="100%">
</picture>

The Part 1 readers label 246,764 headlines of 2018 and 2019 (150,000 stock-days). The fine-tuned model's reading lines up with the price move around the news more than every other reader (daily rank correlation **0.198**, against 0.133 to 0.155). After the news is public, a small effect remains (15 bps a day between bullish and bearish stocks, almost all on bad news). But a memory control fixed in advance, a Qwen model asked with the ticker and the dates only, no headline, is itself linked to that move, so under the rule set before the test, the effect cannot be attributed to reading the news rather than to remembered prices. [Protocol, controls and details](docs/part3.md)

## Part 4, weighing the news in an invented market

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/part4-dark.svg">
  <img alt="Weights each model gives to the base rate, to news, to worthless news and to dramatic wording, against the ideal observer, and calibration of both models" src="docs/figures/part4-light.svg" width="100%">
</picture>

In 4,000 scenarios of an invented market, with nothing to remember, the best possible answer is computed by code from the base rate and each news item's track record. The base model gives the news about **half** the weight it deserves (0.51, ideal 1) and the base rate more (0.84), and news worth nothing still moves it. When a track record says that news usually read as bad came before rises, it follows that record at **a quarter** of the usual weight (0.13). The abliterated model answers **7.9 points** higher on average and moves less with the evidence. In the replication, every base model underweights the base rate, but the higher answers of the abliterated model do not replicate (lower with Gemma 4, unchanged with Llama 3.1). The pre-registration was placed in the repository before the test scenarios were generated but committed only with the results, a deviation reported on the part's page. [Design, tests and deviation](docs/part4.md)

## Replication: what holds across families

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/replication-dark.svg">
  <img alt="Effects of abliteration on advice, tweet reading and answers in the invented market, and the base models' weights on the base rate and the news, for Qwen3.6-35B-A3B, Gemma 4 26B-A4B and Llama 3.1 8B" src="docs/figures/replication-light.svg" width="100%">
</picture>

The results of Part 2, the addendum and Part 4 came from one pair of files. The same experiments, inputs and tests were run on two more families, each a base model and its abliterated copy from the same quantizer: **Gemma 4 26B-A4B** (abliterated by the same author as the Qwen pair, so only the family changes) and **Llama 3.1 8B** (abliterated by another author). The [pre-registration](docs/replication-preregistration.md) was pushed before any test run, and the analysis ran once.

| Finding with Qwen | Gemma 4 26B-A4B | Llama 3.1 8B |
|---|---|---|
| The abliterated model gives gamblers riskier advice | **yes**, +30.5 points | **yes**, +8.2 points |
| It moves more when the client pushes for confirmation | **yes**, +11.9 points | not detected, +2.6 points |
| It recommends leverage to fragile clients chasing losses | no, 0% against 0% | no, 0% against 0% |
| It reads tweets that are not bullish as bullish more often | **reversed**, 17.1% against 21.2% | **reversed**, 2.9% against 5.1% |
| It answers higher in the invented market | **reversed**, 49.4% against 52.2% | no, 58.8% against 58.9% |
| It moves less with the evidence (flatter slope) | yes, small | yes, small |
| The base model underweights the base rate | **yes**, weight 0.41 | **yes**, weight 0.84 |
| It follows a reversed track record less | **yes** | yes, small |
| Worthless news moves it | no effect of interest | yes, small |

*Yes: rejected after Holm's correction in the pre-registered direction. Small: below the smallest effect of interest fixed in advance. Reversed: the difference goes the other way. Not detected or no: not rejected, which does not show that the effect is absent.*

**Abliteration's effect on advice generalizes, its effect on reading the market does not.** In all three families the abliterated model follows a client who sounds like a gambler. The optimism found with Qwen does not replicate: on labelled tweets it reverses with both new families, and in the invented market Gemma's abliterated model answers lower and Llama's the same. In this sample, the optimism belongs to the Qwen pair, not to abliteration in general. All three base models underweight the base rate, but how much they use the news differs widely, from Gemma, which follows the news more than the base rate, to Llama 3.1 8B, which hardly uses it. [Full results, criteria and power](docs/replication.md)

## Limits

- Parts 2 to 4 use Qwen3.6-35B-A3B; the replication adds Gemma 4 26B-A4B and Llama 3.1 8B. Three families in all, one quantization each, abliterations made by third parties (two by the same author).
- English only, thinking disabled, answers in JSON, single-turn questions.
- Part 3 cannot rule out that the Qwen models remember the prices of the test years. Part 4 is synthetic by design: it measures how models use stated information, not whether they predict real markets.
- Each part's page lists its own limits.

## Related work

The closest studies are [Fafuła (2026)](https://arxiv.org/abs/2607.17427), who finds abliterated models more optimistic in their own stock calls (the replication here finds that optimism with Qwen only), [Lopez-Lira and Tang (2023)](https://arxiv.org/abs/2304.07619) on headlines and returns, and the belief-updating literature reviewed by [Benjamin (2019)](https://www.nber.org/papers/w25200). The full list, with how each study differs from this one, is in [docs/related-work.md](docs/related-work.md).

## Reproduce

Windows 11 + WSL2 Ubuntu, see [docs/setup-wsl.md](docs/setup-wsl.md).

```bash
bash setup/setup_wsl.sh               # venv, PyTorch for CUDA 12.8, tests, GPU check
source ~/.venvs/bbg/bin/activate
python scripts/prepare_data.py        # download and freeze the splits
python scripts/baselines_classic.py   # majority class, TF-IDF + logistic regression
python scripts/finbert.py
python scripts/zero_shot_qwen.py
python scripts/zero_shot_llamacpp.py  # needs a llama.cpp server (docs/setup-wsl.md)
python scripts/run_lora.py            # selection on dev, 3 seeds, predictions
python scripts/evaluate.py --split val

python scripts/run_advice.py          # part 2: 480 answers from two llama.cpp models
python scripts/analyze_advice.py      # part 2: pre-registered tests
LLAMA_MODEL=qwen3.6-35b-base-i1 python scripts/zero_shot_llamacpp.py --name qwen3.6-35b-i1-zeroshot
LLAMA_MODEL=qwen3.6-35b-abliterated python scripts/zero_shot_llamacpp.py --name qwen3.6-35b-abliterated-zeroshot
python scripts/analyze_abliteration_sentiment.py   # addendum: pre-registered tests

# part 3: FNSPID headlines and prices, downloaded outside the repository (docs/part3-preregistration.md)
python scripts/part3_prepare.py --period test --news ~/fnspid/Stock_news/All_external.csv --prices ~/fnspid/Stock_price/full_history.zip
python scripts/part3_classify.py --period test --model all
python scripts/part3_probe.py --period test       # control C1
python scripts/part3_coverage.py --period test    # control C2
python scripts/part3_analyze.py --period test     # pre-registered tests
python scripts/part3_explore.py --period test     # exploratory, after the results

# part 4: invented market (docs/part4-preregistration.md)
python scripts/part4_generate.py --split dev
python scripts/part4_run.py --split dev        # both llama.cpp models
python scripts/part4_tone.py --split dev       # tone reader, mapping frozen on dev
python scripts/part4_dev_report.py
python scripts/part4_power.py                  # sizing, measurement bias, smallest effects
python scripts/part4_generate.py --split test  # refused until the pre-registration exists
python scripts/part4_run.py --split test
python scripts/part4_tone.py --split test
python scripts/part4_analyze.py                # pre-registered tests, run once
python scripts/part4_explore.py                # exploratory, after the results

# replication on two more families (docs/replication-preregistration.md)
python scripts/replication_power.py            # power and measurement bias, dev answers only
python scripts/replication_checks.py           # format checks on items outside the test
python scripts/replication_run.py --pair llama # refused until the pre-registration is pushed
python scripts/replication_run.py --pair gemma
python scripts/replication_analyze.py          # run once
python scripts/make_figures.py        # README figures
```

Every model writes `results/predictions/<split>/<model>.csv`; `scripts/evaluate.py` is the only place where scores are computed. Each evaluation of the test split is logged in `results/metrics/test_access.log`.

## License

Code under the MIT license. The Part 1 dataset belongs to its authors (MIT license). FNSPID belongs to its authors (CC BY-NC 4.0); its headlines and prices are not redistributed here, only hashed headline ids, labels and results. Models belong to their respective authors.
