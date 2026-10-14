# Replication pre-registration: Part 2, the addendum and Part 4 on two more model families

Status: **frozen when committed and pushed**, before any test run of the new models. The script of the
test runs (`scripts/replication_run.py`) and the analysis (`scripts/replication_analyze.py`) refuse
to start until this file is committed, unchanged since its commit, and contained in `origin/main`:
the push is the public time stamp. Any change after the push goes to "Deviations" at the end.

## Question

On one pair of models, Qwen3.6-35B-A3B and its abliterated copy by the same quantizer, the project
found that abliteration makes the model follow the client (Part 2), read the news as more bullish
(addendum) and answer higher while moving less with the evidence (Part 4), and that the base model
weighs news at about half its worth and reversed track records at about a quarter (Part 4). Do these
results hold in other model families?

## Models

Each new pair is a base model and its abliterated copy, quantized by the same quantizer (mradermacher,
imatrix `i1-Q4_K_M`) with the same method. The original pair used `i1-Q3_K_M` because of its size.

| Pair | Role | Server name | File (Hugging Face repository) | SHA-256 | Repository commit | Made from |
|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B | base | `gemma-4-26b-a4b-base-i1` | `gemma-4-26B-A4B-it.i1-Q4_K_M.gguf` (mradermacher/gemma-4-26B-A4B-it-i1-GGUF) | `01f2546777b7c3f4ab5f5c0c84e1cb6b7a213739d156edcb25e342afada22a59` | `2758390` | google/gemma-4-26B-A4B-it |
| Gemma 4 26B-A4B | abliterated | `gemma-4-26b-a4b-abliterated` | `Huihui-gemma-4-26B-A4B-it-abliterated.i1-Q4_K_M.gguf` (mradermacher/Huihui-gemma-4-26B-A4B-it-abliterated-i1-GGUF) | `0f9077c9c7b40f1b056b89a48422017717190618e3136d2d3c20c758e3603676` | `00f6327` | huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated |
| Llama 3.1 8B | base | `llama-3.1-8b-base-i1` | `Meta-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf` (mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF) | `85465a58bf902b8fe36e1e719fd1ddc23305110a6b19233f10560dfd1180f77a` | `15a6b2b` | meta-llama/Llama-3.1-8B-Instruct |
| Llama 3.1 8B | abliterated | `llama-3.1-8b-abliterated` | `Meta-Llama-3.1-8B-Instruct-abliterated.i1-Q4_K_M.gguf` (mradermacher/Meta-Llama-3.1-8B-Instruct-abliterated-i1-GGUF) | `74f27fbf0877b4d76d9708efab5fc6dceb6a8bc1ed2394397a020f6c71d57c9a` | `b659b6c` | mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated |

The SHA-256 come from the Hugging Face file listings and were checked locally on 2026-10-14 (all four
match). Full commits and the same data are in `src/bbg/replication.py` (`FILES`).

**Server.** llama.cpp build b10786 (commit `de8656bd9`, released 2026-09-03), router mode, one model
in memory at a time; for the new models: context 8,192 tokens, flash attention on, KV cache in
`q8_0`, one slot, the chat template embedded in each GGUF file (`--jinja`), no multimodal projector.
Gemma's embedded template predates Google's July 2026 template update (tool calls, turn closures);
both files of the Gemma pair carry the same version, and the requests here use neither tools nor
several turns.

**Why these pairs.** Gemma 4 26B-A4B changes only the family: its abliteration has the same author as
the original pair's (Huihui), and it is also a mixture of experts with about 4B active parameters
(Qwen: 3B). Llama 3.1 8B changes the family and the abliteration recipe (mlabonne). If Llama does not
replicate, family and recipe cannot be told apart.

## What is replicated, unchanged

- **Inputs.** Part 2: the 48 scenarios x 5 seeds (`bbg.advice.build_scenarios`). Addendum: the 2,388
  tweets of the Part 1 test split. Part 4: the 4,000 frozen test scenarios (`data/part4/test/scenarios.jsonl`,
  seed 42), the first 500 of them in the history version, and the 120 reading prompts.
- **Requests.** The original code: `bbg.advice`, `bbg.llamacpp` and `bbg.part4_llm`, with the same
  system prompts, schemas and grammar. Sampling: Part 2 temperature 0.7, top_p 0.8, top_k 20, seeds
  0 to 4 (the setting Qwen recommends, kept for every family so that only the model changes);
  addendum temperature 0 with the grammar; Part 4 temperature 0, top_k 1, seed 0, JSON schema, at most
  32 tokens. Thinking disabled (`enable_thinking: false`; Gemma 4's template honours it, Llama 3.1
  has no thinking mode).
- **Extraction rules.** The original validation: `parse_advice`, `parse_answer` and
  `part4_llm.parse`. Part 2 and the addendum stop on an invalid answer, as in the original campaigns;
  Part 4 records each failure with its reason.
- **Same items, same conditions.** Every cache carries a fingerprint of model, prompts, schema and
  sampling. The analysis refuses a cache whose fingerprint differs from the one the code computes,
  and a pair whose two models did not answer exactly the same items.

## Hypotheses and tests

All the original hypotheses are kept, including the one the original pair did not confirm, with the
original tests, directions and corrections (`docs/part2-preregistration.md`,
`docs/part1-abliteration-preregistration.md`, `docs/part4-preregistration.md`).

| Experiment | Hypothesis | Test (original code) | Original pair |
|---|---|---|---|
| Part 2 | H1 gamblers and loss-chasers get riskier advice from the abliterated model | two-sided Wilcoxon, 24 scenario means | confirmed |
| Part 2 | H2 a fragile situation lowers the advice less for the abliterated model | two-sided Wilcoxon, 24 units | not confirmed |
| Part 2 | H3 a client pushing for confirmation moves the abliterated model more | two-sided Wilcoxon, 24 units | confirmed |
| Part 2 | H4 more leverage recommended to fragile clients chasing losses | exact McNemar, 30 paired answers | confirmed |
| Addendum | H1 tweets that are not bullish read as bullish more often | exact McNemar | confirmed |
| Addendum | H2 all tweets read as bullish more often | exact McNemar | confirmed |
| Addendum | H3 macro-F1 differs | paired bootstrap, two-sided | confirmed (lower) |
| Part 4 | H1 the base model underweights the base rate | weight minus 1, robust errors | confirmed |
| Part 4 | H2 worthless news moves the base model | weight, robust errors | confirmed |
| Part 4 | H3 a reversed track record is followed less | difference of weights, robust errors | confirmed |
| Part 4 | H4 the abliterated model answers higher | shift, stacked, clustered by scenario | confirmed |
| Part 4 | H5 the abliterated model's slope on the ideal answer differs | slope, two-sided, clustered by scenario | confirmed (flatter) |

Holm's correction runs within each experiment and each pair, exactly as in the original (H1 to H4 in
Part 2, H1 to H3 in the addendum, H1 to H5 in Part 4). There is no correction across the two pairs:
each pair is an independent replication with its own test. In Part 4, each pair's null values are
moved by its own measurement bias (below), the manipulation check is run (if it fails, H1 to H3 are
reported as uninterpretable for that pair), and the original sensitivity analyses are rerun (the base
model's own reading of the news, the 12 clear kinds, and failures replaced by 50 and by the base rate
if any failure occurs), through the functions of `scripts/part4_analyze.py`.

## Replication criteria

- **Criterion A, statistical replication:** the hypothesis is rejected after Holm's correction, in
  the pre-registered direction. For the two-sided tests (Part 4 H5, addendum H3), the estimate must
  also have the sign the original pair found (a flatter slope; a lower macro-F1). In Part 2 the
  direction is read on the mean difference (H1 to H3) and on the difference of rates (H4).
- **Criterion B, substantive replication (Part 4 only):** criterion A is met and the estimated effect
  reaches the smallest effect of interest. These are the Part 4 values, set by the generator and the
  same for every model: H1 0.118, H2 0.075, H3 0.110, H4 0.100, H5 0.063. Part 2 and the addendum
  have no smallest effect of interest fixed in advance, so only criterion A applies to them.
- **Part 2 H2**, not confirmed in the original pair, has no effect to replicate: its result in a new
  family is reported as a test of the hypothesis in that family, not as a replication.

## Reading the results

Each hypothesis is classified over the two new pairs, separately for criterion A and criterion B:

- **2 of 2:** the result holds in the three families tested.
- **1 of 2:** the result depends on the family.
- **0 of 2:** neither replication met the fixed criterion. This does not show that the effect is
  specific to Qwen.

**A test that is not rejected is not evidence that the effect is absent.** It is reported with its
estimate and interval and read as "not detected at this power". In Part 4, the original reading
rules also apply to each pair (`bbg.part4_stats.interpret`): "no effect of interest" requires the
upper end of the effect's 90% interval to lie below the smallest effect of interest; otherwise a
non-rejection is "inconclusive". In Part 2 and the addendum, where no smallest effect exists, a
non-rejection is reported as not detected, with the power below.

## Dev run and power

**Use of the dev run.** The four new models answered the Part 4 dev run (300 scenarios, 100 of them
in the history version, and the 120 reading prompts) with no failure. The dev answers serve only to
estimate the variance, the measurement bias due to rounding and clipping, and the power. The effects
seen on dev do not select hypotheses, change tests or move thresholds. If a test's power falls below
80%, the test stays unchanged and the limit is written here, before the test.

**Method, Part 4.** `bbg.part4_stats.simulate_power`, the method of the original Part 4, run on each
pair's dev answers (`scripts/replication_power.py`). Each simulated test run draws 4,000 scenarios from
the 300 dev scenarios with replacement, each scenario carrying its regressors and the residuals of
both models of the pair on it: the resampling unit is the scenario, so the clustering of the two
models' answers by scenario is kept. Answers are built from the base model's dev belief-updating
regression with each tested quantity set at its smallest effect of interest (or at its null value),
the abliterated model adds a shift and a flatter slope, answers are rounded to whole percentages and
clipped, 0% or 5% of them are dropped as failures, and the exact pre-registered tests run with Holm's
correction. 300 simulated runs per cell. A first stage, with every null true, measures the bias that
rounding and clipping create; each pair's tests move their null values by it.

**Method, Part 2 and addendum.** There is no dev run for these two experiments. Their power is
computed for an effect as large as the original pair's, and half as large, with the original pair's
spread: Part 2 H1 and H3 by resampling the original 24 units (Wilcoxon), Part 2 H4 and the addendum's
H1 and H2 exactly from the original counts (McNemar), the addendum's H3 from the original bootstrap
interval (normal approximation). Each test is taken at alpha / k of its family (Bonferroni), a lower
bound of its power under Holm.

These figures are estimates conditional on the dev data (Part 4) or on the original pair's data
(Part 2, addendum). They are not a guarantee of detecting the true effect.

**Results** (`results/replication/power.json` and `power.md`, written before the test):

| Part 4, 4,000 scenarios | H1 | H2 | H3 | H4 | H5 | Any rejected under the null |
|---|---|---|---|---|---|---|
| Gemma, at the smallest effects (0% and 5% failures) | 100% | 100% | 100% | 100% | 100% | |
| Gemma, every null true (0% / 5% failures) | | | | | | 4.3% / 3.0% |
| Llama, at the smallest effects (0% and 5% failures) | 100% | 100% | 100% | 100% | 100% | |
| Llama, every null true (0% / 5% failures) | | | | | | 4.0% / 1.7% |

Measurement biases (null values of each pair's tests): Gemma H1 -0.0039, H2 +0.0017, H3 +0.0039,
H4 -0.0018, H5 +0.0025; Llama H1 -0.0012, H2 -0.0006, H3 -0.0018, H4 +0.0047, H5 +0.0001.

| Part 2 and addendum (each test at alpha / k) | Original effect | Half the effect |
|---|---|---|
| Part 2 H1 pressure | 100% | 100% |
| Part 2 H2 fragility | no original effect | no original effect |
| Part 2 H3 push | 100% | 43% |
| Part 2 H4 leverage | 98% | 32% |
| Addendum H1 false optimism | 100% | 100% |
| Addendum H2 bullish share | 100% | 100% |
| Addendum H3 macro-F1 | 100% | 100% |

No test is below 80% at its smallest effect of interest (Part 4) or at the original effect (Part 2,
addendum). **Limit, written before the test:** if the effect in a new family is half the original
one, Part 2 H3 (43%) and H4 (32%) will probably miss it; a non-rejection of these two tests is weak
evidence.

## Checks before the test

- **Format, on no test item** (`scripts/replication_checks.py`, `results/replication/checks.json`):
  each model answers a Part 2 control scenario written for the check, outside the 48 (another amount,
  €15,000, and another wording), with the 5 seeds, and classifies the first 20 tweets of the dev
  split. Result: no failure for any of the four models (5 control answers each, at most 104 of the
  1,024 allowed tokens, so no answer comes close to truncation; 20 dev tweets each).
- **Code, on simulated data** (`tests/test_replication.py`): the analysis finds planted effects and
  finds nothing where nothing is planted; it applies Holm's correction and the sign rule of the
  two-sided tests; it refuses answers to different items or produced with other requests
  (fingerprints); it uses the pre-registered units (24, 24, 24 and 30 in Part 2; one row and one
  cluster per scenario in Part 4) and gives the same p-values as the original code called directly;
  the runs are refused until this file is committed and pushed.

## Procedure

1. Commit and push this file, the replication code, the dev answers of the four models,
   `results/replication/power.json` and `checks.json`.
2. `python scripts/replication_run.py --pair gemma`, then `--pair llama` (refused before step 1).
3. `python scripts/replication_analyze.py`, run once; every run is logged in
   `results/metrics/test_access.log`.
4. A README section and the commit of every file above.
5. Delete the four model files.

## Known limits

- Two more families, one quantization each, English only, thinking disabled, JSON answers, single
  questions.
- For Llama, family and abliteration recipe change together.
- Part 2's sampling is the one Qwen recommends, kept for all families; it may not be each family's
  preferred setting.
- Llama 3.1 (2024) is older than the two other families.
- The addendum's test tweets and Part 4's test scenarios were already used for the original pair;
  nothing was chosen with them, and the new models never answered them before the test run.
- The power of Part 2 and the addendum rests on the original pair's spread.

## Deviations

None so far.
