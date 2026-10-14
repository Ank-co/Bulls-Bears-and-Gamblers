"""Replication, before the test: power of every pre-registered test, from data other than the new test.

    python scripts/replication_power.py

Part 4: for each replication pair, the simulation of scripts/part4_power.py (bbg.part4_stats.simulate_power)
on that pair's 300 dev scenarios, at the size of the frozen test (4,000 scenarios), with 0% and 5%
failures. It resamples whole scenarios with both models' residuals on them (clusters by scenario) and
runs the exact pre-registered tests with Holm's correction. It also gives the pair's measurement bias,
which moves the null values of the pair's tests. The smallest effects of interest are the frozen ones
(results/part4/dev/power.json), from the generator, identical for every pair.
Part 2 and the addendum have no dev run: their power is computed for an effect as large as the
original pair's, and half as large, with the original pair's spread (bbg.replication).
The power is an estimate conditional on these data, not a guarantee of detecting the true effect.
Writes results/replication/power.json and power.md. Reads no test answer of the new models.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from bbg import config
from bbg import part4_stats as S
from bbg import replication as R
from bbg.part4_llm import fingerprint, scenario_jobs

TEST_SCENARIOS = 4000


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def dev_frames(pair: str, scenarios: list[dict]) -> dict:
    jobs = scenario_jobs(scenarios, "stated")
    frames = {}
    for role in R.ROLES:
        short, server = R.PAIRS[pair]["short"][role], R.PAIRS[pair]["server"][role]
        path = config.RESULTS_DIR / "part4" / "dev" / "answers" / short / "stated.jsonl"
        R.check_fingerprint(path.with_suffix(".meta.json"), fingerprint(server, "probability", jobs), f"{pair} {role} dev")
        answers = read_jsonl(path)
        R.check_same_items([j["id"] for j in jobs], [a["id"] for a in answers], f"{pair} {role} dev")
        frames[role] = S.stated_frame(scenarios, answers)
    return frames


def main(reps: int = 300) -> int:
    scenarios = read_jsonl(config.DATA_DIR / "part4" / "dev" / "scenarios.jsonl")
    frozen = json.loads((config.RESULTS_DIR / "part4" / "dev" / "power.json").read_text(encoding="utf-8"))
    sesoi = frozen["smallest_effects"]["effects"]
    out = {"smallest_effects": sesoi, "test_scenarios": TEST_SCENARIOS, "reps": reps, "part4": {}}
    for pair in R.REPLICATIONS:
        frames = dev_frames(pair, scenarios)
        out["part4"][pair] = S.simulate_power(frames["base"], frames["abliterated"], sesoi,
                                              sizes=(TEST_SCENARIOS,), reps=reps)
        print(f"{pair}: done", flush=True)
    advice = pd.read_csv(config.RESULTS_DIR / "advice" / "answers.csv")
    addendum = json.loads((config.METRICS_DIR / "abliteration_sentiment.json").read_text(encoding="utf-8"))
    out["without_dev"] = R.power_without_dev(advice, addendum)
    R.OUT.mkdir(parents=True, exist_ok=True)
    (R.OUT / "power.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    hs = list(S.DIRECTIONS)
    lines = ["# Replication: power of the pre-registered tests (before the test)", "",
             "Estimates conditional on the data used, not guarantees of detecting the true effect.", "",
             f"## Part 4 ({TEST_SCENARIOS:,} test scenarios, {reps} simulated runs per cell, dev noise of each pair)", "",
             "Smallest effects of interest (frozen, from the generator): "
             + ", ".join(f"{h} {v:.3f}" for h, v in sesoi.items()) + ".", "",
             "| Pair | World | Failures | " + " | ".join(hs) + " | Any rejected | All rejected |",
             "|---|---|---|" + "---|" * (len(hs) + 2)]
    low = []
    for pair, sim in out["part4"].items():
        for r in sim["runs"]:
            lines.append(f"| {pair} | {r['world']} | {100 * r['failure_rate']:.0f}% | "
                         + " | ".join(f"{100 * r['rejection_rate'][h]:.0f}%" for h in hs)
                         + f" | {100 * r['any_rejected']:.0f}% | {100 * r['all_rejected']:.0f}% |")
            if r["world"] == "at_smallest_effects" and r["failure_rate"] == 0:
                low += [f"Part 4 {h} ({pair}): {100 * r['rejection_rate'][h]:.0f}%" for h in hs if r["rejection_rate"][h] < 0.8]
    lines += [""] + [f"Measurement bias of {pair} (null values of its tests): "
                     + ", ".join(f"{h} {v:+.4f}" for h, v in sim["measurement_bias"].items()) + "."
                     for pair, sim in out["part4"].items()]
    wd = out["without_dev"]
    lines += ["", "## Part 2 and addendum (no dev run: effect of the original pair, and half of it)", "",
              "Each test at alpha / k of its family, a lower bound of its power under Holm.", "",
              "| Experiment | Test | Original effect | Half the effect |", "|---|---|---|---|"]
    for exp in ("part2", "addendum"):
        for h in wd[exp]["x1"]:
            lines.append(f"| {exp} | {h} | {R.fmt_pct(wd[exp]['x1'][h])} | {R.fmt_pct(wd[exp]['x0.5'][h])} |")
            if wd[exp]["x1"][h] is not None and wd[exp]["x1"][h] < 0.8:
                low.append(f"{exp} {h} at the original effect: {100 * wd[exp]['x1'][h]:.0f}%")
    lines += ["", "Part 2 H2 has no power figure: it was not confirmed in the original pair, so there is no "
              "original effect to size it on.", "",
              "Tests below 80% power (they stay unchanged; the limit is reported before the test): "
              + ("; ".join(low) if low else "none") + "."]
    (R.OUT / "power.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
