"""Part 4: smallest effects of interest and power of the exact confirmatory analysis, by simulation.

    python scripts/part4_power.py

The smallest effect of interest of each hypothesis comes from one criterion (an effect that moves a
typical answer by 2.5 points of probability), converted with the sizes of the regressors in the
generator. The power simulation resamples the dev scenarios with replacement, each with its own
regressors and both models' residuals (so the errors keep their shape, their link with the
regressors and their dependence between the two models), builds answers from the base model's
belief-updating regression fitted on dev with the tested quantities set by the world, adds the
abliterated model's shift and slope, rounds answers to whole percentages, drops failures, and runs
the pre-registered tests with Holm's correction over the five. A null world, with every tested
quantity at its null value, gives the family-wise error.
Writes results/part4/dev/power.json and power.md.
"""
from __future__ import annotations

import json
import sys

from bbg import config
from bbg import part4_stats as S

DEV = config.RESULTS_DIR / "part4" / "dev"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main(reps: int = 300) -> int:
    scenarios = read_jsonl(config.DATA_DIR / "part4" / "dev" / "scenarios.jsonl")
    frames = {m: S.stated_frame(scenarios, read_jsonl(DEV / "answers" / m / "stated.jsonl"))
              for m in ("base", "abliterated")}
    sesoi = S.smallest_effects()
    sim = S.simulate_power(frames["base"], frames["abliterated"], sesoi["effects"], reps=reps)
    out = {"smallest_effects": sesoi, "simulation": sim}
    DEV.mkdir(parents=True, exist_ok=True)
    (DEV / "power.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    hs = list(S.DIRECTIONS)
    lines = ["# Part 4, test size: power of the exact analysis (simulation, dev noise)", "",
             f"Smallest effects of interest (criterion: {sesoi['points']} points of probability on a typical answer): "
             + ", ".join(f"{h} {v:.3f}" for h, v in sesoi["effects"].items()) + ".", "",
             f"{reps} simulated test runs per cell, resampling the {sim['dev_scenarios']} dev scenarios with both "
             "models' residuals. Holm's correction over the five tests, alpha 0.05.", "",
             "| World | Scenarios | Failures | " + " | ".join(hs) + " | Any rejected | All rejected |",
             "|---|---|---|" + "---|" * (len(hs) + 2)]
    for r in sim["runs"]:
        lines.append(f"| {r['world']} | {r['scenarios']} | {100 * r['failure_rate']:.0f}% | "
                     + " | ".join(f"{100 * r['rejection_rate'][h]:.0f}%" for h in hs)
                     + f" | {100 * r['any_rejected']:.0f}% | {100 * r['all_rejected']:.0f}% |")
    raw = sim["null_without_correction"]
    lines += ["", f"Measurement bias under the null (rounding to whole percentages, clipping to 1..99), used as null "
              f"values: " + ", ".join(f"{h} {v:+.4f}" for h, v in sim["measurement_bias"].items()) + ".",
              f"Without that correction, {raw['scenarios']} scenarios, no failures: any rejected "
              f"{100 * raw['any_rejected']:.0f}% (" + ", ".join(f"{h} {100 * v:.0f}%" for h, v in raw["rejection_rate"].items()) + ")."]
    (DEV / "power.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
