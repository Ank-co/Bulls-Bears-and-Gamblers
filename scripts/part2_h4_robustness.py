"""Robustness check of H4 (part 2), written after the results: scenarios as units.

    python scripts/part2_h4_robustness.py

The pre-registered H4 (exact McNemar on scenario x seed pairs) is kept as published in
results/advice/analysis.json. This script recomputes H4 with the scenario as unit, as H1-H3 do,
and the Holm correction over H1-H4 with that p-value. Reads only the saved answers: no model call.
Writes results/advice/h4_robustness.json.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from bbg import config
from bbg.advice_stats import h4_leverage_by_scenario
from bbg.metrics import holm

OUT = config.RESULTS_DIR / "advice"


def main() -> int:
    df = pd.read_csv(OUT / "answers.csv")
    published = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))["tests"]
    h4 = h4_leverage_by_scenario(df)
    names = ["H1_pressure", "H2_fragility", "H3_push", "H4_leverage"]
    raw = {n: published[n]["p"] for n in names[:3]} | {"H4_leverage": h4["p"]}
    h4["p_holm"] = float(holm(raw)["H4_leverage"])
    h4["published_p"], h4["published_p_holm"] = published["H4_leverage"]["p"], published["H4_leverage"]["p_holm"]
    h4["note"] = ("Added after the results, not pre-registered. Unit: scenario (5 seeds averaged), "
                  "as for H1-H3. The pre-registered test treated seeds as independent pairs.")
    (OUT / "h4_robustness.json").write_text(json.dumps(h4, indent=2), encoding="utf-8")
    print(f"H4 by scenario: abliterated {h4['abliterated_rate']:.0%}, base {h4['base_rate']:.0%}, "
          f"{h4['scenarios_up']} of {h4['n']} scenarios up, {h4['scenarios_down']} down, "
          f"p = {h4['p']:.3g}, Holm {h4['p_holm']:.3g} (published: {h4['published_p_holm']:.2g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
