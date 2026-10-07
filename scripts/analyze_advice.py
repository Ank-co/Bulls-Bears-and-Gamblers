"""Pre-registered analysis of the part 2 answers (docs/part2-preregistration.md).

    python scripts/analyze_advice.py

Writes results/advice/analysis.json, analysis.md and examples.md.
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

from bbg import config
from bbg.advice_stats import A, B, regression, run_all

OUT = config.RESULTS_DIR / "advice"
LABELS = {"H1_pressure": "H1 gambler and chasing profiles: abliterated minus base",
          "H2_fragility": "H2 effect of a fragile situation: abliterated minus base",
          "H3_push": "H3 effect of a client pushing for confirmation: abliterated minus base"}


def main() -> int:
    df = pd.read_csv(OUT / "answers.csv")
    res = run_all(df)
    reg = regression(df)
    (OUT / "analysis.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    t = res["tests"]
    lines = [f"## Part 2: investment advice ({res['n_answers']} answers)", "",
             "Pre-registered tests (docs/part2-preregistration.md). Allocation in percentage points.", "",
             "| Test | n | Mean difference | Median difference | p | p (Holm) |", "|---|---|---|---|---|---|"]
    for key, label in LABELS.items():
        r = t[key]
        lines.append(f"| {label} | {r['n']} | {r['mean_diff']:+.1f} | {r['median_diff']:+.1f} "
                     f"| {r['p']:.3g} | {r['p_holm']:.3g} |")
    h4 = t["H4_leverage"]
    lines += ["", f"H4 leverage recommended to a fragile client chasing losses: abliterated "
              f"{h4['abliterated_rate']:.0%}, base {h4['base_rate']:.0%} ({h4['n']} paired answers, "
              f"{h4['abliterated_only']} vs {h4['base_only']} discordant, p = {h4['p']:.3g}, "
              f"Holm {h4['p_holm']:.3g})."]

    lines += ["", "### Mean recommended allocation to high-risk products (%)", ""]
    for model in (B, A):
        d = res["descriptives"][model]
        cell = d["mean_alloc_by_profile_situation"]
        lines += [f"**{model}**", "", "| Profile | Solid situation | Fragile situation |", "|---|---|---|"]
        for prof in ("cautious", "neutral", "gambler", "chasing"):
            lines.append(f"| {prof} | {cell['solid'][prof]:.1f} | {cell['fragile'][prof]:.1f} |")
        lines += ["", f"Pressure effect {d['pressure_effect']:+.1f}, fragility effect "
                  f"{d['fragility_effect']:+.1f}, push effect {d['push_effect']:+.1f}.", ""]
    if reg is not None:
        lines += ["### Secondary: OLS, standard errors clustered by scenario", "",
                  "| Term | Coefficient [95% CI] | p |", "|---|---|---|"]
        for term, row in reg.iterrows():
            lines.append(f"| {pretty(term)} | {row.coef:+.1f} [{row.ci_low:+.1f}, {row.ci_high:+.1f}] "
                         f"| {row.p:.3g} |")
    md = "\n".join(lines) + "\n"
    (OUT / "analysis.md").write_text(md, encoding="utf-8")
    write_examples(df)
    print(md)
    return 0


def pretty(term: str) -> str:
    """C(profile, Treatment('cautious'))[T.chasing] -> profile=chasing; ':' marks interactions."""
    parts = [re.sub(r"C\((\w+), Treatment\('\w+'\)\)\[T\.(\w+)\]", r"\1=\2", p) for p in term.split(":")]
    return " × ".join(parts)


def write_examples(df: pd.DataFrame) -> None:
    """The most demanding cell, first phrasing and first seed, for both models. Not cherry-picked."""
    sub = df[(df["profile"] == "chasing") & (df["situation"] == "fragile") & (df["phrasing"] == 0)
             & (df["seed"] == 0)].sort_values(["request", "model"])
    lines = ["## Examples: fragile client chasing losses (phrasing 0, seed 0)", ""]
    for _, r in sub.iterrows():
        lines += [f"**{r['model']}, request {r['request']}**: {r['risky_allocation_pct']}% high-risk, "
                  f"leverage {'yes' if r['leverage'] else 'no'}", "", f"> {r['advice']}", ""]
    (OUT / "examples.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
