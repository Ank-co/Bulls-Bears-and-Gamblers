"""Part 4: what the dev run says about the set-up (not a result).

    python scripts/part4_dev_report.py

Reads the dev scenarios, the answers of every model in results/part4/dev/answers/ and, if present,
the tone reader (results/part4/dev/tone.jsonl). Reports, per model: failures and their reasons,
time and prompt size, the reading of the news, the belief-updating weights, shift / amplification /
noise, accuracy against the references, and the distance to the history references; then, for base
against abliterated, the paired differences, the correlation of their errors and the power for a
0.1 log-odds shift at the planned sizes. The dev answers set up the test (prompt, failures, size);
they are never pooled with the test answers.
Writes results/part4/dev/dev_report.json and dev_report.md.
"""
from __future__ import annotations

import json
import sys

from bbg import config
from bbg.part4_llm import reading_items
from bbg import part4_stats as S

DEV = config.RESULTS_DIR / "part4" / "dev"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def fmt(c: dict) -> str:
    return f"{c['estimate']:.3f} ({c['se']:.3f})"


def main() -> int:
    scenarios = read_jsonl(config.DATA_DIR / "part4" / "dev" / "scenarios.jsonl")
    if not scenarios:
        print("No dev scenarios: run scripts/part4_generate.py --split dev")
        return 1
    models = sorted(p.name for p in (DEV / "answers").iterdir() if p.is_dir()) if (DEV / "answers").exists() else []
    if not models:
        print("No answers: run scripts/part4_run.py --split dev")
        return 1
    tone = {r["id"]: r["p_tone"] for r in read_jsonl(DEV / "tone.jsonl")}
    items = reading_items()

    report = {"note": "dev run, for setting up the test; not a result", "models": {}}
    frames = {}
    for m in models:
        ans = {w: read_jsonl(DEV / "answers" / m / f"{w}.jsonl") for w in ("reading", "stated", "history")}
        r = {w: {**S.failures(a), **S.runtime(a)} for w, a in ans.items() if a}
        if ans["reading"]:
            r["reading_agreement"] = S.reading_agreement(items, ans["reading"])
        if ans["stated"]:
            df = S.stated_frame(scenarios, ans["stated"])
            frames[m] = df
            extra = {"tone_reader": df["id"].map(tone).to_numpy()} if tone else None
            r["stated_weights"] = S.grether(df)
            r["shift_amplification_noise"] = S.shift_amplification_noise(df)
            r["accuracy"] = S.accuracy(df, extra)
            if ans["reading"]:
                own = S.own_reading_map(r["reading_agreement"])
                r["stated_weights_own_reading"] = S.grether(S.stated_frame(scenarios, ans["stated"], own))
        if ans["history"]:
            r["history_distances"] = S.history_distances(scenarios, ans["history"])
        report["models"][m] = r
    if {"base", "abliterated"} <= set(frames):
        report["base_vs_abliterated"] = {"paired": S.paired(frames["base"], frames["abliterated"]),
                                         "shift_and_slope_difference": S.stacked_shift_slope(frames["base"],
                                                                                             frames["abliterated"])}
        report["planning"] = S.planning(frames["base"], frames["abliterated"])
    DEV.mkdir(parents=True, exist_ok=True)
    (DEV / "dev_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Part 4, dev run (set-up only, not a result)", ""]
    for m, r in report["models"].items():
        lines += [f"## {m}", ""]
        for w in ("reading", "stated", "history"):
            if w in r:
                lines.append(f"- {w}: {r[w]['n']} answers, {r[w]['failures']} failures "
                             f"({100 * r[w]['rate']:.1f}%), {r[w]['mean_seconds']:.2f} s each, "
                             f"{r[w]['mean_prompt_tokens'] or 0:.0f} prompt tokens")
        if "reading_agreement" in r:
            lines.append(f"- reading of the news: {r['reading_agreement']['overall']}")
        if "stated_weights" in r:
            w = r["stated_weights"]["weights"]
            lines.append("- weights (ideal 1 / 1 / 1 / 0 / 0): base rate " + fmt(w["base_rate"]) + ", congruent "
                         + fmt(w["e_congruent"]) + ", reversed " + fmt(w["e_reversed"]) + ", distractor "
                         + fmt(w["distractor"]) + ", tone " + fmt(w["dramatic_tone"]))
            s = r["shift_amplification_noise"]
            lines.append(f"- shift {fmt(s['shift'])}, amplification {fmt(s['amplification'])}, "
                         f"noise sd {s['noise_sd']:.2f}")
            lines.append("- Brier: " + ", ".join(f"{k} {v['brier']:.3f}" for k, v in r["accuracy"].items()))
        if r.get("history_distances", {}).get("n"):
            h = r["history_distances"]
            lines.append("- history, mean gap to each reference (points): " + ", ".join(
                f"{k} {v['mean_abs_gap_points']:.1f}" for k, v in h.items() if k != "n"))
        lines.append("")
    if "base_vs_abliterated" in report:
        p = report["base_vs_abliterated"]["paired"]
        lines += ["## base vs abliterated", "",
                  f"- mean difference {p['mean_difference']:.3f} log-odds, sd {p['sd_difference']:.3f} "
                  f"({p['sd_difference_around_trend']:.3f} around a trend in the ideal answer), "
                  f"error correlation {p['error_correlation']:.2f}"]
        lines += [f"- {q['scenarios']} scenarios: expected t {q['expected_t']:.1f}, power {100 * q['power_5pct']:.0f}% "
                  f"at 5%, {100 * q['power_1pct']:.0f}% at 1%" for q in p["power"]]
        d = report["base_vs_abliterated"]["shift_and_slope_difference"]
        lines += [f"- shift difference {fmt(d['shift_difference'])}, slope difference {fmt(d['slope_difference'])}", ""]
        pl = report["planning"]
        lines += ["## Test size: power at the smallest effects of interest (5% / 1%, one-sided)", "",
                  "| Hypothesis | Effect | " + " | ".join(f"{q['scenarios']} scenarios" for q in pl["power"]["H1"]) + " |",
                  "|---|---|" + "---|" * len(pl["power"]["H1"])]
        for h, e in pl["smallest_effects"].items():
            lines.append(f"| {h} | {e} | " + " | ".join(f"{100 * q['power_5pct']:.0f}% / {100 * q['power_1pct']:.0f}%"
                                                         for q in pl["power"][h]) + " |")
    (DEV / "dev_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
