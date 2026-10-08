"""README figures, rebuilt from the result files (light and dark versions for GitHub).

    python scripts/make_figures.py

Reads results/metrics/test.json and results/advice/answers.csv, writes docs/figures/*.svg.
Requires matplotlib.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bbg import config  # noqa: E402

THEMES = {
    "light": {"bg": "#ffffff", "ink": "#1f2328", "ink2": "#59636e", "muted": "#8c959f",
              "grid": "#d8dee4", "ours": "#2a78d6", "base": "#2a78d6", "abl": "#eb6834",
              "baseline": "#8c959f"},
    "dark": {"bg": "#0d1117", "ink": "#e6edf3", "ink2": "#9198a1", "muted": "#6e7681",
             "grid": "#30363d", "ours": "#3987e5", "base": "#3987e5", "abl": "#d95926",
             "baseline": "#6e7681"},
}

PART1_NAMES = {
    "qwen3-1.7b-lora-s0": "Qwen3-1.7B + LoRA, seed 0",
    "qwen3-1.7b-lora-s1": "Qwen3-1.7B + LoRA, seed 1",
    "qwen3-1.7b-lora-s2": "Qwen3-1.7B + LoRA, seed 2",
    "tfidf-logreg": "TF-IDF + logistic regression",
    "qwen3.6-35b-a3b-zeroshot": "Qwen3.6-35B-A3B, zero-shot",
    "finbert": "FinBERT, off the shelf",
    "qwen3-1.7b-zeroshot": "Qwen3-1.7B, zero-shot",
    "majority": "Majority class",
}
PROFILES = [("cautious", "Cautious"), ("neutral", "Neutral"), ("gambler", "Gambler"),
            ("chasing", "Chasing losses")]


def style(ax, t, grid_axis):
    ax.set_facecolor(t["bg"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def part1(test: dict, t: dict, out: Path) -> None:
    models = test["models"]
    order = sorted(PART1_NAMES, key=lambda k: models[k]["macro_f1"])
    y = range(len(order))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True,
                                 gridspec_kw={"width_ratios": [3, 1.3]}, facecolor=t["bg"])
    for i, k in zip(y, order):
        m = models[k]
        c = t["ours"] if "lora" in k else t["baseline"]
        lo, hi = m["macro_f1_ci95"]
        a1.plot([lo, hi], [i, i], color=c, linewidth=2, solid_capstyle="round")
        a1.scatter([m["macro_f1"]], [i], s=60, color=c, edgecolors=t["bg"], linewidths=2, zorder=3)
        a1.text(hi + 0.012, i, f"{m['macro_f1']:.3f}", va="center", fontsize=9, color=t["ink"])
        a2.barh(i, m["direction_errors"], height=0.5, color=c)
        a2.text(m["direction_errors"] + 2, i, str(m["direction_errors"]), va="center", fontsize=9,
                color=t["ink"])
    a1.set_yticks(list(y), [PART1_NAMES[k] for k in order], color=t["ink"], fontsize=9.5)
    a1.set_ylim(-0.6, len(order) - 0.4)
    a1.set_xlim(0.2, 1.0)
    a1.set_xlabel("Macro-F1 with 95% bootstrap interval", color=t["ink2"], fontsize=9)
    a2.tick_params(axis="y", left=False, labelleft=False)
    a2.set_xlim(0, 85)
    a2.set_xlabel("Direction errors (bullish ↔ bearish)", color=t["ink2"], fontsize=9)
    style(a1, t, "x")
    style(a2, t, "x")
    fig.text(0.01, 0.955, f"Market sentiment on {test['n']:,} held-out tweets", color=t["ink"],
             fontsize=12.5, fontweight="semibold")
    fig.text(0.01, 0.895, "Blue: fine-tuned on the laptop GPU (three seeds). Grey: baselines.",
             color=t["ink2"], fontsize=9.5)
    fig.subplots_adjust(left=0.25, right=0.985, top=0.84, bottom=0.13, wspace=0.06)
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)


def part2(ans: pd.DataFrame, t: dict, out: Path) -> None:
    means = ans.groupby(["request", "model", "profile"])["risky_allocation_pct"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True, facecolor=t["bg"])
    titles = {"open": "Open question", "push": "Client pushes for confirmation"}
    w = 0.36
    for ax, req in zip(axes, ("open", "push")):
        for j, (model, color, label) in enumerate((("base", t["base"], "Base model"),
                                                   ("abliterated", t["abl"], "Abliterated model"))):
            xs = [p + (j - 0.5) * (w + 0.02) for p in range(len(PROFILES))]
            vals = [means[(req, model, prof)] for prof, _ in PROFILES]
            ax.bar(xs, vals, width=w, color=color, label=label)
            for x, v in zip(xs, vals):
                ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5, color=t["ink"])
        ax.set_xticks(range(len(PROFILES)), [n for _, n in PROFILES], color=t["ink"], fontsize=9.5)
        ax.set_title(titles[req], color=t["ink"], fontsize=10.5, loc="left")
        ax.set_ylim(0, 100)
        style(ax, t, "y")
    axes[0].set_ylabel("Mean share of savings in high-risk products (%)", color=t["ink2"], fontsize=9)
    leg = axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    fig.text(0.01, 0.955, "Advice to a client with €20,000 of savings", color=t["ink"],
             fontsize=12.5, fontweight="semibold")
    fig.text(0.01, 0.895, "Each bar: mean of 30 answers (2 situations x 3 phrasings x 5 seeds).",
             color=t["ink2"], fontsize=9.5)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.78, bottom=0.09, wspace=0.05)
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=config.ROOT / "docs" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["Inter", "Segoe UI", "DejaVu Sans"], "svg.fonttype": "path",
                         "svg.hashsalt": "bbg"})
    test = json.loads((args.results / "metrics" / "test.json").read_text(encoding="utf-8"))
    ans = pd.read_csv(args.results / "advice" / "answers.csv")
    for name, t in THEMES.items():
        part1(test, t, args.out / f"part1-{name}.svg")
        part2(ans, t, args.out / f"part2-{name}.svg")
    print(f"figures written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
