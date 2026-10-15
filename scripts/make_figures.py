"""README figures, rebuilt from the result files (light and dark versions for GitHub).

    python scripts/make_figures.py

Reads results/metrics/test.json, results/advice/answers.csv and, when they exist, the Part 3 and Part 4
test analyses (results/part3/test/analysis.json, results/part4/test/analysis.json); writes
docs/figures/*.svg. Requires matplotlib.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import pandas as pd  # noqa: E402

from bbg import config  # noqa: E402

THEMES = {
    "light": {"bg": "#ffffff", "card": "#f6f8fa", "border": "#d0d7de", "ink": "#1f2328",
              "ink2": "#59636e", "grid": "#d8dee4", "ours": "#2a78d6", "base": "#2a78d6",
              "abl": "#eb6834", "baseline": "#818b98",
              "qwen": "#2a78d6", "gemma": "#1baf7a", "llama": "#eda100"},
    "dark": {"bg": "#0d1117", "card": "#161b22", "border": "#30363d", "ink": "#e6edf3",
             "ink2": "#9198a1", "grid": "#30363d", "ours": "#3987e5", "base": "#3987e5",
             "abl": "#d95926", "baseline": "#6e7681",
             "qwen": "#3987e5", "gemma": "#199e70", "llama": "#c98500"},
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


def style(ax, t, grid_axis, face="bg"):
    ax.set_facecolor(t[face])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def card(fig, x0, x1, title, t, y0=0.03, y1=0.83):
    """Rounded panel behind one chart (figure coordinates), with its title inside."""
    w, h = fig.get_size_inches()
    fig.patches.append(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.008",
        mutation_aspect=w / h, transform=fig.transFigure, facecolor=t["card"],
        edgecolor=t["border"], linewidth=1, zorder=-1))
    fig.text(x0 + 0.016, y1 - 0.075, title, color=t["ink"], fontsize=10.5, fontweight="semibold")


def header(fig, title, subtitle, t):
    fig.text(0.008, 0.95, title, color=t["ink"], fontsize=12.5, fontweight="semibold")
    fig.text(0.008, 0.893, subtitle, color=t["ink2"], fontsize=9.5)


def part1(test: dict, t: dict, out: Path) -> None:
    models = test["models"]
    order = sorted(PART1_NAMES, key=lambda k: models[k]["macro_f1"])
    y = range(len(order))
    h = 0.62
    fig = plt.figure(figsize=(10.4, 4.8), facecolor=t["bg"])
    header(fig, f"Market sentiment on {test['n']:,} held-out tweets",
           "Blue: fine-tuned on the laptop GPU (three seeds). Grey: baselines.", t)
    card(fig, 0.008, 0.645, "Macro-F1, higher is better", t)
    card(fig, 0.675, 0.992, "Direction errors, lower is better", t)
    a1 = fig.add_axes((0.215, 0.165, 0.405, 0.555))
    a2 = fig.add_axes((0.695, 0.165, 0.275, 0.555), sharey=a1)
    for i, k in zip(y, order):
        m = models[k]
        c = t["ours"] if "lora" in k else t["baseline"]
        lo, hi = m["macro_f1_ci95"]
        a1.barh(i, m["macro_f1"], height=h, color=c)
        a1.plot([lo, hi], [i, i], color=t["ink"], linewidth=1.5, solid_capstyle="butt", zorder=3)
        for x in (lo, hi):
            a1.plot([x, x], [i - 0.12, i + 0.12], color=t["ink"], linewidth=1.5, zorder=3)
        a1.text(hi + 0.015, i, f"{m['macro_f1']:.3f}", va="center", fontsize=9, color=t["ink"])
        a2.barh(i, m["direction_errors"], height=h, color=c)
        a2.text(m["direction_errors"] + 2, i, str(m["direction_errors"]), va="center", fontsize=9,
                color=t["ink"])
    a1.set_yticks(list(y), [PART1_NAMES[k] for k in order], color=t["ink"], fontsize=9.5)
    a1.set_ylim(-0.6, len(order) - 0.4)
    a1.set_xlim(0, 1.04)
    a1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a1.set_xlabel("error bars: 95% bootstrap interval", color=t["ink2"], fontsize=9)
    a2.tick_params(axis="y", left=False, labelleft=False)
    a2.set_xlim(0, 85)
    a2.set_xlabel("bullish read as bearish, or the reverse", color=t["ink2"], fontsize=9)
    style(a1, t, "x", face="card")
    style(a2, t, "x", face="card")
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


PART3_ROWS = [("qwen3-1.7b-lora-s0", "Qwen3-1.7B + LoRA"), ("tfidf-logreg", "TF-IDF + logistic regression"),
              ("finbert", "FinBERT"), ("qwen3-1.7b-zeroshot", "Qwen3-1.7B, zero-shot"),
              ("probe", "Memory probe: ticker and dates only")]


def part3(res: dict, t: dict, out: Path) -> None:
    """Daily IC of each reader, reaction window and drift window, with 95% Newey-West intervals."""
    probe = res["controls"]["C1_memory_probe"]
    rows = list(reversed(PART3_ROWS))
    y = range(len(rows))
    h = 0.62
    fig = plt.figure(figsize=(10.4, 4.8), facecolor=t["bg"])
    header(fig, f"Headline readings and stock returns, {res['n_ticker_dates']:,} stock-days of 2018 and 2019",
           "Daily rank correlation (IC) between the reading and the market-adjusted return. "
           "Blue: fine-tuned. Grey: other readers. Outline: control.", t)
    card(fig, 0.008, 0.592, "Around the news: last close to next open", t)
    card(fig, 0.612, 0.992, "After it, open to close: prediction", t)
    a1 = fig.add_axes((0.25, 0.165, 0.315, 0.555))
    a2 = fig.add_axes((0.64, 0.165, 0.33, 0.555), sharey=a1)
    for ax, w, xmax, pad in ((a1, "reaction", 0.24, 0.006), (a2, "drift", 0.045, 0.0015)):
        for i, (key, _) in zip(y, rows):
            r = probe[w] if key == "probe" else res["models"][key][w]["ic"]
            m, half = r["mean"], 1.96 * r["se"]
            if key == "probe":
                ax.barh(i, m, height=h, facecolor=t["card"], edgecolor=t["baseline"], linewidth=1.5)
            else:
                ax.barh(i, m, height=h, color=t["ours"] if "lora" in key else t["baseline"])
            ax.plot([m - half, m + half], [i, i], color=t["ink"], linewidth=1.5, zorder=3)
            for x in (m - half, m + half):
                ax.plot([x, x], [i - 0.12, i + 0.12], color=t["ink"], linewidth=1.5, zorder=3)
            ax.text(max(m + half, 0) + pad, i, f"{m:+.3f}", va="center", fontsize=9, color=t["ink"])
        ax.axvline(0, color=t["ink2"], linewidth=1)
        ax.set_xlim(min(-0.01, ax.get_xlim()[0]) if w == "drift" else -0.02, xmax)
        style(ax, t, "x", face="card")
    a1.set_yticks(list(y), [name for _, name in rows], color=t["ink"], fontsize=9.5)
    a1.set_ylim(-0.6, len(rows) - 0.4)
    a1.set_xlabel("mean daily IC, error bars: 95% interval", color=t["ink2"], fontsize=9)
    a2.set_xlabel("mean daily IC, error bars: 95% interval", color=t["ink2"], fontsize=9)
    a2.tick_params(axis="y", left=False, labelleft=False)
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)


PART4_ROWS = [("base_rate", "Base rate", 1.0), ("e_congruent", "News, usual direction", 1.0),
              ("e_reversed", "News, reversed direction", 1.0), ("distractor", "Worthless news", 0.0),
              ("dramatic_tone", "Dramatic wording", 0.0)]
PART4_MODELS = [("base", "base", "Base model"), ("abliterated", "abl", "Abliterated model")]


def part4(res: dict, t: dict, out: Path) -> None:
    """Belief-updating weights of both models against the ideal observer, and their calibration."""
    fig = plt.figure(figsize=(10.4, 4.8), facecolor=t["bg"])
    header(fig, f"Reading the news in an invented market, {res['scenarios']:,} scenarios",
           "Blue: base model. Orange: abliterated model. Vertical marks and diagonal: the ideal observer, "
           "computed by code.", t)
    card(fig, 0.008, 0.592, "Weight given to each input", t)
    card(fig, 0.612, 0.992, "When the model says x%, the share rises y%", t)
    a1 = fig.add_axes((0.215, 0.165, 0.35, 0.555))
    a2 = fig.add_axes((0.68, 0.165, 0.29, 0.555))

    rows = list(reversed(PART4_ROWS))
    h = 0.34
    for i, (key, _, ideal) in enumerate(rows):
        for j, (m, color, _) in enumerate(PART4_MODELS):
            w = res["descriptive"][m]["weights"]["weights"][key]
            y = i + (0.19 if j == 0 else -0.19)
            a1.barh(y, w["estimate"], height=h, color=t[color])
            lo, hi = w["estimate"] - 1.96 * w["se"], w["estimate"] + 1.96 * w["se"]
            a1.plot([lo, hi], [y, y], color=t["ink"], linewidth=1.2, zorder=3)
            for x in (lo, hi):
                a1.plot([x, x], [y - 0.07, y + 0.07], color=t["ink"], linewidth=1.2, zorder=3)
            a1.text(max(hi, 0) + 0.02, y, f"{w['estimate']:.2f}", va="center", fontsize=8.5, color=t["ink"])
        a1.plot([ideal, ideal], [i - 0.45, i + 0.45], color=t["ink"], linewidth=2.2, zorder=4,
                solid_capstyle="butt")
    a1.text(1.0, len(rows) - 0.5, "ideal", ha="center", va="bottom", fontsize=8.5, color=t["ink2"])
    a1.axvline(0, color=t["ink2"], linewidth=1)
    a1.set_yticks(range(len(rows)), [name for _, name, _ in rows], color=t["ink"], fontsize=9.5)
    a1.set_ylim(-0.6, len(rows) - 0.4)
    a1.set_xlim(-0.03, 1.12)
    a1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_xlabel("weight in the belief-updating regression, error bars: 95% interval", color=t["ink2"], fontsize=9)
    style(a1, t, "x", face="card")

    a2.plot([0, 1], [0, 1], color=t["ink"], linewidth=1.2, zorder=2)
    for m, color, label in PART4_MODELS:
        cal = res["descriptive"][m]["calibration"]
        x = [c["mean_answer"] for c in cal]
        y = [c["frequency_up"] for c in cal]
        a2.plot(x, y, color=t[color], linewidth=1.6, zorder=3, label=label)
        a2.scatter(x, y, s=[max(6.0, c["n"] / 8) for c in cal], color=t[color], edgecolor=t["card"],
                   linewidth=0.8, zorder=4)
    top = max(res["descriptive"]["abliterated"]["calibration"], key=lambda c: c["n"])
    lo, hi = (int(v) for v in top["bin"].split("-"))
    a2.annotate(f"{top['n']:,} answers from {lo}% to {hi if hi == 100 else hi - 1}%:\n"
                f"the share rose in {100 * top['frequency_up']:.0f}% of the weeks",
                xy=(top["mean_answer"], top["frequency_up"]), xytext=(0.98, 0.05), ha="right", fontsize=8.5,
                color=t["ink"], arrowprops={"arrowstyle": "-", "color": t["ink2"], "linewidth": 1})
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1)
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    a2.set_xticks(ticks, [f"{100 * v:.0f}%" for v in ticks])
    a2.set_yticks(ticks, [f"{100 * v:.0f}%" for v in ticks])
    a2.set_xlabel("mean answer in each 10-point bin, dot area: answers", color=t["ink2"], fontsize=9)
    a2.set_ylabel("share of weeks up", color=t["ink2"], fontsize=9)
    style(a2, t, "both", face="card")
    a2.spines["left"].set_visible(True)
    a2.spines["left"].set_color(t["grid"])
    fig.savefig(out, facecolor=t["bg"])
    plt.close(fig)


FAMILIES = [("qwen", "Qwen3.6-35B-A3B (original pair)"), ("gemma", "Gemma 4 26B-A4B"), ("llama", "Llama 3.1 8B")]


def mean_answer(desc: dict) -> float:
    """Mean answer in points, from the calibration table of one model (n-weighted bin means)."""
    cal = desc["calibration"]
    return 100 * sum(c["n"] * c["mean_answer"] for c in cal) / sum(c["n"] for c in cal)


def replication_values(rep: dict, q2: dict, qa: dict, q4: dict) -> tuple[list, list]:
    """Effects of abliteration (abliterated minus base, points) and base-model weights, per family,
    read from the original analyses (Qwen) and from results/replication/analysis.json."""
    src = {"qwen": {"part2": q2, "addendum": qa, "part4": q4}}
    src.update({p: rep["pairs"][p] for p in ("gemma", "llama")})
    effects = []
    for label, get in (
            ("Riskier advice to gamblers", lambda s: s["part2"]["tests"]["H1_pressure"]["mean_diff"]),
            ("Moves when the client pushes", lambda s: s["part2"]["tests"]["H3_push"]["mean_diff"]),
            ("Non-bullish tweets read as bullish", lambda s: 100 * s["addendum"]["tests"]["H1_false_optimism"]["diff"]),
            ("Mean answer, invented market", lambda s: mean_answer(s["part4"]["descriptive"]["abliterated"])
             - mean_answer(s["part4"]["descriptive"]["base"]))):
        effects.append((label, {f: get(src[f]) for f, _ in FAMILIES}))
    weights = []
    for label, key in (("Base rate", "base_rate"), ("News, usual direction", "e_congruent"),
                       ("News, reversed direction", "e_reversed")):
        weights.append((label, {f: src[f]["part4"]["descriptive"]["base"]["weights"]["weights"][key]["estimate"]
                                for f, _ in FAMILIES}))
    return effects, weights


def replication(rep: dict, q2: dict, qa: dict, q4: dict, t: dict, out: Path) -> None:
    """Abliteration effects and base-model weights for the three families (point estimates)."""
    effects, weights = replication_values(rep, q2, qa, q4)
    fig = plt.figure(figsize=(10.4, 5.0), facecolor=t["bg"])
    header(fig, "Replication on two more model families",
           "Point estimates; the pre-registered tests and their results are in docs/replication.md.", t)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t[f]) for f, _ in FAMILIES]
    leg = fig.legend(handles, [name for _, name in FAMILIES], loc="upper right", bbox_to_anchor=(0.995, 0.985),
                     ncol=3, frameon=False, fontsize=9, handlelength=1.0, handleheight=1.0, columnspacing=1.2)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    card(fig, 0.008, 0.548, "What abliteration changes: abliterated minus base, points", t, y1=0.84)
    card(fig, 0.566, 0.992, "Base model, weight given to each input", t, y1=0.84)
    a1 = fig.add_axes((0.215, 0.13, 0.31, 0.6))
    a2 = fig.add_axes((0.74, 0.13, 0.235, 0.6))
    h = 0.24
    for ax, rows, fmt, pad in ((a1, effects, "{:+.1f}", 1.2), (a2, weights, "{:.2f}", 0.02)):
        n = len(rows)
        for i, (label, vals) in enumerate(rows):
            y0 = n - 1 - i
            for j, (f, _) in enumerate(FAMILIES):
                y = y0 + (1 - j) * (h + 0.04)
                v = vals[f]
                ax.barh(y, v, height=h, color=t[f])
                ax.text(v + (pad if v >= 0 else -pad), y, fmt.format(v).replace("-", "\u2212"), va="center",
                        ha="left" if v >= 0 else "right", fontsize=8.5, color=t["ink"])
        ax.set_yticks(range(n), [label for label, _ in reversed(rows)], color=t["ink"], fontsize=9.5)
        ax.set_ylim(-0.6, n - 0.4)
        ax.axvline(0, color=t["ink2"], linewidth=1)
        style(ax, t, "x", face="card")
    a1.set_xlim(-12, 52)
    a1.set_xlabel("percentage points; positive is the direction found with Qwen", color=t["ink2"], fontsize=9)
    a2.set_xlim(0, 1.12)
    a2.set_xticks([0, 0.5, 1.0])
    for i in range(len(weights)):
        a2.plot([1, 1], [i - 0.45, i + 0.45], color=t["ink"], linewidth=2.2, zorder=4, solid_capstyle="butt")
    a2.text(1.0, len(weights) - 0.5, "ideal", ha="center", va="bottom", fontsize=8.5, color=t["ink2"])
    a2.set_xlabel("belief-updating weight, ideal 1", color=t["ink2"], fontsize=9)
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
    p3 = args.results / "part3" / "test" / "analysis.json"
    if p3.exists():
        res3 = json.loads(p3.read_text(encoding="utf-8"))
        for name, t in THEMES.items():
            part3(res3, t, args.out / f"part3-{name}.svg")
    p4 = args.results / "part4" / "test" / "analysis.json"
    if p4.exists():
        res4 = json.loads(p4.read_text(encoding="utf-8"))
        for name, t in THEMES.items():
            part4(res4, t, args.out / f"part4-{name}.svg")
    rp = args.results / "replication" / "analysis.json"
    if rp.exists() and p4.exists():
        rep = json.loads(rp.read_text(encoding="utf-8"))
        q2 = json.loads((args.results / "advice" / "analysis.json").read_text(encoding="utf-8"))
        qa = json.loads((args.results / "metrics" / "abliteration_sentiment.json").read_text(encoding="utf-8"))
        for name, t in THEMES.items():
            replication(rep, q2, qa, res4, t, args.out / f"replication-{name}.svg")
    print(f"figures written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
