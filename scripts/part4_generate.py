"""Part 4: generate the scenarios of a split.

    python scripts/part4_generate.py --split dev
    python scripts/part4_generate.py --split test     # refused until the pre-registration exists

Writes data/part4/<split>/scenarios.jsonl (one scenario per line: the numbers, the two texts, the
ideal answer and the history references) and results/part4/<split>/generation_report.json.
The scenarios are synthetic and invented here, so they are committed with the code.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np

from bbg import config, fictional

PREREGISTRATION = config.ROOT / "docs" / "part4-preregistration.md"


def report(scs: list[dict]) -> dict:
    items = [it for s in scs for it in s["items"]]
    words = {v: [len(s[v].split()) for s in scs] for v in ("stated", "history")}
    return {
        "scenarios": len(scs),
        "history_subset": sum(s["in_history_subset"] for s in scs),
        "share_up": float(np.mean([s["outcome"] for s in scs])),
        "mean_ideal_stated": float(np.mean([s["ideal_stated"] for s in scs])),
        "items": dict(Counter(it["role"] + ("-reversed" if it["reversed"] else "") for it in items)),
        "dramatic_by_role": {r: sum(it["dramatic"] for it in items if it["role"] == r)
                             for r in ("evidence", "distractor")},
        "topics": dict(Counter(it["topic"] for it in items)),
        "framings": dict(Counter(s["framing"] for s in scs)),
        "words": {v: {"mean": float(np.mean(w)), "max": int(np.max(w))} for v, w in words.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(fictional.SPLITS), required=True)
    args = ap.parse_args()
    if args.split == "test" and not PREREGISTRATION.exists():
        print(f"Refused: the test scenarios are generated after {PREREGISTRATION.name} is committed.")
        return 1

    scs = fictional.scenarios(args.split)
    out = config.DATA_DIR / "part4" / args.split
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scenarios.jsonl", "w", encoding="utf-8") as f:
        for s in scs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    rep = report(scs)
    res = config.RESULTS_DIR / "part4" / args.split
    res.mkdir(parents=True, exist_ok=True)
    (res / "generation_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
