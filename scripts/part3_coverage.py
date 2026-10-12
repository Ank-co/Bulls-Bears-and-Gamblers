"""Part 3, control C2: are the ticker-dates dropped for lack of prices different from the kept ones?

    python scripts/part3_coverage.py --period dev

The dropped ticker-dates have no returns, so no test is possible on them. This compares what both
groups do have: number of headlines, publishers, kind of ticker symbol, and the tone of the
headlines read by the TF-IDF model of Part 1 (checked against Part 1 first, like every model).
Writes results/part3/<period>/coverage.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from bbg import config, part3

SCRIPTS = Path(__file__).resolve().parent


def make_tfidf():
    spec = importlib.util.spec_from_file_location("bbg_script_part3_classify", SCRIPTS / "part3_classify.py")
    classify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classify)
    fn = classify.make_classifier("tfidf-logreg", "cpu", 0)
    agreement = classify.check_against_part1("tfidf-logreg", fn)
    if agreement < classify.MIN_AGREEMENT:
        raise RuntimeError(f"TF-IDF agreement with Part 1 is {agreement:.4f}")
    return fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(config.PART3_PERIODS), required=True)
    args = ap.parse_args()

    work = part3.work_dir(args.period)
    exc_path = work / "excluded_headlines.parquet"
    if not exc_path.exists():
        print(f"{exc_path} not found: run scripts/part3_prepare.py --period {args.period} first")
        return 1
    inc = pd.read_parquet(work / "headlines.parquet", columns=["id", "ticker", "date", "publisher"])
    exc = pd.read_parquet(exc_path)
    inc_labels = part3.read_labels(part3.labels_path(args.period, "tfidf-logreg"), inc["id"])
    tfidf = make_tfidf()
    exc_labels = pd.DataFrame({"id": exc["id"], "y_pred": tfidf(exc["title"].tolist())})
    res = part3.coverage(inc, inc_labels, exc, exc_labels)
    out = part3.results_dir(args.period) / "coverage.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
