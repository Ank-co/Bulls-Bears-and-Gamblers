"""Part 3, step 3: the pre-registered analysis (docs/part3-preregistration.md).

    python scripts/part3_analyze.py --period dev     # pipeline check, not a test
    python scripts/part3_analyze.py --period test    # once, after the pre-registration commit

Joins the returns of step 1 with the labels of step 2 and the two controls (part3_probe.py,
part3_coverage.py), and writes results/part3/<period>/analysis.json
and analysis.md. Every run on the test period is appended to results/metrics/test_access.log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import pandas as pd

from bbg import config, part3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(config.PART3_PERIODS), required=True)
    args = ap.parse_args()

    work = part3.work_dir(args.period)
    ret = pd.read_parquet(work / "returns.parquet")
    heads = pd.read_parquet(work / "headlines.parquet", columns=["id", "ticker", "date"])
    labels = {m: part3.read_labels(part3.labels_path(args.period, m), heads["id"]) for m in part3.MODELS}
    probe_path = part3.probe_path(args.period)
    cov_path = part3.results_dir(args.period) / "coverage.json"
    for path, script in ((probe_path, "part3_probe.py"), (cov_path, "part3_coverage.py")):
        if not path.exists():
            print(f"{path} not found: run scripts/{script} --period {args.period} first")
            return 1
    probe = pd.read_csv(probe_path, parse_dates=["date"])
    if len(probe) != min(config.PART3_PROBE_SIZE, len(ret)):
        print(f"{probe_path.name}: {len(probe)} scores, the probe run is not finished")
        return 1
    cov = json.loads(cov_path.read_text(encoding="utf-8"))

    if args.period == "test":
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.METRICS_DIR / "test_access.log", "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} part3_analyze models={part3.MODELS}\n")

    res = part3.analyze(ret, heads, labels, probe, cov)
    out = part3.results_dir(args.period)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = part3.render(res, args.period)
    (out / "analysis.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
