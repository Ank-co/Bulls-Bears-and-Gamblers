"""Download the dataset and write the frozen splits + manifest.

    python scripts/prepare_data.py                      # from the Hugging Face Hub
    python scripts/prepare_data.py --train-csv a.csv --test-csv b.csv   # offline, same schema

Refuses to overwrite existing splits unless --force is given: splits are frozen once created.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from bbg import config
from bbg.data import build_splits, save_splits


def from_hub() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from datasets import load_dataset

    ds = load_dataset(config.HF_DATASET)
    source = {"hub": config.HF_DATASET, "hub_splits": {"train": "train", "test": "validation"}}
    try:
        from huggingface_hub import HfApi
        source["revision"] = HfApi().dataset_info(config.HF_DATASET).sha
    except Exception as exc:  # the revision is informative, not required
        source["revision"] = f"unknown ({type(exc).__name__})"
    return ds["train"].to_pandas(), ds["validation"].to_pandas(), source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv")
    ap.add_argument("--test-csv")
    ap.add_argument("--force", action="store_true", help="overwrite existing frozen splits")
    args = ap.parse_args()

    if (config.SPLITS_DIR / "manifest.json").exists() and not args.force:
        print("Splits already exist and are frozen. Use --force only if you really mean it.")
        return 1

    if args.train_csv and args.test_csv:
        train, test = pd.read_csv(args.train_csv), pd.read_csv(args.test_csv)
        source = {"csv": {"train": args.train_csv, "test": args.test_csv}}
    else:
        train, test, source = from_hub()

    splits, report = build_splits(train, test)
    manifest = save_splits(splits, report, source)
    print(json.dumps(report.__dict__, indent=2))
    print(f"Manifest written to {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
