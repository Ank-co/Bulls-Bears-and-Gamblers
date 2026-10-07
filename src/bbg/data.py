"""Frozen train/val/test splits.

Protocol
- test  = the official `validation` split of the dataset, untouched. It is used once, at the end.
- train / val = a stratified split of the official `train` split (val is for model selection only).
- Leakage guard: any train row whose normalized text also appears in test is dropped.
  Exact duplicates inside train are dropped too. Every removal is counted in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

_URL = re.compile(r"https?://\S+")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Key used only for duplicate detection: lowercase, URLs removed, whitespace collapsed."""
    text = _URL.sub(" ", str(text).lower())
    return _SPACES.sub(" ", text).strip()


def validate_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    bad = set(df["label"].unique()) - set(config.LABEL_IDS)
    if bad:
        raise ValueError(f"{name}: unexpected labels {sorted(bad)}")
    if df["text"].isna().any():
        raise ValueError(f"{name}: {int(df['text'].isna().sum())} empty texts")
    return df[["text", "label"]].astype({"text": str, "label": int}).reset_index(drop=True)


@dataclass
class SplitReport:
    removed_train_test_overlap: int = 0
    removed_train_duplicates: int = 0
    conflicting_label_groups_in_train: int = 0
    duplicate_texts_inside_test: int = 0
    counts: dict = field(default_factory=dict)


def build_splits(official_train: pd.DataFrame, official_test: pd.DataFrame,
                 seed: int = config.SPLIT_SEED,
                 val_fraction: float = config.VAL_FRACTION) -> tuple[dict[str, pd.DataFrame], SplitReport]:
    train = validate_frame(official_train, "train")
    test = validate_frame(official_test, "test")
    report = SplitReport()

    train["id"] = [f"tr{i:05d}" for i in range(len(train))]
    test["id"] = [f"te{i:05d}" for i in range(len(test))]
    train["_key"] = train["text"].map(normalize_text)
    test["_key"] = test["text"].map(normalize_text)

    # Test stays untouched; duplicates inside it are only reported.
    report.duplicate_texts_inside_test = int(test["_key"].duplicated().sum())

    # 1. Train/test leakage
    leaked = train["_key"].isin(set(test["_key"]))
    report.removed_train_test_overlap = int(leaked.sum())
    train = train[~leaked]

    # 2. Duplicates inside train (keep the first occurrence)
    groups = train.groupby("_key")["label"].nunique()
    report.conflicting_label_groups_in_train = int((groups > 1).sum())
    dup = train["_key"].duplicated(keep="first")
    report.removed_train_duplicates = int(dup.sum())
    train = train[~dup]

    # 3. Stratified train/val split
    tr, va = train_test_split(train, test_size=val_fraction, random_state=seed,
                              stratify=train["label"])
    splits = {
        "train": tr.sort_values("id"),
        "val": va.sort_values("id"),
        "test": test.sort_values("id"),
    }
    splits = {k: v[["id", "text", "label"]].reset_index(drop=True) for k, v in splits.items()}

    report.counts = {
        name: {config.LABELS[c]: int((df["label"] == c).sum()) for c in config.LABEL_IDS}
        | {"total": int(len(df))}
        for name, df in splits.items()
    }
    assert_no_overlap(splits)
    return splits, report


def assert_no_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """Hard check, run on every build: no shared ids and no shared normalized texts across splits."""
    keys = {name: set(df["text"].map(normalize_text)) for name, df in splits.items()}
    ids = {name: set(df["id"]) for name, df in splits.items()}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if ids[a] & ids[b]:
                raise AssertionError(f"id overlap between {a} and {b}")
            if keys[a] & keys[b]:
                raise AssertionError(f"text overlap between {a} and {b}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_splits(splits: dict[str, pd.DataFrame], report: SplitReport, source: dict,
                out_dir: Path = config.SPLITS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, df in splits.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        files[name] = {"file": path.name, "sha256": sha256(path)}
    manifest = {
        "source": source,
        "seed": config.SPLIT_SEED,
        "val_fraction": config.VAL_FRACTION,
        "labels": config.LABELS,
        "files": files,
        "report": report.__dict__,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def load_split(name: str, splits_dir: Path = config.SPLITS_DIR) -> pd.DataFrame:
    """Load a frozen split and verify it against the manifest checksum."""
    if name not in config.SPLITS:
        raise ValueError(f"unknown split {name!r}")
    manifest = json.loads((splits_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["files"][name]
    path = splits_dir / entry["file"]
    if sha256(path) != entry["sha256"]:
        raise RuntimeError(f"{path} does not match the manifest checksum: splits were modified")
    return pd.read_parquet(path)
