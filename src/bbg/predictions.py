"""Standard prediction files.

Models only write (id, y_pred). They never see or write the gold label and never compute their
own score: the evaluator is the single authority that joins predictions with gold labels.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config
from .data import load_split

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def prediction_path(model: str, split: str, root: Path = config.PREDICTIONS_DIR) -> Path:
    if not _NAME.match(model):
        raise ValueError(f"model name {model!r}: use lowercase letters, digits, '.', '_' or '-'")
    return root / split / f"{model}.csv"


def write_predictions(model: str, split: str, ids: Iterable[str], y_pred: Iterable[int],
                      root: Path = config.PREDICTIONS_DIR,
                      splits_dir: Path = config.SPLITS_DIR) -> Path:
    df = pd.DataFrame({"id": list(ids), "y_pred": [int(v) for v in y_pred]})
    _check(df, split, splits_dir)
    path = prediction_path(model, split, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("id").to_csv(path, index=False)
    return path


def save_scores(model: str, split: str, ids: Iterable[str], scores, root: Path = config.RESULTS_DIR) -> Path:
    """Log-probabilities of each answer (n_items x 3), kept for calibration analysis."""
    from .prompts import ANSWERS
    path = root / "scores" / split / f"{model}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(scores.numpy(), columns=[f"logp_{a}" for a in ANSWERS.values()])
    df.insert(0, "id", list(ids))
    df.to_csv(path, index=False, float_format="%.5f")
    return path


def read_predictions(path: Path, split: str, splits_dir: Path = config.SPLITS_DIR) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str, "y_pred": int})
    _check(df, split, splits_dir)
    return df


def _check(df: pd.DataFrame, split: str, splits_dir: Path) -> None:
    gold_ids = set(load_split(split, splits_dir)["id"])
    ids = set(df["id"])
    if df["id"].duplicated().any():
        raise ValueError("duplicate ids in predictions")
    if ids != gold_ids:
        raise ValueError(f"predictions cover {len(ids)} ids, split {split} has {len(gold_ids)}"
                         f" (missing {len(gold_ids - ids)}, unknown {len(ids - gold_ids)})")
    bad = set(df["y_pred"]) - set(config.LABEL_IDS)
    if bad:
        raise ValueError(f"invalid predicted labels {sorted(bad)}")
