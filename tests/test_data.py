import pandas as pd
import pytest

from bbg.data import assert_no_overlap, build_splits, normalize_text


def make_frames():
    train = pd.DataFrame({
        "text": [f"$AAPL headline number {i} https://t.co/x{i}" for i in range(300)]
                + ["Stocks rally on Fed news https://t.co/aaa",      # leaks into test (other URL)
                   "duplicate tweet", "Duplicate   tweet",            # duplicate inside train
                   "conflict", "CONFLICT"],                           # same text, different labels
        "label": [i % 3 for i in range(300)] + [1, 2, 2, 0, 1],
    })
    test = pd.DataFrame({
        "text": ["stocks rally on fed news https://t.co/bbb"] + [f"test tweet {i}" for i in range(60)],
        "label": [1] + [i % 3 for i in range(60)],
    })
    return train, test


def test_normalize_text_ignores_case_urls_spaces():
    assert normalize_text("Stocks  RALLY https://t.co/a") == normalize_text("stocks rally http://x.y")


def test_build_splits_removes_leakage_and_duplicates():
    splits, report = build_splits(*make_frames())
    assert report.removed_train_test_overlap == 1
    assert report.removed_train_duplicates == 2
    assert report.conflicting_label_groups_in_train == 1
    assert len(splits["test"]) == 61                       # test is never modified
    assert report.counts["train"]["total"] + report.counts["val"]["total"] == 302
    assert_no_overlap(splits)


def test_split_is_stratified_and_deterministic():
    a, _ = build_splits(*make_frames())
    b, _ = build_splits(*make_frames())
    assert a["val"]["id"].tolist() == b["val"]["id"].tolist()
    shares = a["val"]["label"].value_counts(normalize=True)
    assert shares.min() > 0.25


def test_rejects_unknown_labels():
    train, test = make_frames()
    train.loc[0, "label"] = 7
    with pytest.raises(ValueError):
        build_splits(train, test)
