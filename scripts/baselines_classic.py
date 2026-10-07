"""Classic baselines: majority class and TF-IDF + logistic regression.

Model selection uses the val split only (macro-F1). The selected model is trained on train,
then writes predictions for val and test. Scores are computed by scripts/evaluate.py, not here.
"""
from __future__ import annotations

import itertools
import json
import re
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from bbg import config
from bbg.data import load_split
from bbg.metrics import macro_f1
from bbg.predictions import write_predictions

TOKEN_PATTERN = r"(?u)\$?\b\w\w+\b"   # keeps cashtags such as $AAPL as single tokens
_URL = re.compile(r"https?://\S+")


def _clean(text: str) -> str:
    return _URL.sub(" URL ", text.lower())


def make_pipeline(features: str, C: float, class_weight) -> Pipeline:
    word = TfidfVectorizer(preprocessor=_clean, token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
                           min_df=2, sublinear_tf=True)
    if features == "word":
        vec = word
    else:
        char = TfidfVectorizer(preprocessor=_clean, analyzer="char_wb", ngram_range=(2, 5),
                               min_df=2, sublinear_tf=True)
        vec = FeatureUnion([("word", word), ("char", char)])
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=4000)
    return Pipeline([("tfidf", vec), ("clf", clf)])


GRID = {
    "features": ["word", "word+char"],
    "C": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "class_weight": [None, "balanced"],
}


def main() -> int:
    train, val, test = (load_split(s) for s in config.SPLITS)
    out_dir = config.RESULTS_DIR / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Majority class (learned on train)
    majority = int(train["label"].mode().iloc[0])
    for split, df in (("val", val), ("test", test)):
        write_predictions("majority", split, df["id"], np.full(len(df), majority))
    print(f"majority: always predicts {config.LABELS[majority]}")

    # 2. TF-IDF + logistic regression, selected on val
    log = []
    best = None
    for features, C, cw in itertools.product(*GRID.values()):
        t0 = time.time()
        pipe = make_pipeline(features, C, cw).fit(train["text"], train["label"])
        score = macro_f1(val["label"].to_numpy(), pipe.predict(val["text"]))
        log.append({"features": features, "C": C, "class_weight": cw,
                    "val_macro_f1": score, "fit_seconds": round(time.time() - t0, 2)})
        print(f"  {features:9s} C={C:<5} cw={str(cw):8s} val macro-F1={score:.4f}")
        if best is None or score > best[0]:
            best = (score, features, C, cw, pipe)

    score, features, C, cw, pipe = best
    for split, df in (("val", val), ("test", test)):
        write_predictions("tfidf-logreg", split, df["id"], pipe.predict(df["text"]))

    t0 = time.perf_counter()
    pipe.predict(test["text"])
    ms_per_item = 1000 * (time.perf_counter() - t0) / len(test)

    selection = {"selected": {"features": features, "C": C, "class_weight": cw},
                 "val_macro_f1": score, "inference_ms_per_item_cpu": ms_per_item, "grid": log}
    (out_dir / "tfidf-logreg.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"tfidf-logreg selected: {features}, C={C}, class_weight={cw} (val macro-F1 {score:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
