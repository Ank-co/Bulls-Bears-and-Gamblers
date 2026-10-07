"""Project-wide constants. Every script reads its paths and seeds from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"

HF_DATASET = "zeroshot/twitter-financial-news-sentiment"

# Label ids exactly as published in the dataset card.
LABELS = {0: "bearish", 1: "bullish", 2: "neutral"}
LABEL_IDS = sorted(LABELS)

SPLIT_SEED = 42          # seed of the train/val split, never changed after the first run
VAL_FRACTION = 0.10      # share of the official train set held out for model selection
BOOTSTRAP_SEED = 2026
BOOTSTRAP_ITERS = 10_000

SPLITS = ("train", "val", "test")
