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

# Part 3: headlines and returns (FNSPID). Data stay outside the repository (CC BY-NC 4.0).
PART3_PERIODS = {"dev": ("2016-01-01", "2017-12-31"), "test": ("2018-01-01", "2019-12-31")}
PART3_SAMPLE_SIZE = {"dev": 30_000, "test": 150_000}    # ticker-dates
PART3_SEED = 31
PART3_MARKET = "SPY"
PART3_WINSOR = (0.005, 0.995)
PART3_NW_LAGS = 5
PART3_DIR = RESULTS_DIR / "part3"
