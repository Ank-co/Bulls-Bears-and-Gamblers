"""The full fine-tuning protocol, in one command. Safe to re-run: finished runs are skipped.

1. Selection on the dev split, seed 0: learning rate x class weighting (4 runs).
2. The configuration with the best dev macro-F1 is chosen by code, not by hand.
3. That configuration is trained with two more seeds.
4. The three seeds write their val and test predictions as qwen3-1.7b-lora-s0/s1/s2.

    python scripts/run_lora.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from bbg import config
from bbg.lora import training_log_path

GRID = {"lr": [1e-4, 2e-4], "weighting": ["none", "balanced"]}
SEEDS = [0, 1, 2]
FINAL = "qwen3-1.7b-lora"
HERE = Path(__file__).parent


def run_name(lr: float, weighting: str, seed: int) -> str:
    return f"lr{lr:.0e}".replace("e-0", "e-") + f"-{weighting}-s{seed}"


def is_done(name: str) -> bool:
    path = training_log_path(name)
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("done", False)


def call(script: str, *args: str) -> None:
    cmd = [sys.executable, str(HERE / script), *args]
    print("\n$ " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra arguments passed to train_lora.py (debug)")
    args = ap.parse_args()
    common = ["--device", args.device, *args.extra]

    # 1. Selection
    candidates = []
    for lr, weighting in itertools.product(GRID["lr"], GRID["weighting"]):
        name = run_name(lr, weighting, 0)
        if not is_done(name):
            call("train_lora.py", "--name", name, "--lr", str(lr), "--weighting", weighting,
                 "--seed", "0", *common)
        best = json.loads(training_log_path(name).read_text(encoding="utf-8"))["best"]
        candidates.append({"run": name, "lr": lr, "weighting": weighting,
                           "dev_macro_f1": best["dev_macro_f1"], "step": best["step"]})

    # 2. Choice by code: highest dev macro-F1, ties broken by grid order
    chosen = max(candidates, key=lambda c: c["dev_macro_f1"])
    summary = {"grid": GRID, "candidates": candidates, "chosen": chosen, "seeds": SEEDS}
    out = config.RESULTS_DIR / "selection" / "lora.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nchosen: lr={chosen['lr']} weighting={chosen['weighting']} "
          f"(dev macro-F1 {chosen['dev_macro_f1']:.4f})")

    # 3. Seeds and 4. predictions
    for seed in SEEDS:
        name = run_name(chosen["lr"], chosen["weighting"], seed)
        if not is_done(name):
            call("train_lora.py", "--name", name, "--lr", str(chosen["lr"]),
                 "--weighting", chosen["weighting"], "--seed", str(seed), *common)
        call("predict_lora.py", "--run", name, "--as", f"{FINAL}-s{seed}", "--device", args.device)
    print("\nDone. Next: python scripts/evaluate.py --split val")
    return 0


if __name__ == "__main__":
    sys.exit(main())
