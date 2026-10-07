"""Train one LoRA run on the train split, keep the adapter with the best dev macro-F1.

    python scripts/train_lora.py --name sel-lr2e-4-balanced-s0 --lr 2e-4 --weighting balanced --seed 0

Writes checkpoints/<name>/best (adapter, not committed) and results/training/<name>.json (log).
Does not touch the test split. Use scripts/run_lora.py to run the whole protocol.
"""
from __future__ import annotations

import argparse
import sys

import torch

from bbg.data import load_split
from bbg.lora import TrainConfig, checkpoint_dir, load_base, train, training_log_path, wrap_lora


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-id", default=TrainConfig.model_id)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--weighting", choices=["none", "balanced"], default=TrainConfig.weighting)
    ap.add_argument("--epochs", type=float, default=TrainConfig.epochs)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--grad-accum", type=int, default=TrainConfig.grad_accum)
    ap.add_argument("--evals-per-epoch", type=int, default=TrainConfig.evals_per_epoch)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train", type=int, help="debug only: subsample the train split")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available: run scripts/check_gpu.py first.")
        return 1

    cfg = TrainConfig(name=args.name, model_id=args.model_id, seed=args.seed, lr=args.lr,
                      weighting=args.weighting, epochs=args.epochs, batch_size=args.batch_size,
                      grad_accum=args.grad_accum, evals_per_epoch=args.evals_per_epoch,
                      device=args.device)
    train_df, dev_df = load_split("train"), load_split("val")
    if args.max_train:
        train_df = train_df.sample(n=args.max_train, random_state=args.seed)

    model, tok = load_base(cfg.model_id, cfg.device)
    model = wrap_lora(model, cfg)
    model.print_trainable_parameters()
    log = train(cfg, model, tok, train_df, dev_df, checkpoint_dir(cfg.name), training_log_path(cfg.name))
    b = log["best"]
    print(f"{cfg.name}: best dev macro-F1 {b['dev_macro_f1']:.4f} at step {b['step']} "
          f"({log['seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
