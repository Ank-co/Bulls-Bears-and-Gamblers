"""LoRA fine-tuning of a causal LM on the sentiment task.

Loss = minus the log-probability of the gold answer, computed by the same function that scores
answers at evaluation time (`llm_scoring.sequence_logprobs`). Optionally weighted by class so
that minority classes count as much as the majority one (the primary metric is macro-F1).

Model selection uses the dev split only: the adapter with the best dev macro-F1 is kept.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import config
from .llm_scoring import encode_pair, pad_id_of, pad_left, predict, sequence_logprobs
from .metrics import accuracy, macro_f1
from .prompts import ANSWERS


@dataclass
class TrainConfig:
    name: str
    model_id: str = "Qwen/Qwen3-1.7B"
    seed: int = 0
    lr: float = 2e-4
    weighting: str = "balanced"          # "none" or "balanced"
    epochs: float = 2.0
    batch_size: int = 8
    grad_accum: int = 4
    evals_per_epoch: int = 2
    warmup_ratio: float = 0.05
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    max_grad_norm: float = 1.0
    eval_batch_size: int = 16
    device: str = "cuda"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels: np.ndarray, mode: str) -> torch.Tensor:
    k = len(ANSWERS)
    if mode == "none":
        return torch.ones(k)
    if mode != "balanced":
        raise ValueError(f"unknown weighting {mode!r}")
    counts = np.bincount(labels, minlength=k).astype(float)
    return torch.tensor(len(labels) / (k * counts), dtype=torch.float)


def batch_loss(model, tokenizer, texts: list[str], labels: list[int], weights: torch.Tensor) -> torch.Tensor:
    pairs = [encode_pair(tokenizer, t, ANSWERS[y]) for t, y in zip(texts, labels)]
    input_ids, mask = pad_left([p[0] for p in pairs], pad_id_of(tokenizer))
    lp = sequence_logprobs(model, input_ids, mask, [p[1] for p in pairs])
    w = weights[torch.tensor(labels)].to(lp.device)
    return -(w * lp).sum() / w.sum()


def lr_lambda(total_steps: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    return f


def train(cfg: TrainConfig, model, tokenizer, train_df, dev_df, out_dir: Path, log_path: Path) -> dict:
    """Trains LoRA adapters on `model` (already wrapped by peft). Returns the training log."""
    set_seed(cfg.seed)
    weights = class_weights(train_df["label"].to_numpy(), cfg.weighting)
    texts, labels = train_df["text"].tolist(), train_df["label"].tolist()
    dev_texts, dev_y = dev_df["text"].tolist(), dev_df["label"].to_numpy()

    micro_per_epoch = math.ceil(len(texts) / cfg.batch_size)
    steps_per_epoch = math.ceil(micro_per_epoch / cfg.grad_accum)
    total_steps = math.ceil(steps_per_epoch * cfg.epochs)
    eval_every = max(1, steps_per_epoch // cfg.evals_per_epoch)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total_steps, int(cfg.warmup_ratio * total_steps)))
    use_amp = cfg.device == "cuda"
    gen = torch.Generator().manual_seed(cfg.seed)

    log = {"config": asdict(cfg), "class_weights": weights.tolist(), "total_steps": total_steps,
           "evals": [], "best": None, "done": False}
    t0 = time.time()
    step, running, n_running = 0, 0.0, 0
    model.train()
    while step < total_steps:
        order = torch.randperm(len(texts), generator=gen).tolist()
        micro = [order[i:i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
        for start in range(0, len(micro), cfg.grad_accum):
            group = micro[start:start + cfg.grad_accum]
            for idx in group:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = batch_loss(model, tokenizer, [texts[i] for i in idx],
                                      [labels[i] for i in idx], weights)
                (loss / len(group)).backward()
                running += loss.item()
                n_running += 1
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % eval_every == 0 or step == total_steps:
                model.eval()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    preds, _ = predict(model, tokenizer, dev_texts, cfg.eval_batch_size)
                model.train()
                entry = {"step": step, "epoch": round(step / steps_per_epoch, 3),
                         "train_loss": running / max(1, n_running),
                         "dev_macro_f1": macro_f1(dev_y, np.array(preds)),
                         "dev_accuracy": accuracy(dev_y, np.array(preds)),
                         "elapsed_s": round(time.time() - t0, 1)}
                running, n_running = 0.0, 0
                log["evals"].append(entry)
                if log["best"] is None or entry["dev_macro_f1"] > log["best"]["dev_macro_f1"]:
                    log["best"] = entry
                    model.save_pretrained(out_dir / "best")
                print(f"  step {step}/{total_steps}  loss {entry['train_loss']:.4f}  "
                      f"dev macro-F1 {entry['dev_macro_f1']:.4f}", flush=True)
                write_log(log, log_path)
            if step >= total_steps:
                break

    log["seconds"] = round(time.time() - t0, 1)
    if cfg.device == "cuda":
        log["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    log["done"] = True
    write_log(log, log_path)
    return log


def write_log(log: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")


def load_base(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.config.use_cache = False
    return model, tok


def wrap_lora(model, cfg: TrainConfig):
    """Adds LoRA adapters. The seed is set first: it controls the adapter initialization too."""
    from peft import LoraConfig, get_peft_model
    set_seed(cfg.seed)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=cfg.target_modules, task_type="CAUSAL_LM")
    return get_peft_model(model, lcfg)


def checkpoint_dir(name: str) -> Path:
    return config.ROOT / "checkpoints" / name


def training_log_path(name: str) -> Path:
    return config.RESULTS_DIR / "training" / f"{name}.json"
