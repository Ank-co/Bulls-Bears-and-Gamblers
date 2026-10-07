"""Fine-tuning checks on a tiny Qwen3, CPU only."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from bbg.lora import TrainConfig, batch_loss, class_weights, load_base, train, wrap_lora  # noqa: E402
from bbg.llm_scoring import predict  # noqa: E402
from bbg.metrics import macro_f1  # noqa: E402

WORDS = {0: "plunge drop", 1: "jump rally", 2: "holds update"}


def toy_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1, 2], size=n, p=[0.2, 0.2, 0.6])
    return pd.DataFrame({"text": [f"$AAPL {WORDS[int(y)]} {i}" for i, y in enumerate(labels)],
                         "label": labels})


def test_class_weights_balanced_and_none():
    y = np.array([0] + [1] * 2 + [2] * 7)
    w = class_weights(y, "balanced")
    assert torch.allclose(w, torch.tensor([10 / 3, 10 / 6, 10 / 21]))
    assert torch.equal(class_weights(y, "none"), torch.ones(3))
    with pytest.raises(ValueError):
        class_weights(y, "weird")


def test_only_lora_parameters_are_trainable(tiny_qwen_dir):
    cfg = TrainConfig(name="t", model_id=str(tiny_qwen_dir), device="cpu")
    model, _ = load_base(cfg.model_id, "cpu")
    model = wrap_lora(model, cfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)


def test_training_learns_and_saved_adapter_reproduces_dev_score(tiny_qwen_dir, tmp_path):
    from peft import PeftModel

    train_df, dev_df = toy_frame(96, 0), toy_frame(48, 1)
    cfg = TrainConfig(name="t", model_id=str(tiny_qwen_dir), device="cpu", lr=5e-3, epochs=6,
                      batch_size=8, grad_accum=2, evals_per_epoch=1, weighting="balanced")
    model, tok = load_base(cfg.model_id, "cpu")
    model = wrap_lora(model, cfg)
    w = class_weights(train_df["label"].to_numpy(), "balanced")
    with torch.no_grad():
        before = batch_loss(model, tok, train_df["text"].tolist(), train_df["label"].tolist(), w).item()

    log_path = tmp_path / "log.json"
    log = train(cfg, model, tok, train_df, dev_df, tmp_path / "ckpt", log_path)
    with torch.no_grad():
        after = batch_loss(model.eval(), tok, train_df["text"].tolist(), train_df["label"].tolist(), w).item()

    assert log["done"] and json.loads(log_path.read_text())["done"]
    assert after < 0.7 * before, (before, after)
    majority = macro_f1(dev_df["label"].to_numpy(), np.full(len(dev_df), 2))
    assert log["best"]["dev_macro_f1"] > majority + 0.2
    assert log["best"]["dev_macro_f1"] == max(e["dev_macro_f1"] for e in log["evals"])

    base, tok2 = load_base(cfg.model_id, "cpu")
    reloaded = PeftModel.from_pretrained(base, tmp_path / "ckpt" / "best").eval()
    preds, _ = predict(reloaded, tok2, dev_df["text"].tolist(), batch_size=16)
    assert macro_f1(dev_df["label"].to_numpy(), np.array(preds)) == pytest.approx(
        log["best"]["dev_macro_f1"], abs=1e-9)


def test_training_is_reproducible_with_same_seed(tiny_qwen_dir, tmp_path):
    train_df, dev_df = toy_frame(32, 0), toy_frame(16, 1)
    losses = []
    for run in ("a", "b"):
        cfg = TrainConfig(name=run, model_id=str(tiny_qwen_dir), device="cpu", lr=1e-3, epochs=1,
                          batch_size=8, grad_accum=1, evals_per_epoch=1, seed=7)
        model, tok = load_base(cfg.model_id, "cpu")
        torch.manual_seed(run == "a" and 1 or 2)    # global RNG differs: the seed must still win
        model = wrap_lora(model, cfg)
        log = train(cfg, model, tok, train_df, dev_df, tmp_path / run, tmp_path / f"{run}.json")
        losses.append([e["train_loss"] for e in log["evals"]])
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)


def test_different_seeds_give_different_adapter_init(tiny_qwen_dir):
    inits = []
    for seed in (0, 1):
        cfg = TrainConfig(name="t", model_id=str(tiny_qwen_dir), device="cpu", seed=seed)
        model, _ = load_base(cfg.model_id, "cpu")
        model = wrap_lora(model, cfg)
        inits.append(next(p for n, p in model.named_parameters() if "lora_A" in n).detach().clone())
    assert not torch.equal(inits[0], inits[1])
