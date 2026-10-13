"""Part 4: the tone reader (FinBERT) and the tone manipulation check.

    python scripts/part4_tone.py --split dev      # labels, fits the mapping and freezes it
    python scripts/part4_tone.py --split test     # labels, applies the mapping frozen on dev

FinBERT labels each news sentence as the model sees it (dramatic wording included, track record
excluded): positive +1, negative -1, neutral 0. A scenario's tone is the sum over its news. On dev, a
logistic regression of the outcome on that sum gives the tone reader's probability; the mapping is
written to results/part4/tone_reader.json and reused unchanged on the test split. The manipulation
check compares FinBERT's polarity (P(positive) - P(negative)) on dramatic and neutral sentences.
Writes results/part4/<split>/tone.jsonl and tone_check.json.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import statsmodels.api as sm

from bbg import config, fictional

MAPPING = config.RESULTS_DIR / "part4" / "tone_reader.json"
TONE = {"positive": 1, "negative": -1, "neutral": 0}


def make_finbert(device: str = "cpu", batch_size: int = 64):
    """Returns texts -> list of (tone, polarity), with labels read by name from the model config."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device).eval()
    names = {int(i): n.lower() for i, n in model.config.id2label.items()}
    if set(names.values()) != set(TONE):
        raise ValueError(f"unexpected FinBERT labels: {names}")
    pos = next(i for i, n in names.items() if n == "positive")
    neg = next(i for i, n in names.items() if n == "negative")

    @torch.no_grad()
    def classify(texts: list[str]) -> list[tuple[int, float]]:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(device)
            probs = model(**enc).logits.softmax(-1).cpu().numpy()
            out += [(TONE[names[int(p.argmax())]], float(p[pos] - p[neg])) for p in probs]
        return out
    return classify


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(fictional.SPLITS), required=True)
    ap.add_argument("--device", default="cpu", help="cpu by default: the GPU may still hold the 35B model")
    args = ap.parse_args()
    path = config.DATA_DIR / "part4" / args.split / "scenarios.jsonl"
    scenarios = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if args.split != "dev" and not MAPPING.exists():
        print(f"Refused: {MAPPING.name} is fitted on dev first (python scripts/part4_tone.py --split dev).")
        return 1

    sentences = sorted({it["sentence"] for sc in scenarios for it in sc["items"]})
    labelled = dict(zip(sentences, make_finbert(args.device)(sentences)))
    tone_sum = np.array([sum(labelled[it["sentence"]][0] for it in sc["items"]) for sc in scenarios], float)
    y = np.array([sc["outcome"] for sc in scenarios])

    if args.split == "dev":
        fit = sm.Logit(y, sm.add_constant(tone_sum)).fit(disp=0)
        mapping = {"intercept": float(fit.params[0]), "slope": float(fit.params[1]),
                   "fitted_on": "dev", "scenarios": int(len(y))}
        MAPPING.parent.mkdir(parents=True, exist_ok=True)
        MAPPING.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    else:
        mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    p_tone = 1 / (1 + np.exp(-(mapping["intercept"] + mapping["slope"] * tone_sum)))

    out = config.RESULTS_DIR / "part4" / args.split
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tone.jsonl", "w", encoding="utf-8") as f:
        for sc, s, p in zip(scenarios, tone_sum, p_tone):
            f.write(json.dumps({"id": sc["id"], "tone_sum": int(s), "p_tone": float(p),
                                "item_tones": [labelled[it["sentence"]][0] for it in sc["items"]]}) + "\n")

    items = [it for sc in scenarios for it in sc["items"]]
    check = {}
    for name, sel in (("dramatic", [it for it in items if it["dramatic"]]),
                      ("neutral", [it for it in items if not it["dramatic"]])):
        pol = np.array([labelled[it["sentence"]][1] for it in sel])
        check[name] = {"items": len(sel), "mean_abs_polarity": float(np.mean(np.abs(pol))),
                       "share_not_neutral": float(np.mean([labelled[it["sentence"]][0] != 0 for it in sel])),
                       "share_reading_as_conventional": float(np.mean(
                           [labelled[it["sentence"]][0] == it["textbook"] for it in sel]))}
    check["dramatic_more_polarized"] = check["dramatic"]["mean_abs_polarity"] > check["neutral"]["mean_abs_polarity"]
    check["mapping"] = mapping
    (out / "tone_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print(json.dumps(check, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
