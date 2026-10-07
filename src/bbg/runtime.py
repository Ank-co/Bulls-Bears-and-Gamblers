"""Cost measurements written next to the predictions: latency, throughput, peak VRAM."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

from . import config


@contextmanager
def timed():
    box = {}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - t0


def gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    return {"device": torch.cuda.get_device_name(0),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}


def write_runtime(model: str, split: str, n_items: int, seconds: float, **extra) -> None:
    path = config.RESULTS_DIR / "runtime" / f"{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[split] = {"items": n_items, "seconds": round(seconds, 3),
                   "ms_per_item": round(1000 * seconds / max(n_items, 1), 3), **extra}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
