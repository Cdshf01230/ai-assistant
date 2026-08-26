#!/usr/bin/env python3
"""Evaluate the configured depth model on public DIODE and NYU samples."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/home/ubuntu/ai-assistant/models/hf")

ROOT = Path("/home/ubuntu/ai-assistant")
DIODE = ROOT / "benchmarks/data/diode/val"
NYU = ROOT / "benchmarks/data/nyu/nyu_depth_v2_labeled.mat"


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(truth) & (truth > 0) & (pred > 0)
    pred, truth = pred[mask], truth[mask]
    ratio = np.maximum(pred / truth, truth / pred)
    return {
        "pixels": int(mask.sum()),
        "abs_rel": float(np.mean(np.abs(pred - truth) / truth)),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def predict(processor, model, image, size: tuple[int, int]) -> np.ndarray:
    import torch

    inputs = processor(images=image, return_tensors="pt").to("cuda:0")
    inputs["pixel_values"] = inputs["pixel_values"].half()
    with torch.inference_mode():
        output = model(**inputs)
    post = processor.post_process_depth_estimation(output, target_sizes=[size])
    return np.squeeze(post[0]["predicted_depth"].float().cpu().numpy())


def run_diode(processor, model, limit: int) -> dict[str, object]:
    from PIL import Image

    pairs = []
    for depth_path in sorted(DIODE.rglob("*_depth.npy")):
        rgb_path = depth_path.with_name(depth_path.name.replace("_depth.npy", ".png"))
        mask_path = depth_path.with_name(depth_path.name.replace("_depth.npy", "_depth_mask.npy"))
        if rgb_path.exists() and mask_path.exists():
            pairs.append((rgb_path, depth_path, mask_path))
    pairs = pairs[:limit]
    values = []
    for rgb_path, depth_path, mask_path in pairs:
        image = Image.open(rgb_path).convert("RGB")
        truth = np.squeeze(np.load(depth_path).astype(np.float32))
        valid = np.squeeze(np.load(mask_path).astype(bool))
        pred = predict(processor, model, image, truth.shape)
        values.append(metrics(pred[valid], truth[valid]))
    return aggregate(values, len(pairs))


def run_nyu(processor, model, limit: int) -> dict[str, object]:
    import h5py
    from PIL import Image

    values = []
    with h5py.File(NYU, "r") as handle:
        count = min(limit, handle["images"].shape[0])
        for index in range(count):
            rgb = np.transpose(handle["images"][index], (2, 1, 0)).astype(np.uint8)
            truth = np.squeeze(np.array(handle["depths"][index], dtype=np.float32).T)
            values.append(metrics(predict(processor, model, Image.fromarray(rgb), truth.shape), truth))
    return aggregate(values, min(limit, count))


def aggregate(rows: list[dict[str, float]], count: int) -> dict[str, object]:
    if not rows:
        return {"samples": 0, "error": "no valid pairs"}
    return {"samples": count, **{key: float(np.mean([row[key] for row in rows])) for key in ("abs_rel", "rmse", "delta1")}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    parser.add_argument("--output", default="/home/ubuntu/ai-assistant/benchmarks/public_depth_report.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model, dtype=torch.float16).to("cuda:0")
    model.eval()
    report = {"model": args.model, "limit": args.limit, "diode": run_diode(processor, model, args.limit), "nyu": run_nyu(processor, model, args.limit)}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
