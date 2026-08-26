#!/usr/bin/env python3
"""Dump free-space cho MỘT checkpoint depth trên toàn bộ frame có nhãn.

Cùng schema với logs/depth_*.json của 11_depth_probe.py + thêm đặc trưng
tương đối (extent, per_col_ratio) để fusion dùng tín hiệu scale-free.
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/home/ubuntu/ai-assistant/models/hf")

ROOT = Path('/home/ubuntu/ai-assistant')
sys.path.insert(0, str(ROOT / 'scripts'))
import depth_path as dp


def main() -> int:
    model_id = sys.argv[1]
    out_path = Path(sys.argv[2])
    from PIL import Image
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=torch.float16).to('cuda:0').eval()

    files = sorted((ROOT / 'assets/frames').glob('*.jpg'))
    gt = json.load(open(ROOT / 'assets/frames/ground_truth.json'))['frames']
    rows = {}
    t0 = time.perf_counter()
    lat = []
    for f in files:
        name = f.name
        g = gt.get(name, {})
        if g.get('hazard') is None or g.get('excluded'):
            continue
        img = Image.open(f).convert('RGB')
        w, h = img.size
        if max(w, h) > dp_free_long_edge():
            s = dp_free_long_edge() / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        z = proc(images=img, return_tensors='pt').to('cuda:0')
        z['pixel_values'] = z['pixel_values'].half()
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            out = model(**z)
        dep = proc.post_process_depth_estimation(out, target_sizes=[(img.height, img.width)])[0]['predicted_depth'].float().cpu().numpy().squeeze()
        torch.cuda.synchronize(); lat.append((time.perf_counter() - t) * 1000)
        band, col = dp.path_masks(*dep.shape)
        free, grid = dp.free_space(dep, band, col)
        s = dp.summarize_free(free)
        finite = grid[np.isfinite(grid)]
        extent = float(np_max(finite))
        s['per_col_ratio'] = [None if v is None or extent <= 0 else round(v / extent, 3) for v in s['per_col']]
        s['extent'] = round(extent, 2)
        rows[name] = s
        print(f"  {name:<26} free3={s['free3']:7.2f} extent={s['extent']:7.2f} ratio={s['per_col_ratio']}")

    report = {
        'checkpoint': model_id,
        'metric': True,
        'pct': dp.DEPTH_PCT,
        'long_edge': 640,
        'latency_median_ms': sorted(lat)[len(lat)//2] if lat else None,
        'frames': rows,
    }
    out_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f"\n{len(rows)} frame -> {out_path}")
    return 0


def dp_free_long_edge() -> int:
    return 640


def np_max(a):
    import numpy as np
    return float(np.max(a)) if a.size else 0.0


if __name__ == '__main__':
    raise SystemExit(main())
