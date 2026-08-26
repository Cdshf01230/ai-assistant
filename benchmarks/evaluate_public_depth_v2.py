#!/usr/bin/env python3
"""Stratified public depth benchmark with raw and median-scale metrics."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HOME", "/home/ubuntu/ai-assistant/models/hf")

ROOT = Path('/home/ubuntu/ai-assistant')
DIODE = ROOT / 'benchmarks/data/diode/val'
NYU = ROOT / 'benchmarks/data/nyu/nyu_depth_v2_labeled.mat'

def score(pred, truth):
    mask = np.isfinite(pred) & np.isfinite(truth) & (pred > 0) & (truth > 0)
    pred, truth = pred[mask], truth[mask]
    if not len(pred): return {'pixels': 0}
    ratio = np.maximum(pred / truth, truth / pred)
    scale = np.median(truth / pred)
    aligned = pred * scale
    aligned_ratio = np.maximum(aligned / truth, truth / aligned)
    return {
        'pixels': int(len(pred)),
        'scale_median': float(scale),
        'abs_rel': float(np.mean(np.abs(pred - truth) / truth)),
        'rmse': float(np.sqrt(np.mean((pred - truth) ** 2))),
        'delta1': float(np.mean(ratio < 1.25)),
        'aligned_abs_rel': float(np.mean(np.abs(aligned - truth) / truth)),
        'aligned_rmse': float(np.sqrt(np.mean((aligned - truth) ** 2))),
        'aligned_delta1': float(np.mean(aligned_ratio < 1.25)),
    }

def mean_scores(rows):
    keys = ('abs_rel','rmse','delta1','aligned_abs_rel','aligned_rmse','aligned_delta1')
    return {'samples': len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys}}

def predict(proc, model, image, shape):
    import torch
    x = proc(images=image, return_tensors='pt').to('cuda:0')
    x['pixel_values'] = x['pixel_values'].half()
    with torch.inference_mode(): out = model(**x)
    post = proc.post_process_depth_estimation(out, target_sizes=[shape])
    return np.squeeze(post[0]['predicted_depth'].float().cpu().numpy())

def diode(proc, model, limit):
    from PIL import Image
    result = {}
    for domain in ('indoors','outdoor'):
        rows = []
        for dp in sorted((DIODE/domain).rglob('*_depth.npy'))[:limit]:
            rgb = dp.with_name(dp.name.replace('_depth.npy','.png'))
            mask = dp.with_name(dp.name.replace('_depth.npy','_depth_mask.npy'))
            if not rgb.exists() or not mask.exists(): continue
            truth = np.squeeze(np.load(dp).astype('float32'))
            valid = np.squeeze(np.load(mask).astype(bool))
            pred = predict(proc, model, Image.open(rgb).convert('RGB'), truth.shape)
            rows.append(score(pred[valid], truth[valid]))
        result[domain] = mean_scores(rows)
    return result

def nyu(proc, model, limit):
    import h5py
    from PIL import Image
    rows = []
    with h5py.File(NYU, 'r') as f:
        for i in range(min(limit, f['images'].shape[0])):
            rgb = np.transpose(f['images'][i], (2,1,0)).astype('uint8')
            truth = np.squeeze(np.array(f['depths'][i], dtype='float32').T)
            rows.append(score(predict(proc, model, Image.fromarray(rgb), truth.shape), truth))
    return mean_scores(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModelForDepthEstimation.from_pretrained(a.model, dtype=torch.float16).to('cuda:0').eval()
    report = {'model': a.model, 'limit_per_split': a.limit, 'diode': diode(proc, model, a.limit), 'nyu': nyu(proc, model, a.limit)}
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
