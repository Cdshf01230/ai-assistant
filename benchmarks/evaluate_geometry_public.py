#!/usr/bin/env python3
"""Đánh giá tín hiệu hình học free-space trên depth GT công khai (DIODE indoor).

Câu hỏi: near_rank suy ra từ depth DỰ ĐOÁN có khớp với near_rank từ depth GT
không? Đây là cách kiểm chứng nửa hình học của fusion mà KHÔNG cần data thật:
GT của DIODE là mét thật, nên nếu rank khớp thì tín hiệu thứ tự dùng được bất
chấp thang mét của checkpoint sai 4-5 lần (logs/depth_round2.md).

Kết quả -> logs/geometry_diode.json + .md
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path('/home/ubuntu/ai-assistant')
sys.path.insert(0, str(ROOT / 'scripts'))
import fusion
import mvp_config as cfg


def list_diode_indoor(limit: int):
    base = ROOT / 'benchmarks/data/diode/val/indoors'
    pairs = []
    for gt in sorted(base.glob('*/*/*_depth.npy')):
        rgb = str(gt).replace('_depth.npy', '.png')
        if not Path(rgb).exists():
            continue
        pairs.append((rgb, str(gt)))
        if len(pairs) >= limit:
            break
    return pairs


def prep_image(path: str):
    from PIL import Image
    image = Image.open(path).convert('RGB')
    w, h = image.size
    if max(w, h) > cfg.FRAME_LONG_EDGE:
        s = cfg.FRAME_LONG_EDGE / max(w, h)
        image = image.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=60)
    ap.add_argument('--output', default=str(ROOT / 'logs/geometry_diode.json'))
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf'
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=torch.float16).to('cuda:0').eval()
    masks_cache: dict[tuple[int, int], tuple] = {}

    def masks_for(h, w):
        if (h, w) not in masks_cache:
            masks_cache[(h, w)] = dpath_masks(h, w)
        return masks_cache[(h, w)]

    from depth_path import path_masks as dpath_masks

    pairs = list_diode_indoor(args.limit)
    print(f'DIODE indoor: {len(pairs)} cặp RGB-GT | model {model_id}')

    rows = []
    t0 = time.perf_counter()
    for i, (rgb_path, gt_path) in enumerate(pairs):
        image = prep_image(rgb_path)
        gt = np.load(gt_path)[..., 0].astype(np.float32)
        # GT ở 768x1024 -> resize về kích thước ảnh đã prep để cùng lưới toạ độ
        gt_img = Image.fromarray(gt).resize(image.size, Image.BILINEAR)
        gt_resized = np.asarray(gt_img, dtype=np.float32)

        z = proc(images=image, return_tensors='pt').to('cuda:0')
        z['pixel_values'] = z['pixel_values'].half()
        with torch.inference_mode():
            out = model(**z)
        pred = proc.post_process_depth_estimation(out, target_sizes=[(image.height, image.width)])
        pred = pred[0]['predicted_depth'].float().cpu().numpy().squeeze()
        pred = np.asarray(Image.fromarray(pred).resize(image.size, Image.BILINEAR), dtype=np.float32)

        ev_pred = fusion.depth_evidence(pred, masks_for(*pred.shape))
        ev_gt = fusion.depth_evidence(gt_resized, masks_for(*gt_resized.shape))
        rows.append({
            'sample': Path(rgb_path).name,
            'pred': {k: ev_pred.get(k) for k in ('valid', 'near_rank', 'col', 'coverage')},
            'gt': {k: ev_gt.get(k) for k in ('valid', 'near_rank', 'col', 'coverage')},
        })
        if (i + 1) % 20 == 0:
            print(f'  {i + 1}/{len(pairs)} ({time.perf_counter() - t0:.0f}s)')

    ok = [r for r in rows if r['pred']['valid'] and r['gt']['valid']]
    rp = np.array([r['pred']['near_rank'] for r in ok])
    rg = np.array([r['gt']['near_rank'] for r in ok])
    from scipy.stats import spearmanr, pearsonr
    sp = spearmanr(rp, rg); pe = pearsonr(rp, rg)
    flag_p = rp >= 0.5; flag_g = rg >= 0.5
    agree = float((flag_p == flag_g).mean())
    col_agree = float(np.mean([r['pred']['col'] == r['gt']['col'] for r in ok]))
    summary = {
        'model': model_id,
        'samples': len(rows), 'valid_pairs': len(ok),
        'spearman_rho': round(float(sp.statistic), 4), 'spearman_p': float(sp.pvalue),
        'pearson_r': round(float(pe.statistic), 4),
        'median_abs_rank_error': round(float(np.median(np.abs(rp - rg))), 4),
        'near_flag_agreement_at_0.5': round(agree, 4),
        'column_agreement': round(col_agree, 4),
        'latency_per_frame_ms': round((time.perf_counter() - t0) * 1000 / max(len(rows), 1)),
    }
    Path(args.output).write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2) + '\n')
    md = args.output.replace('.json', '.md')
    Path(md).write_text('# Geometry vs GT (DIODE indoor)\n\n```\n'
                        + json.dumps(summary, indent=2) + '\n```\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
