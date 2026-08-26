#!/usr/bin/env python3
"""Eval ngữ nghĩa + fusion trên COCO val2017 (thay data thật không thu được).

Nhãn suy ra từ GT bbox theo quy tắc của coco_select_download.py (xem đầu file đó).
⚠️ Đây là proxy: ảnh COCO không phải toàn bộ góc nhìn người đi bộ, nhãn "vật
trong vùng lối đi" ≠ hoàn toàn "nguy hiểm cho người khiếm thị". Dùng để so SÁNH
TƯƠNG ĐỐI giữa các chiến lược trên phân phối rộng hơn 27 frame tự tạo.

Chạy: benchmarks/evaluate_vlm_public.py [--limit N] [--threshold M]
Kết quả -> logs/vlm_public_coco.json + .md
"""
from __future__ import annotations

import argparse, io, json, os, statistics, sys, time
from pathlib import Path

for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
    os.environ.pop(k, None)

ROOT = Path('/home/ubuntu/ai-assistant')
sys.path.insert(0, str(ROOT / 'scripts'))
import mvp_config as cfg
import fusion
import depth_path as dp

VLM_MODEL = cfg.VLM_MODEL
DEPTH_MODEL = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf'
LONG_EDGE, JPEG_Q = cfg.FRAME_LONG_EDGE, cfg.JPEG_QUALITY


def prep(image):
    from PIL import Image
    image = image.convert('RGB'); w, h = image.size
    if max(w, h) > LONG_EDGE:
        s = LONG_EDGE / max(w, h); image = image.resize((round(w * s), round(h * s)), Image.LANCZOS)
    b = io.BytesIO(); image.save(b, format='JPEG', quality=JPEG_Q); b.seek(0)
    return Image.open(b).convert('RGB')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--threshold', type=float, default=7.0)   # đã hiệu chuẩn Indoor-Large
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoProcessor, AutoModelForImageTextToText

    labels = {r['file']: r for r in json.load(open(ROOT / 'logs/coco_public_labels.json'))}
    files = [f for f in sorted(labels) if (ROOT / 'benchmarks/data/coco/val2017' / f).exists()][:args.limit]
    print(f'COCO val2017: {len(files)} ảnh ({sum(1 for f in files if labels[f]["hazard"])} dương)')

    vp = AutoProcessor.from_pretrained(VLM_MODEL)
    vm = AutoModelForImageTextToText.from_pretrained(VLM_MODEL, dtype=torch.float16,
                                                     attn_implementation=cfg.ATTN_IMPL, device_map='cuda:0').eval()
    dproc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    dmodel = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, dtype=torch.float16).to('cuda:0').eval()

    rows = []
    for i, fname in enumerate(files):
        image = prep(Image.open(ROOT / 'benchmarks/data/coco/val2017' / fname))
        messages = [{'role': 'system', 'content': [{'type': 'text', 'text': cfg.VLM_PROMPT_PIPE}]},
                    {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': 'Frame:'}]}]
        x = vp.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors='pt').to(vm.device)
        n = x['input_ids'].shape[-1]
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            out = vm.generate(**x, max_new_tokens=cfg.VLM_MAX_NEW_TOKENS, do_sample=False)
        raw = vp.decode(out[0][n:], skip_special_tokens=True).strip()
        torch.cuda.synchronize(); t_vlm = (time.perf_counter() - t) * 1000
        parsed = cfg.parse_pipe4(raw)

        z = dproc(images=image, return_tensors='pt').to('cuda:0'); z['pixel_values'] = z['pixel_values'].half()
        t = time.perf_counter()
        with torch.inference_mode():
            dout = dmodel(**z)
        dep = dproc.post_process_depth_estimation(dout, target_sizes=[(image.height, image.width)])[0]['predicted_depth'].float().cpu().numpy().squeeze()
        t_depth = (time.perf_counter() - t) * 1000
        ev_geo = fusion.depth_evidence(dep)

        rows.append({'file': fname, 'truth': labels[fname]['hazard'], 'types_gt': labels[fname]['types'],
                     'vlm_raw': raw, 'vlm': parsed, 'ms_vlm': round(t_vlm), 'ms_depth': round(t_depth),
                     'free_min': ev_geo.get('free_min'), 'near_rank': ev_geo.get('near_rank'),
                     'col': ev_geo.get('col')})
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(files)}')

    def metrics(said_fn):
        tp = fp = tn = fn_ = 0
        for r in rows:
            said = said_fn(r)
            if r['truth'] and said: tp += 1
            elif r['truth'] and not said: fn_ += 1
            elif not r['truth'] and said: fp += 1
            else: tn += 1
        rec = tp / max(tp + fn_, 1); spec = tn / max(tn + fp, 1)
        return {'recall': round(rec, 4), 'specificity': round(spec, 4),
                'balanced': round((rec + spec) / 2, 4), 'tp': tp, 'fp': fp, 'fn': fn_, 'tn': tn}

    thr = args.threshold
    strategies = {
        'A_vlm_near_mid': lambda r: bool(r['vlm'] and r['vlm']['hazard'] and r['vlm'].get('distance') in ('near', 'mid')),
        'B_vlm_near_only': lambda r: bool(r['vlm'] and r['vlm']['hazard'] and r['vlm'].get('distance') == 'near'),
        'E_fusion_calibrated': lambda r: (
            lambda p, geo: bool(p and p['hazard'] and (
                (geo['free_min'] is not None and geo['free_min'] < thr)
                or (geo['near_rank'] is not None and geo['near_rank'] >= 0.5)
                or (p.get('type') in dp.DEPTH_BLIND_TYPES and p.get('distance') == 'near')))
        )(r['vlm'], r),
    }
    summary = {'n': len(rows), 'n_positive': sum(1 for r in rows if r['truth']),
               'depth_threshold_m': thr,
               'latency_ms': {
                   'vlm_median': round(statistics.median(r['ms_vlm'] for r in rows)),
                   'vlm_p95': round(sorted(r['ms_vlm'] for r in rows)[int(len(rows) * .95)]),
                   'depth_median': round(statistics.median(r['ms_depth'] for r in rows))},
               }
    for name, fn in strategies.items():
        m = metrics(fn)
        summary[name] = m
        bal = m['balanced']
        print(f"{name:<20} recall {m['recall']:.0%} spec {m['specificity']:.0%} "
              f"balanced {bal:.0%}  tp/fp/fn/tn={m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")

    out = ROOT / 'logs/vlm_public_coco.json'
    out.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1) + '\n')
    (ROOT / 'logs/vlm_public_coco.md').write_text(
        '# VLM+depth trên COCO val2017 (nhãn từ GT bbox)\n\n```\n'
        + json.dumps(summary, indent=2) + '\n```\n')
    print('-> ', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
