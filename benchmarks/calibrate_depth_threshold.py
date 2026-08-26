#!/usr/bin/env python3
"""Hiệu chuẩn ngưỡng depth cho checkpoint Indoor-Large trên 27 frame có nhãn.

Quy tắc chọn ngưỡng như logs/depth_round2.md: trong các ngưỡng còn giữ
recall 100%, chọn ngưỡng specificity cao nhất. An toàn trước, không tối ưu
balanced accuracy (đã từng đánh mất 2 frame vật cản thật).
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/ubuntu/ai-assistant')
sys.path.insert(0, str(ROOT / 'scripts'))
import mvp_config as cfg
import depth_path as dp


def scan(depth_dump_path: Path):
    dump = json.load(open(depth_dump_path))
    frames = {k: v['free3'] for k, v in dump['frames'].items()}
    gt = json.load(open(ROOT / 'assets/frames/ground_truth.json'))['frames']
    vlm = json.load(open(ROOT / 'logs/eval_vlm_qwen3vl4b.json'))['results']['pipe_v6']
    parsed = {r['frame']: cfg.parse_pipe4(r['output']) for r in vlm}

    truth = {}
    for name, g in gt.items():
        if g.get('hazard') is None:
            continue
        truth[name] = (bool(g['hazard']), bool(g.get('hard', False)))

    print(f"{'thr':>6}  recall specif sp.khó  balanc   bỏ sót / báo thừa")
    best = None
    for thr in [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 26, 30]:
        tp = fp = tn = fn = hard_fp = 0
        miss, extra = [], []
        for name, (t, hard) in truth.items():
            p = parsed.get(name)
            vlm_hazard = bool(p and p.get('hazard'))
            escape = bool(p) and p.get('type') in dp.DEPTH_BLIND_TYPES and p.get('distance') == 'near'
            free = frames.get(name, float('nan'))
            geo_near = free == free and free < thr
            said = vlm_hazard and (geo_near or escape)
            if t and said: tp += 1
            elif t and not said: fn += 1; miss.append(name)
            elif not t and said:
                fp += 1; extra.append(name)
                if hard: hard_fp += 1
            else: tn += 1
        rec = tp / (tp + fn); spec = tn / (tn + fp)
        bal = (rec + spec) / 2
        mark = ''
        if rec == 1.0 and (best is None or spec > best[2]):
            best = (thr, rec, spec, bal)
            mark = ' <-'
        print(f"{thr:>5}m  {rec:5.0%} {spec:5.0%} {1-hard_fp/max(sum(1 for n in truth if not truth[n][0] and truth[n][1]),1):5.0%}  {bal:5.0%}   "
              f"{','.join(m.split('.')[0] for m in miss) or '—'} / {','.join(e.split('.')[0] for e in extra) or '—'}{mark}")
    return best


def dump_ckpt(p):
    import json as j
    return j.load(open(p))['checkpoint'].split('-V2-')[1].replace('-hf', '')


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'indoor_p3'
    path = ROOT / f'logs/depth_{which}.json'
    if not path.exists():
        raise SystemExit(f'không thấy {path}')
    best = scan(path)
    if best:
        print(f"\n=> ngưỡng đề xuất cho {dump_ckpt(path)}: {best[0]}m "
              f"(recall {best[1]:.0%}, spec {best[2]:.0%})")


if __name__ == '__main__':
    main()
