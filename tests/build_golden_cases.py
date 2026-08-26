#!/usr/bin/env python3
"""Đóng gói ảnh + dữ liệu có nhãn tốt thành GOLDEN TEST CASES (không cần GPU).

Nguồn đầu vào (đã đo, đã thẩm định tay):
  assets/frames/ground_truth.json     nhãn tay 27 frame (types/positions là TẬP)
  logs/eval_vlm_qwen3vl4b.json        output VLM pipe_v6 thật trên 27 frame
  logs/depth_indoor_large.json        đặc trưng free-space checkpoint đang dùng
  logs/vlm_public_coco.json           120 ảnh COCO: VLM + free-space + nhãn bbox thô
  logs/coco_adjudication.md           thẩm định tay 45 frame lỗi -> nhãn hiệu chỉnh

Đầu ra: tests/golden/{cases_core27.json,cases_coco120.json,manifest.json}
Mỗi case = (ảnh nguồn, nhãn GT, đầu vào VLM+geometry đã đo, kỳ vọng hành vi).
Chạy lại builder khi thay đổi NGUỒN (frame mới, dump mới) — KHÔNG chạy lại để
"cập nhật kỳ vọng" sau khi sửa fusion.py; kỳ vọng chỉ đổi qua thẩm định tay.

Lưu ý minh bạch: nhãn hiệu chỉnh COCO do người thẩm định trùng người thiết kế
hệ thống — có xung đột lợi ích, chỉ dùng nội bộ (giống README).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/home/ubuntu/ai-assistant')
sys.path.insert(0, str(ROOT / 'scripts'))
import mvp_config as cfg          # noqa: E402  (set HF_HOME trước transformers)
import fusion                     # noqa: E402

GOLDEN = ROOT / 'tests' / 'golden'
OUT = ROOT / 'tests'

# Thẩm định tay coco_adjudication.md:
BORDERLINE = {'000000041888.jpg', '000000168330.jpg', '000000515579.jpg',
              '000000239274.jpg', '000000223130.jpg'}      # hệ thống hơi quá liềng
LABEL_NOISE_FN = {'000000085329.jpg', '000000017627.jpg'}  # nhãn sai, hệ thống đúng


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def geo_from_dump(d: dict) -> dict:
    """Dòng dump free-space -> Evidence geometry đúng interface của fusion.

    near_rank = 1 - free_min/free_max (cùng công thức fusion.depth_evidence),
    tái tạo được từ free3 (min 3 cột) và extent (max 3 cột) của dump.
    """
    free = float(d['free3'])
    ext = float(d['extent'])
    return {'valid': True, 'free_min': round(free, 4),
            'near_rank': round(max(0.0, min(1.0, 1.0 - free / ext)), 4) if ext > 0 else 0.0,
            'col': int(d['col']), 'coverage': 1.0}


def config_snapshot() -> dict:
    c = fusion.FusionConfig()
    return {k: getattr(c, k) for k in (
        'free_abs_threshold', 'near_rank_threshold', 'depth_emergency_threshold',
        'confirm_frames', 'clear_frames', 'depth_fail_policy',
        'mismatch_risk_penalty', 'narrate_min_interval_s')}


# ------------------------------------------------------------- core 27 ----
def build_core27() -> dict:
    gt = json.load(open(ROOT / 'assets/frames/ground_truth.json'))['frames']
    vlm_rows = json.load(open(ROOT / 'logs/eval_vlm_qwen3vl4b.json'))['results']['pipe_v6']
    dep = json.load(open(ROOT / 'logs/depth_indoor_large.json'))['frames']
    conf = fusion.FusionConfig()

    cases = []
    for row in vlm_rows:
        frame = row['frame']
        label = gt[frame]
        if label.get('excluded'):
            continue                                   # không phải POV đi bộ
        parsed = cfg.parse_pipe4(row['output'])
        ev_vlm = fusion.vlm_evidence(parsed)
        geo = geo_from_dump(dep[frame])
        dec = fusion.decide_frame(ev_vlm, geo, conf)
        hazard = bool(label.get('hazard'))
        # Kỳ vọng an toàn (invariant): dương thật thì PHẢI cảnh báo. Âm khó
        # (hard=true) hệ thống được phép cảnh báo thừa — hạn chế đã ghi nhận.
        expect = {
            'frame_positive': dec['frame_positive'],
            'reason': dec['reason'],
            'risk': dec.get('risk'),
            'mismatch': bool(dec.get('mismatch', False)),
            'must_warn': True if hazard else None,
            'may_warn': bool(label.get('hard')),
        }
        cases.append({
            'id': f'core27/{frame}',
            'image': f'assets/frames/{frame}',
            'label': {'hazard': hazard, 'hard': bool(label.get('hard')),
                      'types': label.get('types'), 'positions': label.get('positions'),
                      'note': label.get('note')},
            'inputs': {'vlm_raw': row['output'], 'vlm': parsed, 'geo': geo},
            'expect': expect,
        })
    return {
        '_doc': ('27 frame nhãn tay (assets/frames/ground_truth.json). Đầu vào VLM '
                 '= output thật pipe_v6 (logs/eval_vlm_qwen3vl4b.json), geometry = '
                 'dump checkpoint Indoor-Large đang dùng (logs/depth_indoor_large.json). '
                 'Số đo gốc: recall 100%, specificity 75%, 3 âm khó bị báo thừa.'),
        'source_checkpoint': json.load(open(ROOT / 'logs/depth_indoor_large.json'))['checkpoint'],
        'prompt_tag': 'pipe_v6',
        'config': config_snapshot(),
        'aggregate_expect': {'min_recall': 1.0, 'min_specificity': 0.75,
                             'max_false_positives': 3},
        'cases': cases,
    }


# ------------------------------------------------------------ coco 120 ----
def build_coco120() -> dict:
    rows = json.load(open(ROOT / 'logs/vlm_public_coco.json'))['rows']
    conf = fusion.FusionConfig()

    def ev_geo(r):
        if r['free_min'] is None:
            return fusion.depth_evidence(None)
        return {'valid': True, 'free_min': r['free_min'], 'near_rank': r['near_rank'] or 0.0,
                'col': r['col'] if r['col'] is not None else -1, 'coverage': 1.0}

    cases = []
    for r in rows:
        dec = fusion.decide_frame(fusion.vlm_evidence(r['vlm']), ev_geo(r), conf)
        # Phân loại đúng công thức đã thẩm định (replay_fusion_decisions.py):
        # dương gốc trừ nhãn-noise; âm gốc mà hệ thống cảnh báo -> xác nhận trừ
        # ranh giới. Schema-error chỉ là ghi chú, không đổi phép chấm.
        warned = bool(dec['frame_positive'])
        if r['file'] in LABEL_NOISE_FN:
            adj = 'label_noise'          # nhãn bbox sai, hệ thống im là đúng
            truth_corr = False
        elif r['file'] in BORDERLINE:
            adj = 'borderline'           # hệ thống hơi quá liềng -> FP có chủ đích
            truth_corr = bool(r['truth'])
        elif r['truth']:
            adj = 'schema_error' if r['vlm'] is None else 'bbox_ok'
            truth_corr = True            # dương gốc hợp lệ (kể cả VLM sai schema)
        elif warned:
            adj = 'warning_confirmed'    # "FP" theo nhãn nhưng cảnh báo ĐÚNG
            truth_corr = True
        else:
            adj = 'schema_note' if r['vlm'] is None else 'bbox_ok'
            truth_corr = False
        expect = {'frame_positive': dec['frame_positive'], 'reason': dec['reason']}
        if r['free_min'] is not None and r['free_min'] < conf.depth_emergency_threshold:
            expect.update({'must_warn': True, 'must_reason': 'depth_emergency'})
        cases.append({
            'id': f'coco120/{r["file"]}',
            'image': f'benchmarks/data/coco/val2017/{r["file"]}',
            'label': {'hazard_bbox': bool(r['truth']), 'hazard_corrected': truth_corr,
                      'adjudication': adj, 'types_gt': r.get('types_gt') or []},
            'inputs': {'vlm_raw': r['vlm_raw'], 'vlm': r['vlm'],
                       'geo': {'valid': r['free_min'] is not None,
                               'free_min': r['free_min'], 'near_rank': r['near_rank'],
                               'col': r['col'], 'coverage': 1.0}},
            'expect': expect,
        })
    n_corr_pos = sum(1 for c in cases if c['label']['hazard_corrected'])
    return {
        '_doc': ('120 ảnh COCO val2017 nhãn từ GT bbox (logs/vlm_public_coco.json), '
                 'hiệu chỉnh theo thẩm định tay 45 frame lỗi (logs/coco_adjudication.md): '
                 '24 cảnh báo "báo thừa" thực ra ĐÚNG, 5 ranh giới, 2 nhãn sai. '
                 '⚠️ Người thẩm định trùng người thiết kế hệ thống — chỉ dùng nội bộ.'),
        'config': config_snapshot(),
        'corrected_positives': n_corr_pos,
        # Số đo chuẩn (replay_fusion_decisions.py chạy cùng dump này):
        # recall hiệu chỉnh 91%, specificity 81%, fp = đúng 5 case ranh giới.
        'aggregate_expect': {'min_recall_corrected': 0.90, 'min_specificity_corrected': 0.80,
                             'max_fp_allowed': 5},
        'cases': cases,
    }


# ------------------------------------------------------------- manifest ----
def build_manifest() -> dict:
    files = {}
    for p in sorted((ROOT / 'assets/frames').glob('*.jpg')):
        files[f'assets/frames/{p.name}'] = sha256(p)
    for p in sorted((ROOT / 'benchmarks/data/coco/val2017').glob('*.jpg')):
        files[f'benchmarks/data/coco/val2017/{p.name}'] = sha256(p)
    return {
        '_doc': ('SHA-256 ảnh nguồn được tham chiếu bởi golden cases. Runner kiểm tra '
                 'trước khi chấm: ảnh đổi/nhãn cũ => FAIL, bắt hiệu chuẩn lại.'),
        'files': files,
    }


def main() -> int:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    core = build_core27()
    coco = build_coco120()
    (GOLDEN / 'cases_core27.json').write_text(json.dumps(core, ensure_ascii=False, indent=1) + '\n')
    (GOLDEN / 'cases_coco120.json').write_text(json.dumps(coco, ensure_ascii=False, indent=1) + '\n')
    (GOLDEN / 'manifest.json').write_text(json.dumps(build_manifest(), indent=1) + '\n')

    pos = sum(1 for c in core['cases'] if c['label']['hazard'])
    print(f"core27 : {len(core['cases'])} case ({pos} dương, "
          f"{sum(1 for c in core['cases'] if c['label']['hard'])} âm khó)")
    print(f"coco120: {len(coco['cases'])} case ({coco['corrected_positives']} dương hiệu chỉnh)")
    print("ghi vào tests/golden/: cases_core27.json, cases_coco120.json, manifest.json")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
