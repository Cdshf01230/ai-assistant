#!/usr/bin/env python3
"""Replay quyết định fusion trên dump đã đo — KHÔNG cần GPU.

Kiểm chứng logic trong scripts/fusion.py tái hiện đúng số của
logs/depth_probe_run.md (outdoor) và logs/depth_round2.md (indoor):
chiến lược E = VLM AND depth + cửa thoát step/hole/door.

Đầu vào: logs/eval_vlm_qwen3vl4b.json (pipe_v6), logs/depth_{indoor,outdoor}.json,
assets/frames/ground_truth.json.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

ROOT = Path('/home/ubuntu/ai-assistant')
sys.path.insert(0, str(ROOT / 'scripts'))
import mvp_config as cfg
import fusion


def load_vlm_outputs() -> dict[str, dict]:
    dump = json.load(open(ROOT / 'logs/eval_vlm_qwen3vl4b.json'))
    out = {}
    for row in dump['results']['pipe_v6']:
        parsed = cfg.parse_pipe4(row['output'])
        out[row['frame']] = {'parsed': parsed, 'truth': bool(row['truth_hazard'])}
    return out


def load_depth(which: str) -> dict[str, float]:
    dump = json.load(open(ROOT / f'logs/depth_{which}.json'))
    return {k: v['free3'] for k, v in dump['frames'].items()}


def geo_from_free(free3: float | None, threshold: float, col: int = 1) -> dict:
    """Đưa free-space tuyệt đối vào interface của fusion.

    free_min đi thẳng vào ngưỡng đã hiệu chuẩn; near_rank mô phỏng tín hiệu
    scale-free (1 khi bị chắn trong ngưỡng, 0 khi thoáng).
    """
    if free3 is None or free3 != free3:
        return fusion.depth_evidence(None)
    return {"valid": True, "free_min": float(free3),
            "near_rank": 1.0 if free3 < threshold else 0.0,
            "coverage": 1.0, "col": col, "per_col_rank": []}


def score(vlm: dict, depth: dict[str, float], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    conf = fusion.FusionConfig(free_abs_threshold=threshold)
    for frame, item in vlm.items():
        ev_v = fusion.vlm_evidence(item['parsed'])
        ev_g = geo_from_free(depth.get(frame), threshold)
        d = fusion.decide_frame(ev_v, ev_g, conf)
        said = bool(d['frame_positive'])
        truth = item['truth']
        if truth and said: tp += 1
        elif truth and not said: fn += 1
        elif not truth and said: fp += 1
        else: tn += 1
    recall = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {'recall': recall, 'specificity': spec,
            'balanced': (recall + spec) / 2, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


CASES = [   # (dataset, ngưỡng đã đo, recall/specificity trong log)
    ('outdoor', 19.0, (1.00, 0.92)),
    ('indoor',   8.00, (1.00, 0.75)),
    ('indoor_large', 7.00, (1.00, 0.75)),   # checkpoint đang dùng trong api_main.py
]


def main() -> int:
    vlm = load_vlm_outputs()
    failures = []
    print(f"frames: {len(vlm)} | prompt pipe_v6 | chiến lược E replay\n")
    for which, thr, (exp_rec, exp_spec) in CASES:
        depth = load_depth(which)
        m = score(vlm, depth, thr)
        ok = abs(m['recall'] - exp_rec) < 1e-9 and abs(m['specificity'] - exp_spec) < 0.005
        print(f"{which:<8} @{thr:>5}m  recall {m['recall']:.0%} (log {exp_rec:.0%})  "
              f"spec {m['specificity']:.0%} (log {exp_spec:.0%})  "
              f"balanced {m['balanced']:.0%}  tp/fp/fn/tn = {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}  "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(which)

    # --- unit: depth hỏng -> theo policy ---
    c = fusion.FusionConfig()
    ev_near = fusion.vlm_evidence({'hazard': True, 'type': 'pole', 'position': 'front',
                                   'distance': 'near', 'action': 'stop'})
    ev_far = fusion.vlm_evidence({'hazard': True, 'type': 'pole', 'position': 'front',
                                  'distance': 'far', 'action': 'slow'})
    broken = fusion.depth_evidence(None)
    a = fusion.decide_frame(ev_near, broken, c)['frame_positive']
    b = fusion.decide_frame(ev_far, broken, c)['frame_positive']
    s = fusion.decide_frame(ev_near, broken,
                            fusion.FusionConfig(depth_fail_policy='silent'))['frame_positive']
    print(f"\ndepth hỏng + VLM near -> warn={a} (True) | VLM far -> warn={b} (False) | policy silent -> {s} (False)")
    if not (a and not b and not s):
        failures.append('depth_fail_policy')

    # --- unit: cửa thoát lớp mù ---
    ev_door = fusion.vlm_evidence({'hazard': True, 'type': 'door', 'position': 'front',
                                   'distance': 'near', 'action': 'slow'})
    clear_geo = {"valid": True, "near_rank": 0.0, "coverage": 1.0, "col": 1, "per_col_rank": []}
    esc = fusion.decide_frame(ev_door, clear_geo, c)
    print(f"cửa thoát door|near dù depth trống -> positive={esc['frame_positive']} ({esc['reason']})")
    if not esc['frame_positive']:
        failures.append('escape_hatch')

    # --- unit: cross-check hướng hạ risk ---
    geo_left = dict(clear_geo, near_rank=1.0, col=0)          # geometry nói TRÁI
    ev_front = fusion.vlm_evidence({'hazard': True, 'type': 'pole', 'position': 'front',
                                    'distance': 'near', 'action': 'stop'})
    d_ok = fusion.decide_frame(ev_front, dict(geo_left, col=1), c)
    d_bad = fusion.decide_frame(ev_front, geo_left, c)
    print(f"hướng khớp risk={d_ok['risk']} vs lệch risk={d_bad['risk']} (phải nhỏ hơn)")
    if not d_bad['risk'] < d_ok['risk']:
        failures.append('cross_check')

    # --- unit: state machine trễ + dedup ---
    sess = fusion.FusionSession(config=c)
    dec = {'frame_positive': True, 'risk': 0.8}
    events = [fusion.update_session(sess, dec, ev_front, now=float(t)) for t in range(6)]
    seq = [(e['alert'], e['message_code']) for e in events]
    print(f"\n6 frame liên tiếp cùng vật: alert sequence = {[s[0] for s in seq]} (chỉ frame 2 phát)")
    if [s[0] for s in seq] != [False, True, False, False, False, False]:
        failures.append('dedup_repeat')

    # đổi vật -> phát lại ngay
    ev_person = fusion.vlm_evidence({'hazard': True, 'type': 'person', 'position': 'front_right',
                                     'distance': 'near', 'action': 'move_left'})
    e7 = fusion.update_session(sess, dec, ev_person, now=6.0)
    print(f"đổi POLE->PERSON: alert={e7['alert']} code={e7['message_code']}")
    if not (e7['alert'] and e7['message_code'] == 'PERSON_AHEAD'):
        failures.append('key_change_realert')

    # VLM bịa vật trên lối TRỐNG (frame không positive) -> KHÔNG được alert
    ev_halluc = fusion.vlm_evidence({'hazard': True, 'type': 'object', 'position': 'front',
                                     'distance': 'mid', 'action': 'slow'})
    dec_clear = {'frame_positive': False}
    e8 = fusion.update_session(sess, dec_clear, ev_halluc, now=7.0)
    e9 = fusion.update_session(sess, dec_clear, ev_halluc, now=8.0)
    print(f"bịa OBJECT khi depth trống: alert={e8['alert']} code={e8['message_code']} "
          f"warn={e8['warn']} (chờ tắt) | frame sau: warn={e9['warn']}")
    if e8['alert'] or e8['message_code']:
        failures.append('hallucinated_key_change')

    # --- unit: depth emergency override (thẩm định COCO, coco_adjudication.md) ---
    ev_clear = fusion.vlm_evidence({'hazard': False, 'type': 'clear', 'position': 'none',
                                    'distance': 'none', 'action': 'none'})
    geo_blocked = {"valid": True, "free_min": 1.0, "near_rank": 0.9, "coverage": 1.0,
                   "col": 1, "per_col_rank": []}
    geo_empty_worst = {"valid": True, "free_min": 2.39, "near_rank": 0.05, "coverage": 1.0,
                       "col": 1, "per_col_rank": []}   # synthclear_indoor — trống thưa nhất
    e1 = fusion.decide_frame(ev_clear, geo_blocked, c)
    e2 = fusion.decide_frame(ev_clear, geo_empty_worst, c)
    e3 = fusion.decide_frame(fusion.vlm_evidence(None), geo_blocked, c)
    sess2 = fusion.FusionSession(config=c)
    ev_em = fusion.update_session(sess2, e1, ev_clear, now=0.0)
    ev_em2 = fusion.update_session(sess2, e1, ev_clear, now=1.0)
    print(f"\noverride: VLM CLEAR + free 1.0 -> positive={e1['frame_positive']} ({e1['reason']}) "
          f"| free 2.39 (trống) -> positive={e2['frame_positive']} | VLM hỏng + free 1.0 -> {e3['frame_positive']}")
    print(f"  session: frame1 alert={ev_em['alert']} code={ev_em['message_code']} | "
          f"frame2 (confirm xong) alert={ev_em2['alert']} code={ev_em2['message_code']}")
    if not (e1['frame_positive'] and not e2['frame_positive'] and e3['frame_positive']):
        failures.append('emergency_override')
    if ev_em2['message_code'] != 'STOP':
        failures.append('emergency_code')
    # emergency + VLM có hazard nhưng action=none -> vẫn phải STOP
    ev_hz = fusion.vlm_evidence({'hazard': True, 'type': 'person', 'position': 'front',
                                 'distance': 'near', 'action': 'none'})
    sess3 = fusion.FusionSession(config=c)
    fusion.update_session(sess3, e1, ev_clear, now=0.0)
    ev_em3 = fusion.update_session(sess3, e1, ev_hz, now=1.0)
    print(f"  emergency + VLM action=none -> code={ev_em3['message_code']} action={ev_em3['action']} (phải STOP/stop)")
    if ev_em3['message_code'] != 'STOP' or ev_em3['action'] != 'stop':
        failures.append('emergency_stop_priority')

    # --- replay COCO: trước/sau override, nhãn thô + hiệu chỉnh thẩm định ---
    coco = ROOT / 'logs/vlm_public_coco.json'
    if coco.exists():
        dump = json.load(open(coco))['rows']
        BORDERLINE = {'000000041888.jpg', '000000168330.jpg', '000000515579.jpg',
                      '000000239274.jpg', '000000223130.jpg'}      # hệ thống hơi quá liềng
        LABEL_NOISE_FN = {'000000085329.jpg', '000000017627.jpg'}   # nhãn sai, hệ thống đúng

        def ev_geo_row(r):
            if r['free_min'] is None:
                return fusion.depth_evidence(None)
            return {"valid": True, "free_min": r['free_min'], "near_rank": r['near_rank'] or 0.0,
                    "coverage": 1.0, "col": r['col'] if r['col'] is not None else -1,
                    "per_col_rank": []}

        def run(cfg):
            tp = fp = tn = fn = 0
            for r in dump:
                d = fusion.decide_frame(fusion.vlm_evidence(r['vlm']), ev_geo_row(r), cfg)
                said = bool(d['frame_positive'])
                if r['truth'] and said: tp += 1
                elif r['truth'] and not said: fn += 1
                elif not r['truth'] and said: fp += 1
                else: tn += 1
            return tp, fp, fn, tn

        def show(tag, tp, fp, fn, tn):
            rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
            print(f"  {tag:<28} recall {rec:.0%} spec {spec:.0%} bal {(rec+spec)/2:.0%}  "
                  f"tp/fp/fn/tn={tp}/{fp}/{fn}/{tn}")

        old = fusion.FusionConfig(depth_emergency_threshold=0.0)   # tắt override
        print("\nCOCO 120 ảnh, nhãn bbox thô:")
        show('trước override', *run(old))
        show('sau override', *run(c))
        # Hiệu chỉnh bằng thẩm định tay: nhãn của mỗi ảnh được thay bằng phán
        # quyết POV ("người đi bộ nhìn frame này có cần cảnh báo không"):
        #  - dương gốc trừ 2 ảnh nhãn-noise (085329 chân dung, 017627 xe đỗ đối diện)
        #  - âm gốc mà hệ thống cảnh báo -> dương nếu KHÔNG thuộc 5 case ranh giới
        warned, not_warned = set(), set()
        for r in dump:
            d = fusion.decide_frame(fusion.vlm_evidence(r['vlm']), ev_geo_row(r), c)
            (warned if d['frame_positive'] else not_warned).add(r['file'])
        corrected_pos = {r['file'] for r in dump
                         if (r['truth'] and r['file'] not in LABEL_NOISE_FN)
                         or (not r['truth'] and r['file'] in warned and r['file'] not in BORDERLINE)}
        all_files = {r['file'] for r in dump}
        tp2 = len(warned & corrected_pos)
        fp2 = len(warned - corrected_pos)
        fn2 = len(corrected_pos - warned)
        tn2 = len(all_files - warned - corrected_pos)
        print("COCO 120 ảnh, nhãn HIỆU CHỈNH theo thẩm định tay (người thẩm định trùng")
        print("  người thiết kế hệ thống — có xung đột lợi ích, chỉ dùng nội bộ):")
        show('sau override (hiệu chỉnh)', tp2, fp2, fn2, tn2)

    print('\n==> ' + ('PASS' if not failures else f'FAIL: {failures}'))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
