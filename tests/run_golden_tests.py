#!/usr/bin/env python3
"""Chạy golden test cases — hồi quy fusion KHÔNG cần GPU.

Bộ case được đóng gói bởi tests/build_golden_cases.py từ ảnh có nhãn tay
(27 frame) + COCO 120 ảnh (nhãn bbox + hiệu chỉnh thẩm định). Mỗi case ghim
đầu vào VLM+geometry ĐÃ ĐO và kỳ vọng hành vi; runner so với scripts/fusion.py.

Kiểm tra theo 3 mức:
  A. INVARIANT an toàn  — mọi dương thật phải được cảnh báo; frame free_min <
     emergency threshold phải STOP kể cả VLM CLEAR; depth hỏng thì theo policy.
  B. REGRESSION         — fingerprint từng case (positive/reason/risk) khớp bản
     ghi; lệch => sửa fusion đã đổi hành vi, phải thẩm định lại rồi mới chạy
     `build_golden_cases.py` để cập nhật có chủ đích.
  C. AGGREGATE          — recall/specificity tổng không tụt dưới ngưỡng đã đo.

Cách chạy:  python tests/run_golden_tests.py            (exit 0 = PASS)
            python tests/run_golden_tests.py -v         (in từng case lệch)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/home/ubuntu/ai-assistant')
sys.path.insert(0, str(ROOT / 'scripts'))
import mvp_config as cfg   # noqa: E402
import fusion              # noqa: E402

GOLDEN = ROOT / 'tests' / 'golden'
FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)


def load_cases(name: str) -> dict:
    return json.load(open(GOLDEN / name))


# ------------------------------------------------------- manifest check ----
def check_manifest() -> None:
    man = load_cases('manifest.json')
    missing = broken = 0
    for rel, digest in man['files'].items():
        p = ROOT / rel
        if not p.exists():
            missing += 1
            continue
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        if h.hexdigest() != digest:
            broken += 1
    n = len(man['files'])
    print(f"[manifest] {n} ảnh: thiếu {missing}, sai checksum {broken}")
    if missing or broken:
        fail(f"manifest: {missing} thiếu, {broken} sai checksum — ảnh nguồn đã đổi, "
             f"nhãn cũ không còn đúng; chạy lại build_golden_cases.py sau khi thẩm định.")


# ------------------------------------------------------------ replay -------
def ev_geo(case: dict) -> dict:
    g = case['inputs']['geo']
    if not g.get('valid'):
        return fusion.depth_evidence(None)
    return {'valid': True, 'free_min': g['free_min'], 'near_rank': g['near_rank'] or 0.0,
            'col': g['col'] if g['col'] is not None else -1,
            'coverage': g.get('coverage', 1.0)}


def run_suite(tag: str, suite: dict, verbose: bool) -> tuple[int, int]:
    conf_d = suite['config']
    conf = fusion.FusionConfig(**{k: v for k, v in conf_d.items()})
    strict = tag.startswith('core27')     # bộ nhãn tay: chấm per-case; COCO: aggregate
    label_key = 'hazard' if strict else 'hazard_corrected'
    tp = fp = tn = fn = 0
    for case in suite['cases']:
        dec = fusion.decide_frame(fusion.vlm_evidence(case['inputs']['vlm']),
                                  ev_geo(case), conf)
        said = bool(dec['frame_positive'])
        truth = bool(case['label'][label_key])
        exp = case['expect']

        # --- A. invariant ---
        if truth:
            if said:
                tp += 1
            else:
                fn += 1
                if strict:
                    fail(f"{case['id']}: DƯƠNG THẬT bị bỏ sót (reason={dec['reason']})")
        elif said:
            fp += 1
            if strict and not case['label'].get('hard'):
                fail(f"{case['id']}: cảnh báo trên âm (không phải hard-negative)")
        else:
            tn += 1

        # emergency override: free_min < ngưỡng => PHẢI positive vì depth_emergency
        if exp.get('must_warn') and not said:
            fail(f"{case['id']}: must_warn nhưng im (reason={dec['reason']})")
        if exp.get('must_reason') and dec['reason'] != exp['must_reason']:
            fail(f"{case['id']}: reason {dec['reason']} != {exp['must_reason']}")

        # --- B. regression fingerprint ---
        for key in ('frame_positive', 'reason', 'risk'):
            if key in exp and dec.get(key) != exp[key]:
                fail(f"{case['id']}: {key}={dec.get(key)!r} golden={exp[key]!r}")
                if verbose:
                    print(f"  LỆCH {case['id']}: {key} {dec.get(key)!r} != {exp[key]!r}")
        if 'mismatch' in exp and bool(dec.get('mismatch')) != exp['mismatch']:
            fail(f"{case['id']}: mismatch={bool(dec.get('mismatch'))} golden={exp['mismatch']}")
            if verbose:
                print(f"  LỆCH {case['id']}: mismatch")

    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    agg = suite['aggregate_expect']
    min_rec = agg.get('min_recall', agg.get('min_recall_corrected'))
    min_spec = agg.get('min_specificity', agg.get('min_specificity_corrected'))
    ok_rec = rec >= min_rec
    ok_spec = spec >= min_spec
    fp_ok = fp <= agg.get('max_fp_allowed', 10**9)
    print(f"[{tag}] recall {rec:.0%} (min {min_rec:.0%}) "
          f"{'OK' if ok_rec else 'TỤT'} | specificity {spec:.0%} "
          f"(min {min_spec:.0%}) {'OK' if ok_spec else 'TỤT'} | "
          f"fp={fp} (max {agg.get('max_fp_allowed', '—')})"
          f"{' OK' if fp_ok else ' VƯỢT'} | tp/fp/fn/tn={tp}/{fp}/{fn}/{tn}")
    if not ok_rec:
        fail(f"{tag}: recall {rec:.0%} < {min_rec:.0%}")
    if not ok_spec:
        fail(f"{tag}: specificity {spec:.0%} < {min_spec:.0%}")
    if not fp_ok:
        fail(f"{tag}: fp={fp} > {agg['max_fp_allowed']} case ranh giới đã thẩm định")
    return tp + fn, tn + fp


# ------------------------------------------------- unit hành vi fusion -----
def run_units(verbose: bool) -> None:
    c = fusion.FusionConfig()
    geo_clear = {"valid": True, "free_min": 30.0, "near_rank": 0.0, "coverage": 1.0,
                 "col": 1}
    # 3.0: trong khoảng (emergency, ngưỡng chắn) để đi qua đường fusion thường
    geo_blocked = {"valid": True, "free_min": 3.0, "near_rank": 0.95, "coverage": 1.0,
                   "col": 1}
    geo_emergency = dict(geo_blocked, free_min=1.0)

    def ev(obj, pos, dist, act):
        return fusion.vlm_evidence({'hazard': obj != 'clear', 'type': obj,
                                    'position': pos, 'distance': dist, 'action': act})

    # 1. depth hỏng -> theo policy
    a = fusion.decide_frame(ev('pole', 'front', 'near', 'stop'), fusion.depth_evidence(None), c)
    b = fusion.decide_frame(ev('pole', 'front', 'far', 'slow'), fusion.depth_evidence(None), c)
    s = fusion.decide_frame(ev('pole', 'front', 'near', 'stop'), fusion.depth_evidence(None),
                            fusion.FusionConfig(depth_fail_policy='silent'))
    if verbose:
        print(f"  depth hỏng: near->{a['frame_positive']} far->{b['frame_positive']} silent->{s['frame_positive']}")
    if not (a['frame_positive'] and not b['frame_positive'] and not s['frame_positive']):
        fail('unit depth_fail_policy')

    # 2. cửa thoát lớp mù step/hole/door
    esc = fusion.decide_frame(ev('door', 'front', 'near', 'slow'), geo_clear, c)
    if not esc['frame_positive']:
        fail('unit escape_blind_type')
    no_esc = fusion.decide_frame(ev('pole', 'front', 'near', 'stop'),
                                 dict(geo_clear, free_min=30.0), c)
    if no_esc['frame_positive']:
        fail('unit depth_veto_pole_on_clear_path')

    # 3. cross-check hướng
    d_ok = fusion.decide_frame(ev('pole', 'front', 'near', 'stop'), dict(geo_blocked, col=1), c)
    d_bad = fusion.decide_frame(ev('pole', 'front', 'near', 'stop'), dict(geo_blocked, col=0), c)
    if not d_bad['risk'] < d_ok['risk']:
        fail('unit cross_check_penalty')
    if d_bad['risk'] >= 0.75:
        fail('unit mismatch_blocks_escalation_risk')

    # 4. emergency override + ưu tiên STOP
    e1 = fusion.decide_frame(ev('clear', 'none', 'none', 'none'), geo_emergency, c)
    e2 = fusion.decide_frame(ev('clear', 'none', 'none', 'none'),
                             dict(geo_blocked, free_min=2.39), c)
    e3 = fusion.decide_frame(fusion.vlm_evidence(None), geo_emergency, c)
    if not (e1['frame_positive'] and e1['reason'] == 'depth_emergency'):
        fail('unit emergency_override')
    if e2['frame_positive']:
        fail('unit emergency_threshold_margin (2.39 phải còn thoáng)')
    if not e3['frame_positive']:
        fail('unit emergency_when_vlm_invalid')
    sess = fusion.FusionSession(config=c)
    r1 = fusion.update_session(sess, e1, ev('clear', 'none', 'none', 'none'), now=0.0)
    r2 = fusion.update_session(sess, e1, ev('person', 'front', 'near', 'none'), now=1.0)
    if r2['message_code'] != 'STOP' or r2['action'] != 'stop':
        fail(f"unit emergency_stop_priority (code={r2['message_code']} action={r2['action']})")
    if r1['alert']:
        fail('unit confirm_frames (frame 1 chưa đủ trễ thì chưa alert)')
    if verbose:
        print(f"  emergency: code={r2['message_code']} action={r2['action']} | frame1 alert={r1['alert']}")

    # 5. state machine: trễ confirm/clear + dedup + đổi vật phát lại
    sess = fusion.FusionSession(config=c)
    dec_pos = {'frame_positive': True, 'risk': 0.8}
    seq = [fusion.update_session(sess, dec_pos, ev('pole', 'front', 'near', 'move_left'),
                                 now=float(t)) for t in range(6)]
    alerts = [x['alert'] for x in seq]
    if alerts != [False, True, False, False, False, False]:
        fail(f'unit dedup_repeat ({alerts})')
    e7 = fusion.update_session(sess, dec_pos, ev('person', 'front_right', 'near', 'move_left'),
                               now=6.0)
    if not (e7['alert'] and e7['message_code'] == 'PERSON_AHEAD'):
        fail('unit key_change_realert')
    # VLM bịa vật khi depth trống -> không alert, chờ clear_frames mới tắt
    e8 = fusion.update_session(sess, {'frame_positive': False},
                               ev('object', 'front', 'mid', 'slow'), now=7.0)
    if e8['alert'] or e8['message_code']:
        fail('unit hallucinated_no_alert')
    if verbose:
        print(f"  session: alerts={[int(x) for x in alerts]} đổi vật alert={e7['alert']} bịa vật alert={e8['alert']}")

    # 6. parse schema: token lạ là None, không đoán
    if cfg.parse_pipe4('POLE|FRONT|SUPER_NEAR|STOP') is not None:
        fail('unit parse_rejects_unknown_token')
    if cfg.parse_pipe4('CLEAR|NONE|NONE|NONE').get('hazard'):
        fail('unit parse_clear_not_hazard')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    print('=== Golden test cases — fusion regression (no GPU) ===\n')
    check_manifest()
    core = load_cases('cases_core27.json')
    coco = load_cases('cases_coco120.json')
    run_suite('core27 ', core, args.verbose)
    run_suite('coco120', coco, args.verbose)
    run_units(args.verbose)

    print('\n==> ' + ('PASS — fusion tái hiện đúng toàn bộ hành vi đã đóng gói'
                       if not FAILS else f'FAIL: {len(FAILS)} lỗi'))
    for f in FAILS[:20]:
        print(f'  - {f}')
    return 0 if not FAILS else 1


if __name__ == '__main__':
    raise SystemExit(main())
