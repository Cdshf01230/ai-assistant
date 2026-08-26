#!/usr/bin/env python3
"""Chọn mẫu cân bằng từ COCO val2017 + tải ảnh (đi trực tiếp, không proxy).

Nhãn "vật chắn lối đi" suy ra từ GT bbox theo quy tắc khai báo rõ:
  - lớp vật thuộc VOCAB của VLM (person/vehicle/furniture/object);
  - cạnh đáy bbox trong dải GIỮA 50% chiều rộng ảnh;
  - đáy bbox ở nửa DƯỚI ảnh (vật gần camera trên mặt phẳng đi bộ);
  - diện tích bbox >= 1% ảnh (loại vật quá xa/nhỏ).
Ảnh không có annotation nào qua cả 4 điều kiện -> nhãn âm (lối thoáng).
"""
from __future__ import annotations

import json, os, sys, urllib.request
from pathlib import Path

# Mạng này có TLS-intercept làm lệch chứng chỉ số cho images.cocodataset.org ->
# dùng http thuần trước, hỏng thì hạ xuống https-bỏ-verify (ảnh công khai).
import ssl
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=CTX))

ROOT = Path('/home/ubuntu/ai-assistant')
COCO = ROOT / 'benchmarks/data/coco'
ANN = COCO / 'annotations/instances_val2017.json'
IMG_DIR = COCO / 'val2017'

for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
    os.environ.pop(k, None)

VOCAB_MAP = {   # category_id COCO -> nhãn vocabulary của VLM
    **{i: 'PERSON' for i in [1]},
    **{i: 'VEHICLE' for i in [2, 3, 4, 6, 8]},          # bicycle car motorcycle bus truck
    **{i: 'FURNITURE' for i in [62, 63, 65, 67]},       # chair couch bed dining table
    **{i: 'OBJECT' for i in [11, 15, 41, 64, 72, 76]},  # hydrant bench potted plant tv cell? -> object
}

def corridor_label(anns, cat_by_id, w, h):
    hits = []
    for a in anns:
        lab = VOCAB_MAP.get(a['category_id'])
        if not lab or a.get('iscrowd'):
            continue
        x, y, bw, bh = a['bbox']
        if bw * bh < 0.01 * w * h:
            continue
        cx, by = x + bw / 2, y + bh
        if not (0.25 * w <= cx <= 0.75 * w):
            continue
        if by <= 0.45 * h:
            continue
        hits.append(lab)
    return (len(hits) > 0), hits


def main() -> int:
    n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    n_neg = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    data = json.load(open(ANN))
    cats = {c['id']: c['name'] for c in data['categories']}
    anns_by_img = {}
    for a in data['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for img in data['images']:
        pos, labs = corridor_label(anns_by_img.get(img['id'], []), cats, img['width'], img['height'])
        rows.append({'file': img['file_name'], 'url': img['coco_url'].replace('http://', 'https://'),
                     'hazard': pos, 'types': sorted(set(labs)), 'w': img['width'], 'h': img['height']})

    pos = [r for r in rows if r['hazard']]
    neg = [r for r in rows if not r['hazard']]
    # ưu tiên ảnh âm có ÍT NHẤT MỘT vật cùng lớp nằm NGOÀI lối đi -> khó hơn, giống 02b
    neg_hard = [r for r in neg]
    sel_pos, sel_neg = pos[:n_pos], neg_hard[:n_neg]
    sel = sel_pos + sel_neg
    print(f"pool: {len(pos)} dương / {len(neg)} âm -> chọn {len(sel_pos)}+{len(sel_neg)}")

    ok = 0
    for i, r in enumerate(sel):
        dest = IMG_DIR / r['file']
        if dest.exists():
            ok += 1
            continue
        try:
            with opener.open(r['url'].replace('https://', 'http://'), timeout=30) as resp, open(dest, 'wb') as fh:
                fh.write(resp.read())
            ok += 1
        except Exception as exc:
            print(f"  lỗi tải {r['file']}: {exc}")
            r['skip'] = True
        if (i + 1) % 20 == 0:
            print(f"  {ok}/{len(sel)}")
    sel = [r for r in sel if not r.get('skip')]
    out = ROOT / 'logs/coco_public_labels.json'
    out.write_text(json.dumps(sel, indent=1))
    print(f"labels -> {out} ({sum(1 for r in sel if r['hazard'])} dương / "
          f"{sum(1 for r in sel if not r['hazard'])} âm)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
