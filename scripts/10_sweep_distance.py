#!/usr/bin/env python3
"""Quét ngưỡng khoảng cách trên kết quả đã lưu, không cần chạy lại GPU.

Schema 4 trường trả NEAR/MID/FAR. Decision Engine (§16) phải chọn: báo khi NEAR,
hay báo cả MID? Đây là đánh đổi recall <-> specificity, và với thiết bị dẫn đường
thì bỏ sót đắt hơn báo thừa — nên phải xem con số trước khi chốt.

Dùng:
    python scripts/10_sweep_distance.py logs/eval_vlm_qwen3vl4b.json
"""

from __future__ import annotations

import json
import os
import sys

FRAMES_DIR = "/home/ubuntu/ai-assistant/assets/frames"


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(FRAMES_DIR, "ground_truth.json"), encoding="utf-8") as f:
        gt = json.load(f)["frames"]

    print(f"Model: {data['model']}\n")
    for pname, rows in data["results"].items():
        parts0 = rows[0]["output"].split("|")
        if len(parts0) != 4:
            continue  # chỉ schema 4 trường mới có gì để quét
        print(f"--- {pname} ---")
        print(f"  {'ngưỡng':<16}{'recall':>8}{'specif':>8}{'sp.khó':>8}{'balanced':>10}")
        for name, keep in (("chỉ NEAR", {"near"}),
                           ("NEAR + MID", {"near", "mid"}),
                           ("mọi khoảng cách", {"near", "mid", "far"})):
            tp = fn = tn = fp = hard_tn = hard_fp = 0
            for r in rows:
                g = gt[r["frame"]]
                obj, _pos, dist, _act = r["output"].split("|")
                said = obj != "clear" and dist in keep
                hard = g.get("hard", False)
                if g["hazard"]:
                    tp += said
                    fn += not said
                else:
                    fp += said
                    tn += not said
                    hard_fp += said and hard
                    hard_tn += (not said) and hard
            rec = tp / (tp + fn) if tp + fn else 0.0
            spc = tn / (tn + fp) if tn + fp else 0.0
            hsp = hard_tn / (hard_tn + hard_fp) if hard_tn + hard_fp else 0.0
            print(f"  {name:<16}{rec:7.0%}{spc:7.0%}{hsp:7.0%}{(rec + spc) / 2:9.0%}")

        dists = [r["output"].split("|")[2] for r in rows]
        n_clear = sum(d == "none" for d in dists)
        print(f"  phân bố khoảng cách: near={dists.count('near')} mid={dists.count('mid')} "
              f"far={dists.count('far')} none(=CLEAR)={n_clear} / {len(dists)} frame")
        print("  (near chiếm gần hết => trường khoảng cách gần như không mang thông tin)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
