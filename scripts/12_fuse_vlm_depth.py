#!/usr/bin/env python3
"""Ghép VLM (nhận dạng vật) với depth (khoảng cách) rồi chấm lại — offline, không GPU.

Phân công theo đúng chỗ mỗi model mạnh, số đo đã chỉ rõ:
  - VLM  : gọi tên vật 85%, hướng trái/phải 85%. Nhưng khoảng cách thì mù hoàn toàn.
  - Depth: cho khoảng cách bằng hình học. Nhưng không biết vật đó là gì.

Nên Decision Engine (§16) lấy `type`/`position` từ VLM và `distance` từ depth,
thay vì tin trường khoảng cách VLM tự bịa ra.

So 5 chiến lược trên cùng bộ nhãn:
  A. chỉ VLM, ngưỡng near+mid   <- baseline đang dùng
  B. chỉ VLM, ngưỡng near
  C. chỉ depth (VLM không tham gia)
  D. VLM AND depth              <- depth được phủ quyết mọi thứ
  E. D + cửa thoát cho lớp vật depth vốn mù (step/hole/door)  <- đề xuất

Ngưỡng ghép được QUÉT và chọn theo an toàn (giữ recall 100% trước, rồi mới tối đa
specificity), không chọn theo balanced accuracy — lần chạy đầu chọn theo balanced
và nó đã đổi mất 2 frame có vật cản thật.

Dùng:
    python scripts/12_fuse_vlm_depth.py logs/eval_vlm_qwen3vl4b.json logs/depth_outdoor.json
    python scripts/12_fuse_vlm_depth.py <vlm.json> <depth.json> --prompt pipe_v6
    python scripts/12_fuse_vlm_depth.py <vlm.json> <depth.json> --thr 20   # cố định, bỏ quét
"""

from __future__ import annotations

import argparse
import json
import os
import sys

FRAMES_DIR = "/home/ubuntu/ai-assistant/assets/frames"
COL_TO_POS = {0: {"left", "front_left"},
              1: {"front", "front_left", "front_right"},
              2: {"right", "front_right"}}

# Những lớp vật mà depth KHÔNG được phép phủ quyết VLM.
#
# Phương pháp free-space đo "tia nhìn dọc lối đi đi được bao xa trước khi bị chắn".
# Ba lớp này không chắn tia đó, nên depth đọc ra khoảng trống phía sau chúng:
#   step  bậc thang thấp, trải ngang mặt đất, gần như không có chiều cao
#   hole  hố/khuyết mặt đường, còn "sâu hơn" mặt đường nên càng không chắn
#   door  cửa mở hoặc cửa kính — tia nhìn xuyên qua vào không gian bên trong
# Đo được: door_00.jpg (bậc thang trước cửa, cách ~3 bước) depth đọc 19.70 vì
# nhìn thẳng vào sảnh, trong khi VLM trả đúng `door|front|near|slow`.
# Đây là lỗ cấu trúc của phương pháp, không phải chọn ngưỡng sai — cả hai frame
# bị bỏ sót đều nằm lọt trong khoảng của frame trống nên không ngưỡng nào cứu được.
DEPTH_BLIND_TYPES = frozenset({"step", "hole", "door"})


def load_gt() -> dict:
    with open(os.path.join(FRAMES_DIR, "ground_truth.json"), encoding="utf-8") as f:
        gt = json.load(f)["frames"]
    return {k: v for k, v in gt.items() if not v.get("excluded")}


def confusion(gt: dict, decide) -> tuple:
    """decide(fname) -> (said, type_or_None, col_or_None). Trả các chỉ số."""
    tp = fn = tn = fp = hard_tn = hard_fp = 0
    t_ok = p_ok = n_hit = 0
    misses, extras = [], []
    for fname, g in gt.items():
        said, vtype, pos_set = decide(fname)
        hard = g.get("hard", False)
        if g["hazard"]:
            if said:
                tp += 1
                n_hit += 1
                t_ok += vtype in g["types"] if vtype else 0
                p_ok += bool(pos_set & set(g["positions"])) if pos_set else 0
            else:
                fn += 1
                misses.append(fname)
        else:
            if said:
                fp += 1
                hard_fp += hard
                extras.append(fname + (" (âm khó)" if hard else ""))
            else:
                tn += 1
                hard_tn += hard
    rec = tp / (tp + fn) if tp + fn else 0.0
    spc = tn / (tn + fp) if tn + fp else 0.0
    hsp = hard_tn / (hard_tn + hard_fp) if hard_tn + hard_fp else 0.0
    return (rec, spc, hsp, (rec + spc) / 2,
            t_ok / n_hit if n_hit else 0.0, p_ok / n_hit if n_hit else 0.0,
            misses, extras)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vlm_dump")
    ap.add_argument("depth_dump")
    ap.add_argument("--prompt", default=None, help="mặc định: prompt 4 trường đầu tiên")
    ap.add_argument("--thr", type=float, default=None, help="ngưỡng depth, mặc định lấy từ dump")
    ap.add_argument("--key", default="free3", choices=["free3", "freeC"],
                    help="free3 = min 3 cột (nhạy); freeC = chỉ cột giữa (đúng §16 hơn)")
    args = ap.parse_args()

    gt = load_gt()
    with open(args.vlm_dump, encoding="utf-8") as f:
        vd = json.load(f)
    with open(args.depth_dump, encoding="utf-8") as f:
        dd = json.load(f)

    pname = args.prompt
    if pname is None:
        for k, rows in vd["results"].items():
            if rows and len(rows[0]["output"].split("|")) == 4:
                pname = k
                break
    if pname is None:
        sys.exit("Dump VLM không có prompt nào dùng schema 4 trường — chạy 09 với pipe_v6.")

    # Ngưỡng nằm trong dict để hàm quyết định đọc được khi quét.
    state = {"thr": args.thr if args.thr is not None else (
        dd["best_threshold_center"] if args.key == "freeC" else dd["best_threshold"])}
    vlm = {}
    for r in vd["results"][pname]:
        parts = r["output"].split("|")
        vlm[r["frame"]] = parts if len(parts) == 4 else None
    depth = dd["frames"]

    common = [f for f in gt if f in vlm and f in depth]
    print(f"VLM   : {vd['model']} / {pname}")
    print(f"Depth : {dd['checkpoint']} | {args.key} | p{dd.get('pct', '?')} | "
          f"{dd.get('latency_median_ms', '?')} ms")
    print(f"Frames: {len(common)} khớp cả hai dump\n")

    def d_near(fname):
        v = depth[fname].get(args.key)
        return v is not None and v == v and v < state["thr"]

    def dec_vlm(keep):
        def f(fname):
            p = vlm[fname]
            if p is None:                       # sai schema -> im lặng (§16)
                return False, None, None
            obj, pos, dist, _act = p
            said = obj != "clear" and dist in keep
            return said, obj, {pos}
        return f

    def dec_depth(fname):
        c = depth[fname]["col"]
        return d_near(fname), None, COL_TO_POS.get(c)

    def dec_fused(fname):
        p = vlm[fname]
        if p is None:
            return False, None, None
        obj, pos, _dist, _act = p
        # VLM quyết CÓ VẬT GÌ, depth quyết CÓ ĐỦ GẦN KHÔNG.
        return (obj != "clear") and d_near(fname), obj, {pos}

    def dec_fused_or(fname):
        """Như D, nhưng depth không được phủ quyết lớp vật nó vốn mù (xem
        DEPTH_BLIND_TYPES). Với các lớp đó thì tin nhãn near của VLM."""
        p = vlm[fname]
        if p is None:
            return False, None, None
        obj, pos, dist, _act = p
        if obj == "clear":
            return False, obj, {pos}
        blind = obj in DEPTH_BLIND_TYPES and dist == "near"
        return d_near(fname) or blind, obj, {pos}

    strategies = [
        ("A. VLM near+mid", dec_vlm({"near", "mid"})),
        ("B. VLM chỉ near", dec_vlm({"near"})),
        ("C. chỉ depth", dec_depth),
        ("D. VLM AND depth", dec_fused),
        ("E. D + cửa thoát", dec_fused_or),
    ]

    sub = {k: v for k, v in gt.items() if k in common}

    # Quét ngưỡng ghép, chọn theo AN TOÀN: giữ recall 100% rồi mới tối đa
    # specificity. Chọn theo balanced accuracy như lần đầu đã đổi mất 2 frame có
    # vật cản thật — sai chiều với thiết bị dẫn đường.
    if args.thr is None:
        thrs = [8.0, 10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 19.0, 19.5,
                20.0, 21.0, 22.0, 24.0, 26.0]
        print(f"--- Quét ngưỡng ghép ({args.key}) ---")
        print(f"  {'ngưỡng':>8}{'D recall':>10}{'D spec':>8}{'D khó':>7}"
              f"{'E recall':>10}{'E spec':>8}{'E khó':>7}")
        rows = []
        for t in thrs:
            state["thr"] = t
            rd = confusion(sub, dec_fused)
            re_ = confusion(sub, dec_fused_or)
            rows.append((t, rd, re_))
            print(f"  {t:7.2f}{rd[0]:9.0%}{rd[1]:8.0%}{rd[2]:7.0%}"
                  f"{re_[0]:9.0%}{re_[1]:8.0%}{re_[2]:7.0%}")

        # Ưu tiên E (có cửa thoát) vì nó đạt recall 100% ở ngưỡng thấp hơn,
        # nghĩa là giữ được nhiều specificity hơn.
        full_e = [(t, e) for t, _d, e in rows if e[0] >= 1.0]
        full_d = [(t, d) for t, d, _e in rows if d[0] >= 1.0]
        if full_e:
            pick_t, pick_r = max(full_e, key=lambda x: (x[1][1], x[1][2]))
            rule = "E giữ recall 100% ở ngưỡng thấp nhất"
        elif full_d:
            pick_t, pick_r = max(full_d, key=lambda x: (x[1][1], x[1][2]))
            rule = "chỉ D đạt recall 100%"
        else:
            pick_t, pick_r = max(((t, e) for t, _d, e in rows),
                                 key=lambda x: (x[1][3], x[1][2]))
            rule = "KHÔNG cấu hình nào đạt recall 100% -> đành lấy balanced cao nhất"
        state["thr"] = pick_t
        print(f"  quy tắc chọn: {rule}")
        print(f"  => ngưỡng {pick_t:.2f}"
              f"{'m' if dd.get('metric') else ''} (recall {pick_r[0]:.0%}, "
              f"specificity {pick_r[1]:.0%}, âm khó {pick_r[2]:.0%})")

        # Dư địa phải đo với vật cản mà DEPTH tự phải bắt bằng ngưỡng — không tính
        # frame được cửa thoát cứu, vì ngưỡng không quyết định số phận frame đó.
        by_thr = []
        for f in sub:
            if not sub[f]["hazard"]:
                continue
            p = vlm.get(f)
            if p and p[0] in DEPTH_BLIND_TYPES and p[2] == "near":
                continue                    # cửa thoát lo, ngưỡng không liên quan
            v = depth[f].get(args.key)
            if v is not None and v == v:
                by_thr.append(v)
        if by_thr and pick_r[0] >= 1.0:
            worst = max(by_thr)
            m = (pick_t - worst) / pick_t
            print(f"  dư địa tới vật cản ngưỡng-phải-bắt xa nhất ({worst:.2f}): "
                  f"{pick_t - worst:+.2f} = {m:.1%}"
                  f"{'   <- QUÁ SÁT, ngưỡng đang bị ghim bởi 1 frame' if m < 0.10 else ''}")
        print()

    print(f"Ngưỡng dùng: {args.key} < {state['thr']:.2f}"
          f"{'m' if dd.get('metric') else ''}\n")
    print(f"{'chiến lược':<18}{'recall':>8}{'specif':>8}{'sp.khó':>8}{'balanc':>9}"
          f"{'type':>7}{'hướng':>7}")
    results = {}
    for name, fn_ in strategies:
        r = confusion(sub, fn_)
        results[name] = r
        print(f"{name:<18}{r[0]:7.0%}{r[1]:7.0%}{r[2]:7.0%}{r[3]:8.0%}{r[4]:6.0%}{r[5]:6.0%}")

    print()
    for name, _ in strategies:
        r = results[name]
        print(f"{name}")
        print(f"   bỏ sót  ({len(r[6])}): {', '.join(r[6]) or '—'}")
        print(f"   báo thừa({len(r[7])}): {', '.join(r[7]) or '—'}")

    a = results["A. VLM near+mid"]
    d = results["D. VLM AND depth"]
    e = results["E. D + cửa thoát"]
    print("\n=== Kết luận ===")
    print(f"  {'':<14}{'recall':>8}{'specif':>8}{'sp.khó':>8}{'balanc':>8}")
    for label, r in (("A baseline", a), ("D AND thuần", d), ("E đề xuất", e)):
        print(f"  {label:<14}{r[0]:7.0%}{r[1]:7.0%}{r[2]:7.0%}{r[3]:7.0%}")
    print(f"  âm khó: {a[2]:.0%} -> {e[2]:.0%}   <- đây là chỗ VLM đơn độc bằng 0")

    if e[0] < a[0]:
        print(f"  ⚠️ Ghép depth vẫn làm recall TỤT ({a[0]:.0%} -> {e[0]:.0%}): mất "
              f"{len(e[6]) - len(a[6])} frame có vật cản ({', '.join(e[6])}). "
              "Với thiết bị dẫn\n     đường thì bỏ sót đắt hơn báo thừa — chưa "
              "chốt được cấu hình này.")
    elif e[1] > a[1]:
        print(f"  => Recall giữ nguyên {e[0]:.0%}, specificity {a[1]:.0%} -> "
              f"{e[1]:.0%}, âm khó {a[2]:.0%} -> {e[2]:.0%}.")
        print("     Không đổi gì lấy gì: depth cấp đúng tín hiệu VLM đang thiếu.")
    if e[0] > d[0]:
        cost = (f"specificity {d[1]:.0%} -> {e[1]:.0%}" if e[1] < d[1]
                else "không mất specificity")
        print(f"  Cửa thoát cho {'/'.join(sorted(DEPTH_BLIND_TYPES))} cứu được "
              f"{len(d[6]) - len(e[6])} frame mà AND thuần bỏ sót, giá: {cost}.")
    print(f"\nLƯU Ý: {sum(1 for f in sub if not sub[f]['hazard'])} frame âm — "
          "specificity vẫn nhảy từng bước lớn. §8 vẫn cần 100-500 frame thật.")
    print("LƯU Ý: ngưỡng trên KHÔNG phải mét thật (checkpoint VKITTI, camera gắn xe)"
          " — chỉ\n       dùng được thứ tự, và phải hiệu chuẩn lại cho từng máy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
