#!/usr/bin/env python3
"""Thử Depth-Anything-V2 xem có giải được chỗ VLM mù khoảng cách hay không.

Bối cảnh: 09_eval_vlm.py đã đo được VLM (cả 2B và 4B) KHÔNG suy luận khoảng cách —
cùng một cây cột ở chân trời và ngay trước mặt đều trả `pole|front|near|move_left`,
nhãn `far` không bao giờ được dùng. Nên §16 không có gì để lọc cảnh báo.
Script này kiểm tra giả thuyết: một model depth nhẹ có cấp được tín hiệu đó không.

Cách đo — KHÔNG hỏi model "xa hay gần", mà tính bằng hình học:

  1. Depth map cho cả frame (mét, nếu dùng checkpoint metric).
  2. Khoanh lối đi thành hình thang, đúng polygon mà 02b dùng để dựng frame
     synthetic — nghĩa là hình học đã biết trước, không phải đoán.
  3. Chia hình thang thành 10 dải ngang (gần -> xa) x 3 cột dọc (trái/giữa/phải).
  4. Quét từng cột từ gần ra xa. Mặt đường trống thì depth phải TĂNG dần theo
     chiều lên ảnh; vật cản thì chặn tầm nhìn nên mọi dải phía sau nó đều đọc ra
     xấp xỉ khoảng cách của chính nó. Nên "xa nhất nhìn thấy được dọc lối đi"
     = khoảng cách tới vật cản. Đó là free space.
  5. Báo cáo hai biến thể:
       free_min3   = min trên cả 3 cột  -> nhạy, bắt cả vật lệch sang bên
       free_center = chỉ cột giữa       -> đúng §16 hơn (lệch bên thì chưa cảnh báo)
     Có cả hai vì frame synthside_pole cho thấy hình thang ở các hàng THẤP rộng
     gần hết khung, nên vật "lệch hẳn sang bên" vẫn lọt vào cột trái.

Điểm mấu chốt là bước 4: nó biến "vật này xa hay gần" — thứ VLM không làm được —
thành phép so sánh đơn điệu trên depth, thứ hoàn toàn hình học.

Dùng:
    python scripts/11_depth_probe.py --dump logs/depth_outdoor.json
    python scripts/11_depth_probe.py --which indoor
    python scripts/11_depth_probe.py --which relative
    python scripts/11_depth_probe.py --pct 3        # vật mảnh (cột đèn) dễ lọt p10
    python scripts/11_depth_probe.py --save-viz
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_path as dp  # noqa: E402
import mvp_config as cfg  # noqa: E402

FRAMES_DIR = "/home/ubuntu/ai-assistant/assets/frames"
GT_PATH = os.path.join(FRAMES_DIR, "ground_truth.json")
VIZ_DIR = "/home/ubuntu/ai-assistant/assets/depth"

CHECKPOINTS = {
    "outdoor": dp.DEPTH_CHECKPOINT,
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "relative": "depth-anything/Depth-Anything-V2-Small-hf",
}

# Hình học lối đi + hàm quét free space nằm ở scripts/depth_path.py — dùng chung
# với app/ để hằng số hình thang không trôi giữa script đo và server.
N_BANDS = dp.N_BANDS
COL_NAMES = dp.COL_NAMES
COL_TO_POS = dp.COL_TO_POS
path_masks = dp.path_masks
free_space = dp.free_space
PATH_TOP_Y = dp.PATH_TOP_Y
PATH_BOT_HALFW = dp.PATH_BOT_HALFW
PATH_TOP_HALFW = dp.PATH_TOP_HALFW
COL_EDGE = dp.COL_EDGE
MIN_PIX = dp.MIN_PIX
BREAK_RATIO = dp.BREAK_RATIO


def load_ground_truth() -> dict:
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)["frames"]
    return {k: v for k, v in gt.items() if not v.get("excluded")}


def load_frames(gt: dict, long_edge: int):
    from PIL import Image

    out = []
    for name in sorted(gt):
        path = os.path.join(FRAMES_DIR, name)
        if not os.path.exists(path):
            print(f"  (thiếu file {name}, bỏ qua)")
            continue
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > long_edge:
            s = long_edge / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        # Round-trip JPEG y như 03/09: depth phải chạy trên đúng thứ server nhận.
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=cfg.JPEG_QUALITY)
        out.append((name, Image.open(buf).convert("RGB")))
    return out


def score(rows, gt, thr, key):
    """rows = {fname: {"free3","freeC","col"}}. free < thr => cảnh báo."""
    tp = fn = tn = fp = hard_tn = hard_fp = 0
    pos_ok = pos_n = 0
    misses, extras = [], []
    for fname, r in rows.items():
        g = gt[fname]
        v = r[key]
        # nan = không đọc được lối đi -> im lặng (§16: thà im hơn đoán)
        said = v == v and v < thr
        hard = g.get("hard", False)
        if g["hazard"]:
            if said:
                tp += 1
                pos_n += 1
                pos_ok += bool(COL_TO_POS.get(r["col"], set()) & set(g["positions"]))
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
    pac = pos_ok / pos_n if pos_n else 0.0
    return rec, spc, hsp, (rec + spc) / 2, pac, misses, extras


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="outdoor", choices=list(CHECKPOINTS))
    ap.add_argument("--long-edge", type=int, default=cfg.FRAME_LONG_EDGE)
    ap.add_argument("--pct", type=int, default=10,
                    help="percentile depth trong mỗi ô; nhỏ hơn = bắt vật mảnh hơn")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--save-viz", action="store_true")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if not torch.cuda.is_available():
        sys.exit("Không có CUDA — dừng.")

    repo = CHECKPOINTS[args.which]
    gt = load_ground_truth()
    frames = load_frames(gt, args.long_edge)
    n_pos = sum(1 for n, _ in frames if gt[n]["hazard"])

    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForDepthEstimation.from_pretrained(repo, dtype=torch.float16).to("cuda:0")
    model.eval()
    torch.cuda.synchronize()
    vram = (torch.cuda.memory_allocated() - base) / 1024**2

    metric = args.which != "relative"
    print(f"Depth : {repo}")
    print(f"Load  : {time.perf_counter() - t0:.1f}s | VRAM {vram:.0f} MiB | fp16")
    print(f"Frames: {len(frames)} ({n_pos} có vật cản / {len(frames) - n_pos} trống) "
          f"| {args.long_edge}px | p{args.pct} | "
          f"{'mét' if metric else 'đơn vị tương đối'}\n")

    def depth_of(img):
        inputs = proc(images=img, return_tensors="pt").to("cuda:0")
        inputs["pixel_values"] = inputs["pixel_values"].half()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            out = model(**inputs)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        post = proc.post_process_depth_estimation(
            out, target_sizes=[(img.height, img.width)])
        d = np.squeeze(post[0]["predicted_depth"].float().cpu().numpy())
        if not metric:
            # Model tương đối trả inverse depth (lớn = gần). Đổi sang "khoảng cách
            # giả" để dùng chung một logic quét đơn điệu với bản metric.
            d = 1.0 / np.maximum(d, 1e-3)
        return d, ms

    depth_of(frames[0][1])  # warmup

    band = col = None
    lat: list[float] = []
    per: dict[str, dict] = {}
    for _ in range(args.repeat):
        for fname, img in frames:
            d, ms = depth_of(img)
            lat.append(ms)
            if band is None or band.shape != d.shape:
                band, col = path_masks(*d.shape)
            free, _grid = free_space(d, band, col, args.pct)
            c = -1 if np.all(np.isnan(free)) else int(np.nanargmin(free))
            per.setdefault(fname, {
                "free3": float(np.nanmin(free)) if c >= 0 else float("nan"),
                "freeC": float(free[1]),
                "col": c,
                "per_col": [None if v != v else round(float(v), 2) for v in free],
            })

    print(f"--- Latency depth ({len(lat)} lần) ---")
    print(f"  median {statistics.median(lat):.0f} ms | min/max "
          f"{min(lat):.0f}/{max(lat):.0f} ms   (VLM 4B đang là ~845 ms)\n")
    if metric:
        # Quét mịn ở 14-26 vì đó là chỗ ranh giới thật nằm: trên bộ frame hiện tại
        # vật cản cao nhất đọc ra 19.70 và frame trống thấp nhất là 17.17.
        thrs = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 15.0,
                16.0, 18.0, 19.0, 19.5, 20.0, 21.0, 22.0, 24.0, 26.0]
    else:
        vals = sorted(v["free3"] for v in per.values() if v["free3"] == v["free3"])
        lo, hi = vals[0], vals[-1]
        thrs = [round(lo + (hi - lo) * k / 10, 3) for k in range(1, 10)]

    # Chọn ngưỡng theo AN TOÀN, không theo balanced accuracy.
    # Lần chạy đầu chọn theo balanced và nó đã đổi recall 100% -> 87%: mất 2 frame
    # có vật cản thật để lấy specificity. Với thiết bị dẫn đường thì đó là đổi sai
    # chiều — bỏ sót một bậc thang đắt hơn nhiều lần một cảnh báo thừa.
    # Quy tắc: trong các ngưỡng còn giữ recall 100%, lấy ngưỡng cho specificity cao
    # nhất (vì recall tăng dần theo ngưỡng và specificity giảm dần, đó chính là
    # ngưỡng NHỎ nhất đạt recall 100%). Không ngưỡng nào đạt thì mới xét balanced.
    best = {}
    for key, label in (("free3", "min 3 cột"), ("freeC", "chỉ cột giữa")):
        rows = []
        for thr in thrs:
            rec, spc, hsp, bal, pac, _m, _e = score(per, gt, thr, key)
            rows.append((thr, rec, spc, hsp, bal, pac))

        full = [r for r in rows if r[1] >= 1.0]
        if full:
            pick = max(full, key=lambda r: (r[2], r[3]))
            rule = "recall 100% + specificity cao nhất"
        else:
            pick = max(rows, key=lambda r: (r[4], r[3]))
            rule = "KHÔNG ngưỡng nào đạt recall 100% -> đành lấy balanced cao nhất"
        best[key] = pick

        print(f"--- Quét ngưỡng, {label} (free < ngưỡng => cảnh báo) ---")
        print(f"  {'ngưỡng':>10}{'recall':>8}{'specif':>8}{'sp.khó':>8}"
              f"{'balanc':>8}{'hướng':>7}")
        for r in rows:
            star = "  <- chọn" if r is pick else ""
            print(f"  {r[0]:9.2f}{'m' if metric else ' '}{r[1]:7.0%}{r[2]:7.0%}"
                  f"{r[3]:7.0%}{r[4]:7.0%}{r[5]:6.0%}{star}")
        print(f"  quy tắc chọn: {rule}")

        # Ngưỡng ghim bởi một frame duy nhất thì không phải biên an toàn.
        haz = sorted(v[key] for f, v in per.items()
                     if gt[f]["hazard"] and v[key] == v[key])
        if haz and pick[1] >= 1.0:
            margin = (pick[0] - haz[-1]) / pick[0]
            print(f"  dư địa tới vật cản xa nhất ({haz[-1]:.2f}): "
                  f"{pick[0] - haz[-1]:+.2f} = {margin:.1%}"
                  f"{'   <- QUÁ SÁT, ngưỡng bị ghim bởi 1 frame' if margin < 0.10 else ''}")
        print()

    if metric:
        hz = sorted(v["free3"] for f, v in per.items()
                    if gt[f]["hazard"] and v["free3"] == v["free3"])
        cl = sorted(v["free3"] for f, v in per.items()
                    if not gt[f]["hazard"] and v["free3"] == v["free3"])
        if hz and cl:
            print("--- Thang đo này có thật là mét không? ---")
            print(f"  frame có vật cản : {hz[0]:6.2f} .. {hz[-1]:6.2f}")
            print(f"  frame lối trống  : {cl[0]:6.2f} .. {cl[-1]:6.2f}")
            print(f"  Vật cản gần nhất trong bộ frame cách chừng 1-2 bước, nhưng đọc"
                  f" ra {hz[0]:.1f} 'mét'.")
            print("  Checkpoint metric train trên VKITTI (camera gắn trên xe, tầm"
                  " 0-80m), nên camera")
            print("  cầm tay ở tầm người là out-of-distribution: sai số thang chừng"
                  " 4-5 lần. Chỉ THỨ TỰ")
            print("  dùng được. §16 KHÔNG ngưỡng được theo '3 bước chân' — ngưỡng"
                  " phải hiệu chuẩn")
            print("  cho từng độ cao/góc chúc camera và sẽ trôi khi đổi cấu hình.\n")

    print("--- So với baseline ---")
    print(f"  {'cấu hình':<24}{'recall':>8}{'specif':>8}{'sp.khó':>8}{'balanc':>8}")
    print(f"  {'VLM 4B v6 (near+mid)':<24}{1.00:7.0%}{0.58:7.0%}{0.00:7.0%}{0.79:7.0%}")
    print(f"  {'VLM 4B v6 (chỉ near)':<24}{0.87:7.0%}{0.75:7.0%}{0.25:7.0%}{0.81:7.0%}")
    for key, label in (("free3", "depth min-3-cột"), ("freeC", "depth cột-giữa")):
        b = best[key]
        print(f"  {label + f' @{b[0]:.1f}':<24}{b[1]:7.0%}{b[2]:7.0%}{b[3]:7.0%}{b[4]:7.0%}")

    # Phép thử quyết định: cùng MỘT cây cột, chỉ khác khoảng cách và vị trí ngang.
    trio = ["synthnear_pole.jpg", "synthfar_pole.jpg", "synthside_pole.jpg"]
    if all(t in per for t in trio):
        print("\n--- Cùng-1-cây-cột (phép thử VLM đã trượt) ---")
        for t in trio:
            r = per[t]
            lbl = COL_NAMES[r["col"]] if r["col"] >= 0 else "?"
            print(f"  {t:<22} min3={r['free3']:6.2f} giữa={r['freeC']:6.2f} "
                  f"3cột={r['per_col']} gần-nhất={lbl}")
        vn, vf, vs = (per[t]["free3"] for t in trio)
        cn, cf, cs = (per[t]["freeC"] for t in trio)
        ok_far = vf == vf and vn == vn and vf > vn * 1.3
        ok_side = cs == cs and cn == cn and cs > cn * 1.3
        print(f"  cột-xa xa hơn cột-gần?          "
              f"{'CÓ' if ok_far else 'KHÔNG'}  ({vf:.2f} vs {vn:.2f})")
        print(f"  cột-lệch-bên bị cột-giữa bỏ qua? "
              f"{'CÓ' if ok_side else 'KHÔNG'}  ({cs:.2f} vs {cn:.2f})")
        if ok_far:
            print("  => Depth phân biệt được xa/gần, thứ VLM không làm được.")
        else:
            print("  => Chưa tách được xa/gần. Lưu ý 3 frame này là ảnh synthetic màu"
                  "\n     phẳng, depth model thiếu manh mối quang học thật — xem bảng"
                  "\n     frame thật bên dưới trước khi kết luận.")

    thr3 = best["free3"][0]
    for group, keep in (("FRAME THẬT", lambda n: not n.startswith("synth")),
                        ("FRAME SYNTHETIC", lambda n: n.startswith("synth"))):
        print(f"\n--- {group} (ngưỡng min-3-cột {thr3:.2f}) ---")
        for fname in sorted(f for f in per if keep(f)):
            r = per[fname]
            truth = gt[fname]["hazard"]
            said = r["free3"] == r["free3"] and r["free3"] < thr3
            mark = "ok" if truth == said else ("BỎ SÓT" if truth else "BÁO THỪA")
            hard = " (âm khó)" if gt[fname].get("hard") else ""
            lbl = COL_NAMES[r["col"]] if r["col"] >= 0 else "?"
            print(f"  {'HAZ' if truth else 'CLR'} {fname:<26} min3={r['free3']:6.2f} "
                  f"giữa={r['freeC']:6.2f} cột={lbl:<5} {mark}{hard}")

    if args.save_viz:
        from PIL import Image

        os.makedirs(VIZ_DIR, exist_ok=True)
        for fname, img in frames:
            d, _ = depth_of(img)
            lo, hi = np.percentile(d, 2), np.percentile(d, 98)
            norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            vis = np.stack([norm, 1 - norm, np.zeros_like(norm)], -1)
            b, _c = path_masks(*d.shape)
            vis[b < 0] *= 0.35            # tối phần ngoài lối đi cho dễ nhìn
            Image.fromarray((vis * 255).astype("uint8")).save(
                os.path.join(VIZ_DIR, f"{os.path.splitext(fname)[0]}_depth.png"))
        print(f"\nẢnh depth (đỏ=xa, xanh=gần, vùng tối=ngoài lối đi) -> {VIZ_DIR}/")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": repo, "metric": metric, "pct": args.pct,
                       "long_edge": args.long_edge,
                       "best_threshold": best["free3"][0],
                       "best_threshold_center": best["freeC"][0],
                       "latency_median_ms": round(statistics.median(lat)),
                       "frames": per}, f, ensure_ascii=False, indent=2)
        print(f"Kết quả thô -> {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
