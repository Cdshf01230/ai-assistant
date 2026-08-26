#!/usr/bin/env python3
"""Hình học free-space trên depth map + luật quyết định đã đo được.

Module này là NGUỒN DUY NHẤT cho hình thang lối đi và cách quét free space.
`11_depth_probe.py` (đo/chấm điểm) và `app/` (server) đều import từ đây — nếu để
mỗi bên một bản copy thì hằng số hình thang sẽ trôi.

⚠️ PHÂN CÔNG HIỆN TỜI (đừng nhầm):
  - PRODUCTION dùng `fusion.FusionConfig` (scripts/fusion.py) với checkpoint
    Indoor-Large (mvp_config.DEPTH_MODEL), free_abs_threshold=7.0 đã hiệu chuẩn
    trên nó + depth_emergency_threshold=2.0 (COCO adjudication).
  - Các hằng số DEPTH_CHECKPOINT/DEPTH_THRESHOLD=19.0 bên dưới là của PHA NGHIÊN
    CỨU round-2 (checkpoint Outdoor-Small, logs/depth_round2.md) — chỉ còn các
    script benchmark lịch sử 11_depth_probe.py / 13_gpu_sharing.py dùng lại.
    should_warn_fused()/depth_says_near() cũng vậy: production đi qua
    fusion.decide_frame(), không qua hàm này nữa.

Vì sao tính bằng hình học chứ không hỏi VLM: 09_eval_vlm.py đo được VLM (cả 2B và
4B) KHÔNG suy luận khoảng cách — cùng một cây cột ở chân trời và ngay trước mặt đều
trả `pole|front|near|move_left`, nhãn `far` không bao giờ được dùng trên 27 frame.

Cách quét: mặt đường trống thì depth TĂNG dần theo chiều lên ảnh; vật cản chặn tầm
nhìn nên mọi dải phía sau nó đọc ra xấp xỉ khoảng cách của chính nó. Nên "xa nhất
nhìn thấy được dọc lối đi" = khoảng cách tới vật cản.
"""

from __future__ import annotations

# --- Checkpoint (LEGACY round-2 — production dùng mvp_config.DEPTH_MODEL) ------
# ĐÃ ĐO (logs/depth_round2.md): bản outdoor (VKITTI) cho specificity 92% / âm khó
# 75%; bản indoor (Hypersim) chỉ 75% / 25% dù thang mét của nó sát tầm người hơn.
# Thang đo không quan trọng, khả năng PHÂN BIỆT mới quan trọng => giữ outdoor.
DEPTH_CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

# Hình thang lối đi, toạ độ chuẩn hoá [0,1]. Lấy đúng polygon của 02b:
#   [(0,360),(640,360),(420,190),(220,190)] trên khung 640x360
# => đáy y=1.0 rộng hết khung; đỉnh y=0.528 rộng x 0.344..0.656.
PATH_TOP_Y = 0.528
PATH_BOT_HALFW = 0.50
PATH_TOP_HALFW = 0.156
N_BANDS = 10          # số dải ngang, gần -> xa
COL_EDGE = 1 / 3      # ranh giới trái|giữa|phải, tính theo nửa bề rộng lối đi
MIN_PIX = 20          # ô ít pixel hơn thế thì bỏ, quá nhiễu
BREAK_RATIO = 0.75    # depth tụt dưới 75% mức xa nhất đã thấy = coi như bị chắn
DEPTH_PCT = 10        # percentile depth trong mỗi ô

COL_NAMES = ("trái", "giữa", "phải")
COL_TO_POS = {                      # cột hình học -> nhãn hướng chấp nhận được
    0: {"left", "front_left"},
    1: {"front", "front_left", "front_right"},
    2: {"right", "front_right"},
}

# --- Ngưỡng quyết định -------------------------------------------------------
# ĐÃ ĐO (logs/depth_round2.md, 12_fuse_vlm_depth.py, 27 frame có nhãn):
#
#   chiến lược            recall  specif  âm khó  balanced
#   A. chỉ VLM near+mid     100%     58%      0%      79%   <- baseline cũ
#   B. chỉ VLM near          87%     75%     25%      81%
#   C. chỉ depth             93%     50%     75%      72%
#   D. VLM AND depth         93%     92%     75%      92%
#   E. D + cửa thoát        100%     92%     75%      96%   <- đang dùng
#
# E thắng baseline ở MỌI cột, không đổi gì lấy gì. Ngưỡng chọn theo an toàn: trong
# các ngưỡng còn giữ recall 100%, lấy ngưỡng cho specificity cao nhất — KHÔNG chọn
# theo balanced accuracy (chọn kiểu đó đã đổi mất 2 frame có vật cản thật).
#
# ⚠️ HAI CẢNH BÁO PHẢI ĐỌC TRƯỚC KHI TIN 19.00:
#
# 1. Đây KHÔNG phải 19 mét thật. Checkpoint train trên VKITTI (camera gắn trên xe,
#    tầm 0-80m) nên camera cầm tay ở tầm người là out-of-distribution: cây cột cách
#    ~2 bước đọc ra 8.88, cửa kính cách 1-2 bước đọc ra 7.98 — lệch chừng 4-5 lần.
#    Chỉ THỨ TỰ dùng được. Đổi độ cao/góc chúc camera là phải hiệu chuẩn lại.
# 2. Ngưỡng đang bị ghim bởi ĐÚNG MỘT frame: clear_02 (người nằm trên lối đi) đọc
#    18.80, dư địa 1.1%. Specificity thì phẳng 92% suốt từ 10 đến 19 nên trục ổn
#    định; trục mong manh là recall. Phải đo lại trên 100-500 frame thật của §8.
DEPTH_THRESHOLD = 19.0
DEPTH_KEY = "free3"   # min trên 3 cột; "freeC" = chỉ cột giữa (kém hơn, xem log)

# Những lớp vật mà depth KHÔNG được phép phủ quyết VLM.
#
# Free-space đo "tia nhìn dọc lối đi đi được bao xa trước khi bị chắn". Ba lớp này
# không chắn tia đó nên depth đọc ra khoảng trống PHÍA SAU chúng:
#   step  bậc thang thấp, trải ngang mặt đất, gần như không có chiều cao
#   hole  hố/khuyết mặt đường, còn sâu hơn mặt đường nên càng không chắn
#   door  cửa mở hoặc cửa kính — tia nhìn xuyên qua vào không gian bên trong
# Đo được: door_00.jpg (bậc thang trước cửa, ~3 bước) depth đọc 19.70 vì nhìn thẳng
# vào sảnh, còn VLM trả đúng `door|front|near|slow`. Đây là lỗ CẤU TRÚC của phương
# pháp, không phải chọn ngưỡng sai — cả hai frame bị bỏ sót đều nằm lọt trong
# khoảng của frame trống (17.17-53.06) nên không ngưỡng nào cứu được.
# Kiểm tra trên dump: chỉ door_00 và door_01 được gán nhãn `door` trong cả 27 frame,
# cả hai đều là vật cản thật => cửa thoát này không mất specificity nào.
DEPTH_BLIND_TYPES = frozenset({"step", "hole", "door"})


def path_masks(h: int, w: int):
    """Trả (band, col) cho từng pixel; -1 = ngoài lối đi."""
    import numpy as np

    ys = (np.arange(h) + 0.5) / h
    xs = (np.arange(w) + 0.5) / w
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    # t = 0 ở đáy ảnh (chân người dùng), t = 1 ở đỉnh hình thang (xa nhất)
    t = (1.0 - yy) / (1.0 - PATH_TOP_Y)
    inside_y = (t >= 0.0) & (t <= 1.0)
    halfw = PATH_BOT_HALFW + np.clip(t, 0, 1) * (PATH_TOP_HALFW - PATH_BOT_HALFW)
    u = (xx - 0.5) / halfw                      # -1..1 bên trong lối đi
    inside = inside_y & (np.abs(u) <= 1.0)

    band = np.full((h, w), -1, dtype=np.int8)
    band[inside] = np.clip((t[inside] * N_BANDS).astype(np.int8), 0, N_BANDS - 1)

    col = np.full((h, w), -1, dtype=np.int8)
    col[inside] = 1
    col[inside & (u < -COL_EDGE)] = 0
    col[inside & (u > COL_EDGE)] = 2
    return band, col


def free_space(depth, band, col, pct: int = DEPTH_PCT):
    """Quét đơn điệu depth từ gần ra xa trên 3 cột.

    Trả (free_per_col, grid). free_per_col[c] = xa nhất nhìn thấy được ở cột c.
    """
    import numpy as np

    grid = np.full((N_BANDS, 3), np.nan)
    for i in range(N_BANDS):
        for c in range(3):
            sel = depth[(band == i) & (col == c)]
            if sel.size >= MIN_PIX:
                grid[i, c] = np.percentile(sel, pct)

    free = []
    for c in range(3):
        run = 0.0
        for i in range(N_BANDS):                # i tăng = đi ra xa
            d = grid[i, c]
            if np.isnan(d):
                continue
            if run == 0.0 or d > run * BREAK_RATIO:
                run = max(run, d)               # lối đi còn lùi ra xa được
            else:
                break                           # depth thôi tăng -> bị chắn
        free.append(run if run > 0 else np.nan)
    return np.array(free), grid


def summarize_free(free) -> dict:
    """free_per_col -> {'free3','freeC','col','per_col'} như dump của 11."""
    import numpy as np

    all_nan = bool(np.all(np.isnan(free)))
    c = -1 if all_nan else int(np.nanargmin(free))
    return {
        "free3": float("nan") if all_nan else float(np.nanmin(free)),
        "freeC": float(free[1]),
        "col": c,
        "per_col": [None if v != v else round(float(v), 2) for v in free],
    }


def depth_says_near(free_val: float | None,
                    thr: float = DEPTH_THRESHOLD) -> bool:
    """NaN/None = không đọc được lối đi -> KHÔNG khẳng định gần.

    §16 "thà im hơn đoán": khi depth mù thì để VLM và cửa thoát quyết.
    """
    if free_val is None or free_val != free_val:
        return False
    return free_val < thr


def should_warn_fused(parsed: dict | None, free_val: float | None,
                     thr: float = DEPTH_THRESHOLD) -> bool:
    """Decision Engine (§16) — chiến lược E đã đo được.

    Phân công theo đúng chỗ mỗi model mạnh:
      VLM   quyết CÓ VẬT GÌ (gọi đúng tên 85%, đúng hướng 85%)
      depth quyết CÓ ĐỦ GẦN KHÔNG (VLM mù hoàn toàn khoản này)
    Cộng thêm cửa thoát: depth không được phủ quyết lớp vật nó vốn mù.

    parsed=None (sai schema) -> im lặng, tuyệt đối không đoán.
    """
    if parsed is None or not parsed.get("hazard"):
        return False
    obj = parsed.get("type")
    if obj in DEPTH_BLIND_TYPES and parsed.get("distance") == "near":
        return True                     # cửa thoát: tin VLM, depth mù lớp này
    return depth_says_near(free_val, thr)
