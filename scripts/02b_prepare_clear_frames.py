#!/usr/bin/env python3
"""Bổ sung frame ÂM (đường thông thoáng) cho bộ eval.

Vì sao cần riêng script này: bộ frame của 02_prepare_frames.py lệch nặng — 12 frame
có vật cản / 2 frame trống. Với chỉ 2 frame âm thì specificity nhảy theo bước 50%,
đo được 100% cũng không nói lên gì. Muốn biết VLM thật sự NHÌN ảnh hay chỉ đọc
prior của prompt thì phải có đủ frame âm.

Ba loại frame âm, khó dần:
  clear_*  — lối đi trống hẳn (âm dễ)
  far_*    — có vật nhưng ở XA (§16: không được báo) (âm khó)
  side_*   — có vật nhưng lệch hẳn sang bên, không chắn đường (âm khó)

Thêm cả synthetic clear nhiều biến thể để loại giả thuyết "model chỉ đang phản ứng
với ảnh trơn màu phẳng" thay vì thật sự đánh giá lối đi.

Dùng:
    python scripts/02b_prepare_clear_frames.py
    python scripts/02b_prepare_clear_frames.py --per-scenario 3
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 02_prepare_frames không import được bằng tên (bắt đầu bằng số) -> nạp bằng đường dẫn.
_spec = importlib.util.spec_from_file_location(
    "prep_frames", os.path.join(HERE, "02_prepare_frames.py")
)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

FRAMES_DIR = prep.FRAMES_DIR

NEG_SCENARIOS = {
    "clear": "filetype:bitmap empty sidewalk pavement no people",
    "clearhall": "filetype:bitmap empty corridor hallway indoor",
    "clearpath": "filetype:bitmap empty footpath park walkway",
    "far": "filetype:bitmap long straight sidewalk distant pedestrians",
    "side": "filetype:bitmap sidewalk parked cars along road empty pavement",
}


def make_synthetic_clear() -> None:
    """Nhiều biến thể 'đường trống' khác nhau về màu / hình học / texture.

    Nếu model trả CLEAR cho cả 5 biến thể này nhưng vẫn báo hazard cho các frame
    dương thì đó là dấu hiệu nó thật sự phân biệt, không phải ăn may.
    """
    from PIL import Image, ImageDraw

    variants = {
        # (màu trời, màu lối đi, màu nền hai bên, có vạch kẻ giữa đường)
        "synthclear_grey.jpg": ((170, 190, 210), (105, 105, 110), (150, 160, 170), False),
        "synthclear_warm.jpg": ((235, 205, 165), (150, 130, 110), (110, 130, 90), False),
        "synthclear_dark.jpg": ((35, 40, 55), (55, 55, 60), (30, 35, 40), True),
        "synthclear_tiled.jpg": ((200, 215, 230), (185, 180, 172), (140, 150, 140), True),
        "synthclear_indoor.jpg": ((230, 228, 222), (205, 200, 192), (215, 210, 200), False),
    }
    for name, (sky, path, side, marks) in variants.items():
        img = Image.new("RGB", (640, 360), side)
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, 640, 190), fill=sky)
        d.polygon([(0, 360), (640, 360), (420, 190), (220, 190)], fill=path)
        if marks:  # vạch kẻ dọc giữa lối đi — thêm chi tiết nhưng vẫn không có vật cản
            for y0, y1 in ((200, 230), (250, 300), (320, 360)):
                t = (y0 - 190) / 170
                w = 4 + 10 * t
                cx = 320
                d.polygon([(cx - w, y1), (cx + w, y1), (cx + w * 0.7, y0), (cx - w * 0.7, y0)],
                          fill=tuple(min(255, c + 45) for c in path))
        img.save(os.path.join(FRAMES_DIR, name), quality=85)
        print(f"  {name} (synthetic clear)")


def make_synthetic_hard() -> None:
    """Âm KHÓ + dương đối chứng, dựng bằng hình học nên ground truth tuyệt đối chắc.

    §16 nói không được cảnh báo vật ở xa hoặc lệch hẳn sang bên. Ảnh Commons không
    kiểm soát được khoảng cách, còn ở đây thì kiểm soát được: cùng một cây cột, chỉ
    đổi kích thước/vị trí. Nếu model trả cùng đáp án cho cả 3 thì nó không hề suy
    luận về khoảng cách — nó chỉ đang phát hiện "có vật thể".

    Lối đi là polygon [(0,360),(640,360),(420,190),(220,190)]:
      y=190 -> lối đi rộng x 220..420 (xa)
      y=250 -> lối đi rộng x 142..498
      y=360 -> lối đi rộng x 0..640 (ngay chân)
    """
    from PIL import Image, ImageDraw

    POLE = (90, 90, 95)
    scenes = {
        # ÂM: cột nằm trong lối đi nhưng ở tận đường chân trời -> chưa cần cảnh báo
        "synthfar_pole.jpg": [("rectangle", (324, 172, 337, 214), POLE)],
        # ÂM: người ở xa, nhỏ, gần đường chân trời
        "synthfar_person.jpg": [("ellipse", (336, 190, 345, 199), (200, 170, 150)),
                                ("rectangle", (335, 199, 347, 226), (40, 60, 130))],
        # ÂM: cột to nhưng nằm HẲN ngoài lối đi, bên trái (x<142 tại y=250)
        "synthside_pole.jpg": [("rectangle", (55, 110, 100, 300), POLE)],
        # DƯƠNG đối chứng: đúng cây cột đó nhưng ngay trước mặt, trong lối đi
        "synthnear_pole.jpg": [("rectangle", (288, 145, 342, 360), POLE)],
    }
    for name, shapes in scenes.items():
        img = Image.new("RGB", (640, 360), (150, 160, 170))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, 640, 190), fill=(170, 190, 210))
        d.polygon([(0, 360), (640, 360), (420, 190), (220, 190)], fill=(105, 105, 110))
        for kind, box, color in shapes:
            getattr(d, kind)(box, fill=color)
        img.save(os.path.join(FRAMES_DIR, name), quality=85)
        print(f"  {name} (synthetic hard)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=3)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    os.makedirs(FRAMES_DIR, exist_ok=True)
    print(f"Frame âm -> {FRAMES_DIR}\n")

    total = 0
    if not args.no_download:
        for scenario, query in NEG_SCENARIOS.items():
            total += prep.download_scenario(scenario, query, args.per_scenario)

    print()
    make_synthetic_clear()
    make_synthetic_hard()
    print(f"\nTải được {total} ảnh thật + 5 synthetic clear + 4 synthetic hard.")
    print("BƯỚC TIẾP: xem từng ảnh rồi thêm nhãn vào assets/frames/ground_truth.json — "
          "ảnh Commons không đảm bảo là trống thật, phải mắt người xác nhận. "
          "(Đã gặp thực tế: query 'empty sidewalk' trả về ảnh có người nằm giữa lối đi.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
