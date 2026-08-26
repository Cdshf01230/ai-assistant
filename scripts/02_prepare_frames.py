#!/usr/bin/env python3
"""Chuẩn bị frame test cho VLM.

Tải ảnh thật từ Wikimedia Commons theo đúng các điều kiện §8 yêu cầu
(trong nhà / ngoài đường / ánh sáng yếu / đông người / cột / xe / bậc thang / cửa),
kèm vài ảnh synthetic để có case "đường thông thoáng" xác định trước.

ĐÂY CHỈ LÀ BỘ TEST TẠM để verify pipeline và đo latency. §8 yêu cầu 100–500 frame
THẬT do chính camera điện thoại chụp ở góc nhìn người đi bộ — ảnh Commons chụp
bằng máy ảnh ở góc khác nên không thay thế được. Bỏ frame thật vào assets/frames/.

Dùng:
    python scripts/02_prepare_frames.py
    python scripts/02_prepare_frames.py --per-scenario 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

FRAMES_DIR = "/home/ubuntu/ai-assistant/assets/frames"
UA = "mvp-blind-assistant/0.1 (model benchmark; contact: dev)"

# scenario -> query Commons. Tên file sẽ mang tiền tố scenario để dễ đọc kết quả.
SCENARIOS = {
    "street": "filetype:bitmap sidewalk pedestrian street",
    "crowd": "filetype:bitmap crowded pedestrian street market",
    "pole": "filetype:bitmap utility pole sidewalk obstruction",
    "stairs": "filetype:bitmap outdoor staircase steps public",
    "door": "filetype:bitmap building entrance glass door",
    "indoor": "filetype:bitmap indoor corridor hallway chairs",
    "lowlight": "filetype:bitmap street night pedestrian dim",
    "vehicle": "filetype:bitmap motorcycle parked sidewalk",
}


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def fetch_bytes(url: str, retries: int = 4) -> bytes:
    """Commons rate-limit khá gắt (429) — backoff rồi thử lại."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 503):
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last if last else RuntimeError("fetch failed")


def download_scenario(scenario: str, query: str, limit: int) -> int:
    api = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": query, "gsrlimit": limit, "gsrnamespace": 6,
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 800,
    })
    try:
        data = fetch_json(api)
    except Exception as exc:
        print(f"  {scenario}: API lỗi ({type(exc).__name__})")
        return 0

    got = 0
    for i, page in enumerate((data.get("query", {}).get("pages") or {}).values()):
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl")
        if not url:
            continue
        dest = os.path.join(FRAMES_DIR, f"{scenario}_{i:02d}.jpg")
        if os.path.exists(dest):
            got += 1
            continue
        try:
            blob = fetch_bytes(url)
        except Exception as exc:
            code = getattr(exc, "code", "")
            print(f"  {scenario}_{i:02d}: bỏ qua ({type(exc).__name__} {code})")
            continue
        with open(dest, "wb") as f:
            f.write(blob)
        lic = (info.get("extmetadata") or {}).get("LicenseShortName", {}).get("value", "?")
        print(f"  {scenario}_{i:02d}.jpg  {len(blob) / 1024:5.0f} KB  [{lic}]")
        got += 1
        time.sleep(0.8)  # tôn trọng rate limit của Commons
    return got


def make_synthetic() -> None:
    """Case xác định trước — dùng để đo latency ổn định, KHÔNG dùng đánh giá accuracy."""
    from PIL import Image, ImageDraw

    scenes = {
        "synth_clear.jpg": [],
        "synth_pole_right.jpg": [("rectangle", (470, 120, 510, 400), (90, 90, 95))],
        "synth_person_front.jpg": [
            ("ellipse", (300, 140, 340, 180), (200, 170, 150)),
            ("rectangle", (295, 180, 345, 320), (40, 60, 130)),
        ],
    }
    for name, shapes in scenes.items():
        img = Image.new("RGB", (640, 360), (150, 160, 170))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, 640, 190), fill=(170, 190, 210))          # trời
        d.polygon([(0, 360), (640, 360), (420, 190), (220, 190)], fill=(105, 105, 110))  # lối đi
        for kind, box, color in shapes:
            getattr(d, kind)(box, fill=color)
        img.save(os.path.join(FRAMES_DIR, name), quality=85)
        print(f"  {name} (synthetic)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=2)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    os.makedirs(FRAMES_DIR, exist_ok=True)
    print(f"Frames -> {FRAMES_DIR}\n")

    total = 0
    if not args.no_download:
        for scenario, query in SCENARIOS.items():
            total += download_scenario(scenario, query, args.per_scenario)

    print()
    make_synthetic()

    files = sorted(f for f in os.listdir(FRAMES_DIR) if f.lower().endswith((".jpg", ".png")))
    print(f"\nTổng {len(files)} frame ({total} ảnh thật + 3 synthetic)")
    if total == 0:
        print("Không tải được ảnh thật — chỉ có synthetic, latency vẫn đo được "
              "nhưng KHÔNG đánh giá được accuracy.")
    print("\n§8: cần 100–500 frame THẬT từ camera điện thoại ở góc người đi bộ. "
          "Bỏ thêm vào thư mục trên.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
