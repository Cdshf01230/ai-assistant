#!/usr/bin/env python3
"""Smoke test + benchmark TTS (VieNeu-TTS v3 Turbo) trên CPU.

Theo §24, TTS chạy CPU để nhường toàn bộ T4 cho VLM. VieNeu SDK v3.3.0 tự
chọn ONNX (torch-free, int8) khi không thấy GPU — và với câu ngắn thì đường
CPU/ONNX còn nhanh hơn GPU vì GPU chỉ thắng khi batch text dài.

Ở đây ta cố tình ẩn CUDA để chắc chắn không chiếm VRAM của VLM.

Dùng:
    python scripts/04_smoke_tts.py
    python scripts/04_smoke_tts.py --list-voices
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

# PHẢI set trước khi vieneu/onnxruntime khởi tạo: ẩn GPU khỏi TTS.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mvp_config as cfg  # noqa: E402

OUT_DIR = "/home/ubuntu/ai-assistant/assets/tts_test"

# Hai loại câu theo §9: critical ngắn (đáng cache) và dynamic dài (buộc phải TTS).
CRITICAL = ["Dừng lại.", "Rẽ trái.", "Có vật cản phía trước bên phải."]
DYNAMIC = [
    "Khoảng ba mươi mét nữa rẽ phải vào đường Nguyễn Trãi.",
    "Đang dẫn đường tới Vincom Bà Triệu, còn khoảng bốn trăm mét.",
]
# Dùng lại làm input cho STT smoke test (§5)
COMMANDS = [
    "Đưa tôi đến Vincom.",
    "Xung quanh có gì?",
    "Đọc biển phía trước.",
    "Dừng dẫn đường.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp32"])
    args = ap.parse_args()

    try:
        from vieneu import Vieneu
    except ImportError:
        sys.exit("Chưa cài `vieneu`. Chạy: .venv/bin/pip install vieneu")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Model: {cfg.TTS_MODEL} (SDK vieneu, CPU/ONNX, precision={args.precision})")
    t0 = time.perf_counter()
    tts = Vieneu() if args.precision == "int8" else Vieneu(precision="fp32")
    print(f"Load : {time.perf_counter() - t0:.1f}s")

    if args.list_voices:
        for label, voice_id in tts.list_preset_voices():
            print(f"  {voice_id:<28} {label}")
        return 0

    sr = getattr(tts, "sample_rate", 48000)
    results: list[tuple[str, str, float, float]] = []

    for group, texts in (("critical", CRITICAL), ("dynamic", DYNAMIC), ("command", COMMANDS)):
        print(f"\n--- {group} ---")
        for i, text in enumerate(texts):
            t = time.perf_counter()
            audio = tts.infer(text, voice_id=cfg.TTS_VOICE)
            ms = (time.perf_counter() - t) * 1000

            path = os.path.join(OUT_DIR, f"{group}_{i:02d}.wav")
            tts.save(audio, path)

            n = len(audio) if hasattr(audio, "__len__") else 0
            dur_s = n / sr if n else 0.0
            rtf = (ms / 1000) / dur_s if dur_s else float("nan")
            results.append((group, text, ms, rtf))
            print(f"  {ms:6.0f} ms | audio {dur_s:4.1f}s | RTF {rtf:4.2f} | {text}")

    crit = [ms for g, _, ms, _ in results if g == "critical"]
    dyn = [ms for g, _, ms, _ in results if g == "dynamic"]
    print("\n--- Kết luận TTS ---")
    print(f"  câu critical: median {statistics.median(crit):.0f} ms")
    print(f"  câu dynamic : median {statistics.median(dyn):.0f} ms")
    print(f"  RTF median  : {statistics.median([r for *_, r in results]):.2f} (<1.0 = nhanh hơn realtime)")
    print(f"\n  WAV -> {OUT_DIR}")
    print("  => Câu critical vẫn nên cache sẵn (§9): chạy scripts/05_build_alert_cache.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
