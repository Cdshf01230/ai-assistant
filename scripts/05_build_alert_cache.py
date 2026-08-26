#!/usr/bin/env python3
"""Sinh sẵn toàn bộ audio cảnh báo tối quan trọng (§9, §17.3).

Decision Engine chỉ trả message_code; server/frontend phát WAV tương ứng,
không inference TTS -> latency gần như bằng 0 cho đúng nhóm câu quan trọng nhất.

Sinh ra:
    assets/audio/STOP.wav, TURN_LEFT.wav, ...
    assets/audio/manifest.json   (message_code -> file, text, duration)

Dùng:
    python scripts/05_build_alert_cache.py
    python scripts/05_build_alert_cache.py --voice <voice_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # cache generation cũng chạy CPU

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mvp_config as cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=cfg.TTS_VOICE, help="voice_id preset")
    ap.add_argument("--force", action="store_true", help="ghi đè file đã có")
    args = ap.parse_args()

    try:
        from vieneu import Vieneu
    except ImportError:
        sys.exit("Chưa cài `vieneu`. Chạy: .venv/bin/pip install vieneu")

    os.makedirs(cfg.ALERT_AUDIO_DIR, exist_ok=True)

    print(f"Sinh {len(cfg.ALERT_PHRASES)} câu cảnh báo -> {cfg.ALERT_AUDIO_DIR}")
    tts = Vieneu()
    sr = getattr(tts, "sample_rate", 48000)
    kwargs = {"voice_id": args.voice} if args.voice else {}

    manifest: dict[str, dict] = {}
    t_all = time.perf_counter()

    for code, text in cfg.ALERT_PHRASES.items():
        path = os.path.join(cfg.ALERT_AUDIO_DIR, f"{code}.wav")
        if os.path.exists(path) and not args.force:
            print(f"  bỏ qua {code} (đã có)")
        else:
            audio = tts.infer(text, **kwargs)
            tts.save(audio, path)

        size = os.path.getsize(path)
        # WAV PCM16 mono: (bytes - 44 byte header) / 2 / sample_rate
        dur = max(0.0, (size - 44) / 2 / sr)
        manifest[code] = {
            "file": f"{code}.wav",
            "text": text,
            "duration_s": round(dur, 2),
            "bytes": size,
        }
        print(f"  {code:<22} {dur:4.1f}s  {size / 1024:6.0f} KB  {text}")

    manifest_path = os.path.join(cfg.ALERT_AUDIO_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {"sample_rate": sr, "voice": args.voice or "default", "alerts": manifest},
            f, ensure_ascii=False, indent=2,
        )

    total_kb = sum(m["bytes"] for m in manifest.values()) / 1024
    print(f"\nXong trong {time.perf_counter() - t_all:.1f}s | tổng {total_kb:.0f} KB")
    print(f"Manifest: {manifest_path}")
    print("\nFrontend nên preload toàn bộ file này khi mở app để phát tức thì.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
