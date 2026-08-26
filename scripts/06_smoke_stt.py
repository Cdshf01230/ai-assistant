#!/usr/bin/env python3
"""Smoke test + benchmark STT (PhoWhisper-small) trên CPU.

§24: STT chạy CPU, chỉ đẩy lên GPU nếu đo thấy quá chậm — vì obstacle warning
(VLM) ưu tiên hơn STT. §5: chỉ cần command ngắn, push-to-talk, không cần
continuous recognition.

Input audio lấy từ assets/tts_test/command_*.wav (do 04_smoke_tts.py sinh),
hoặc chỉ định --audio-dir tới thư mục WAV thật của bạn.

Dùng:
    python scripts/06_smoke_stt.py
    python scripts/06_smoke_stt.py --device cuda        # so sánh nếu CPU chậm
    python scripts/06_smoke_stt.py --model vinai/PhoWhisper-medium
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mvp_config as cfg  # noqa: E402

DEFAULT_AUDIO_DIR = "/home/ubuntu/ai-assistant/assets/tts_test"

# Ground truth cho audio do 04_smoke_tts.py sinh ra (COMMANDS trong script đó)
EXPECTED = {
    "command_00.wav": "đưa tôi đến vincom",
    "command_01.wav": "xung quanh có gì",
    "command_02.wav": "đọc biển phía trước",
    "command_03.wav": "dừng dẫn đường",
}

# §5: intent parser rule-based, không cần model riêng.
INTENT_RULES = [
    ("NAVIGATE", ("đưa tôi đến", "đi đến", "dẫn tôi đến", "đến")),
    ("DESCRIBE", ("xung quanh", "có gì", "phía trước là gì", "mô tả")),
    ("READ_TEXT", ("đọc biển", "đọc chữ", "đọc giúp", "đọc")),
    ("STOP_NAV", ("dừng dẫn đường", "hủy dẫn đường", "dừng lại")),
    ("WHERE_AM_I", ("tôi đang ở đâu", "đang ở đâu", "vị trí")),
]

CONFIDENCE_THRESHOLD = 0.60  # §5: dưới ngưỡng thì hỏi lại, không đoán


def parse_intent(text: str) -> tuple[str, str | None]:
    low = text.lower().strip(" .!?,")
    for intent, keys in INTENT_RULES:
        for k in keys:
            if k in low:
                slot = low.split(k, 1)[1].strip(" .!?,") if intent == "NAVIGATE" else None
                return intent, (slot or None)
    return "UNKNOWN", None


def normalize(s: str) -> str:
    return " ".join(s.lower().strip(" .!?,").split())


def wer(ref: str, hyp: str) -> float:
    """Word error rate đơn giản (Levenshtein trên token)."""
    r, h = normalize(ref).split(), normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.STT_MODEL)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    import librosa
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    if not os.path.isdir(args.audio_dir):
        sys.exit(f"Không thấy {args.audio_dir} — chạy scripts/04_smoke_tts.py trước.")
    wavs = sorted(f for f in os.listdir(args.audio_dir) if f.lower().endswith((".wav", ".mp3", ".m4a")))
    if not wavs:
        sys.exit(f"{args.audio_dir} không có file audio nào.")

    # PhoWhisper-small chỉ có pytorch_model.bin (không có safetensors).
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print(f"Model : {args.model}")
    print(f"Device: {args.device} | dtype={str(dtype).replace('torch.', '')} | threads={torch.get_num_threads()}")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=dtype)
    model.to(args.device).eval()
    print(f"Load  : {time.perf_counter() - t0:.1f}s\n")

    sr = processor.feature_extractor.sampling_rate  # 16000
    lat: list[float] = []
    wers: list[float] = []

    for wav in wavs:
        path = os.path.join(args.audio_dir, wav)
        speech, _ = librosa.load(path, sr=sr, mono=True)
        dur = len(speech) / sr

        per_run = []
        text = ""
        for _ in range(args.runs):
            t = time.perf_counter()
            inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
            feats = inputs.input_features.to(args.device, dtype=dtype)
            with torch.inference_mode():
                ids = model.generate(feats, language="vi", task="transcribe", max_new_tokens=64)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            per_run.append((time.perf_counter() - t) * 1000)

        ms = statistics.median(per_run)
        lat.append(ms)
        rtf = (ms / 1000) / dur if dur else float("nan")

        intent, slot = parse_intent(text)
        line = f"  {wav:<18} {dur:4.1f}s audio | {ms:6.0f} ms | RTF {rtf:4.2f} | {intent}"
        if slot:
            line += f"(dest={slot!r})"
        print(line)
        print(f"    -> {text!r}")

        if wav in EXPECTED:
            e = wer(EXPECTED[wav], text)
            wers.append(e)
            print(f"       ref={EXPECTED[wav]!r}  WER={e:.0%}")

    print("\n--- Kết luận STT ---")
    print(f"  latency median {statistics.median(lat):.0f} ms | max {max(lat):.0f} ms")
    if wers:
        print(f"  WER trung bình trên {len(wers)} command: {statistics.mean(wers):.0%}")
        print("  (lưu ý: audio là TTS-synth, sạch hơn thực tế — cần thu ngoài đường để đánh giá thật)")
    print(f"  ngưỡng confidence đề xuất: {CONFIDENCE_THRESHOLD} -> dưới mức này phát REPEAT_PLEASE.wav")
    med = statistics.median(lat)
    # ĐÃ ĐO cả hai đường: CPU 2710 ms / GPU 165 ms (chênh 16x). §24 giả định STT
    # nằm ở CPU, nhưng số đo nói ngược: PhoWhisper-small fp16 chỉ ~0.5 GiB trên T4
    # đang còn trống ~10 GiB, và STT là push-to-talk (bursty) nên không giành
    # VRAM liên tục với VLM. => Khuyến nghị: STT lên GPU.
    if args.device == "cpu":
        if med > 2000:
            print(f"  => CPU {med:.0f} ms là quá chậm cho một câu lệnh ngắn. "
                  "Đã đo GPU = 165 ms (16x nhanh hơn) => ĐẨY STT LÊN GPU, "
                  "sửa lại giả định của §24.")
        else:
            print(f"  => CPU {med:.0f} ms là chấp nhận được, giữ CPU để trọn T4 cho VLM (§24).")
    else:
        print(f"  => GPU {med:.0f} ms. Tốn thêm ~0.5 GiB VRAM (T4 còn dư ~10 GiB sau VLM) "
              "và STT chỉ chạy lúc push-to-talk nên không cản VLM. Nên chọn đường này.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
