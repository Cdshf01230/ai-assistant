#!/usr/bin/env python3
"""Tune latency VLM: tách TTFT / decode và sweep resolution × format output.

Lý do: 03_smoke_vlm.py cho ~1100 ms trên T4, đã vượt ngân sách §12 trước khi
cộng network + TTS. Cần biết thời gian nằm ở đâu mới tối ưu đúng chỗ:

  T_prefill (TTFT) = vision tower + encode prompt  -> giảm bằng cách hạ resolution
  T_decode         = sinh token tuần tự            -> giảm bằng cách rút ngắn output

§7 đã gợi ý format cực ngắn `HAZARD|FRONT_RIGHT|MOVE_LEFT` thay cho JSON —
script này đo xem tiết kiệm được bao nhiêu ms thật.

Dùng:
    python scripts/07_tune_vlm.py
    python scripts/07_tune_vlm.py --frames 4 --repeat 3
"""

from __future__ import annotations

import argparse
import io
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mvp_config as cfg  # noqa: E402

FRAMES_DIR = "/home/ubuntu/ai-assistant/assets/frames"

# Format A: JSON schema như §7 mô tả (dễ parse, ~23 token)
PROMPT_JSON = cfg.VLM_SYSTEM_PROMPT

# Format B: pipe-compact như §7 gợi ý ở cuối (ít token nhất).
# LƯU Ý: viết schema thành các dòng "TYPE: ..." kiểu template khiến model
# copy nguyên template ra thay vì điền. Phải diễn đạt bằng câu + few-shot.
PROMPT_PIPE = (
    "You are the vision module of a mobility aid for blind pedestrians. "
    "Report ONLY the single most urgent hazard in the walking path.\n"
    "Answer with exactly one line: three uppercase fields joined by '|'.\n"
    "First field is the object: PERSON, VEHICLE, POLE, STEP, DOOR, FURNITURE, "
    "HOLE, WALL, OBJECT, or CLEAR if the path is free.\n"
    "Second field is where it is: FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, RIGHT, or NONE.\n"
    "Third field is what the walker should do: NONE, SLOW, STOP, MOVE_LEFT, or MOVE_RIGHT.\n"
    "Valid answers look like POLE|FRONT_RIGHT|MOVE_LEFT or VEHICLE|FRONT|STOP "
    "or CLEAR|NONE|NONE.\n"
    "Output that one line only, no other words."
)

CONFIGS = [
    # (label, long_edge, prompt, max_new_tokens)
    ("448 / json", 448, PROMPT_JSON, 40),
    ("448 / pipe", 448, PROMPT_PIPE, 12),
    ("512 / pipe", 512, PROMPT_PIPE, 12),
    ("640 / json", 640, PROMPT_JSON, 40),
    ("640 / pipe", 640, PROMPT_PIPE, 12),
]


def load_frames(long_edge: int, limit: int):
    from PIL import Image

    names = sorted(f for f in os.listdir(FRAMES_DIR) if f.lower().endswith((".jpg", ".png")))[:limit]
    out = []
    for n in names:
        img = Image.open(os.path.join(FRAMES_DIR, n)).convert("RGB")
        w, h = img.size
        if max(w, h) > long_edge:
            s = long_edge / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=cfg.JPEG_QUALITY)
        out.append((n, Image.open(buf).convert("RGB"), buf.tell() / 1024))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.VLM_MODEL)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        sys.exit("Cần CUDA.")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation=cfg.ATTN_IMPL, device_map="cuda:0",
    )
    model.eval()
    print(f"{args.model} | fp16 | {torch.cuda.get_device_name(0)}\n")

    def build(img, prompt):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "Frame:"}]},
        ]
        return processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

    def timed(inputs, max_new):
        """Trả (ttft_ms, total_ms, n_tokens). TTFT đo bằng 1 forward pass prefill."""
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - t) * 1000

        prompt_len = inputs["input_ids"].shape[-1]
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        torch.cuda.synchronize()
        total = (time.perf_counter() - t) * 1000
        gen = out[0][prompt_len:]
        return ttft, total, len(gen), processor.decode(gen, skip_special_tokens=True).strip()

    # Warmup một lần cho toàn bộ phiên
    warm = load_frames(448, 1)
    timed(build(warm[0][1], PROMPT_PIPE), 16)

    print(f"{'config':<12} {'img tok':>8} {'TTFT':>8} {'decode':>8} {'total':>8} {'tok':>5}  ví dụ output")
    print("-" * 92)

    rows = []
    for label, edge, prompt, max_new in CONFIGS:
        frames = load_frames(edge, args.frames)
        ttfts, totals, ntoks, sample, plens = [], [], [], "", []
        for _ in range(args.repeat):
            for _name, img, _kb in frames:
                inputs = build(img, prompt)
                plens.append(inputs["input_ids"].shape[-1])
                ttft, total, n, text = timed(inputs, max_new)
                ttfts.append(ttft)
                totals.append(total)
                ntoks.append(n)
                sample = sample or text
        med_ttft = statistics.median(ttfts)
        med_total = statistics.median(totals)
        rows.append((label, med_total, med_ttft, statistics.mean(ntoks)))
        print(f"{label:<12} {statistics.mean(plens):8.0f} {med_ttft:7.0f}ms "
              f"{med_total - med_ttft:7.0f}ms {med_total:7.0f}ms {statistics.mean(ntoks):5.1f}  "
              f"{sample[:34]!r}")

    best = min(rows, key=lambda r: r[1])
    base = next((r for r in rows if r[0] == "640 / json"), best)
    print(f"\nBaseline hiện tại (640/json): {base[1]:.0f} ms")
    print(f"Cấu hình nhanh nhất ({best[0]}): {best[1]:.0f} ms "
          f"-> tiết kiệm {base[1] - best[1]:.0f} ms ({(1 - best[1] / base[1]) * 100:.0f}%)")
    verdict = ("rất tốt" if best[1] < 300 else "tốt" if best[1] < 600
               else "demo được" if best[1] < 1000 else "vẫn CHƯA ĐẠT")
    print(f"§12: mức '{verdict}' cho riêng VLM (chưa cộng upload + TTS)")
    print("\nGhi chú: hạ resolution làm giảm TTFT (ít vision token), rút ngắn output làm")
    print("giảm decode (T4 decode ~memory-bandwidth-bound, mỗi token tốn ms cố định).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
