#!/usr/bin/env python3
"""A/B prompt cho VLM: đo LATENCY và số token của từng biến thể prompt.

⚠️ KHÔNG dùng script này để chọn prompt. Nó chỉ đo tốc độ.
   Chọn prompt thì chạy 09_eval_vlm.py (chấm theo nhãn tay).

Vì sao: bản đầu của script này chấm prompt theo "tỉ lệ trả CLEAR", với đúng MỘT
frame ground-truth âm. Cách chấm đó thưởng cho prompt nào nói CLEAR nhiều nhất —
kể cả khi nó bỏ sót cột điện giữa đường. Nó đã chọn ra một prompt bỏ sót 11/15 vật
cản thật. Với thiết bị dẫn đường cho người khiếm thị, đó là kiểu sai nguy hiểm nhất.
09_eval_vlm.py tách recall khỏi specificity trên 27 frame có nhãn tay và thay thế
hoàn toàn phần "chọn prompt" ở đây.

Cột "CLEAR rate" bên dưới giữ lại làm chỉ dấu thô để phát hiện prompt bị suy sụp
(0% = luôn báo động, 100% = luôn im lặng), KHÔNG phải số đo accuracy.

Dùng:
    python scripts/08_ab_prompt.py
    python scripts/08_ab_prompt.py --repeat 2
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
import vlm_prompts as vp  # noqa: E402

FRAMES_DIR = "/home/ubuntu/ai-assistant/assets/frames"

# Prompt để ở vlm_prompts.py để 08 (latency) và 09 (accuracy) không lệch nhau.
PROMPTS = vp.PROMPTS


def load_frames(long_edge: int):
    from PIL import Image

    names = sorted(f for f in os.listdir(FRAMES_DIR) if f.lower().endswith((".jpg", ".png")))
    out = []
    for n in names:
        img = Image.open(os.path.join(FRAMES_DIR, n)).convert("RGB")
        w, h = img.size
        if max(w, h) > long_edge:
            s = long_edge / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=cfg.JPEG_QUALITY)
        out.append((n, Image.open(buf).convert("RGB")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.VLM_MODEL)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation=cfg.ATTN_IMPL, device_map="cuda:0",
    )
    model.eval()
    frames = load_frames(cfg.FRAME_LONG_EDGE)
    print(f"{args.model} | fp16 | {len(frames)} frame | repeat={args.repeat}\n")

    def infer(img, prompt, max_new):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "Frame:"}]},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        plen = inputs["input_ids"].shape[-1]
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        gen = out[0][plen:]
        return processor.decode(gen, skip_special_tokens=True).strip(), ms, len(gen)

    infer(frames[0][1], PROMPTS["pipe_v2"], 12)  # warmup

    summary = []
    for name, prompt in PROMPTS.items():
        lat, toks, bad, clear = [], [], 0, 0
        per_frame: dict[str, str] = {}

        for _ in range(args.repeat):
            for fname, img in frames:
                max_new = vp.MAX_NEW_TOKENS.get(name, cfg.VLM_MAX_NEW_TOKENS)
                text, ms, n = infer(img, prompt, max_new)
                lat.append(ms)
                toks.append(n)
                parsed = cfg.parse_pipe4(vp.split_v5(text))
                if parsed is None:
                    bad += 1
                    per_frame.setdefault(fname, f"SAI: {text[:24]}")
                    continue
                is_clear = not parsed["hazard"]
                clear += is_clear
                per_frame.setdefault(
                    fname, "CLEAR" if is_clear
                    else f"{parsed['type']}|{parsed['position']}|{parsed['action']}"
                )

        n_tot = len(lat)
        med = statistics.median(lat)
        clear_rate = clear / n_tot
        summary.append((name, med, statistics.mean(toks), clear_rate, bad))

        print(f"--- {name} ---")
        print(f"  latency median {med:.0f} ms | token mean {statistics.mean(toks):.1f} "
              f"| CLEAR {clear}/{n_tot} ({clear_rate:.0%}) | sai schema {bad}")
        for fname in sorted(per_frame):
            print(f"    {fname:<26} {per_frame[fname]}")
        print()

    print("=== Tổng hợp (chỉ latency — KHÔNG phải bảng chọn prompt) ===")
    print(f"{'prompt':<10} {'latency':>9} {'token':>6} {'CLEAR rate':>11} {'sai schema':>11}")
    for name, med, tk, cr, bad in summary:
        note = ""
        if cr == 0:
            note = "  <- luôn báo động, suy sụp"
        elif cr == 1:
            note = "  <- luôn im lặng, suy sụp"
        print(f"{name:<10} {med:8.0f}ms {tk:6.1f} {cr:10.0%} {bad:11d}{note}")

    fastest = min(summary, key=lambda s: s[1])
    print(f"\nNhanh nhất: {fastest[0]} — {fastest[1]:.0f} ms, {fastest[2]:.1f} token.")
    print("Nhanh KHÔNG có nghĩa là đúng. Chốt prompt bằng 09_eval_vlm.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
