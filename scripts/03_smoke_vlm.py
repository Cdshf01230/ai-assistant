#!/usr/bin/env python3
"""Smoke test + benchmark latency cho VLM (Qwen3-VL-2B-Instruct) trên T4.

Kiểm chứng đúng những gì §7, §12, §13 yêu cầu:
  - load được bằng fp16 + sdpa trên sm_75;
  - ép được structured output ngắn (1 dòng JSON);
  - đo VRAM, TTFT và tổng latency ở resolution đã resize.

Dùng:
    python scripts/03_smoke_vlm.py
    python scripts/03_smoke_vlm.py --model HuggingFaceTB/SmolVLM2-2.2B-Instruct
    python scripts/03_smoke_vlm.py --runs 20 --max-new-tokens 48
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
import mvp_config as cfg  # noqa: E402  (phải set HF_HOME trước khi import torch/transformers)

FRAMES_DIR = "/home/ubuntu/ai-assistant/assets/frames"


def load_frames(long_edge: int) -> list[tuple[str, "Image.Image"]]:
    from PIL import Image

    if not os.path.isdir(FRAMES_DIR):
        sys.exit(f"Chưa có {FRAMES_DIR} — chạy scripts/02_prepare_frames.py trước.")
    names = sorted(f for f in os.listdir(FRAMES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        sys.exit(f"{FRAMES_DIR} rỗng — chạy scripts/02_prepare_frames.py trước.")

    out = []
    for n in names:
        img = Image.open(os.path.join(FRAMES_DIR, n)).convert("RGB")
        w, h = img.size
        if max(w, h) > long_edge:
            s = long_edge / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        # Round-trip qua JPEG để mô phỏng đúng thứ server nhận được từ WebSocket (§13)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=cfg.JPEG_QUALITY)
        kb = buf.tell() / 1024
        out.append((f"{n} [{img.size[0]}x{img.size[1]}, {kb:.0f}KB]", Image.open(buf).convert("RGB")))
    return out


def parse_vlm_output(text: str, fmt: str) -> dict | None:
    if fmt == "pipe":
        # parse_pipe4 nhận cả schema 3 và 4 trường.
        return cfg.parse_pipe4(text)
    # JSON: model đôi khi bọc trong ```json ... ``` — cắt lấy object đầu tiên.
    t = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j <= i:
        return None
    try:
        return json.loads(t[i : j + 1])
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.VLM_MODEL)
    ap.add_argument("--runs", type=int, default=3, help="số lần lặp qua toàn bộ frame")
    ap.add_argument("--format", default="pipe", choices=["pipe", "json"],
                    help="pipe = production (7-9 token); json = đối chứng (chậm 2.5x)")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--long-edge", type=int, default=cfg.FRAME_LONG_EDGE)
    args = ap.parse_args()

    prompt = cfg.VLM_PROMPT_PIPE if args.format == "pipe" else cfg.VLM_PROMPT_JSON
    max_new = args.max_new_tokens or (cfg.VLM_MAX_NEW_TOKENS if args.format == "pipe" else 40)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        sys.exit("Không có CUDA — dừng, chạy VLM trên CPU vô nghĩa cho MVP.")

    dtype = torch.float16  # sm_75: bf16 không native (bị emulate, chậm ~11x)
    print(f"Model : {args.model}")
    print(f"Device: {torch.cuda.get_device_name(0)} | dtype=float16 | attn={cfg.ATTN_IMPL}")
    print(f"Format: {args.format} | max_new_tokens={max_new} | long_edge={args.long_edge}")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=cfg.ATTN_IMPL,
        device_map="cuda:0",
    )
    model.eval()
    torch.cuda.synchronize()
    print(f"Load  : {time.perf_counter() - t0:.1f}s")
    print(f"VRAM  : {torch.cuda.memory_allocated() / 1024**3:.2f} GiB weights+, "
          f"reserved {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

    frames = load_frames(args.long_edge)
    print(f"Frames: {len(frames)}\n")

    def infer(img) -> tuple[str, float, int]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "Frame:"}]},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,          # deterministic: cảnh báo an toàn không được random
                temperature=None,
                top_p=None,
                top_k=None,
            )
        torch.cuda.synchronize()
        total = (time.perf_counter() - t_start) * 1000

        gen = out[0][prompt_len:]
        text = processor.decode(gen, skip_special_tokens=True)
        return text, total, len(gen)

    # Warmup: lần đầu luôn tính cả CUDA autotune / kernel compile.
    print("Warmup...", flush=True)
    w_text, w_ms, _ = infer(frames[0][1])
    print(f"  warmup {w_ms:.0f} ms -> {w_text.strip()[:90]}\n")

    lat: list[float] = []
    toks: list[int] = []
    bad_schema = 0

    for r in range(args.runs):
        for label, img in frames:
            text, ms, n_tok = infer(img)
            lat.append(ms)
            toks.append(n_tok)
            parsed = parse_vlm_output(text, args.format)
            if parsed is None:
                bad_schema += 1
            if r == 0:
                if parsed:
                    shown = (json.dumps(parsed, ensure_ascii=False)
                             + ("  -> CẢNH BÁO" if cfg.should_warn(parsed) else "  -> im lặng"))
                else:
                    shown = f"SAI SCHEMA: {text!r}"
                print(f"  {label}\n    {ms:7.0f} ms | {n_tok:2d} tok | {shown}")

    n = len(lat)
    lat_sorted = sorted(lat)
    print(f"\n--- Latency VLM ({n} lần infer, format={args.format}, max_new_tokens={max_new}) ---")
    print(f"  mean   {statistics.mean(lat):7.0f} ms")
    print(f"  median {statistics.median(lat):7.0f} ms")
    print(f"  p90    {lat_sorted[min(n - 1, int(n * 0.9))]:7.0f} ms")
    print(f"  min/max{min(lat):7.0f} / {max(lat):.0f} ms")
    print(f"  tokens sinh ra: mean {statistics.mean(toks):.1f}")
    print(f"  schema parse fail: {bad_schema}/{n}")
    print(f"  VRAM peak: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")

    # Đối chiếu bảng mục tiêu §12
    med = statistics.median(lat)
    budget = "rất tốt" if med < 300 else "tốt" if med < 600 else "demo được" if med < 1000 else "CHƯA ĐẠT"
    print(f"\n  => Riêng phần VLM: {med:.0f} ms — mức '{budget}' theo §12")
    print("     (chưa gồm T_upload + T_encode + T_TTS; ngân sách end-to-end < 1000 ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
