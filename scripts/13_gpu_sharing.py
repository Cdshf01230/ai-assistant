#!/usr/bin/env python3
"""Thí nghiệm chia GPU: VLM + depth + STT trên cùng một T4.

§22 nói rõ Docker KHÔNG chia T4 thành GPU ảo độc lập — các process vẫn tranh VRAM,
CUDA compute, CUDA context và memory bandwidth. §23 nói không dùng MIG. §24 bảo ưu
tiên GPU cho vision, STT xếp hàng nếu GPU bận. Nhưng cả ba mục đều không cho SỐ.
Script này đo, vì kết quả quyết định cách viết FastAPI:

  Q1. VRAM: nạp cả VLM 4B + depth + STT lên T4 14.6 GiB có vừa không?
  Q2. Tranh chấp: khi STT chạy chen vào lúc VLM đang decode thì VLM chậm đi bao
      nhiêu? Đây là con số §24 cần mà chưa ai đo.
  Q3. Serialize có cứu được không? So 3 chế độ:
        a. chạy riêng           -> baseline
        b. song song tự do      -> hai thread cùng đâm vào GPU
        c. có lock ưu tiên      -> đúng ý §24, vision đi trước
      Nếu (b) đã đủ tốt thì không cần lock, server viết đơn giản hơn nhiều.

Lưu ý về CUDA: hai thread trong CÙNG một process dùng chung một CUDA context, GPU
xen kẽ kernel của chúng. Hai PROCESS riêng thì phải context-switch, đắt hơn. Ta đo
đường 1-process vì §18/§25 chốt là 1 FastAPI app.

Dùng:
    python scripts/13_gpu_sharing.py
    python scripts/13_gpu_sharing.py --rounds 10
    python scripts/13_gpu_sharing.py --skip-stt      # chỉ đo VLM + depth
"""

from __future__ import annotations

import argparse
import io
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_path as dp  # noqa: E402
import mvp_config as cfg  # noqa: E402

FRAMES_DIR = "/home/ubuntu/ai-assistant/assets/frames"
AUDIO_DIR = "/home/ubuntu/ai-assistant/assets/tts_test"


def pct(vals: list[float], p: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8, help="số lần lặp mỗi chế độ")
    ap.add_argument("--skip-stt", action="store_true")
    ap.add_argument("--vlm", default=cfg.VLM_MODEL)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                              AutoModelForImageTextToText, AutoProcessor,
                              WhisperForConditionalGeneration)

    if not torch.cuda.is_available():
        sys.exit("Không có CUDA — dừng.")

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU   : {torch.cuda.get_device_name(0)} | {total_vram:.1f} GiB")
    print(f"Chế độ: {args.rounds} vòng mỗi chế độ\n")

    # ---- Q1: nạp cả ba, đo VRAM từng bước ------------------------------------
    print("=== Q1. VRAM khi nạp chồng cả ba model ===")
    marks = []

    def mark(label: str):
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv = torch.cuda.memory_reserved() / 1024**3
        marks.append((label, alloc, resv))
        print(f"  {label:<28} alloc {alloc:5.2f} GiB | reserved {resv:5.2f} GiB "
              f"| còn ~{total_vram - resv:5.2f} GiB")

    mark("(trống)")

    t0 = time.perf_counter()
    vlm_proc = AutoProcessor.from_pretrained(args.vlm)
    vlm = AutoModelForImageTextToText.from_pretrained(
        args.vlm, dtype=torch.float16, attn_implementation=cfg.ATTN_IMPL,
        device_map="cuda:0")
    vlm.eval()
    t_vlm = time.perf_counter() - t0
    mark(f"+ VLM 4B fp16")

    t0 = time.perf_counter()
    d_proc = AutoImageProcessor.from_pretrained(dp.DEPTH_CHECKPOINT)
    depth = AutoModelForDepthEstimation.from_pretrained(
        dp.DEPTH_CHECKPOINT, dtype=torch.float16).to("cuda:0")
    depth.eval()
    t_depth = time.perf_counter() - t0
    mark("+ depth fp16")

    stt = stt_proc = None
    t_stt = 0.0
    if not args.skip_stt:
        t0 = time.perf_counter()
        stt_proc = AutoProcessor.from_pretrained(cfg.STT_MODEL)
        stt = WhisperForConditionalGeneration.from_pretrained(
            cfg.STT_MODEL, dtype=torch.float16).to("cuda:0")
        stt.eval()
        t_stt = time.perf_counter() - t0
        mark("+ PhoWhisper-small fp16")

    print(f"\n  Thời gian nạp: VLM {t_vlm:.1f}s | depth {t_depth:.1f}s | "
          f"STT {t_stt:.1f}s  (cold start của server)")

    # ---- Chuẩn bị input ------------------------------------------------------
    img = Image.open(os.path.join(FRAMES_DIR, "pole_00.jpg")).convert("RGB")
    w, h = img.size
    s = cfg.FRAME_LONG_EDGE / max(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=cfg.JPEG_QUALITY)
    img = Image.open(buf).convert("RGB")

    speech = None
    if stt is not None:
        import librosa
        wavs = sorted(f for f in os.listdir(AUDIO_DIR) if f.startswith("command_"))
        if not wavs:
            print("  (không có command_*.wav — bỏ phần STT)")
            stt = None
        else:
            sr = stt_proc.feature_extractor.sampling_rate
            speech, _ = librosa.load(os.path.join(AUDIO_DIR, wavs[0]), sr=sr, mono=True)

    band_col: dict = {}

    def run_vision() -> float:
        """VLM + depth, đúng thứ server sẽ làm cho mỗi frame."""
        torch.cuda.synchronize()
        t = time.perf_counter()
        msgs = [{"role": "system", "content": [{"type": "text", "text": cfg.VLM_PROMPT}]},
                {"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": "Frame:"}]}]
        inputs = vlm_proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(vlm.device)
        with torch.inference_mode():
            vlm.generate(**inputs, max_new_tokens=cfg.VLM_MAX_NEW_TOKENS,
                         do_sample=False, temperature=None, top_p=None, top_k=None)

        d_in = d_proc(images=img, return_tensors="pt").to("cuda:0")
        d_in["pixel_values"] = d_in["pixel_values"].half()
        with torch.inference_mode():
            d_out = depth(**d_in)
        post = d_proc.post_process_depth_estimation(
            d_out, target_sizes=[(img.height, img.width)])
        dmap = np.squeeze(post[0]["predicted_depth"].float().cpu().numpy())
        if "bc" not in band_col or band_col["shape"] != dmap.shape:
            band_col["bc"] = dp.path_masks(*dmap.shape)
            band_col["shape"] = dmap.shape
        dp.free_space(dmap, *band_col["bc"])
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    def run_stt() -> float:
        torch.cuda.synchronize()
        t = time.perf_counter()
        inp = stt_proc(speech, sampling_rate=16000, return_tensors="pt")
        feats = inp.input_features.to("cuda:0", dtype=torch.float16)
        with torch.inference_mode():
            stt.generate(feats, language="vi", task="transcribe", max_new_tokens=64)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    print("\nWarmup...", flush=True)
    run_vision()
    if stt is not None:
        run_stt()

    # ---- Q2 chế độ a: chạy riêng --------------------------------------------
    print("\n=== Q2. Latency khi chạy RIÊNG (baseline) ===")
    solo_v = [run_vision() for _ in range(args.rounds)]
    print(f"  vision (VLM+depth)  median {statistics.median(solo_v):6.0f} ms | "
          f"p90 {pct(solo_v, 0.9):6.0f} ms")
    solo_s: list[float] = []
    if stt is not None:
        solo_s = [run_stt() for _ in range(args.rounds)]
        print(f"  STT                 median {statistics.median(solo_s):6.0f} ms | "
              f"p90 {pct(solo_s, 0.9):6.0f} ms")

    if stt is None:
        print("\n(bỏ qua Q3 vì không có STT)")
        return 0

    # ---- Q3 chế độ b: song song tự do ---------------------------------------
    print("\n=== Q3a. Song song TỰ DO (hai thread cùng đâm vào GPU) ===")
    par_v: list[float] = []
    par_s: list[float] = []

    def vision_loop(n, out):
        for _ in range(n):
            out.append(run_vision())

    def stt_loop(n, out):
        for _ in range(n):
            out.append(run_stt())
            time.sleep(0.05)      # push-to-talk: bursty, không liên tục

    t_wall = time.perf_counter()
    tv = threading.Thread(target=vision_loop, args=(args.rounds, par_v))
    ts = threading.Thread(target=stt_loop, args=(args.rounds, par_s))
    tv.start(); ts.start(); tv.join(); ts.join()
    wall_free = time.perf_counter() - t_wall

    mv, ms_ = statistics.median(par_v), statistics.median(par_s)
    bv, bs = statistics.median(solo_v), statistics.median(solo_s)
    print(f"  vision  median {mv:6.0f} ms  ({mv / bv:4.2f}x so với chạy riêng)")
    print(f"  STT     median {ms_:6.0f} ms  ({ms_ / bs:4.2f}x)")
    print(f"  wall    {wall_free:.1f}s")

    # ---- Q3 chế độ c: lock ưu tiên vision -----------------------------------
    # §24: "vision request -> priority high; STT request -> queue nếu GPU đang bận".
    # Cài bằng một lock GPU + cờ báo có vision đang chờ; STT nhường trước khi vào.
    print("\n=== Q3b. Có LOCK ưu tiên vision (đúng §24) ===")
    gpu_lock = threading.Lock()
    vision_waiting = threading.Event()
    lok_v: list[float] = []
    lok_s: list[float] = []
    stt_waits: list[float] = []

    def vision_loop_locked(n, out):
        for _ in range(n):
            vision_waiting.set()
            with gpu_lock:
                vision_waiting.clear()
                out.append(run_vision())

    def stt_loop_locked(n, out):
        for _ in range(n):
            t_q = time.perf_counter()
            while vision_waiting.is_set():   # nhường vision vào trước
                time.sleep(0.005)
            with gpu_lock:
                stt_waits.append((time.perf_counter() - t_q) * 1000)
                out.append(run_stt())
            time.sleep(0.05)

    t_wall = time.perf_counter()
    tv = threading.Thread(target=vision_loop_locked, args=(args.rounds, lok_v))
    ts = threading.Thread(target=stt_loop_locked, args=(args.rounds, lok_s))
    tv.start(); ts.start(); tv.join(); ts.join()
    wall_lock = time.perf_counter() - t_wall

    lv, ls = statistics.median(lok_v), statistics.median(lok_s)
    print(f"  vision  median {lv:6.0f} ms  ({lv / bv:4.2f}x so với chạy riêng)")
    print(f"  STT     median {ls:6.0f} ms  ({ls / bs:4.2f}x)  "
          f"+ chờ hàng đợi median {statistics.median(stt_waits):.0f} ms")
    print(f"  wall    {wall_lock:.1f}s")

    # ---- Kết luận ------------------------------------------------------------
    print("\n=== Kết luận ===")
    print(f"  {'':<22}{'vision':>10}{'STT':>10}{'vision vs riêng':>18}")
    print(f"  {'chạy riêng':<22}{bv:9.0f}{bs:9.0f}{'1.00x':>18}")
    print(f"  {'song song tự do':<22}{mv:9.0f}{ms_:9.0f}{mv / bv:17.2f}x")
    print(f"  {'lock ưu tiên vision':<22}{lv:9.0f}{ls:9.0f}{lv / bv:17.2f}x")

    peak = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\n  VRAM peak cả ba model + inference: {peak:.2f} / {total_vram:.1f} GiB "
          f"(còn {total_vram - peak:.2f} GiB)")
    if peak < total_vram * 0.85:
        print("  => Q1: VỪA. §22 lo OOM là đúng về nguyên tắc nhưng với bộ model "
              "này thì còn dư.")
    else:
        print("  => Q1: SÁT TRẦN. Phải bỏ bớt model khỏi GPU.")

    slow_free = (mv / bv - 1) * 100
    slow_lock = (lv / bv - 1) * 100
    if slow_free < 15:
        print(f"  => Q2/Q3: song song tự do chỉ làm vision chậm {slow_free:+.0f}%. "
              "KHÔNG cần lock —\n     viết server đơn giản, để CUDA tự xen kẽ.")
    elif slow_lock < slow_free - 10:
        print(f"  => Q2/Q3: tự do làm vision chậm {slow_free:+.0f}%, có lock còn "
              f"{slow_lock:+.0f}%.\n     => CẦN lock ưu tiên vision đúng như §24.")
    else:
        print(f"  => Q2/Q3: tự do {slow_free:+.0f}%, lock {slow_lock:+.0f}% — lock "
              "không cứu được nhiều.\n     Tranh chấp nằm ở compute chứ không ở "
              "thứ tự. Cân nhắc đẩy STT về CPU (đo được 2710 ms)\n     hoặc chấp "
              "nhận vision chậm trong lúc user bấm nói.")

    budget = bv + 0  # vision solo
    print(f"\n  §12 ngân sách 1000 ms: vision riêng {budget:.0f} ms, "
          f"lúc STT chen vào {mv:.0f} ms.")
    if mv > 1000:
        print("  ⚠️ Lúc tranh chấp thì VƯỢT ngân sách §12. Cảnh báo vật cản sẽ trễ "
              "đúng lúc\n     người dùng đang bấm nói — phải xử lý ở tầng thiết kế, "
              "không phải tầng code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
