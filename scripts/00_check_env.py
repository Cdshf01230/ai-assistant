#!/usr/bin/env python3
"""Kiểm tra môi trường trước khi load model.

Điểm quan trọng nhất: T4 là sm_75 -> KHÔNG hỗ trợ bfloat16 native và
KHÔNG hỗ trợ FlashAttention-2. Toàn bộ pipeline GPU phải dùng float16 + sdpa.
"""

import shutil
import sys

OK, WARN, BAD = "[OK]  ", "[WARN]", "[FAIL]"


def main() -> int:
    fails = 0
    print(f"Python: {sys.version.split()[0]}")

    try:
        import torch
    except ImportError:
        print(f"{BAD} torch chưa được cài")
        return 1

    print(f"torch: {torch.__version__} (CUDA build: {torch.version.cuda})")

    if not torch.cuda.is_available():
        print(f"{BAD} CUDA không khả dụng — VLM sẽ phải chạy CPU (quá chậm cho MVP)")
        fails += 1
    else:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"{OK} GPU: {name} | sm_{cap[0]}{cap[1]} | {vram:.1f} GiB VRAM")

        if cap < (7, 5):
            print(f"{BAD} sm_{cap[0]}{cap[1]} quá cũ, CUDA 13 đã bỏ hỗ trợ")
            fails += 1

        # CẢNH BÁO: torch.cuda.is_bf16_supported() mặc định including_emulation=True
        # nên trả True cả trên T4 — nhưng bf16 lúc đó bị emulate và chậm ~11x.
        try:
            bf16_native = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:  # torch cũ chưa có kwarg này
            bf16_native = cap >= (8, 0)

        if bf16_native:
            print(f"{OK} bfloat16 native — có thể dùng bf16")
        else:
            print(f"{WARN} KHÔNG có bfloat16 native (đúng như dự kiến với T4 / sm_75).")
            print("       is_bf16_supported() trả True là do emulation — ĐỪNG tin con số đó.")
            print("       => Bắt buộc load model bằng dtype=torch.float16")

        # sm_75 không có FA2. sdpa (mem-efficient kernel) là lựa chọn đúng.
        print(f"{OK} attn_implementation nên dùng: 'sdpa' (FlashAttention-2 cần sm_80+)")

        # Đo thật fp16 vs bf16 để lời khuyên dtype có số liệu hậu thuẫn
        import time

        def tflops(dtype) -> float:
            n = 4096
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            for _ in range(3):
                torch.matmul(a, a)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                torch.matmul(a, a)
            torch.cuda.synchronize()
            return 2 * n**3 / ((time.perf_counter() - t) / 10) / 1e12

        try:
            f16, bf16 = tflops(torch.float16), tflops(torch.bfloat16)
            print(f"{OK} matmul 4096³: fp16 {f16:.1f} TFLOP/s | bf16 {bf16:.1f} TFLOP/s "
                  f"(fp16 nhanh hơn {f16 / bf16:.1f}x)")
        except Exception as exc:  # pragma: no cover - phụ thuộc driver
            print(f"{BAD} matmul benchmark lỗi: {exc}")
            fails += 1

    try:
        import transformers

        print(f"{OK} transformers: {transformers.__version__}")
    except ImportError:
        print(f"{BAD} transformers chưa cài")
        fails += 1

    for mod, label in [
        ("onnxruntime", "onnxruntime (TTS CPU path)"),
        ("librosa", "librosa (STT audio load)"),
        ("soundfile", "soundfile"),
    ]:
        try:
            __import__(mod)
            print(f"{OK} {label}")
        except ImportError:
            print(f"{WARN} {label} chưa cài")

    free_gb = shutil.disk_usage("/home/ubuntu").free / 1024**3
    tag = OK if free_gb > 30 else WARN
    print(f"{tag} Disk free: {free_gb:.0f} GiB (cần ~15 GiB cho model + cache)")

    print("\n" + ("Môi trường OK." if not fails else f"{fails} lỗi cần xử lý."))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
