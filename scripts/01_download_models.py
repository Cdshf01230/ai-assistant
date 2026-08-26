#!/usr/bin/env python3
"""Tải toàn bộ model của MVP về local cache.

Baseline = ĐỦ để server demo khởi động (khớp mvp_config.py / api_main.py):
    VLM   Qwen/Qwen3-VL-4B-Instruct                       (~8.5 GB, GPU fp16)
    DEPTH depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf  (~1.4 GB)
    STT   vinai/PhoWhisper-medium                         (~1 GB, GPU)
    TTS   pnnbao-ump/VieNeu-TTS-v3-Turbo (onnx int8)      (~250 MB, CPU)
          + OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano(-ONNX) — audio codec bắt buộc
            của VieNeu v3 (vieneu/_v3turbo.py), thiếu là TTS lỗi lúc load.
    Describe (gemini) là API ngoài — không cần tải, chỉ cần GEMINI_API_KEY trong .env.

Backup/cũ (chỉ tải khi truyền --with-backup):
    STT   vinai/PhoWhisper-small
    VLM   Qwen/Qwen3-VL-2B-Instruct, HuggingFaceTB/SmolVLM2-2.2B-Instruct
    DEPTH Metric-Outdoor-Small / Indoor-Small / relative Small

Dùng:
    python scripts/01_download_models.py
    python scripts/01_download_models.py --with-backup
    python scripts/01_download_models.py --only vlm
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# huggingface_hub 1.x dùng hf-xet để tăng tốc download (không cần hf_transfer nữa)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
# Giữ toàn bộ weight trong project để dễ snapshot/EBS backup
os.environ.setdefault("HF_HOME", "/home/ubuntu/ai-assistant/models/hf")

BASELINE = {
    "vlm": ("Qwen/Qwen3-VL-4B-Instruct", None),
    # Checkpoint đang dùng trong api_main.py (đã hiệu chuẩn ngưỡng 7.0 trên nó).
    "depth": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
              ["*.json", "*.safetensors", "*.txt"]),
    "stt": ("vinai/PhoWhisper-medium", ["*.json", "*.txt", "pytorch_model.bin"]),
    # ONNX int8 là path CPU torch-free; kèm speaker_encoder + denoiser cho voice cloning.
    "tts": (
        "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        ["*.json", "*.txt", "onnx_int8/*", "speaker_encoder.onnx", "denoiser.onnx", "tokenizer*"],
    ),
    # Audio codec của VieNeu v3 Turbo: bản ONNX cho đường CPU, bản gốc cho path torch.
    "tts_codec": ("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX", None),
    "tts_codec_torch": ("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano", None),
}

BACKUP = {
    "stt_backup": ("vinai/PhoWhisper-small", ["*.json", "*.txt", "pytorch_model.bin"]),
    "vlm_backup": ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", None),
    "vlm_2b": ("Qwen/Qwen3-VL-2B-Instruct", None),
    "depth_outdoor_small": ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
                            ["*.json", "*.safetensors", "*.txt"]),
    "depth_indoor": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                     ["*.json", "*.safetensors", "*.txt"]),
    "depth_rel": ("depth-anything/Depth-Anything-V2-Small-hf",
                  ["*.json", "*.safetensors", "*.txt"]),
}


def fetch(key: str, repo: str, allow: list[str] | None) -> bool:
    from huggingface_hub import snapshot_download

    print(f"\n=== {key}: {repo} ===", flush=True)
    t0 = time.time()
    try:
        path = snapshot_download(
            repo_id=repo,
            allow_patterns=allow,
            max_workers=4,
        )
    except Exception as exc:
        print(f"  LỖI: {type(exc).__name__}: {exc}")
        return False
    print(f"  -> {path}  ({time.time() - t0:.0f}s)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-backup", action="store_true", help="tải cả model phương án B")
    ap.add_argument("--only", choices=sorted(BASELINE) + sorted(BACKUP), help="chỉ tải 1 model")
    args = ap.parse_args()

    targets = dict(BASELINE)
    if args.with_backup:
        targets.update(BACKUP)
    if args.only:
        merged = {**BASELINE, **BACKUP}
        targets = {args.only: merged[args.only]}

    failed = [k for k, (repo, allow) in targets.items() if not fetch(k, repo, allow)]

    print(f"\nHF_HOME = {os.environ['HF_HOME']}")
    if failed:
        print(f"Thất bại: {', '.join(failed)}")
        return 1
    print(f"Đã tải xong {len(targets)} model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
