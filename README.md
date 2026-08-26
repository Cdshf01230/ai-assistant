# MVP Trợ lý Thị giác AI — demo trong nhà cho người khiếm thị

Backend FastAPI + fusion VLM×depth + web frontend, chạy trên 1× Tesla T4.
Spec gốc: `~/mvp_ai_assistant_blind.md`. Trạng thái: **demo tương tác trong nhà
(cảnh báo vật cản + thuyết minh môi trường) đã chạy end-to-end qua HTTPS/WSS.**

## Phần cứng và hai ràng buộc của T4

```
GPU: 1 × Tesla T4 15 GiB (sm_75) | CPU: 4 vCPU | RAM: 15 GiB | Python 3.14
Driver 595.91.07 (CUDA 13.2)
```

1. **fp16 là bắt buộc**: T4 không có bf16 native. Đo matmul 4096³: fp16 29.2
   TFLOP/s vs bf16 2.6 (chậm 11.3×). Lưu ý `torch.cuda.is_bf16_supported()`
   trả True vì tính cả emulation — phải dùng `including_emulation=False`.
   Hard-coded trong `scripts/mvp_config.py`.
2. **Không có FlashAttention-2** (cần sm_80+) → `attn_implementation="sdpa"`.

## Kiến trúc đang chạy

```text
Điện thoại (web/index.html)
├─ camera → JPEG 640/q80 → WSS /v1/vision (~1 FPS, §13/§14)
│     └─ api_main.py: Qwen3-VL-4B (fp16, GPU) + Depth-Anything-V2 Indoor-Large (GPU)
│        └─ scripts/fusion.py — Decision Engine đã đo (§16):
│           • VLM quyết CÓ GÌ/Ở ĐÂU; depth quyết GẦN/XA (VLM mù khoảng cách)
│           • quy tắc lai: free_min < 7.0 (hiệu chuẩn) OR near_rank ≥ 0.5
│             OR cửa thoát step/hole/door (lớp depth mù cấu trúc)
│           • EMERGENCY: free_min < 2.0 → STOP bất chấp VLM (VLM từng trả
│             CLEAR khi vật 0.3–1.7 m ngay trước mặt — 7/45 frame lỗi COCO)
│           • cross-check hướng, trễ confirm=2/clear=3, dedup §14,
│             narrate gate ≥10s
│        ├─ alert   → message_code → WAV cache phát ngay (4 ms)
│        └─ narrate → POST /v1/describe (Gemini thinking-LOW, ≤2 câu) → TTS
└─ mic giữ-để-nói → WAV 16k encode trên client → POST /v1/stt
      └─ confidence < 0.60 → REPEAT_PLEASE.wav, không đoán (§5)
```

Chạy demo: `./scripts/run_demo_server.sh` → mở `https://<IP>:8443/` trên điện
thoại (cert tự ký). Chi tiết mạng/deploy: `deploy/README.md`.

## Model stack

| Vai trò | Model | Size | License | Chạy ở đâu |
|---|---|---:|---|---|
| VLM | `Qwen/Qwen3-VL-4B-Instruct` | 8.5 GB | Apache-2.0 | GPU fp16 |
| Depth | `Depth-Anything-V2-Metric-Indoor-Large-hf` | ~1.4 GB | Apache-2.0 | GPU fp16 |
| STT | `vinai/PhoWhisper-medium` | ~1 GB | BSD-3 | GPU fp16 |
| TTS | `VieNeu-TTS-v3-Turbo` (int8 ONNX) | ~250 MB | Apache-2.0 | CPU |
| Thuyết minh | `gemini-3.6-flash` (API) | — | — | ngoài |

VRAM tổng khi chạy đủ: ~11.3 GiB / 15 GiB.

## Số đo chính (tất cả đo trên máy này)

### Vì sao cần depth — VLM mù khoảng cách hoàn toàn
Cùng một cây cột ở chân trời và ngay trước mặt trả y một đáp án; nhãn `far`
không bao giờ được dùng (27 frame, cả 2B lẫn 4B — không phụ thuộc tham số).
=> Khoảng cách phải đến từ hình học, không thể prompt ra.

### Fusion trên bộ frame nhãn tay (27 frame, ground_truth.json)

| chiến lược | recall | specif | âm khó | balanced |
|---|---:|---:|---:|---:|
| A. chỉ VLM (near+mid) — baseline cũ | 100% | 58% | 0% | 79% |
| **E. fusion Indoor-Large @7m (đang dùng)** | 100% | 75% | 25% | 88% |
| E với checkpoint outdoor @19m (đối chứng) | 100% | 92% | 75% | 96% |

Quy tắc chọn ngưỡng: giữ recall 100%, lấy specificity cao nhất — KHÔNG tối ưu
balanced (từng đánh mất 2 frame vật cản thật vì tối ưu balanced).

Kiểm chứng logic fusion tái hiện đúng các số trên mà không cần GPU:
`benchmarks/replay_fusion_decisions.py` (PASS, gồm 7 unit test hành vi).

### Đánh giá trên dữ liệu công khai (thay frame thật không thu được)

- **DIODE indoor GT depth, 40 ảnh** (`evaluate_geometry_public.py`): median lỗi
  rank hình học **3.3%**, near-flag agreement 87.5%, agreement cột 82.5%
  → nửa hình học của fusion đúng so với depth GT thật.
- **COCO val2017, 120 ảnh nhãn từ GT bbox** (`evaluate_vlm_public.py`):
  fusion đạt balanced ~62% theo nhãn thô (recall 73%, spec 52%). **Đã thẩm định
  tay cả 45 frame lỗi** (`logs/coco_adjudication.md`): 24/29 "báo thừa" thực ra
  là cảnh báo ĐÚNG (chó/cột/voi/tàu... chắn 0.6–6.7 m mà nhãn bbox gọi là
  thoáng) → hiệu chỉnh lại fusion đạt **balanced ~80%, specificity ~85–88%**.
  Lỗi thật còn lại tập trung ở 1 lớp: **VLM trả CLEAR khi vật chắn <3 m trong
  nhà** (7 case) → **ĐÃ SỬA bằng depth emergency override** (free_min < 2.0 →
  STOP bất chấp VLM): recall hiệu chỉnh 73→**91%**, balanced 80→**86%**, bộ 27
  frame không đổi (100%/75%). Lưu ý minh bạch: người thẩm định trùng người
  thiết kế hệ thống — số hiệu chỉnh có xung đột lợi ích, chỉ dùng nội bộ.

### Latency end-to-end qua HTTPS/WSS thật (`measure_latency_e2e.py`, logs/latency_e2e.json)

| chặng | median | ngân sách §12 |
|---|---:|---|
| Vision RTT (send→decision, bỏ warm-up 4.7 s) | **953 ms** | "demo được" (<1 s) |
| — phần server (VLM+depth) | 929 ms | decode ~33–40 ms/token là bottleneck |
| — upload+TLS+JSON+queue | 31 ms | |
| STT push-to-talk round-trip | 485 ms | |
| TTS cached WAV (kênh cảnh báo) | **4 ms** | ← cache là tất cả (§9) |
| TTS động câu dài (CPU/ONNX) | 1515 ms | |
| Describe (Gemini LOW + TTS) | ~11 s | kênh thuyết minh, non-critical |

Chưa gồm T_camera và thời gian phát audio trên thiết bị thật. Describe từng
19 s với prompt dài mặc định — đã giảm bằng thinking_level='LOW' + giới hạn
2 câu; muốn nhanh nữa thì stream TTS hoặc rút ngắn output.

### Prompt VLM — hai bài học đắt giá
1. **Prompt lệch thì model lệch theo**: "Report ONLY the most urgent hazard"
   → báo động 25/27 frame kể cả ảnh trống; "Most frames need no warning" → bỏ
   sót 11/15. Prompt production phải TRUNG TÍNH. Đổi prompt = BẮT BUỘC chạy lại eval.
2. Output pipe-compact `OBJ|WHERE|DIST|ACTION` nhanh hơn JSON 2.4× (468 vs
   1103 ms) vì decode là bottleneck, không phải resolution (640 giữ nguyên).

### STT / TTS
- PhoWhisper-medium: GPU 165 ms vs CPU 2710 ms (chênh 16×) → STT lên GPU,
  push-to-talk không tranh GPU với vision. WER 21% trên audio TTS-synth sạch;
  lỗi thật: tên riêng ("vincom"→"viện cồm") → cần prompt bias/fuzzy-match POI.
  Audio synth sạch hơn thực tế nhiều — WER đường phố sẽ cao hơn.
- VieNeu CPU int8: RTF 0.59–0.61. 682 ms cho "Dừng lại." là quá chậm lúc sắp
  va chạm → 16 câu critical sinh sẵn WAV (`assets/audio/`), lúc chạy là 4 ms.

## Cấu trúc repo

```
ai-assistant/
├── api_main.py              # FastAPI: /health /v1/alerts /v1/stt /v1/tts
│                            #          /v1/describe  WS /v1/vision  /audio/*
├── web/index.html           # frontend mobile 1 file (camera/mic/audio)
├── Dockerfile               # code vào image; weights/.env/certs/logs mount ngoài
├── docker-compose.yml       # GPU (nvidia runtime), port 8443
├── scripts/
│   ├── mvp_config.py        # model id, prompt production, ALERT_PHRASES, ngưỡng
│   ├── fusion.py            # Evidence + decide_frame + update_session (§14/§16)
│   ├── depth_path.py        # hình thang lối đi + free-space scan (nguồn duy nhất)
│   ├── vlm_prompts.py       # biến thể prompt cho A/B (08/09)
│   ├── run_demo_server.sh   # TLS 0.0.0.0:8443 (đã bỏ proxy — xem deploy/README.md)
│   ├── 01_download_models.py # tải ĐỦ 6 model production về models/hf (~12 GB)
│   └── 00..13_*.py          # pipeline chuẩn bị model + benchmark từng thành phần
├── benchmarks/
│   ├── replay_fusion_decisions.py     # hồi quy fusion trên dump (không cần GPU)
│   ├── dump_depth_features.py         # free-space dump cho 1 checkpoint
│   ├── calibrate_depth_threshold.py   # hiệu chuẩn ngưỡng tuyệt đối (quét ngưỡng)
│   ├── evaluate_geometry_public.py    # DIODE GT: kiểm chứng nửa hình học
│   ├── evaluate_vlm_public.py         # COCO: ngữ nghĩa + fusion trên phân phối rộng
│   ├── coco_select_download.py        # chọn mẫu cân bằng + tải val2017 subset
│   └── measure_latency_e2e.py         # latency qua HTTPS/WSS thật
├── tests/golden/               # TEST CASE đóng gói: ảnh+nhãn+input đã đo+kỳ vọng
│   │                           # (27 frame tay + 120 COCO hiệu chỉnh, SHA-256 manifest)
│   ├── build_golden_cases.py   # tái tạo golden khi đổi NGUỒN (có chủ đích)
│   └── run_golden_tests.py     # hồi quy fusion không cần GPU — PASS trước khi deploy
├── models/hf/               # HF_HOME — toàn bộ weight trong project (~19 GB)
├── assets/{frames,audio,tts_test}/
├── logs/                    # mọi kết quả eval thô (JSON/MD)
├── certs/                   # cert tự ký (key.pem chmod 600)
└── deploy/                  # systemd unit + hướng dẫn truy cập từ điện thoại
```

## Chạy từ đầu

```bash
cd /home/ubuntu/ai-assistant && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                        # điền GEMINI_API_KEY (kênh thuyết minh)
python scripts/01_download_models.py        # đủ 6 model production (~12 GB)
python scripts/05_build_alert_cache.py      # 16 WAV cảnh báo

./scripts/run_demo_server.sh                # demo TLS :8443
TLS=0 ./scripts/run_demo_server.sh          # dev HTTP :8000

# hoặc Docker (weights mount ngoài image): docker compose up -d --build

python tests/run_golden_tests.py            # hồi quy fusion — phải PASS trước deploy
```

Sau khi ĐỔI CHECKPOINT DEPTH hoặc CAMERA/ĐỘ CAO/GÓC CHỤC: bắt buộc hiệu chuẩn
lại ngưỡng — `dump_depth_features.py` rồi `calibrate_depth_threshold.py`.

## Việc còn lại

- [ ] Đo trên điện thoại thật (T_camera + audio playback), test người dùng
- [x] Docker image hoá toàn stack (weights mount ngoài image) — `docker compose up -d --build`, xem `deploy/README.md`
- [ ] Google Maps/navigation (§15) — ngoài phạm vi demo v1
- [ ] Mở rộng eval COCO sang subset walking-POV lọc kỹ hơn để bớt nhiễu nhãn
