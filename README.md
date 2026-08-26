# Trợ lý Thị giác AI cho người khiếm thị

> Demo trong nhà: camera điện thoại → **cảnh báo vật cản bằng giọng nói** (bộ nhớ
> đệm WAV, 4 ms) + **thuyết minh môi trường** (Gemini + TTS). Chạy hoàn toàn trên
> 1× Tesla T4. Đã chạy end-to-end qua HTTPS/WSS.

| | |
|---|---|
| Backend | FastAPI (`api_main.py`) |
| Frontend | 1 file HTML mobile (`web/index.html`) |
| AI | Fusion **VLM × depth** — xem phần [Cách hệ thống quyết định](#cách-hệ-thống-quyết-định) |
| Deploy | Docker Compose (GPU) hoặc chạy trực tiếp |

---

## Chạy nhanh

### Cách 1 — Docker (khuyên dùng)

Yêu cầu: máy có GPU NVIDIA ≥ 15 GiB VRAM + driver + nvidia-container-toolkit.

```bash
git clone https://github.com/Cdshf01230/ai-assistant.git && cd ai-assistant
cp .env.example .env                 # điền GEMINI_API_KEY (chỉ kênh thuyết minh cần)
docker compose up -d --build         # ~10 phút build lần đầu
docker compose logs -f               # đợi tới "status":"ok" (~2 phút load model)
```

Mở trên điện thoại: `https://<IP-máy-chủ>:8443/` — cert tự ký, chọn *Advanced →
Proceed*. Camera/mic chỉ hoạt động trên HTTPS. Chi tiết mạng/tunnel:
[`deploy/README.md`](deploy/README.md).

### Cách 2 — chạy trực tiếp

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                        # điền GEMINI_API_KEY
python scripts/01_download_models.py        # tải đủ 6 model production (~12 GB)
python scripts/05_build_alert_cache.py      # sinh 16 WAV cảnh báo

./scripts/run_demo_server.sh                # HTTPS :8443
TLS=0 ./scripts/run_demo_server.sh          # dev HTTP :8000
```

### Kiểm tra trước khi deploy

```bash
python tests/run_golden_tests.py     # hồi quy fusion trên golden cases — phải PASS
```

Không cần GPU cho bước này: test case đóng gói sẵn đầu vào VLM+depth đã đo
(27 frame nhãn tay + 120 ảnh COCO thẩm định). Xem [`tests/README.md`](tests/README.md).

---

## Cách hệ thống quyết định

```text
Điện thoại (web/index.html)
├─ camera → JPEG 640/q80 → WSS /v1/vision (~1 FPS)
│     └─ api_main.py: Qwen3-VL-4B (fp16) + Depth-Anything-V2 Indoor-Large
│        └─ scripts/fusion.py — Decision Engine:
│           • VLM quyết CÓ GÌ / Ở ĐÂU      (gọi đúng vật 85%, đúng hướng ~90%)
│           • Depth quyết GẦN / XA          (VLM MÙ khoảng cách hoàn toàn)
│           • free_min < 7.0 hoặc near_rank ≥ 0.5 → chắn lối đi
│           • EMERGENCY: free_min < 2.0 → STOP bất chấp VLM nói CLEAR
│           • trễ confirm=2/clear=3 frame, dedup cảnh báo trùng, gate thuyết minh 10 s
│        ├─ alert   → message_code → WAV có sẵn phát ngay (4 ms)
│        └─ narrate → Gemini (≤2 câu) → TTS
└─ mic giữ-để-nói → WAV 16k → POST /v1/stt
      └─ độ tin cậy < 0.60 → "Xin nói lại", không đoán
```

**Vì sao cần 2 model?** Đã đo trên 27 frame: VLM trả *y một đáp án* cho cây cột
ở chân trời và cây cột ngay trước mặt — khoảng cách không thể prompt ra, phải
đến từ hình học (depth monocular). Ngược lại depth không biết vật là gì và mù
cấu trúc (bậc thang/hố/cửa kính không chắn tia nhìn) — nên lớp này tin VLM.

## Model stack

| Vai trò | Model | Size | License | Thiết bị |
|---|---|---:|---|---|
| VLM | `Qwen/Qwen3-VL-4B-Instruct` | 8.5 GB | Apache-2.0 | GPU fp16 |
| Depth | `Depth-Anything-V2-Metric-Indoor-Large-hf` | 1.4 GB | Apache-2.0 | GPU fp16 |
| STT | `vinai/PhoWhisper-medium` | 1 GB | BSD-3 | GPU fp16 |
| TTS | `VieNeu-TTS-v3-Turbo` int8 ONNX | 250 MB | Apache-2.0 | CPU |
| Thuyết minh | `gemini-3.6-flash` (API) | — | — | ngoài |

VRAM tổng khi chạy đủ: **~11.3 / 15 GiB**.

## Số đo chính

### Fusion so với baseline (27 frame nhãn tay)

| Chiến lược | Recall | Specificity | Balanced |
|---|---:|---:|---:|
| A. chỉ VLM (near+mid) — baseline cũ | 100% | 58% | 79% |
| **E. fusion Indoor-Large @7 m — đang dùng** | **100%** | **75%** | **88%** |
| E với checkpoint outdoor @19 m (đối chứng) | 100% | 92% | 96% |

Nguyên tắc chọn ngưỡng: **giữ recall 100%, lấy specificity cao nhất** — không tối
ưu balanced (đã từng đánh mất 2 vật cản thật vì tối ưu theo balanced).

### Trên dữ liệu công khai

- **DIODE GT depth, 40 ảnh**: median lỗi rank hình học 3.3%, near-flag agreement
  87.5% → nửa hình học của fusion đúng so với ground truth thật.
- **COCO val2017, 120 ảnh**: nhãn bbox thô cho balanced 62%; sau khi thẩm định tay
  cả 45 frame lỗi ([logs/coco_adjudication.md](logs/coco_adjudication.md)) — 24/29
  "báo thừa" thực ra là cảnh báo ĐÚNG — và thêm **depth emergency override**
  (free_min < 2.0 → STOP bất chấp VLM): **recall hiệu chỉnh 91%, balanced 86%**.
  ⚠️ Người thẩm định trùng người thiết kế hệ thống — số hiệu chỉnh chỉ dùng nội bộ.

### Latency end-to-end (HTTPS/WSS thật)

| Chặng | Median | Ghi chú |
|---|---:|---|
| Vision RTT (send → quyết định) | **953 ms** | decode ~33–40 ms/token là bottleneck |
| STT push-to-talk round-trip | 485 ms | |
| TTS câu cảnh báo (WAV cache) | **4 ms** | cache là tất cả |
| TTS câu dài động (CPU ONNX) | 1515 ms | chỉ dùng cho kênh không khẩn cấp |
| Describe (Gemini + TTS) | ~11 s | kênh thuyết minh, non-critical |

### Bài học đắt giá (để khỏi trả giá lần nữa)

1. **Prompt lệch thì model lệch theo**: "Report ONLY the most urgent hazard" → báo
   động cả ảnh trống (25/27); "Most frames need no warning" → bỏ sót 11/15. Prompt
   phải trung tính, và **đổi prompt = bắt buộc chạy lại eval**.
2. **Số token sinh ra mới là bottleneck**, không phải resolution: pipe-compact
   `OBJ|WHERE|DIST|ACTION` nhanh hơn JSON 2.4× (468 vs 1103 ms).
3. STT lên GPU (165 ms vs 2710 ms CPU — chênh 16×); WER 21% trên audio synth sạch,
   lỗi chính là tên riêng → cần fuzzy-match POI về sau.

---

## Cấu trúc repo

```
ai-assistant/
├── api_main.py              # FastAPI: /health /v1/alerts /v1/stt /v1/tts
│                            #          /v1/describe  WS /v1/vision  /audio/*
├── web/index.html           # frontend mobile 1 file (camera/mic/audio)
├── Dockerfile               # code vào image; weights/.env/certs/logs mount ngoài
├── docker-compose.yml       # GPU (nvidia runtime), port 8443
├── scripts/
│   ├── mvp_config.py        # model id, prompt production, ngưỡng, ALERT_PHRASES
│   ├── fusion.py            # Evidence + decide_frame + update_session
│   ├── depth_path.py        # hình thang lối đi + free-space scan
│   ├── run_demo_server.sh   # TLS 0.0.0.0:8443
│   ├── 01_download_models.py# tải đủ 6 model production về models/hf
│   └── 00..13_*.py          # pipeline chuẩn bị model + benchmark từng thành phần
├── benchmarks/              # replay_fusion_decisions, DIODE/COCO eval,
│                            # calibrate_depth_threshold, measure_latency_e2e...
├── tests/
│   ├── golden/              # test case đóng gói: ảnh+nhãn+input đã đo+kỳ vọng
│   └── run_golden_tests.py  # hồi quy fusion KHÔNG cần GPU — PASS trước mọi deploy
├── assets/{frames,audio}/   # 32 frame nhãn tay + 16 WAV cảnh báo
├── logs/                    # kết quả eval thô (JSON/MD) — golden builder cần
└── deploy/                  # systemd unit + hướng dẫn truy cập từ điện thoại
```

Không nằm trong git: `models/` (weights ~19 GB, tải bằng script), `.env`, `certs/`,
dataset lớn (`benchmarks/data/` — chỉ 120 ảnh COCO có nhãn được track).

## Yêu cầu phần cứng & ràng buộc T4

```
GPU: 1 × Tesla T4 15 GiB (sm_75) | CPU: 4 vCPU | RAM: 15 GiB | Python 3.14
```

1. **fp16 là bắt buộc** — T4 không có bf16 native (matmul 4096³: fp16 29.2 vs
   bf16 2.6 TFLOP/s, chậm 11.3×). Lưu ý `torch.cuda.is_bf16_supported()` trả True
   vì tính cả emulation — phải dùng `including_emulation=False`.
   Hard-code trong `scripts/mvp_config.py`.
2. **Không có FlashAttention-2** (cần sm_80+) → `attn_implementation="sdpa"`.

⚠️ **Khi nào phải hiệu chuẩn lại ngưỡng depth:** sau khi ĐỔI checkpoint depth,
camera, độ cao hoặc góc chúc — thang mét của checkpoint sai 4–5 lần so với thực tế
nên ngưỡng gắn chặt với đúng bộ (checkpoint × camera) hiện tại. Quy trình:
`benchmarks/dump_depth_features.py` → `calibrate_depth_threshold.py`.

## Việc còn lại

- [ ] Đo trên điện thoại thật (T_camera + audio playback), test người dùng
- [x] Docker image hoá toàn stack (weights mount ngoài image)
- [x] Golden test cases hồi quy fusion không cần GPU
- [ ] Google Maps/navigation — ngoài phạm vi demo v1
- [ ] Mở rộng eval COCO sang subset walking-POV lọc kỹ hơn để bớt nhiễu nhãn
