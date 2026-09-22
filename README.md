# Trợ lý Thị giác AI cho người khiếm thị

> Demo trong nhà: camera điện thoại → **cảnh báo vật cản bằng giọng nói** (bộ nhớ
> đệm WAV, 4 ms) + **thuyết minh môi trường** (Gemini + TTS). Chạy hoàn toàn trên
> 1× Tesla T4. Đã chạy end-to-end qua HTTPS/WSS.

| | |
|---|---|
| Backend | FastAPI (`api_main.py`) |
| Frontend | React mobile (`frontend/`), build sẵn trong `web/dist/` |
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
Điện thoại (web/dist)
├─ camera → JPEG 640/q80 → WSS /v1/vision (~1 FPS)
│     └─ api_main.py: Qwen3-VL-4B (fp16) + Depth-Anything-V2 Indoor-Large
│        └─ scripts/fusion.py — Decision Engine:
│           • VLM quyết CÓ GÌ / Ở ĐÂU      (gọi đúng vật 85%, đúng hướng ~90%)
│           • Depth quyết GẦN / XA          (VLM MÙ khoảng cách hoàn toàn)
│           • free_min < 7.0 hoặc near_rank ≥ 0.5 → chắn lối đi
│           • EMERGENCY: free_min < 2.0 → STOP bất chấp VLM nói CLEAR
│           • vật thường cần 2 kết quả Qwen mới; PERSON cần 3; clear=3 frame
│           • cache Qwen không được đếm lặp; hướng rẽ phải khớp cả VLM và depth
│        ├─ alert   → message_code → WAV có sẵn phát ngay (4 ms)
│        └─ narrate → Gemini (≤2 câu) → TTS
└─ micro/VAD → WAV 16k → POST /v1/stt
      ├─ độ tin cậy < 0.60 → "Xin nói lại", không đoán
      └─ intent theo luật → guide | narration | paused
            └─ câu hỏi ở narration → /v1/describe (Gemini + TTS)
```

**Vì sao cần 2 model?** Đã đo trên 27 frame: VLM trả *y một đáp án* cho cây cột
ở chân trời và cây cột ngay trước mặt — khoảng cách không thể prompt ra, phải
đến từ hình học (depth monocular). Ngược lại depth không biết vật là gì và mù
cấu trúc (bậc thang/hố/cửa kính không chắn tia nhìn) — nên lớp này tin VLM.

### Contract voice-first cho frontend

Bản React hoàn thiện từ Figma Make nằm trong `frontend/`, build sẵn tại
`web/dist/`. FastAPI và notebook Kaggle ưu tiên phục vụ bản build này.
Upload lại `ai-assistant-kaggle.zip` cùng notebook mới để sử dụng. Không cần
cài Node.js trên Kaggle. Hướng dẫn build, cấu trúc và giới hạn trình duyệt:
[frontend/README.md](frontend/README.md).

Backend giữ router intent thuần luật trong `scripts/mvp_config.py`, không gọi thêm
LLM. `GET /v1/bootstrap` trả câu giới thiệu, các câu hệ thống, thứ tự ưu tiên
audio và tham số VAD gợi ý. `POST /v1/intent` route transcript có sẵn để frontend
test độc lập.

`POST /v1/stt` vẫn nhận field `audio` như cũ và nhận thêm hai form field tùy chọn
`current_mode`, `resume_mode`, `turn_id`. Khi STT đủ tin cậy, response có thêm:

```json
{
  "text": "phía trước có gì",
  "confidence": 0.91,
  "repeat": false,
  "intent": "ask_description",
  "mode": "narration",
  "reply_code": "HEARD_THINKING",
  "reply_text": "Mình đã nghe rõ. Đang kiểm tra cảnh trước mặt.",
  "audio_kind": "filler",
  "audio_priority": 20,
  "should_describe": true,
  "question": "phía trước có gì"
}
```

Frontend phát `reply_text` đồng thời gọi `/v1/describe`; không chờ câu đệm phát
xong mới gửi Gemini. Request describe có thể kèm `request_id`; backend echo lại
ID để frontend bỏ response thuộc turn hoặc mode cũ. Cảnh báo vision có priority
100 và phải ngắt audio loại `answer`/`filler`.

`system_audio` trong bootstrap trả URL WAV nếu cache đã có, hoặc `null` để frontend
dùng `/v1/tts` làm fallback. Chạy `scripts/05_build_alert_cache.py` trong môi
trường đã cài VieNeu để tạo cả WAV cảnh báo lẫn WAV giới thiệu/xác nhận.
Notebook hiện tự tạo WAV hệ thống thiếu trong cell inference trước khi mở API.
Frontend dùng cùng các mã này, ưu tiên tải trước WAV để lời chào/câu đệm không
phải chờ TTS. `guide` chạy cảnh báo liên tục; `narration` chỉ chụp ảnh khi có câu hỏi;
Khi đổi từ `guide` sang `narration`, frontend đóng WebSocket vision ngay, hủy
audio cảnh báo đang phát và bỏ mọi response thuộc kênh guide đến muộn. `paused` dừng inference
nhưng vẫn nghe “tiếp tục”. Câu hỏi mô tả cảnh không phải chức năng định tuyến GPS.

## Model stack

| Vai trò | Model | Size | License | Thiết bị |
|---|---|---:|---|---|
| VLM | `Qwen/Qwen3-VL-4B-Instruct` | 8.5 GB | Apache-2.0 | GPU fp16 |
| Depth | `Depth-Anything-V2-Metric-Indoor-Large-hf` | 1.4 GB | Apache-2.0 | GPU fp16 |
| STT | `vinai/PhoWhisper-medium` | 1 GB | BSD-3 | GPU fp16 |
| TTS | `VieNeu-TTS-v3-Turbo` int8 ONNX | 250 MB | Apache-2.0 | CPU |
| Thuyết minh | `gemini-3.5-flash-lite` (API) | — | — | ngoài |

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
| Describe API (Gemini 3.5 Flash Lite) | median 2,11 s | 6/6 request hoàn chỉnh trong smoke test; TTS chạy sau |

### Profile realtime trên Kaggle Dual T4

Notebook Kaggle tách tần số xử lý để độ trễ cảnh báo không bị khóa theo tốc độ
sinh token của Qwen:

- Depth chạy trên mọi frame, mặc định frontend gửi tối đa 5 FPS.
- Frontend chỉ cho phép một frame đang chờ response. Timer kiểm tra camera tối đa
  5 Hz, nhưng chỉ chụp frame tiếp theo sau khi backend trả kết quả; ở backend
  1,5 FPS thì tốc độ gửi thực cũng xấp xỉ 1,5 FPS, không dồn ảnh cũ. Queue một
  phần tử latest-frame-wins ở server là lớp bảo vệ thứ hai.
- Qwen 4B chạy nền trên GPU 0 ở 1 Hz; các frame xen giữa dùng kết quả ngữ nghĩa
  gần nhất, tối đa 2.5 giây. Frame đầu tiên vẫn chờ cả Qwen và depth.
- Bộ xác nhận chỉ tăng streak khi có kết quả Qwen mới. Vật thường cần hai lần
  liên tiếp; nhãn `PERSON` cần ba lần liên tiếp. Một kết quả `CLEAR` mới xóa
  streak, nên cùng một hallucination cache không thể tự biến thành cảnh báo.
- `MOVE_LEFT`/`MOVE_RIGHT` chỉ được phát khi vị trí do Qwen nêu và cột bị chắn
  của depth cùng chỉ về phía đối diện; nếu hai nguồn lệch nhau thì phát
  `SLOW_DOWN`, tránh spam chỉ hướng sai.
- Kết quả trả về có `depth_latency_ms`, `vlm_latency_ms`, `vlm_age_ms` và
  `vlm_fresh`, cùng `queue_wait_ms`, `server_total_ms`, `dropped_frames` để đo
  trực tiếp trên đúng phiên Kaggle.
- URL hỗ trợ `?fps=1`, `?fps=2` hoặc `?fps=5`. Giá trị là số frame mỗi giây,
  không phải mili giây.
- Qwen được giới hạn 128–320 visual token. Ảnh 640×360 thường nằm sẵn trong
  khoảng này, tránh processor tự phóng ảnh lên ngân sách lớn hơn.

Các biến `VLM_REFRESH_HZ`, `VLM_MAX_STALE_S`,
`VLM_MIN_VISUAL_TOKENS` và `VLM_MAX_VISUAL_TOKENS` cho phép benchmark profile
khác. Mốc 5 FPS ở đây là nhịp depth/fusion phản ứng; ngữ nghĩa Qwen vẫn khoảng
1 FPS. Cần lấy p50/p95 từ Kaggle trước khi tuyên bố mốc thực tế.

Dùng nhiều notebook có thể tăng tổng throughput bằng cách chạy nhiều Qwen worker,
nhưng không làm một lần suy luận Qwen nhanh hơn. Quick Tunnel của từng notebook
còn thêm RTT, URL thay đổi theo phiên và yêu cầu sắp thứ tự frame trước khi cập
nhật `FusionSession`. Vì vậy cấu hình một notebook Dual T4 là mặc định; chỉ tách
worker sau khi số đo cho thấy GPU 0 bão hòa và cần ngữ nghĩa mới hơn 1 Hz.

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
├── api_main.py              # FastAPI: /health /v1/bootstrap /v1/intent
│                            # /v1/alerts /v1/stt /v1/tts /v1/describe
│                            # WS /v1/vision và /audio/*
├── frontend/                # source React, voice/camera/audio và test giao thức
├── web/dist/                # frontend đã build để FastAPI/Kaggle phục vụ trực tiếp
├── web/index.html           # trang fallback nếu chưa có web/dist
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
│   ├── run_golden_tests.py  # hồi quy fusion KHÔNG cần GPU — PASS trước mọi deploy
│   └── test_voice_protocol.py # intent/mode voice-first, KHÔNG cần model
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
