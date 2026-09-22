# Trợ lý thị giác cho người khiếm thị

Đây là bản thử nghiệm một **trợ lý thị giác điều khiển bằng giọng nói**, hỗ trợ người khiếm thị nhận biết vật cản và hiểu không gian phía trước khi di chuyển trong nhà.

Người dùng chỉ cần mở ứng dụng trên điện thoại, cấp quyền camera và micro, sau đó có thể sử dụng hai chế độ chính:

- **Dẫn đường:** camera liên tục quan sát phía trước và phát cảnh báo ngắn khi phát hiện vật cản có khả năng chắn lối.
- **Thuyết minh:** người dùng đặt câu hỏi bằng giọng nói như “Phía trước có gì?” và nhận câu trả lời bằng âm thanh.

Ứng dụng được thiết kế theo hướng **ưu tiên âm thanh**, giúp người dùng thực hiện các thao tác chính mà không cần nhìn hoặc chạm nhiều vào màn hình.

> **Lưu ý:** Đây là sản phẩm thử nghiệm trong môi trường trong nhà, chưa phải thiết bị hỗ trợ di chuyển đã được kiểm định an toàn. Ứng dụng không thay thế gậy, chó dẫn đường hoặc người hỗ trợ.

---

## Điểm chính của bản demo

Hệ thống không chỉ nhận diện vật thể trong ảnh mà còn kết hợp **thông tin về khoảng cách và vị trí** để xác định vật nào thực sự có khả năng chắn đường.

Khi phát hiện nguy hiểm, cảnh báo an toàn được ưu tiên phát ngay. Khi người dùng muốn hiểu rõ hơn môi trường xung quanh, họ có thể đặt câu hỏi bằng giọng nói và nhận câu trả lời ngắn gọn bằng tiếng Việt.

Bản demo hiện tập trung vào ba mục tiêu:

- **Nhận biết vật cản phía trước.**
- **Cảnh báo nhanh bằng giọng nói.**
- **Cho phép người dùng hỏi về không gian xung quanh mà không cần nhìn màn hình.**

---

## Trải nghiệm sử dụng

Khi mở ứng dụng lần đầu, người dùng cấp quyền truy cập camera và micro. Sau lời chào, ứng dụng chờ lệnh bằng giọng nói.

### Chế độ dẫn đường

Ở chế độ này, ứng dụng liên tục kiểm tra vùng phía trước và phát các cảnh báo ngắn như:

- “Có vật cản phía trước.”
- “Đi sang trái.”
- “Đi sang phải.”
- “Dừng lại.”

Các cảnh báo quan trọng đã được tạo sẵn thành file âm thanh để có thể phát gần như ngay lập tức mà không cần chờ hệ thống tạo giọng nói.

Nếu đang phát lời thuyết minh mà xuất hiện tình huống nguy hiểm, **cảnh báo an toàn sẽ được ưu tiên và có thể ngắt lời thuyết minh**.

### Chế độ thuyết minh

Người dùng có thể hỏi về cảnh trước mặt, ví dụ:

- “Phía trước có gì?”
- “Bên trái có cửa không?”
- “Có ai đang đứng trước mặt không?”

Ứng dụng chụp một ảnh tại thời điểm người dùng hỏi, gửi ảnh cùng câu hỏi tới **Gemini 3.5 Flash Lite**, sau đó đọc câu trả lời bằng tiếng Việt.

Câu trả lời được giới hạn trong khoảng một đến hai câu để dễ nghe khi người dùng đang di chuyển.

### Điều khiển bằng giọng nói

Các thao tác chính có thể thực hiện mà không cần chạm màn hình:

- **“Dẫn đường”** — bật chế độ cảnh báo vật cản.
- **“Thuyết minh”** — chuyển sang chế độ hỏi về môi trường.
- **“Tạm dừng”** — tạm dừng xử lý camera.
- **“Tiếp tục”** — quay lại chế độ trước đó.
- **“Trợ giúp”** — nghe lại các lệnh có thể sử dụng.

Nếu hệ thống không nghe đủ rõ, ứng dụng sẽ yêu cầu người dùng nói lại thay vì tự suy đoán ý định.

---

## Hệ thống hoạt động như thế nào?

Để quyết định có cần cảnh báo hay không, hệ thống cần trả lời hai câu hỏi:

1. **Trước mặt có vật gì và vật đó nằm ở đâu?**
2. **Vật đó có thực sự ở gần và chắn lối đi không?**

Một mô hình thị giác đảm nhiệm việc nhận biết vật thể và vị trí. Một mô hình ước lượng chiều sâu đánh giá tương đối khoảng cách gần – xa.

Hai nguồn thông tin được kết hợp trước khi hệ thống quyết định phát cảnh báo.

```text
Camera điện thoại
       │
       ▼
Nhận biết vật và vị trí ──┐
                           ├──► Kiểm tra vật có chắn lối ──► Cảnh báo bằng giọng nói
Ước lượng gần và xa ──────┘
```

Cách kết hợp này giúp hạn chế hai loại lỗi thường gặp:

- Nhận diện đúng vật thể nhưng đánh giá sai mức độ nguy hiểm do không biết vật ở gần hay xa.
- Phát hiện một vùng ở gần nhưng không biết đó là người, đồ vật hay cấu trúc của căn phòng.

Ngoài ra, hệ thống chờ kết quả ổn định qua nhiều lần quan sát trước khi phát cảnh báo nhằm giảm các cảnh báo sai do một khung hình bất thường.

Riêng tình huống phát hiện vật ở rất gần sẽ được ưu tiên cảnh báo **“Dừng lại”** ngay.

---

## Những gì bản hiện tại đã làm được

- Giao diện web tối ưu cho điện thoại, chữ lớn và độ tương phản cao.
- Điều khiển các thao tác chính bằng giọng nói.
- Nhận biết vật cản trong môi trường trong nhà.
- Kết hợp nhận diện vật thể với ước lượng chiều sâu trước khi đưa ra cảnh báo.
- Phát cảnh báo an toàn bằng âm thanh có sẵn để giảm độ trễ.
- Thuyết minh cảnh vật bằng Gemini 3.5 Flash Lite.
- Nhận dạng giọng nói và đọc câu trả lời bằng tiếng Việt.
- Ba trạng thái hoạt động rõ ràng: **Dẫn đường**, **Thuyết minh** và **Tạm dừng**.
- Tự ngắt lời thuyết minh khi xuất hiện cảnh báo an toàn.
- Tự thử kết nối lại khi luồng camera bị gián đoạn.
- Có bộ kiểm thử để phát hiện thay đổi làm sai hành vi cảnh báo hoặc hội thoại.
- Có thể chạy trên máy chủ GPU hoặc trong Kaggle Notebook.

---

## Kết quả thử nghiệm hiện tại

Các số liệu dưới đây được dùng để theo dõi tiến độ kỹ thuật của bản demo, **chưa phải kết quả kiểm định sản phẩm trong điều kiện sử dụng thực tế**.

| Hạng mục | Kết quả hiện tại | Ý nghĩa |
|---|---:|---|
| Bộ 27 ảnh trong nhà được gắn nhãn thủ công | Phát hiện đủ 100% vật cản; 75% cảnh trống được giữ im lặng | Kiểm tra nhanh khả năng phát hiện vật cản và hạn chế cảnh báo sai trong tập thử nghiệm nhỏ |
| Bộ kiểm thử 120 ảnh COCO đã thẩm định | Recall 91%; specificity 81% | Phần lớn tình huống có vật cản được phát hiện, đồng thời hệ thống có khả năng giữ im lặng ở nhiều cảnh không cần cảnh báo |
| Thời gian từ lúc gửi ảnh đến khi có quyết định | Trung vị khoảng 953 ms | Trong phép đo hiện tại, một nửa số lượt phản hồi nhanh hơn mốc này và một nửa chậm hơn |
| Phát câu cảnh báo đã lưu sẵn | Khoảng 4 ms | Cảnh báo âm thanh gần như có thể phát ngay sau khi hệ thống ra quyết định |
| Nhận dạng một câu nói ngắn | Khoảng 485 ms | Lệnh giọng nói được xử lý trong thời gian ngắn |
| Gemini trả lời câu hỏi về cảnh | Trung vị khoảng 2,11 giây trong smoke test | Chế độ thuyết minh có độ trễ cao hơn chế độ cảnh báo nên không được dùng cho phản ứng an toàn tức thời |

Mốc 953 ms là kết quả trong môi trường thử nghiệm, không phải thời gian tối đa
hay cam kết cho mọi thiết bị và đường truyền. Để tránh độ trễ tăng dần, ứng dụng
không xếp hàng nhiều ảnh cũ mà chỉ tiếp tục với ảnh mới nhất. Gemini không nằm
trên luồng cảnh báo an toàn; mốc 4 ms chỉ là thời gian lấy file âm thanh đã lưu
sẵn sau khi hệ thống đã đưa ra quyết định.

Các ngưỡng khoảng cách hiện được hiệu chỉnh theo camera và góc đặt đã sử dụng khi thử nghiệm. Khi đổi camera hoặc cách đeo thiết bị, hệ thống cần được hiệu chuẩn lại.

---

## Phạm vi của bản thử nghiệm

Bản hiện tại tập trung vào việc chứng minh trải nghiệm sử dụng trong môi trường trong nhà.

Hệ thống hiện chưa có:

- Chỉ đường theo bản đồ hoặc GPS.
- Nhận biết đầy đủ mọi loại nguy hiểm ngoài đời thực.
- Kiểm định trên số lượng lớn người dùng, môi trường, thiết bị và điều kiện ánh sáng khác nhau.
- Chứng nhận để sử dụng như một thiết bị hỗ trợ di chuyển an toàn.

Chế độ dẫn đường xử lý hình ảnh trên máy chủ riêng.

Chế độ thuyết minh cần kết nối Internet để gọi Gemini. Nếu không cấu hình Gemini, chức năng cảnh báo vật cản vẫn có thể hoạt động nhưng phần hỏi đáp về cảnh sẽ không sử dụng được.

---

## Hướng phát triển

Trong các phiên bản tiếp theo, nhóm mong muốn:

- Tích hợp **bản đồ và định vị** để hỗ trợ cả nhận biết môi trường lẫn định hướng tới điểm đến.
- Mở rộng kiểm thử trên nhiều không gian, thiết bị và điều kiện ánh sáng hơn.
- Cải thiện tốc độ xử lý và độ ổn định của hệ thống.
- Tối ưu khả năng nhận biết các tình huống nguy hiểm phức tạp.
- Thử nghiệm với người dùng thực tế để đánh giá mức độ hữu ích và khả năng sử dụng.

Mục tiêu dài hạn là tạo ra một trải nghiệm di chuyển **liền mạch**, trong đó người dùng có thể nhận cảnh báo, hiểu môi trường và định hướng đường đi thông qua cùng một giao diện giọng nói.

---

# Hướng dẫn chạy bản demo

## Chạy bằng Docker

### Yêu cầu

- Máy Linux có GPU NVIDIA với khoảng **15 GB VRAM**.
- NVIDIA Container Toolkit.
- Một Gemini API key nếu muốn sử dụng chế độ thuyết minh.

### Cài đặt

```bash
git clone https://github.com/Cdshf01230/ai-assistant.git
cd ai-assistant

cp .env.example .env

# Điền GEMINI_API_KEY trong file .env
docker compose up -d --build
docker compose logs -f
```

Khi máy chủ đã sẵn sàng, mở trên điện thoại:

```text
https://<IP-máy-chủ>:8443/
```

Camera và micro trên trình duyệt cần HTTPS.

Hướng dẫn kết nối mạng và tunnel nằm trong [`deploy/README.md`](deploy/README.md).

---

## Chạy trên Kaggle

Notebook [`kaggle_ai_assistant.ipynb`](kaggle_ai_assistant.ipynb) là bản demo dành cho môi trường Kaggle có hai GPU T4.

Notebook sẽ:

1. Nạp các model cần thiết.
2. Khởi động API.
3. Tạo một đường dẫn HTTPS để truy cập từ điện thoại.

Trước khi chạy:

1. Thêm mã nguồn dự án vào Kaggle dưới dạng **Private Dataset**.
2. Chọn accelerator **GPU T4 x2**.
3. Bật Internet.
4. Thêm secret `GEMINI_API_KEY`.
5. Chạy toàn bộ notebook.
6. Mở đường dẫn được in ở cell cuối.

---

# Kiểm tra trước khi triển khai

## Kiểm tra logic cảnh báo

Không cần GPU:

```bash
python tests/run_golden_tests.py
```

## Kiểm tra backend và giao thức giọng nói

```bash
python -m pytest -q
```

## Kiểm tra và build giao diện

```bash
cd frontend

pnpm install
pnpm test
pnpm build
```

Bản build được đặt trong:

```text
web/dist/
```

để backend có thể phục vụ trực tiếp.

---

# Thành phần chính

| Thành phần | Vai trò |
|---|---|
| `frontend/` | Giao diện React sử dụng trên điện thoại |
| `web/dist/` | Bản giao diện đã build sẵn |
| `api_main.py` | API kết nối camera, micro, model và giao diện |
| `scripts/fusion.py` | Kết hợp nhận diện vật thể với thông tin gần – xa |
| `scripts/mvp_config.py` | Cấu hình model, câu nói và ngưỡng cảnh báo |
| `assets/audio/` | Các câu cảnh báo được tạo sẵn |
| `tests/` | Kiểm thử hành vi cảnh báo và hội thoại |
| `benchmarks/` | Công cụ đo độ chính xác và độ trễ |
| `deploy/` | Hướng dẫn triển khai và truy cập từ điện thoại |

---

# Các model đang sử dụng

| Nhiệm vụ | Model | Nơi chạy |
|---|---|---|
| Nhận biết vật và vị trí | Qwen3-VL-4B-Instruct | GPU của máy chủ |
| Ước lượng gần / xa | Depth Anything V2 Indoor Large | GPU của máy chủ |
| Nhận dạng tiếng Việt | PhoWhisper Medium | GPU của máy chủ |
| Đọc câu trả lời | VieNeu-TTS v3 Turbo | CPU của máy chủ |
| Trả lời câu hỏi về cảnh | Gemini 3.5 Flash Lite | Google Gemini API |

---

# Bảo mật và cấu hình

Mã nguồn không chứa:

- Model weights.
- API key.
- Chứng chỉ HTTPS.
- Thông tin bí mật dùng trong quá trình triển khai.

Các model cục bộ được tải riêng khi cài đặt.

Thông tin bí mật được cấu hình thông qua:

```text
.env
```

hoặc:

```text
Kaggle Secrets
```

---

## Trạng thái dự án

Đây là một **prototype phục vụ mục đích trình diễn và thử nghiệm kỹ thuật**.

Bản hiện tại đã chứng minh được luồng trải nghiệm chính:

```text
Camera
  ↓
Hiểu môi trường
  ↓
Đánh giá vật cản
  ↓
Cảnh báo bằng giọng nói
```

song song với:

```text
Câu hỏi bằng giọng nói
  ↓
Ảnh tại thời điểm hỏi
  ↓
Hiểu cảnh
  ↓
Câu trả lời bằng tiếng Việt
```

Nhóm sẽ tiếp tục cải thiện độ chính xác, tốc độ, phạm vi thử nghiệm và khả năng hỗ trợ di chuyển trước khi xem xét sử dụng hệ thống trong các tình huống thực tế.
