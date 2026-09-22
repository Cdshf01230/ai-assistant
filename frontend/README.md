# Frontend Trợ lý thị giác

React + TypeScript + Vite, phát triển từ thiết kế Figma Make trong `Implement this idea.zip`.
Mã nguồn chính nằm ở `frontend/`. Bản production nằm ở `web/dist/`;
`web/index.html` cũ chỉ được dùng nếu chưa có bản build.

## Build và chạy

Yêu cầu Node.js 22.12+ hoặc 24, Corepack/pnpm.

```powershell
cd frontend
corepack pnpm install --frozen-lockfile
corepack pnpm test
corepack pnpm run build
```

Build chạy TypeScript trước Vite. `web/dist/ui-assets/` chứa JS/CSS.
FastAPI độc lập và cell API Kaggle đều phục vụ `/` và `/ui-assets/*`.
Kaggle không cần Node.js: upload ZIP chứa sẵn `web/dist` rồi Run All notebook mới.

Development: `corepack pnpm dev` proxy `/v1`, `/audio`, `/health`
đến `http://127.0.0.1:8000`. Mở cùng origin với backend khi triển khai.
Không host riêng trên Figma/Lovable origin mà vẫn kỳ vọng relative API hoạt động.

## Luồng giọng nói

1. Một lần chạm Bắt đầu: mở AudioContext trong user gesture, xin camera/micro.
2. Bootstrap API, tải WAV cảnh báo/hệ thống, khởi động AudioWorklet.
3. Video gắn stream sau khi màn hình chính mount, phát lời giới thiệu, nghe lệnh.
   Khi vào Dẫn đường hoặc Thuyết minh, camera hiện thành preview chính; micro vẫn nghe lệnh đổi mode.
4. STT gửi `current_mode`, `resume_mode`, `turn_id`; backend quyết định intent.
5. Chỉ guide gửi frame liên tục tối đa `?fps=1..5`. Client không chờ response trước
   khi gửi frame kế tiếp. Chỉ có một frame được phép đang chờ kết quả; nhịp kiểm
   tra camera vẫn tối đa 5 Hz nhưng frame tiếp theo chỉ được chụp sau response,
   nên tốc độ thực tự bám theo inference và không tạo hàng đợi ảnh cũ. Ngoài ra
   dừng gửi nếu bộ đệm WebSocket vượt 1 MiB.
   Server chỉ giữ frame mới nhất trong queue một phần tử; frame cũ chưa infer sẽ bị bỏ.
   Narration giữ camera preview nhưng chỉ chụp một frame khi người dùng đặt câu hỏi.
6. Khi hỏi mô tả, gửi Gemini ngay; câu đệm chạy độc lập. Câu trả lời thay câu đệm ngay khi sẵn sàng.
7. Pause tắt camera inference nhưng micro tiếp tục nghe để nhận “tiếp tục”.
8. Audio đang phát đóng VAD. Chỉ mở lại sau khi queue hết và trễ 300 ms.
9. Emergency chỉ hoạt động trong guide và không hủy STT đang nhận lệnh đổi chế độ.
   Đổi sang narration đóng WebSocket cảnh báo và xóa cảnh báo cũ ngay lập tức.
10. Trang vào background dừng gửi frame/thu câu nói; trở lại foreground tiếp tục khi trình duyệt cho phép.

Lỗi `video.play()` loại `NotAllowedError` trên iOS/WebView không được coi là từ chối
quyền camera. Thẻ video giữ `muted`, `playsInline`, `autoplay` và thử lại khi metadata tới.

`AbortController` hủy chờ HTTP phía client; không bảo đảm dừng GPU/Gemini
đã chạy phía server. ID và epoch vẫn bảo đảm kết quả cũ không được phát. Frontend
cũng tương thích với phản hồi STT cũ thiếu `turn_id`: transcript được gửi qua `/v1/intent`
thay vì bị bỏ qua.

## Tổ chức mã

- `src/App.tsx`: giao diện, trạng thái và nút dự phòng.
- `src/hooks/useAssistant.ts`: điều phối mode, turn, âm thanh, camera và lỗi.
- `src/hooks/useVoiceCapture.ts`: AudioWorklet thu PCM; không dùng ScriptProcessor.
- `src/lib/vad.ts`: pre-roll 300 ms, minimum voiced duration 180 ms,
  silence 800 ms, max utterance 8 s, WAV khai báo sample rate thực tế.
- `src/lib/audio.ts`: hàng đợi AudioContext, ưu tiên số lớn hơn trước.
- `src/hooks/useVisionWebSocket.ts`: một frame chờ, watchdog 15 s, reconnect 1.5 s.
- `src/api/index.ts`: API cùng origin, token query + session cookie; không log token.
- `src/components/`: các thành phần kế thừa và chỉnh từ Figma.

Không dùng Web Speech Recognition hay model STT dịch vụ trình duyệt. WAV gửi PhoWhisper.
VAD hiện dùng ngưỡng năng lượng RMS: không phải bộ phân biệt giọng nói với tiếng ồn.
Phải kiểm tra mức ngưỡng trên điện thoại/môi trường sử dụng thực tế.

## Thiết kế và accessibility

Giữ nền navy và điểm nhấn xanh/vàng của Figma. Chỉnh:
- Chữ tối trên nút màu sáng để tăng tương phản.
- Cảnh báo chữ lớn, nền đặc; câu trả lời có khung đọc riêng.
- Nút chính 56–64 px, focus rõ, tiếng Việt, live region cảnh báo không lồng nhau.
- Nội dung được cuộn thay vì cắt mất khi zoom/màn hình thấp; bố cục ngang hai cột.
- Reduced motion tắt toàn bộ chuyển động; diagnostics không vào live region.

Google Fonts dùng cho Be Vietnam Pro, có system-font fallback. Token trang không gửi
qua Referer nhờ `referrer=no-referrer`.

## Đã kiểm tra và giới hạn

Vitest kiểm tra hàng đợi audio, pre-roll/WAV, pause/resume, camera mount,
STT trùng/độ tin cậy thấp, filler trước Gemini, loại response cũ và WebSocket treo.
Python `tests/test_backend_contract.py` dùng FastAPI thật cho cả server và cell notebook;
STT/Gemini/TTS được giả lập, không đo chất lượng model hoặc độ trễ GPU.

Chẩn đoán hiển thị FPS response thực tế, latency depth/VLM, tổng thời gian server,
thời gian chờ queue và tổng frame cũ đã bỏ. Nếu `Chờ queue` gần 0 nhưng FPS vẫn thấp,
GPU depth là giới hạn; tăng bất đồng bộ sẽ không tăng thêm throughput.

Chưa xác nhận camera/micro/loa trên điện thoại thật. Autoplay, AudioWorklet và
quyền micro cần HTTPS (localhost được phép để phát triển). Đây là half-duplex:
khi trợ lý nói thì chưa hỗ trợ người dùng chen lời bằng giọng nói; nút mode/pause
vẫn có thể ngắt. Trình duyệt có thể đình chỉ trang khi tắt màn hình.
